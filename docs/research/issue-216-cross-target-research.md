# Cross-target decoding of neural video: research report (issue #216)

Status as of 2026-11-05. Roadmap item C14. Branch
`fix/cross-gpu-entropy-determinism` on `ef4dcfd8` (`main` `163441e9`); all work
described here is uncommitted.

This report collects what is known, measured, built and ruled out, so the
research can continue from it. Detailed gate tables live in the evidence files:

- [Issue #216 fix evidence](../../evidence/issue-216-cross-hardware-fix-20261104.md)
  (gates, quality study, speed, GEMM work)
- [Research probe](../../evidence/issue-216-research-probe-20261105.md)
  (temporal-prior ablations, 8-bit screen, literature)
- [Cross-target decode design](../cross-target-decode.md) (the shipped profile)

---

## 1. Summary

| Question | Answer |
|---|---|
| Why do DCVC-RT streams fail across GPUs or providers? | Entropy table choices depend on FP16 network outputs; tiny arithmetic differences flip choices near table boundaries and desynchronize rANS. P-frames depend on nearly the whole decoder through the temporal prior. |
| What happens now on a mismatch? | The decoder fails cleanly with `malformed_bitstream` instead of writing garbage. Native streams are byte-identical to `main`. |
| Can a stream decode bit-identically elsewhere? | Yes, with `--numerics nvcr.det-int.v1`: TensorRT (RTX 4070) and ONNX Runtime CPU decode each other's I/P streams bit-identically, verified over 80 frames across a reference reset. |
| Quality cost? | None measurable: BD-rate -0.089% against FP32 (today's FP16 TensorRT path +0.064%), seven sequences, 96-frame GOPs. Stream overhead about 130 bytes per I-frame. |
| Speed cost? | Large. 1080p decode 18.4 fps exact against 48.7 fps native; encode 30.1 against 64.5 fps (RTX 4070, diagnostic). |
| Can engineering close the gap? | Not on this GPU: measured bounds leave about 10-20% for kernel tuning. The exact path is memory-bound and every 16-bit multiply costs four INT8 products. |
| Can a quality compromise close it without retraining? | No: the best no-training 8-bit compromise saves 6.5% of P-frame time. |
| What could close it? | Codec changes that need training: an entropy model conditioned only on exactly known values (latent prior, as in DCVC-HEM), scale indices transmitted through the hyperprior (as in MLVC), or mixed-precision quantization-aware training (as in JOMP). None is proven quality-neutral for DCVC-RT. |
| Desktop GPU encode, Snapdragon display? | Not possible with DCVC-RT today (native fails, exact is refused by QNN, and DCVC-RT on QNN is verified only to 360p). MLVC cannot desynchronize by construction but runs only through ONNX Runtime in NVCR. |

---

## 2. The problem

### 2.1 Mechanism

DCVC-RT codes every latent symbol with one of 128 Gaussian tables chosen from a
predicted scale:

```text
index = floor((ln(clamp(scale, 0.11, 16)) - ln(0.11)) * 25.516)
```

The scale is a network output. Different GPUs build different TensorRT plans
(kernels, tiling, FP16 accumulation order), and different providers (TensorRT,
ONNX Runtime CPU, QNN) use different arithmetic altogether. Their scales differ
by a few FP16 rounding steps. A scale within that distance of a table boundary
selects the neighbouring table on the decoder, and the rANS decoder reads
garbage from that symbol on. The issue #216 report (RTX 4070 to RTX 5060) shows
identical hyper-latents, first divergence in the I-frame scales, six flipped
indexes, then garbage.

Any deterministic map from a continuous value to a discrete index has
boundaries, so more precision only makes flips rarer. Two families of fixes
exist: transmit the fragile decisions, or compute them identically everywhere.

### 2.2 What must match (the closure)

| Frame | Stages whose output reaches an entropy decision or decoder state | 1080p GMAC |
|---|---|---:|
| I | `i_hyper_synthesis` (hyper decoder, prior fusion, reduction), `i_spatial_prior_1..3` | 281 |
| I | `i_synthesis`, because the decoded I-frame is the first P-frame's reference | 476 |
| P | `p_reference_feature` (feature adaptor, feature extractor) | 105 |
| P | `p_reference_frame` (after an I-frame and at reference resets) | 122 |
| P | `p_prior` (hyper decoder 2.6, temporal prior 6.4, fusion 30.2) | 39 |
| P | `p_spatial_prior` | 22 |
| P | `p_synthesis` (decoder 67.6, reconstruction 112.0) | 180 |

Encoder-only stages (`*_analysis`, `*_hyper_analysis`) do not need to match.
A P-frame's exact work is about 346 GMAC at 1080p; the entropy-parameter
networks alone are about 61 GMAC. MAC counts come from hooks on the pinned
PyTorch model at 1088x1920.

---

## 3. What is built

### 3.1 Clean failure (N0)

- Bounded rANS decoding in `third_party/dcvc_rans` (two upstream memory-safety
  defects fixed, reproduced under ASan first), `RansCodec::finish_decode()`,
  and an end-of-frame check in both DCVC-RT backends: every frame must end at
  the encoder's initial rANS state with every byte consumed.
- Result: a mismatched decode stops with `malformed_bitstream` ("DCVC-RT
  entropy decoding desynchronized") and writes no wrong frames. Before, it
  decoded silently at 13-20 dB.

### 3.2 Exact numerics profile

| Piece | Where | What it does |
|---|---|---|
| Codec declaration | `src/dcvcrt/numerics.*` | DCVC-RT lists ten normative stages and requires `nvcr.det-int.v1` |
| Profile and CPU reference | `src/deterministic/program.*`, `det_math.hpp` | Exact integer dense convolutions (per-output-channel weight scale, per-position power-of-two activation scale, 8 or 16 bits per layer, exact sums); det-fp32 for everything else (fixed operation order, no contraction, exactly rounded `exp`) |
| CUDA executor | `src/deterministic/cuda_executor.*` | INT8 tensor-core GEMM (16-bit values as two INT8 limbs, exact recombination), fused WSiLU / chunk-add / residual epilogues; bit-identical to the CPU reference |
| Programs | `scripts/backends/dcvcrt/export_deterministic_program.py` | Lowers the shipped ONNX stages to a small op set (`conv`, `mul`, `add`, `mul_scalar`, `scale_channels`, `sigmoid`, `clamp`, `slice_channels`, `concat`, `crop`, `depth_to_space`, `space_to_depth`, `im2col`); dense spatial convs become `im2col` plus a pointwise conv |
| Width map | `configs/numerics/dcvcrt-det-int-v1.json` | 8-bit layers only in the I-frame entropy model (from the research branch study); 16-bit everywhere else |
| Providers | `src/provider/tensorrt`, `src/provider/onnxruntime` | TensorRT runs programs through the CUDA executor (SM 8.0+); ONNX Runtime CPU through the CPU reference; ONNX Runtime CUDA and QNN refuse with `incompatible_precision` |
| Stream identity | `include/nvcr/bitstream/numerics_identity.hpp` | Random-access units are NVAU v2 with a required 48-byte section (profile id plus SHA-256 over stage ids and program digests); P units stay NVAU v1 and inherit it |
| Artifacts | `export_normative_programs.sh`, `prepare_artifacts.sh`, `build_tensorrt.sh` | Programs ship in each bundle's `normative/` directory |
| CLI | `cli/main.cpp` | `--numerics nvcr.det-int.v1` on encode; decode reads the profile from the stream |

Separation of concerns follows `AGENTS.md`: the codec declares, the runtime
defines the profile, providers implement it, nothing falls back silently, and
matrix cells come only from conformance results.

### 3.3 Verified results

| Gate | Result |
|---|---|
| Programs against pinned PyTorch modules | All ten agree to FP32 rounding (max relative error 1.5e-6) |
| CUDA against CPU reference | Bit-identical on every op, every width combination, all ten stage programs |
| Cross-provider, FourPeople QCIF, 80 frames, GOP 96, reset at frame 65 | TensorRT encode decodes to `720c2f56…6552` on TensorRT and ONNX Runtime CPU; ONNX Runtime CPU encode decodes to `b9994fec…e76a` on both |
| Refusals | ONNX Runtime CUDA, and a bundle without programs, refuse exact streams before decoding |
| Native parity with `main` | Akiyo QCIF 65 frames: `.nvcr` `afbcbd65…f631`, YUV `f8512e55…fdb3`, identical |
| Suites | ASan/UBSan 31/31; Release 52/54 (two pre-existing failures, identical on `main`) |

### 3.4 Quality

`scripts/backends/dcvcrt/integer_gop_study.py`, every decoder-side dense conv
emulated, seven sequences (BasketballDrive, Kimono, ReadySteadyGo, Jockey,
HoneyBee 1080p; FourPeople, Johnny 720p), 96 frames, QPs 0/21/42/63, BD-rate
against FP32:

| Variant | BD-rate YUV |
|---|---:|
| FP16 (stand-in for TensorRT FP16) | +0.064% |
| All 16-bit | -0.057% |
| Shipped map | -0.089% |
| Shipped map with FP16 tensor storage (rejected) | +0.164% |

### 3.5 Speed

1080p BasketballDrive, 97 frames, GOP 32, QP 32, 10 warm-up frames, ten
independent processes per condition, interleaved (diagnostic, not the
`docs/performance.md` protocol; harness `scripts/backends/dcvcrt/exact_fps_ab.py`):

| Operation | Native codec FPS | Exact codec FPS | Exact / native |
|---|---:|---:|---:|
| Decode | 48.65 | 18.36 | 0.377 |
| Encode | 64.45 | 30.08 | 0.467 |

Per stage, device time (TensorRT from `nvcr decode --profile`, exact from
`nvcr_deterministic_executor_tests --bench-stages`):

| Stage | TensorRT FP16 | Exact | Ratio |
|---|---:|---:|---:|
| `i_hyper_synthesis` | 2.66 ms | 6.27 ms | 2.4 |
| `i_spatial_prior_1..3` | 2.85 ms each | 5.37-6.38 ms | 1.9-2.2 |
| `i_synthesis` | 21.45 ms | 59.26 ms | 2.8 |
| `p_reference_frame` | 6.44 ms | 18.10 ms | 2.8 |
| `p_reference_feature` | 5.27 ms | 15.50 ms | 2.9 |
| `p_prior` | 1.53 ms | 5.22 ms | 3.4 |
| `p_spatial_prior` | 0.73 ms | 2.44 ms | 3.3 |
| `p_synthesis` | 8.10 ms | 24.75 ms | 3.1 |
| P-frame total | 15.6 ms | 47.9 ms | 3.1 |

The research branch's "threshold-0.02 map, about 23 ms" figure is the I-frame
entropy stages alone (four stages, TensorRT 11.0 ms); today those measure
23.6 ms against 11.2 ms. The scope grew to the whole decoder side when P-frames
became exact.

---

## 4. Engineering investigation

### 4.1 Changes kept (scheduling only, exact bits unchanged)

| Change | Effect |
|---|---|
| Dense spatial convs lowered to `im2col` plus integer pointwise conv | p_prior 10.7 to 5.3 ms, p_synthesis 41.7 to 27.6 ms; decode 13.0 to 17.1 fps |
| GEMM grid order (row blocks fastest, activation tiles stay in L2) | 368x736 layer 2.32 to 1.02 ms; I-frame synthesis 77 to 59 ms |
| Quantizer block 16 positions x 16 lanes | 4-5% on wide inputs |

### 4.2 Tried and rejected

| Attempt | Result |
|---|---|
| Transmitted entropy corrections (research branch) | Decodes only at margin 0.5, 37-40 times the payload in side data, reconstructions still differ |
| 8-bit weights per P-frame module (W8A16) | +0.10% to +4.2% BD-rate per module |
| FP16 tensor storage | -9% P-frame time, +0.164% BD-rate |
| GEMM at one block per SM, deeper K per stage | 6-25% slower |
| Quantizer staging in shared memory; reverse-order second pass | Slower; no change |
| Producer-side activation maxima (bound: quantizer max pass removed) | -1.5% at most |
| Near-free GEMM epilogue (bound) | -10% to -14% on K = 256 layers |
| Whole-DepthConvBlock fusion | Not built: ~100 KiB shared memory per SM fits all channels of only 16-32 positions, so every tile re-streams ~1 MB of weight limbs from L2 with little tensor-core reuse |

### 4.3 Performance model

- GEMM throughput depends on K, not output size: K = 256 runs at 38-42 T INT8
  products/s, K = 1024 at 60-64 (RTX 4070 INT8 dense peak about 116). A fit of
  time = a + b K per tile puts the fixed per-tile cost near 250 K-steps, about
  half the time of DCVC-RT's K = 256 layers.
- A 256-channel DepthConvBlock at 1/8 resolution moves about 770 MB per pass
  (float32 outputs, two quantizer reads, limb traffic): about 1.7 ms at DRAM
  bandwidth, against well under a millisecond of integer math at peak.
- Four INT8 products per 16-bit multiply set a compute floor near 17 ms per
  1080p P-frame even with perfect kernels.
- Realistic remaining engineering headroom: 10-20% (epilogue and pipeline),
  taking decode from about 18 to 20-22 fps.

### 4.4 Tooling for kernel work

- `nvcr_deterministic_executor_tests --bench-gemm OUT IN H W [WBITS ABITS [ITER]]`:
  one integer conv (quantizer plus GEMM).
- `nvcr_deterministic_executor_tests --bench-stages PROGRAM_DIR BH BW [ITER]`:
  every stage at (64 BH) x (64 BW).
- `nsys profile` plus `nsys stats -r cuda_gpu_kern_sum` for kernel splits.
  Nsight Compute needs GPU performance-counter permission, which this user
  lacks (`ERR_NVGPUCTRPERM`).
- Any kernel change must keep `nvcr_deterministic_executor` passing and should
  pass a repeated-run determinism check (20+ identical runs); a test-only race
  was found this way.

---

## 5. Research probe (inference only)

### 5.1 Temporal-prior ablations

Replacing the temporal prior that feeds the prior fusion network (pinned model,
no retraining):

- Zeroed: also changes the quantization step the fusion network emits, so the
  codec lands at a different operating point (up to 50 times the bits); not
  interpretable.
- Per-channel spatial mean (keeps global statistics, drops spatial detail):
  +71% to +270% bits at 0.5-3.7 dB lower PSNR.

DCVC-RT depends heavily on a spatially detailed temporal prior. It cannot be
removed or approximated at inference time. Both ablations are distribution
shifts, so they bound what the existing weights can do, not what a retrained
model could do.

### 5.2 8-bit screens

| Module | W8A16 change | W8A8 change |
|---|---:|---:|
| `P.hyper_decoder` (2.6 GMAC) | +0.10% | +0.02% |
| `P.temporal_prior_encoder` (6.4) | +0.17% | +0.10% |
| `I.dec` (476, I-frames) | +0.30% | +3.09% |
| `P.recon_generation_net` (112) | +0.52% | +7.25% |
| `P.y_spatial_prior` (21.7) | +0.88% | +2.29% |
| `P.feature_extractor` (103) | +1.12% | +5.89% |
| `P.y_prior_fusion` (30.2) | +1.15% | +33.0% |
| `P.feature_adaptor_i`, `_p` | +1.52% | +5.12% |
| `P.decoder` (68) | +4.22% | +7.73% |

Screen: Kimono, BasketballDrive 1080p, FourPeople, Johnny 720p, 40 frames;
changes against the shipped map on the same set; noise about 0.1%.

### 5.3 Small-compromise test

The cheapest no-training compromise (W8A16 on reconstruction, I-frame
synthesis, spatial prior; W8A8 on hyper decoder and temporal prior) cuts
P-frame exact stages from 47.9 to 44.8 ms (-6.5%) and I-frame synthesis from
59.3 to 50.7 ms. Width compromises are not a speed lever; its BD-rate was not
measured.

---

## 6. Literature

| Work | Relevance |
|---|---|
| DCVC-RT (CVPR 2025), appendix on model integerization | Upstream's own int16 model for cross-device consistency: about 3% BD-rate loss, 28.3/20.9 fps against 125.2/112.8 fps FP16 at 1080p on an A100 (4-5 times slower). NVCR's exact profile: 2.6 times slower, no BD-rate loss. |
| JPEG AI (variable-rate overview) | All entropy-related calculations in integer precision, built on Koyuncu et al.'s entropy-network quantization (0.3% BD-rate) |
| DCVC-HEM (ACM MM 2022) | Latent prior: entropy model conditioned on the previous frame's decoded latent; removing it costs 10.1%, alone it saves 19.3%. Kept the feature-domain temporal prior too. |
| MLVC (ECCV 2026) | Scale parameters transmitted through the hyperprior and looked up in a fixed codebook; long-term references bound reconstruction drift. A review notes the hyper-latent's factorized model must still be bit-exact. |
| JOMP (arXiv 2606.13110) | Mixed-precision integerization for neural video codecs: bit-exact, cross-platform, DCVC-FM-level rate-distortion with 87.6% fewer bit operations |
| Integer-Centric Neural Video Compression (ICLR 2026) | Integer codec trained from scratch; argues post-hoc quantization loses significant rate |
| Cross-Platform Neural Video Coding: A Case Study (SSF) | Static post-training quantization of the hyperprior path without retraining; about 9-10% BD-rate on the same machine (secondary sources) |
| Quantized decoder for deterministic reconstruction (arXiv 2312.11209) | 16-bit weights and activations for the whole decoder give exact reconstruction with small loss |
| MobileNVC (WACV 2024) | Real-time 1080p int8 on a phone; its int8 model trails its float model substantially |
| Codebook-based cross-platform codec (arXiv 2310.10292) | No entropy model, no drift; compression not convincingly competitive |
| Opportunistic conditional entropy coding (arXiv 2609.21816) | Dual-mode entropy model switching between a previous decoded latent and the hyperprior |
| microsoft/DCVC issue #137 | DCVC-RT training code requested, apparently not public; OpenDCVCs (arXiv 2508.04491) publishes training implementations of earlier DCVC models |

---

## 7. Cross-device scenario: desktop GPU encode, phone display

| Codec, encoder to Snapdragon decoder | Decoding | Picture |
|---|---|---|
| DCVC-RT native, TensorRT to QNN | Fails cleanly (expected; matrix cell `untested`) | none |
| DCVC-RT exact, TensorRT to QNN | Refused before decoding (QNN does not implement the profile) | none |
| MLVC, ONNX Runtime CPU to QNN (C10 evidence) | Cannot desynchronize | Small drift, recovers at long-term reference |
| MLVC, TensorRT to QNN | Not available (NVCR runs MLVC only through ONNX Runtime) | — |

Why MLVC cannot desynchronize, from NVCR's implementation (`src/mlvc/entropy.cpp`,
`host_ops.cpp`): hyper-latent symbols use fixed probability tables from the
bundle selected by the transmitted Q index; every latent's scale index is
`trunc(clamp(|z|, 0, levels - 1))` from those decoded integers; latents are
coded zero-mean. No network output enters entropy decoding, so the encoder's
GPU and arithmetic cannot matter. Reconstructions still differ by device and
the difference accumulates over P-frames until a long-term reference.

DCVC-RT on QNN is verified only to 360p (4.9 fps decode on the S25 NPU), and a
QNN implementation of `nvcr.det-int.v1` would need custom exact Hexagon kernels
for about 346 GMAC per 1080p P-frame; not realistic.

---

## 8. Option space

| Option | Quality | Expected 1080p decode (RTX 4070) | Effort | `AGENTS.md` implications |
|---|---|---|---|---|
| A. Ship as is (exact opt-in) | No loss | 18.4 fps exact, 48.7 native | None | Compliant |
| B. Kernel tuning (epilogue, pipeline) | No loss | 20-22 fps (estimate) | Days | Provider-internal; fusion must key on op patterns, never stage names |
| C. Native display reconstruction | No loss; output not bit-identical across targets (±1) | About 23 fps (estimate) | Days | New codec profile id; output cells cannot be `exact`; reference frames before resets stay exact |
| D. Latent-conditioned entropy model (retrain) | Unknown | About 35 fps (estimate) | Weeks to months | New model profile, pinned reference, golden tests, new roadmap item |
| E. Scale indices through the hyperprior (retrain) | Unknown | Near native | Months | As D |
| F. Mixed-precision QAT, whole decoder integer (retrain) | Unknown for DCVC-RT; JOMP shows near-lossless for DCVC-FM | 1.5-2 times today's exact speed (estimate) | Months | As D |
| G. Use MLVC for cross-device streams | MLVC's own RD (not compared to DCVC-RT yet) | Not measured on desktop | MLVC on TensorRT would be new provider work | MLVC/TensorRT is a new roadmap item |

Every retrained option needs a DCVC-RT training pipeline first; none is
public.

---

## 9. Proposed next experiments

### 9.1 Distillation probe for a latent-conditioned temporal prior (option D, cheap first step)

Goal: measure the rate cost of conditioning the entropy model on the previous
frame's decoded latent instead of the decoded feature, before any end-to-end
training.

1. Data: run the pinned FP32 model on sequences outside the seven-sequence test
   set (for example Cactus and BQTerrace, `/home/oelghati/DCVC/datasets/misc`),
   recording per P-frame the previous decoded latent `y_hat[t-1]`, the current
   hyper-latent `z_hat[t]`, and the target temporal-prior output
   (`temporal_prior_encoder(ctx_t)`, 256 channels at latent resolution).
2. Model: a small network T(`y_hat[t-1]`) producing the same tensor, about
   2-5 GMAC at 1080p.
3. Loss: MSE to the target, then a rate-aware stage minimizing the Gaussian
   bits of `y_hat[t]` under the frozen fusion and spatial-prior networks.
4. Evaluation: add a study variant that substitutes T for the temporal prior in
   both encoder and decoder; BD-rate on the seven-sequence set.
5. Gate: BD-rate within the accepted compromise (for example 1-2%) justifies
   end-to-end fine-tuning and a training-pipeline effort.

What it does not show: end-to-end retraining can only do better; cross-target
reconstruction drift must be measured separately (the entropy-correction study
saw 52.5-57.1 dB against the encoder's decode over 33 frames).

### 9.2 MLVC against DCVC-RT (option G)

Same HD sequences: MLVC (ONNX Runtime) and DCVC-RT (TensorRT native) BD-rate,
plus diagnostic FPS. If MLVC's quality is close enough, it is an immediate
cross-device answer.

### 9.3 RTX 5060 conformance (issue #216's own pair)

On an RTX 5060 with the same `normative/` programs: `nvcr_deterministic_executor`,
then decode a TensorRT RTX 4070 exact stream and compare hashes. Native RTX
4070 streams should fail cleanly there. This is the only way to change that
matrix cell.

### 9.4 Kernel tuning (option B)

Epilogue cost on K = 256 layers (exact int64 recombination and conversion),
then pipeline fill; target the 10-20% bound. Gate each change with the
executor conformance test, the determinism stress check and unchanged stream
hashes.

---

## 10. Reproduction

Environment: RTX 4070 (SM 8.9), driver 580.178.04, CUDA 12.8, TensorRT 10.9.0,
ONNX Runtime 1.26.0, GCC 13.3. Pinned DCVC-RT `1feb52a5` at
`/home/oelghati/DCVC-RT-nvcr-1feb`; study Python at
`/home/oelghati/DCVC-RT/src/venv/bin/python3` (PyTorch 2.9.1+cu126). Datasets
under `/home/oelghati/datasets/{hd,720p,qcif}`.

```bash
# Programs for a bundle
PYTHON=/home/oelghati/DCVC-RT/src/venv/bin/python3 \
  scripts/backends/dcvcrt/export_normative_programs.sh build/models/dcvcrt build/models/dcvcrt/normative
```

```bash
# Programs against the pinned PyTorch modules
/home/oelghati/DCVC-RT/src/venv/bin/python3 scripts/backends/dcvcrt/check_deterministic_programs.py \
  --dcvcrt-root /home/oelghati/DCVC-RT-nvcr-1feb --program-dir build/models/dcvcrt/normative \
  i_hyper_synthesis i_spatial_prior_1 i_synthesis p_reference_frame p_reference_feature p_prior p_spatial_prior p_synthesis
```

```bash
# Quality study (variants: fp32, fp16, wNaM, map=PATH, ...@fp16, fp32@temporal=zero|mean)
/home/oelghati/DCVC-RT/src/venv/bin/python3 scripts/backends/dcvcrt/integer_gop_study.py \
  --dcvcrt-root /home/oelghati/DCVC-RT-nvcr-1feb \
  --sequence /home/oelghati/datasets/hd/Kimono_1920x1080_24.yuv:1920x1080 \
  --frames 96 --variants fp32 map=configs/numerics/dcvcrt-det-int-v1.json --output study.json
```

```bash
# Layer names for width maps (I.<module> and P.<module>)
/home/oelghati/DCVC-RT/src/venv/bin/python3 scripts/backends/dcvcrt/integer_gop_study.py \
  --dcvcrt-root /home/oelghati/DCVC-RT-nvcr-1feb --list-layers
```

```bash
# Executor conformance and stage timing
build-issue216-fix-release/tests/nvcr_deterministic_executor_tests build/models/dcvcrt/normative
build-issue216-fix-release/tests/nvcr_deterministic_executor_tests --bench-stages build/models/dcvcrt/normative 17 30 20
```

```bash
# Exact encode and cross-provider decode
build-issue216-fix-release/cli/nvcr encode -i in.yuv -o exact.nvcr -s 176x144 --frames 80 --gop-size 96 \
  --qp 32 --provider tensorrt --engine-dir build/engines-rtx4070-exact-20260805/qcif --numerics nvcr.det-int.v1
build-issue216-fix-release/cli/nvcr decode -i exact.nvcr -o out.yuv --provider onnxruntime-cpu --engine-dir build/models/dcvcrt
```

```bash
# Native-versus-exact FPS A/B (diagnostic)
python3 scripts/backends/dcvcrt/exact_fps_ab.py fps-ab --nvcr build-issue216-fix-release/cli/nvcr \
  --engine-dir build/engines-rtx4070-exact-20260805/1080p \
  --input /home/oelghati/datasets/hd/BasketballDrive_1920x1080_50.yuv
```

Per-module screens: build a map from the shipped one with every layer of a
module prefix (`--list-layers` names) set to `w8a16` or `w8a8`, then pass
`map=PATH`. Study pitfall: the pinned checkout's fused CUDA kernels bypass
emulated layers; the script disables them and refuses to report if any
emulated layer never ran.

---

## 11. Loose ends

- Nothing on the branch is committed; the git index still holds older staged
  versions of several files (`README.md`, `ROADMAP.md`, docs, tests).
- `build/normative` is a stale copy of the earlier I-frame-only programs.
- `nvcr compatibility check --codec mlvc --provider tensorrt` reports
  "compatible" while the runtime refuses MLVC on TensorRT.
- Published bundle manifests and the catalog do not list `normative/`.
- CUDA-graph capture with exact stages is untested on an integrated GPU
  (graphs are captured only there).
- Scratch outputs (study JSON, streams, profiles) are not retained; the host
  rebooted twice during this work. Rerun from section 10.

## 12. Not verified

- RTX 4070 to RTX 5060, in either mode.
- Bit-exactness of the CPU reference on arm64.
- Exact-path speed under `docs/performance.md`.
- Any retrained variant (options D-F).
- MLVC's rate-distortion against DCVC-RT.

## Sources

- [DCVC-RT: Towards Practical Real-Time Neural Video Compression (arXiv 2502.20762)](https://arxiv.org/abs/2502.20762)
- [Overview of Variable Rate Coding in JPEG AI (arXiv 2503.16288)](https://arxiv.org/pdf/2503.16288)
- [DCVC-HEM: Hybrid Spatial-Temporal Entropy Modelling (arXiv 2207.05894)](https://arxiv.org/pdf/2207.05894)
- [MLVC: Multi-platform Learned Video Codec for Real-World Deployment (arXiv 2606.28027)](https://arxiv.org/abs/2606.28027)
- [MLVC review (Pith)](https://pith.science/paper/2606.28027)
- [JOMP: Jointly-Optimized Mixed-Precision Quantization (arXiv 2606.13110)](https://arxiv.org/abs/2606.13110)
- [Integer-Centric Neural Video Compression (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/integer-centric-neural-video-compression/)
- [Cross-Platform Neural Video Coding: A Case Study (arXiv 2410.20145)](https://arxiv.org/html/2410.20145v1)
- [Quantized Decoder in Learned Image Compression for Deterministic Reconstruction (arXiv 2312.11209)](https://arxiv.org/html/2312.11209v2)
- [Post-Training Quantization for Cross-Platform Learned Image Compression (arXiv 2202.07513)](https://arxiv.org/pdf/2202.07513)
- [MobileNVC (arXiv 2310.01258)](https://arxiv.org/pdf/2310.01258)
- [Effortless Cross-Platform Video Codec: A Codebook-Based Method (arXiv 2310.10292)](https://arxiv.org/pdf/2310.10292)
- [Opportunistic Conditional Entropy Coding (arXiv 2609.21816)](https://arxiv.org/pdf/2609.21816)
- [OpenDCVCs (arXiv 2508.04491)](https://arxiv.org/pdf/2508.04491)
- [microsoft/DCVC issue #137](https://github.com/microsoft/DCVC/issues/137)
- [What Matters in Practical Learned Image Compression (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.pdf)
