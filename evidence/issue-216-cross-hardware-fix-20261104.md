# Issue #216 cross-hardware fix evidence — 2026-11-04

## Scope

- Branch: `fix/cross-gpu-entropy-determinism`, on `ef4dcfd8` (`main` `163441e9`
  plus the `AGENTS.md` codec/provider capability rules). Uncommitted at the
  time of these measurements.
- Roadmap: C12, cross-hardware encode/decode mismatch.
- Two parts: mismatched decodes fail cleanly (N0), and the exact-numerics
  profile `dcvc-rt.exact.v1` makes a stream decode bit-identically on every
  provider that implements `nvcr.det-int.v1`, I- and P-frames included.
- Not on this branch: the transmitted entropy-correction profile, measured and
  rejected (see "Rejected approach"); it stays on
  `research/issue-216-exact-numerics`.

## Root cause

DCVC-RT selects one of 128 entropy tables per symbol from the predicted scale
(`floor((ln(clamp(scale)) - ln 0.11) * 25.516)`). Two providers or targets
compute that scale slightly differently. A scale near a table boundary then
selects the neighbouring table on the decoder, and rANS desynchronizes. P-frame
entropy parameters depend on the previous decoded feature (or, after an I-frame
and at reference resets, the previous decoded frame), so the closure that must
match spans nearly the whole decoder.

The vendored upstream rANS coder also had two memory-safety defects, both
reproduced with the original sources under ASan:

| Defect | Reproduction | ASan report |
|---|---|---|
| Encoder output buffer sized at one byte per symbol | 4,096 rare-escape symbols | heap-buffer-overflow, `rans_byte.h:81` `RansEncRenorm` |
| Decoder reads past a truncated stream | Stream cut in half | heap-buffer-overflow, `rans_byte.h:139` `RansDecAdvance` |

## Changes

**Clean failure (N0).** Bounded rANS decoding (`third_party/dcvc_rans`,
recorded in its `UPSTREAM.md`), `RansCodec::finish_decode()` with per-call
corruption checks, and an end-of-frame check after the last entropy decode of
every I and P frame in both DCVC-RT backends.

**Exact numerics.** Design: [docs/cross-target-decode.md](../docs/cross-target-decode.md).

- DCVC-RT declares ten normative stages (`src/dcvcrt/numerics.*`): the I-frame
  entropy stages and synthesis, and every P-frame stage except the
  encoder-only analysis and hyper analysis. Encoder and decoder therefore hold
  identical reference state, and decoded frames are identical too.
- `nvcr.det-int.v1` (`src/deterministic`): dense convolutions in exact integer
  arithmetic (8 or 16 bits per layer; spatial convolutions lowered to `im2col`
  plus a pointwise conv), everything else in det-fp32. Executors: CPU
  reference, CUDA (integer tensor cores, SM 8.0 and newer).
- Providers: TensorRT runs normative programs through the CUDA executor; ONNX
  Runtime CPU through the CPU reference; ONNX Runtime CUDA and QNN refuse them
  with `incompatible_precision`.
- Programs: ten `<stage>.detprog.json` files with FP16 weights, lowered from the
  shipped ONNX stages, in the bundle's `normative/` directory
  (`export_normative_programs.sh`, wired into `prepare_artifacts.sh` and
  `build_tensorrt.sh`). Width map: `configs/numerics/dcvcrt-det-int-v1.json`.
- Stream: each random-access unit of an exact stream is NVAU v2 with a required
  `numerics_identity` section (profile id plus one SHA-256 over the stage ids
  and program digests, 48 bytes). P-frame units keep NVAU v1 framing and
  inherit the identity, like a parameter set.
- CLI: `--numerics nvcr.det-int.v1` on encode; decode reads the profile from the
  stream. No other flag.

## Rejected approach: transmitted entropy corrections

`dcvc-rt.entropy-correction.v1` sent the decisions near a table boundary as
side data. Between TensorRT and ONNX Runtime CPU on Akiyo and Foreman QCIF
(33 frames, GOP 8, QP 32) it decoded every frame only at exact mode with margin
0.5, at 2.3-2.6 bpp of corrections (37-40 times the 0.06 bpp payload), and
reconstructions still differed (PSNR-Y 52.5-57.1 dB against the encoder's
decode). Smaller margins desynchronized by the first P-frame. The code stays on
the research branch.

## AGENTS.md compliance

| Rule | How the change complies |
|---|---|
| The codec owns sequencing, entropy decisions and payload syntax | DCVC-RT declares its normative stages, owns the exact scale-to-table thresholds and inter-stage glue, and writes the identity; the identity format is runtime-owned |
| Providers stay independent; no codec branches in providers | Providers load any descriptor with a `numerics_profile` through `src/deterministic`; no DCVC-RT names in provider code |
| No GPU, SM, plan or kernel identity in codec semantics | The stream names a profile and program digests only; the SM 8.0 requirement is the TensorRT provider's capability check |
| Provider execution details stay in the provider | The DCVC-RT TensorRT backend no longer disables CUDA graphs for exact sessions: the provider captures graphs only per TensorRT engine `enqueueV3` (on integrated GPUs), never around exact stages, so the codec needs no exact-numerics case. Exact streams are byte-identical with and without the removed gate; graph capture with exact stages is untested on an integrated GPU |
| A missing capability fails before decode with a structured error | A provider without the profile returns `incompatible_precision` at stage load; missing or different programs return `missing_artifact` or `incompatible_target` before decoding |
| No silent fallback | A stream that names the profile never decodes with native arithmetic; an exact decoder refuses a native random-access unit |
| Capability matrix only from retained evidence | `docs/compatibility.md` cells come from the gates below; RTX 4070 to RTX 5060 stays `untested` |

## Environment

- RTX 4070 (SM 8.9), driver 580.178.04, Linux 7.0.0-38 (7.0.0-42 after a
  reboot during the 1080p timing), GCC 13.3.0
- CUDA 12.8.93, TensorRT 10.9.0, ONNX Runtime 1.26.0 (CPU execution provider)
- Engines: `build/engines-rtx4070-exact-20260805/{qcif,720p,1080p}`, each with
  `normative/` copied from the model bundle
- ONNX models and programs: `build/models/dcvcrt`, `build/models/dcvcrt/normative`
- Pinned DCVC-RT `1feb52a5` (`/home/oelghati/DCVC-RT-nvcr-1feb`) for program
  checks and the quality study, run in the DCVC-RT venv (PyTorch 2.9.1+cu126)

## Gates

| Gate | Result |
|---|---|
| CPU Debug, ASan/UBSan, warnings as errors (`build-issue216-fix-asan`) | 31/31 pass |
| Release, TensorRT + ONNX Runtime (`build-issue216-fix-release`), focused | `nvcr_deterministic_executor` (with the bundle programs), `nvcr_numerics_identity`, `nvcr_rans_conformance`, `nvcr_dcvcuf_rans_conformance`, `nvcr_dcvcrt_cross_provider_decode`: 5/5 pass |
| Release full suite (`build-issue216-fix-release`, six exact engine profiles) | 52/54 pass. `nvcr_tensorrt_session_contracts_360p` and `_540p` (static-shape mismatch in the generic stage test) fail identically on `main` `163441e9`, rebuilt from a clean worktree |
| Programs against pinned PyTorch modules (`check_deterministic_programs.py`, det-fp32 interpretation, random inputs, 128x192 frame) | All ten stages agree to FP32 rounding (largest relative error 1.5e-6) |
| C++ CPU reference against the PyTorch interpreter, same inputs | Six stages agree to 2.5e-6 relative (p_synthesis frame output 3.3e-4 after clamp) |
| CUDA against CPU reference | Bit-identical on every op (new: `scale_channels`, `clamp`, `concat`, `crop`, `space_to_depth`, `im2col`), a synthetic data-movement program checked against direct indexing, and all ten stage programs, under det-fp32 and det-int |
| Native byte parity, Akiyo QCIF (`/home/oelghati/datasets/qcif/akiyo_qcif.yuv`), 65 frames, QP 32, GOP 8, TensorRT | Branch and `main` (`163441e9`, built from a clean worktree) both write `.nvcr` `afbcbd65…f631` and YUV `f8512e55…fdb3`; the branch decodes `main`'s stream to the same YUV |
| Native stream decoded by the other provider (CTest fixture, I/P/P) | Both directions rejected with `malformed_bitstream` ("DCVC-RT entropy decoding desynchronized"), no frames written |
| Exact stream decoded by the other provider (CTest fixture, I/P/P) | Both directions byte-identical to the encoder's own decode |
| Exact stream on ONNX Runtime CUDA; on a bundle without `normative/` | Both refused before decoding, no frames written |
| Exact, FourPeople QCIF (80 frames scaled from 720p with ffmpeg, input `8cc861e1…`), GOP 96, QP 32: one I-frame, 79 P-frames, reference reset at frame 65 | TensorRT encode: TensorRT and ONNX Runtime CPU decodes both `720c2f56…6552`. ONNX Runtime CPU encode: both decodes `b9994fec…e76a` |
| Stream size and quality, FourPeople QCIF 80 frames, GOP 32, QP 32, TensorRT | Exact 15,016 against native 14,744 payload bytes (+1.8%: about 130 bytes of identity and v2 framing per random-access unit; P units match native size); PSNR-Y 33.380 against 33.452 dB. One rate point; the BD-rate result is the study below |
| Stream size, BasketballDrive 1080p 33 frames, GOP 32 | Exact 116,967 against native 116,790 payload bytes (+0.15%) |

### Quality study

`scripts/backends/dcvcrt/integer_gop_study.py`, pinned DCVC-RT, every
decoder-side dense convolution emulated (221 layers; the script checks that
every emulated layer ran, after a first run was found to be bypassed by the
checkout's fused CUDA kernels and discarded). BasketballDrive, Kimono,
ReadySteadyGo, Jockey, HoneyBee 1080p, FourPeople and Johnny 720p; 96 frames
(one I-frame, reference reset every 32), QPs 0/21/42/63; BD-rate against FP32.

| Variant | BD-rate YUV | BD-rate Y |
|---|---:|---:|
| FP16 (stands in for the TensorRT FP16 path) | +0.064% | +0.046% |
| All 16-bit | -0.057% | -0.040% |
| `dcvcrt-det-int-v1.json` (I-frame entropy map, 16-bit elsewhere) | -0.089% | -0.071% |

Per-module screen for 8-bit weights (W8A16 on one module group, the map
elsewhere): Kimono and BasketballDrive 1080p, FourPeople and Johnny 720p, 40
frames, same QPs. BD-rate change against the map's own result on this set
(-0.332%):

| Module at W8A16 | Change |
|---|---:|
| `P.hyper_decoder` | +0.10% |
| `P.temporal_prior_encoder` | +0.17% |
| `I.dec` | +0.30% |
| `P.recon_generation_net` | +0.52% |
| `P.y_spatial_prior` | +0.88% |
| `P.feature_extractor` | +1.12% |
| `P.y_prior_fusion` | +1.15% |
| `P.feature_adaptor_i`, `_p` | +1.52% |
| `P.decoder` | +4.22% |

Every module costs rate at 8-bit weights; even the two cheapest together would
move the profile past the FP16 path. The map keeps 16 bits on every P-frame
layer.

### Speed (diagnostic; not the `docs/performance.md` protocol)

Native against exact on one build (`nvcr` `e880a1c5…`) and bundle
(`build/engines-rtx4070-exact-20260805/1080p`, target `rtx4070-ubuntu2404`),
idle GPU. BasketballDrive 1920x1080, 97 frames, GOP 32, QP 32, TensorRT, 10
warm-up frames (`--measurement-json`, contract
`host-yuv420p8-completed-frame-v1`, per-frame synchronized). Ten independent
processes per condition, native and exact interleaved with alternating order.
Process FPS is frames over spawn-to-exit time (includes initialization,
warm-up and YUV I/O); codec FPS is the CLI's synchronized codec interval.
Student-t 95% intervals over n = 10:

| Operation | Path | Process FPS mean (95% CI) | CV | Codec FPS mean (95% CI) | Codec ms/frame | Init | Max RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Decode | Native | 26.65 (26.41-26.89) | 1.27% | 48.65 (48.61-48.69) | 20.6 | 1.05 s | 576 MiB |
| Decode | Exact | 11.54 (11.52-11.57) | 0.26% | 18.36 (18.34-18.37) | 54.5 | 2.09 s | 803 MiB |
| Encode | Native | 27.12 (26.98-27.25) | 0.69% | 64.45 (64.38-64.51) | 15.5 | 1.05 s | 580 MiB |
| Encode | Exact | 11.65 (11.64-11.67) | 0.20% | 30.08 (30.03-30.13) | 33.2 | 2.09 s | 803 MiB |

Exact over native: decode 0.433 (process) and 0.377 (codec); encode 0.430
(process) and 0.467 (codec). At 97 frames, initialization and I/O weigh on
process FPS; exact initialization takes 1 s longer (program loading and integer
weight preparation). Streams: native 389,461 and exact 389,842 file bytes.
Max RSS is host process memory, not GPU memory.

Exact stage device time at 1088x1920 (`nvcr_deterministic_executor_tests
--bench-stages`, 20 iterations, idle GPU), before and after the GEMM work
below: I-frame synthesis 77.4 to 59.3 ms; P-frame stages 51.9 to 47.9 ms
(reference feature 15.5, prior 5.2, spatial prior 2.4, synthesis 24.8).

### GEMM optimization

Single-layer timing (`--bench-gemm OUT IN 136 240`, quantizer plus GEMM, idle
GPU) and Nsight Systems kernel splits. GPU performance counters are not
available to this user, so Nsight Compute could not run.

| Change | Result | Kept |
|---|---|---|
| Grid order: row blocks vary fastest, so resident blocks share one activation tile in L2 instead of streaming every position per row block | 368x736: 2.32 to 1.02 ms; 320x640: 1.29 to 0.78 ms; 256x512: 0.72 to 0.56 ms; 1024x256 unchanged | Yes |
| Quantizer block of 16 positions x 16 lanes (was 32 x 8) | 1024 channels: 641 to 612 us; 736: 397 to 382 us | Yes |
| 16 x 16-bit GEMM at one block per SM (K 64 x 3 stages, or K 32 x 4 stages; removes the 164-byte register spill) | 6-25% slower on every shape | No |
| Quantizer staging its positions in shared memory (one DRAM pass) | Slower (one block per SM): 641 to 719 us | No |
| Quantizer second pass in reverse channel order (L2 reuse) | No change | No |
| FP16 tensor storage (every stored tensor rounded to binary16) | P-frame stages 47.9 to 43.5 ms (-9%), but BD-rate +0.164% against FP32 (map with float32 storage -0.089%, FP16 TensorRT path +0.064%) on the full study set | No: costs rate |

Every kept change is scheduling only: exact streams and reconstructions are
byte-identical before and after (FourPeople QCIF TensorRT stream
`e211269d…ef66`, decodes `720c2f56…6552` and `b9994fec…e76a`; the 1080p exact
stream unchanged).

While testing FP16 storage, a 2-in-20 GPU nondeterminism appeared in the
test-only host upload path (a pageable `cudaMemcpy` on the legacy stream racing
kernels on the executor's non-blocking stream). It was found with a
repeated-run stress test and a per-prefix scan, and fixed by an ordered
asynchronous copy; that FP16 upload path left the code with the FP16 storage
revert. The production path never used it. The float32 executor kept here
passes the same stress test (60 of 60 repeated runs identical).

Bounds measured afterwards with timing-only builds (wrong outputs, reverted):

| Experiment | Result |
|---|---|
| Quantizer max pass removed entirely (upper bound for producer-side activation maxima) | P-frame stages 47.9 to 47.2 ms (-1.5%): not worth building |
| GEMM epilogue reduced to a raw store (upper bound for epilogue work) | 1024x256: 817 to 731 us; 256x256: 226 to 195 us (-10% to -14%) |
| GEMM throughput against K (GEMM kernel only) | K = 256: 38-42 T INT8 products/s; K = 512: 53; K = 1024: 60-64, independent of output size. A fit of time = a + b K per tile puts the fixed per-tile cost (pipeline fill, epilogue, latency) near 250 K-steps, about half the time of DCVC-RT's K = 256 layers |

Whole-block fusion (one thread block per DepthConvBlock tile, intermediates
on chip) is limited by shared memory: about 100 KiB per SM holds every channel
of only 16-32 positions, so each tile re-streams the block's ~1 MB of weight
limbs from L2 with little tensor-core reuse per fragment. It was not built.

Where the time goes now: a 256-channel DepthConvBlock at 1/8 resolution moves
about 770 MB per pass (float32 outputs, the quantizer's two reads, limb
traffic), about 1.7 ms at DRAM bandwidth, while its integer products need well
under a millisecond at INT8 peak. The exact path is memory-bound; the four
INT8 products of every 16 x 16-bit multiply set a compute floor near 17 ms per
1080p P-frame. The bounds above leave about 10-20% for epilogue and pipeline
tuning; larger gains need a codec change (see the
[research probe](issue-216-research-probe-20261105.md)).

## Capability matrix (DCVC-RT)

| Encoder | Decoder | Native numerics | `dcvc-rt.exact.v1` |
|---|---|---|---|
| TensorRT, RTX 4070 | Same target | `native`, bit-identical | `exact` |
| TensorRT, RTX 4070 | ONNX Runtime CPU, same host | `unsupported` (clean failure) | `exact` |
| ONNX Runtime CPU | TensorRT, RTX 4070 | `unsupported` (clean failure) | `exact` |
| TensorRT, RTX 4070 | TensorRT, RTX 5060 | `untested` | `untested` |
| Any | ONNX Runtime CUDA or QNN | `untested` | `unsupported` (refused before decode) |

## Not verified

- RTX 4070 to RTX 5060, the cell issue #216 reports; no second GPU is attached.
  ONNX Runtime CPU on the same host stands in as the second target.
- Exact-path speed under `docs/performance.md`; the numbers above are diagnostic.
- Upstream compatibility of exact streams: pinned Python DCVC-RT computes table
  indexes natively and cannot decode them. Native streams are unchanged.
- Bit errors confined to raw rANS bypass bits are not detected; NVAU has no
  checksum.
- Publication bundles: the engine checksum manifest and catalog do not list
  `normative/` yet, so published archives would fail validation if it were
  added.
