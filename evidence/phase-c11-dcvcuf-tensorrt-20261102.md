# Phase C11 DCVC-UF on the TensorRT provider — 2026-11-02

Status: phase 0 (reference vectors and fidelity gates) complete. No TensorRT
code exists yet. On 2026-11-02 the user directed that DCVC-UF move from the
second `dcvcuf-upstream-cuda` provider to the existing TensorRT provider while
staying as faithful as possible to Microsoft's published Python codec, pinned
`microsoft/DCVC` `cbdae87a5445114cdc7f48816da63ea80bdeac40`.

## Scope found in the current code

- The TensorRT provider session implements the generic `IProviderSession`
  contract, but `load_stage` accepts only DCVC-RT's compiled-in `engine_specs`
  table, in DCVC-RT bundle order and FP16 only, and lives under
  `src/dcvcrt/backend/tensorrt/`. Phase 1 makes stage contracts come from the
  validated bundle manifest. DCVC-RT stream and reconstruction hashes must stay
  identical.
- DCVC-RT splits each frame type into TensorRT plans for its neural
  sub-networks, with codec-specific CUDA glue (`cuda_ops.cu`) and CPU rANS.
  DCVC-UF maps onto the same split:
  - image (DMCI): `enc`, `hyper_enc`, `hyper_dec`, `y_prior_fusion`,
    `y_spatial_prior_reduction`, three `y_spatial_prior_adaptor_k` +
    `y_spatial_prior` steps, `dec`;
  - HT-S: `feature_adaptor_i`, `feature_adaptor_m`, `feature_extractor`,
    `encoder`, `hyper_encoder`, `temporal_prior_encoder`, `hyper_decoder`,
    `y_prior_fusion`, `y_spatial_prior_reduction`, three adaptor + prior steps,
    `decoder`, `recon_head` (eight frame branches and the reset branch);
  - glue: about 25 upstream CUDA kernels (masks, masked quantization,
    skip-aware entropy indexes, pixel shuffles, replicate padding), to be ported
    as codec-local CUDA without libtorch. The UF rANS codec already exists in
    NVCR.

## Fidelity gates

Approved by the user on 2026-11-02 as proposed. All gates use the upstream
recipe NVCR currently implements: intra frame only at frame 0 (upstream
`intra_period -1`), `--reset_interval 32`, `--skip_thres 0`, QPs 0/21/42/63
(upstream `--rate_num 4`).

| Gate | Requirement | Kind |
|---|---|---|
| G1 drift-free | The TensorRT encoder's internal reconstruction equals the TensorRT decoder's output byte for byte on every frame of all four reference sequences at all four QPs (at least 500 frames on BasketballDrive) | pass/fail |
| G2 format | UF payload syntax, UF rANS framing, entropy-parallel field and reset signalling identical to the current NVCR/upstream format; malformed-stream rejection unchanged | pass/fail |
| G3 fidelity | Per sequence and QP against the upstream reference below: mean per-frame PSNR-Y within 0.05 dB, total bits within 1%; per class BD-rate within ±0.5% | pass/fail |
| G4 interoperability | Upstream decoding NVCR TensorRT streams: drift and PSNR change | reported only |
| Precision | TensorRT FP16 first; any layer moved to FP32 is recorded with its reason and cost | recorded |
| Performance | Codec and process FPS against the fused provider (183.8/135.1 codec, 51.6/43.8 process, 1080p encode/decode) and against upstream Python | reported; the user requires the runtime not to be slower |

Byte identity with upstream streams is not a gate: TensorRT FP16 does not
reproduce the fused CUTLASS FP16 accumulation, so latents and therefore streams
will differ. G3 bounds the resulting rate-distortion change.

## Reference 1: sequence-level RD (upstream test_video.py)

`scripts/backends/dcvcuf/capture_rd_reference.py` runs unmodified upstream
`test_video.py` on the hash-pinned sequences in
`scripts/backends/dcvcuf/c8_reference_sequences.json` and checkpoints, verifies
the upstream checkout is clean before and after, and keeps per-frame bpp, PSNR
and frame types plus each stream's SHA-256.

Recipe: intra only at frame 0, `--reset_interval 32`, `--skip_thres 0`,
QPs 0/21/42/63 (upstream QP 0 is the lowest rate). Mean per-frame values from
upstream's own report (bpp, PSNR-Y in dB):

| Sequence | Frames | QP 0 bpp / PSNR-Y | QP 21 | QP 42 | QP 63 |
|---|---:|---|---|---|---|
| BasketballDrive_1920x1080_50 | 500 | 0.00135 / 27.203 | 0.00499 / 31.041 | 0.01710 / 34.095 | 0.05405 / 35.996 |
| Kimono_1920x1080_24 | 240 | 0.00089 / 29.158 | 0.00363 / 33.045 | 0.01314 / 36.768 | 0.04248 / 39.680 |
| Jockey_1920x1080_120fps_420_8bit_YUV | 600 | 0.00105 / 29.213 | 0.00340 / 33.722 | 0.01110 / 37.700 | 0.03507 / 40.984 |
| HoneyBee_1920x1080_120fps_420_8bit_YUV | 600 | 0.00070 / 35.652 | 0.00162 / 40.161 | 0.00403 / 42.488 | 0.01054 / 43.432 |

Checks on this reference:

- Rate and quality rise monotonically with QP on every sequence, and the
  per-frame bits sum to each stream's size.
- Every point has a single intra frame and quality holds over the full length:
  the mean PSNR-Y of the last 50 frames is at most 0.37 dB below the sequence
  mean (Jockey QP 63), and at most 1.66 dB above it (Kimono QP 0, whose content
  settles). There is no collapse like the intra-period-32 schedule's.
- Reproducibility: an independent re-run of all four Kimono points produced
  byte-identical streams and identical per-frame PSNR, so G3 compares against a
  stable target.

G3's BD-rate uses upstream `compare_bd_rate.py` with its defaults
(`--compare_between class`, `--distortion_metrics psnr`, the 6:1:1 weighted
YUV PSNR), this reference as the anchor.

Upstream's README recipe (`--reset_interval 128`, `--skip_thres 0.15`, same
QPs and intra period) was captured with the same tool. Against the recipe above
it spends fewer bits at nearly the same PSNR-Y on every point: 2-13% fewer on
BasketballDrive, 3-20% on Kimono, 2-11% on Jockey and 44-56% on HoneyBee, with
PSNR-Y changes between -0.23 and +0.37 dB. NVCR cannot express this recipe yet
(C7 remaining gate), so it is not a C8 gate input. Faithfulness to Microsoft's
published operating point does, however, require NVCR to support it; the
reference is retained for that comparison.

## Reference 2: per-stage tensors (upstream PyTorch and fused)

`scripts/backends/dcvcuf/capture_stage_vectors.py` runs, for one intra frame
and three eight-frame HT-S groups (the second resets feature memory, upstream's
schedule with `--reset_interval 16`), upstream `forward_one_frame` in FP32 and
in upstream's FP16 channels-last setup with forward hooks on every
sub-network, plus upstream's fused `compress`/`decompress` sequence. QP 32,
skip threshold 0.

Two capture details matter for later comparisons:

- Upstream's PyTorch path has the fused stage structure only when the 1/16
  latent is a multiple of 4; otherwise the fused path replicate-pads it
  (`pad_for_y`) before the hyper network. QCIF does not qualify, so the small
  vector is a 256x256 crop of BasketballDrive at (832, 412); full 1080p
  qualifies.
- Fused outputs are borrowed buffers that the next call overwrites. Upstream
  consumes them immediately; the tool snapshots each one.

| Vector | Mean PSNR-Y vs source: PyTorch FP32 / FP16 / fused | PyTorch vs fused reconstruction PSNR-Y (mean, min) | Quantized symbols differing, FP16 vs FP32 |
|---|---|---|---|
| BasketballDrive 256x256 crop, 25 frames | 32.703 / 32.732 / 32.688 dB | FP32 40.11, 37.03 dB; FP16 40.15, 36.55 dB | 0.31-0.69% |
| BasketballDrive 1080p, 25 frames | 33.493 / 33.486 / 33.481 dB | FP32 40.67, 39.32 dB; FP16 40.64, 39.16 dB | 0.24-0.52% |

Upstream's own valid implementations therefore already differ from each other:
FP16 flips a fraction of a percent of latent symbols, and the reconstructions
sit 37-41 dB apart, while sequence quality agrees within 0.012 dB at 1080p. A
correct TensorRT FP16 port should fall in the same band; the 0.05 dB G3 limit
leaves room for that without hiding a real error.

The per-stage FP16-vs-FP32 errors in the report are cumulative: each FP16 stage
receives inputs that have already diverged, including flipped symbols. Phase 2
checks each exported stage in isolation by feeding it the stored inputs. The
crop vector stores every stage's inputs and outputs in both precisions and the
fused reconstructions (371 arrays); 1080p stores statistics only.

## Retained local outputs

Raw outputs stay in ignored `build/` directories; digests are in the adjacent
JSON evidence.

- `build/dcvcuf-c8-reference-nvcr-20261102/` and
  `build/dcvcuf-c8-reference-readme-20261102/`: RD reports, streams, upstream
  JSON.
- `build/dcvcuf-c8-stage-vectors-20261102/`: stage report and crop tensors.

## Phase 1: codec-agnostic TensorRT provider session

The TensorRT provider session held DCVC-RT's stage table, bundle file list,
model-manifest checks, warm-up shapes, quantization caches, decode staging and
symbol buffers. It now lives in `src/provider/tensorrt/session.{hpp,cpp}`
(`nvcr::tensorrt`) and holds none of them:

- Codecs supply each stage's tensor contract and warm-up shapes. The session
  validates the contract against the deserialized engine's own I/O (names,
  modes, FP16, rank 4, fixed and dynamic axes) and accepts only files that
  `engine.sha256` lists and that passed digest checks, each at most once.
  Generic descriptors warm up at the engine's optimization-profile minimum.
- Creating a session (`make_provider_session`, the registered `tensorrt`
  factory) validates the bundle identity, target, CUDA/TensorRT versions and
  every listed file's digest, and loads no stages. The codec loads its stages
  and calls `finalize_stages()` (shared workspace, warm-up, CUDA graphs).
- DCVC-RT keeps its 14-stage table and warm-up shapes, its exact 20-file
  bundle set and model-manifest identities (checked before any stage loads),
  its device state, its `[nvcr.dcvcrt] [info] TensorRT mode:` log line, and its
  public `register_tensorrt_provider()`/`tensorrt_provider_descriptor()`,
  which now forward to the generic provider. Load order, and therefore stage
  indices, are unchanged.
- Current provider limits: rank-4 FP16 tensors only.

Verification on the RTX 4070 (CUDA 12.8, TensorRT 10.9, Release):

| Check | Before | After |
|---|---|---|
| B7 65-frame Akiyo QCIF GOP-8 stream / decoded YUV, catalog bundle | `98966844...` / `10e02e46...` | identical |
| Same in low-memory context mode | `98966844...` / `10e02e46...` | identical |
| 100-frame BasketballDrive 1080p stream / decoded YUV | `d96d4bcb...` / `53e2d1d6...` | identical |
| 1080p codec FPS encode / decode (5 runs, 10 warm-up) | 56.8 / 45.0 | 57.5 / 44.9 |
| TensorRT build CTest (engine contracts, 1080p round trip, pinned Python I-frame golden) | 31/31 | 32/32 |
| CPU build CTest | 27/27 | 27/27 |

The added `nvcr_tensorrt_session_contracts` test loads a DCVC-RT plan through
a generic descriptor and checks rejection of a mismatched tensor name, a wrong
fixed axis, a file outside the verified bundle and a foreign provider, plus
duplicate loads, execution before finalize, a second finalize and loading after
finalize, then executes the stage. Not yet covered by a test: a bundle with an
extra or missing file, which DCVC-RT still rejects with its exact-set check.

## Phase 2: per-stage ONNX and TensorRT export

`scripts/backends/dcvcuf/tensorrt_stages.py` defines 20 stages (7 image, 13
HT-S) built from pinned upstream modules in the order upstream's inference path
composes them. Glue stays outside, where the fused path has it: the HT-S
hyper-decoder output is cropped before the prior stage fuses it with the
temporal prior; the image prior stage returns fusion and reduction outputs at
the padded latent size (the reduction is 1x1, so cropping both is exact); the
reconstruction head is unclamped, as upstream's PyTorch head.

`export_tensorrt_stages.py` exports FP32 ONNX (opset 17, dynamic H and W),
checks each graph on ONNX Runtime CPU against the PyTorch stage at 256x256 and
128x192 frames, and builds TensorRT 10.9 plans with DCVC-RT's recipe (`--fp16`,
builder optimization level 2, 1 GiB workspace, timing cache, dynamic 1080p
profile with 64x64 minimum) plus explicit FP16 I/O formats. DCVC-RT gets FP16
I/O by exporting FP16 ONNX instead; this is the one recipe difference. It
writes an `nvcr.engine-bundle.v2` manifest (`dcvcuf-cvpr2026-hts`, 20 plans plus
`stages.json`, 307 MiB), which NVCR's generic TensorRT session accepts.

All 20 ONNX graphs match PyTorch FP32 within 2.8e-06 of each output's peak.

`check_tensorrt_stages.py` maps the 41 recorded upstream calls of the crop
vector (one I frame, three HT-S groups including the reset) onto the stages,
feeds each plan the recorded FP32 inputs rounded to FP16 through
`nvcr_tensorrt_stage_runner` (NVCR's generic session, generic descriptors), and
compares with the FP32 outputs. The same FP16 inputs also run through upstream's
own PyTorch stage in its FP16 channels-last configuration:

| Stage | Cases | TensorRT/upstream FP16 mean error | max error |
|---|---:|---|---|
| `i_encoder` | 1 | 3.08-3.08x | 6.39-6.39x |
| `i_hyper_encoder` | 1 | 2.05-2.05x | 1.86-1.86x |
| `i_hyper_decoder` | 2 | 2.20-2.21x | 2.39-2.88x |
| `i_spatial_prior_1` | 1 | 2.26-2.26x | 1.91-1.91x |
| `i_spatial_prior_2` | 1 | 2.30-2.30x | 1.88-1.88x |
| `i_spatial_prior_3` | 1 | 2.36-2.36x | 2.73-2.73x |
| `i_decoder` | 1 | 1.59-1.59x | 2.29-2.29x |
| `p_feature_adaptor_i` | 2 | 2.16-2.20x | 1.60-2.20x |
| `p_feature_adaptor_m` | 1 | 2.72-2.72x | 1.95-1.95x |
| `p_feature_extractor` | 3 | 2.20-2.30x | 1.30-1.53x |
| `p_encoder` | 3 | 4.55-4.64x | 9.02-11.24x |
| `p_hyper_encoder` | 3 | 2.39-2.52x | 1.57-6.44x |
| `p_hyper_decoder` | 3 | 1.67-1.69x | 1.15-1.71x |
| `p_prior` | 6 | 2.53-3.36x | 3.53-6.17x |
| `p_spatial_prior_1` | 3 | 2.55-2.59x | 2.25-3.52x |
| `p_spatial_prior_2` | 3 | 2.56-2.59x | 2.18-3.67x |
| `p_spatial_prior_3` | 3 | 2.60-2.64x | 2.99-7.49x |
| `p_decoder` | 3 | 2.32-2.73x | 1.77-3.34x |
| `p_reconstruction` | 3 | 1.73-1.75x | 1.73-2.17x |
| `p_reset` | 1 | 2.02-2.02x | 1.77-1.77x |

All plans load through generic descriptors, resolve the expected shapes and
produce finite outputs. TensorRT FP16 error is consistently 1.6-3x upstream
PyTorch FP16 error, and 3-4.7x on the two encoders, whose outputs reach the
hundreds. A diagnostic build with FP32 layers (FP16 I/O) brings the encoders and
HT-S prior to 0.3-0.6x, so the gap is TensorRT's FP16 kernels; builder
optimization level 5 does not change it. FP32 costs 2.3-2.6x on those stages at
1080p (image encoder 17.0 to 38.5 ms, HT-S encoder 13.4 to 31.2 ms, HT-S prior
5.9 to 15.4 ms per call).

PyTorch FP16 is a stricter band than upstream's deployed path: the C7 diagnostic
found the fused CUTLASS first convolution at about 6x PyTorch FP16 error against
an FP64 oracle (mean 0.00194 vs 0.00033), while fused and PyTorch sequence
quality agree (32.69 vs 32.73 dB mean PSNR-Y on the crop vector). The per-stage
ratios are therefore an early warning, not a gate. The plans stay FP16; the
sequence-level gates in phase 4 decide, with FP32 for named stages as the
recorded fallback if G3 fails.

## Phase 3: glue kernels and stage boundaries

### Glue port, bit-identical to upstream

`scripts/backends/dcvcuf/port_glue_kernels.py` copies every `__global__` and
`__device__` function from pinned upstream `elementwise/stream.cu`,
`cat_and_pad.cu` and `shuffle.cu` into
`src/dcvcuf/backend/tensorrt/glue_kernels.inc` (25 kernels and one helper),
renaming only `at::Half` to `Half`; `--check` and a CTest case verify the
committed file regenerates from the pinned checkout. `glue_types.cuh`
reproduces what the kernels take from libtorch and upstream headers: a `Half`
with c10::Half's device semantics (round-to-nearest conversions, each
`Half op Half` in float, `Half op float` yielding float), an accessor with
`PackedTensorAccessor32` indexing, and upstream's `load2`/`store2`, intrinsic
wrappers and constants. `glue.cu` replaces upstream's `at::Tensor` wrappers with
libtorch-free launchers using the same instantiations, launch shapes and
dispatch. The kernels compile with upstream's flags: `-O3 --use_fast_math
--extra-device-vectorization` from its `setup.py` and torch's
`COMMON_NVCC_FLAGS` (`__CUDA_NO_HALF*` macros, `--expt-relaxed-constexpr`). The
macros are part of the arithmetic: they disable native `__half` operators, so
comparisons and operations go through float as upstream's do. The target builds
warning-free in the native (CUDA 12.6) and TensorRT (CUDA 12.8) configurations.

`tests/dcvcuf_glue_parity_tests.cpp` (native DCVC-UF build) runs the compiled
upstream kernels and the port on identical channels-last inputs: all 65,536
FP16 patterns through `round_z` and scale-to-index, plus random, exact `.5`
boundary, tiny and large values on four latent shapes, both skip thresholds,
odd sizes, in-place variants and all shuffle/pad/crop paths. All 219 checks are
bit-identical. Injected faults are detected: `SCALE_MAX` 16 to 15 fails 18
checks; round-toward-zero subtraction fails 82.

### Stage boundaries now follow the fused path

- Latent and feature tensors bind as TensorRT `hwc8`, which is NHWC for
  `C % 8 == 0`: upstream's channels-last layout. The glue kernels then run
  verbatim, including their paired-channel loads, and flat kernels such as
  `round_z` read hyper-latent symbols in upstream's HWC stream order. Only the
  3-channel frame input stays CHW. Accuracy and speed are unchanged by the
  layout (error ratios within 2.6%, stage times within 3%).
- The image decoder and HT-S reconstruction head end before their final 8x
  pixel shuffle, as the fused networks do; glue applies upstream's
  `pixel_shuffle_8` with its [-0.5, 0.5] clamp. The fused path builds the HT-S
  reference feature from this clamped I frame, which upstream's PyTorch head
  does not clamp. The reset head stays a separate stage used identically on
  both sides, as the fused decoder's `forward_reset`.
- Plans build at TensorRT builder optimization level 5 instead of DCVC-RT's 2.
  At 1080p this cuts the reconstruction head from 55 to 34 ms, the image
  decoder from 33 to 22 ms, the image encoder from 17 to 12 ms and the HT-S
  encoder from 14 to 10 ms; a fixed-shape profile gains only 2 ms more on the
  reconstruction head. Isolated accuracy is unchanged: mean error
  1.42-4.65x upstream PyTorch FP16, all outputs finite.

### Model-only speed against upstream (RTX 4070, 1080p, median of 50)

| Stage | ms |
|---|---:|
| `p_reconstruction` | 34.32 |
| `i_decoder` | 21.77 |
| `i_encoder` | 11.51 |
| `p_decoder` | 10.81 |
| `p_encoder` | 10.16 |
| `p_feature_adaptor_m` | 9.31 |
| `p_feature_extractor` | 7.21 |
| `p_feature_adaptor_i` | 5.87 |
| `p_reset` | 5.09 |
| `p_prior` | 4.03 |
| `i_spatial_prior_2` | 2.16 |
| `i_spatial_prior_3` | 2.15 |
| `i_spatial_prior_1` | 2.15 |
| `p_spatial_prior_1` | 2.13 |
| `p_spatial_prior_3` | 2.13 |
| `p_spatial_prior_2` | 2.13 |
| `i_hyper_decoder` | 1.93 |
| `p_hyper_decoder` | 0.44 |
| `p_hyper_encoder` | 0.28 |
| `i_hyper_encoder` | 0.16 |

| Operation | TensorRT FP16 stages | Upstream fused, whole operation (test_video.py timer) |
|---|---:|---:|
| I frame encode | 41.8 ms | - |
| I frame decode | 30.2 ms | - |
| 8-frame group encode | 48.6 ms | 31.7 ms |
| 8-frame group decode | 72.5 ms | 43.6 ms |

The TensorRT figures exclude glue, entropy coding and copies; upstream's
include its model, glue and entropy coding. On this GPU TensorRT FP16 is
therefore at least 1.5x slower than upstream's fused CUTLASS kernels for a
group encode and 1.7x for a group decode, before glue and entropy are added.
Upstream tunes those kernels for 2080 Ti, 4090, A100, H100 and B200. TensorRT
FP16 is not the faster mode here; its value is a single provider and
portability to targets without upstream's CUTLASS build.

## Phase 4: execution path — 2026-11-03

`src/dcvcuf/tensorrt_session.cpp` runs DCVC-UF on the generic TensorRT
session. It follows upstream's fused `compress`/`decompress` for the intra
frame and the eight-frame HT-S group step for step: the plans replace the fused
networks, the ported glue kernels run between them, and NVCR's UF rANS coder
codes on the host. It needs no LibTorch. The adapter wraps a TensorRT session
when the CLI's `--engine-dir` names a stage bundle; the native fused path stays
the default for its `.pt` bundle.

- `tables.bin` (added by `add_tensorrt_tables.py`) carries the per-QP scale
  vectors and z/y CDFs from the native bundle, digest-listed like the plans.
- `frames.cu` converts YUV420P8 frames with the native path's arithmetic.
- Glue tensors accept channel-slice views (`stride`), as upstream's chunked
  tensors are.
- The TensorRT session's low-memory mode now releases each context after
  warm-up, as it already did after each run.

Checks on an RTX 4070 shared with a 10 GiB training job, so on plans rebuilt
with a 256x256 maximum profile (builder level 2) instead of the 1080p bundle,
whose contexts do not fit beside it. QP 32, upstream recipe:

| Clip | Native fused bytes / PSNR-Y | TensorRT bytes / PSNR-Y |
|---|---|---|
| Akiyo QCIF, 17 frames | 1770 / 37.440 dB | 1775 / 37.462 dB |
| Akiyo QCIF, 65 frames (feature-memory reset at 33) | 4496 / 37.204 dB | 4526 / 37.293 dB |
| BasketballDrive 200x120 crop, 33 frames (padding, crops) | 2495 / 32.289 dB | 2471 / 32.293 dB |

- Repeated TensorRT encode and decode are byte-identical; quality holds
  through the reset, so encoder and decoder reference states agree.
- The native fused path is byte-identical to the branch before the rebase
  (65-frame Akiyo stream and reconstruction).

### 1080p speed (RTX 4070 idle, 1080p bundle, builder level 5)

BasketballDrive 1080p, 100 frames, QP 32, intra only at frame 0, three runs
each; CLI codec time after the first output, process time from the first run.
Every run of a path produced the same stream.

| Path | Encode fps | Decode fps | Process time enc / dec | Payload | PSNR-Y |
|---|---:|---:|---:|---:|---:|
| Native fused | 156.7-160.0 | 122.7-124.0 | 2.30 / 2.47 s | 274,014 B | 32.6607 dB |
| TensorRT | 112.1-113.3 | 88.6-89.3 | 3.89 / 4.68 s | 273,633 B | 32.6571 dB |

TensorRT runs at about 71% of the fused path's speed in both directions, in
line with the phase-3 per-stage timings (TensorRT FP16 kernels 1.5-1.7x slower
than upstream's CUTLASS kernels). The fused path stays the faster default on
NVIDIA GPUs; TensorRT removes LibTorch and is the sequencing the QNN port
reuses.

### 1080p fidelity gates (G3)

`scripts/backends/dcvcuf/run_tensorrt_gates.py` encodes and decodes the full
reference set (500, 240, 600 and 600 frames) at QPs 0/21/42/63 with both NVCR
paths and the upstream recipe, and measures mean per-frame PSNR on the 8-bit
output (`build/dcvcuf-c11-gates-20261103/gates.json`, SHA-256
`ba2859d3487d0c515913ac924077354355ff57ff839e1edcfd59392495d32b91`).

| Sequence | QP | Native bytes | Native PSNR-Y | Upstream PSNR-Y | TensorRT bytes | TensorRT PSNR-Y | dPSNR-Y | Bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BasketballDrive | 0 | 182,125 | 27.1999 | 27.2030 | 182,475 | 27.2062 | +0.0062 | +0.19% |
| BasketballDrive | 21 | 654,272 | 31.0341 | 31.0414 | 654,991 | 31.0379 | +0.0038 | +0.11% |
| BasketballDrive | 42 | 2,223,145 | 34.0799 | 34.0948 | 2,223,100 | 34.0803 | +0.0004 | -0.00% |
| BasketballDrive | 63 | 7,012,542 | 35.9724 | 35.9958 | 7,012,802 | 35.9726 | +0.0002 | +0.00% |
| Kimono | 0 | 59,166 | 29.1529 | 29.1576 | 59,305 | 29.1416 | -0.0113 | +0.23% |
| Kimono | 21 | 229,105 | 33.0334 | 33.0451 | 228,923 | 33.0333 | -0.0001 | -0.08% |
| Kimono | 42 | 820,699 | 36.7401 | 36.7675 | 820,527 | 36.7402 | +0.0001 | -0.02% |
| Kimono | 63 | 2,646,389 | 39.6271 | 39.6803 | 2,646,125 | 39.6274 | +0.0003 | -0.01% |
| Jockey | 0 | 172,860 | 29.2077 | 29.2128 | 172,943 | 29.2083 | +0.0006 | +0.05% |
| Jockey | 21 | 537,543 | 33.7081 | 33.7223 | 537,788 | 33.7065 | -0.0016 | +0.05% |
| Jockey | 42 | 1,734,885 | 37.6650 | 37.7001 | 1,734,466 | 37.6655 | +0.0005 | -0.02% |
| Jockey | 63 | 5,463,081 | 40.9107 | 40.9839 | 5,463,201 | 40.9104 | -0.0003 | +0.00% |
| HoneyBee | 0 | 118,364 | 35.6313 | 35.6520 | 118,282 | 35.6390 | +0.0076 | -0.07% |
| HoneyBee | 21 | 261,532 | 40.1023 | 40.1605 | 261,615 | 40.1053 | +0.0030 | +0.03% |
| HoneyBee | 42 | 635,007 | 42.3896 | 42.4885 | 635,174 | 42.3894 | -0.0002 | +0.03% |
| HoneyBee | 63 | 1,647,726 | 43.3099 | 43.4324 | 1,647,152 | 43.3099 | -0.0001 | -0.03% |

**G3 passes.** Largest |dPSNR-Y| 0.0113 dB (limit 0.05), largest bit
change 0.23% (limit 1%), BD-rate (6:1:1 YUV PSNR,
TensorRT against native) HEVC-B +0.003% and UVG -0.029% (limit ±0.5%); per
sequence: BasketballDrive -0.007%, Kimono +0.013%, Jockey +0.051%, HoneyBee -0.109%.

The native path's streams are byte-identical to upstream's (C7), yet its
8-bit-output PSNR-Y sits up to 0.12 dB below upstream's report, growing
with quality: upstream measures its own reconstruction, not the rounded 8-bit
frames. Both NVCR paths are measured the same way, so the gate is unaffected.

G1 was run on 2026-11-07 under roadmap C13; see
[G1 below](#g1-encoder-internal-reconstruction-equals-decoder-output--2026-11-07).
G4 (upstream decoding NVCR streams) remains open in C13.

### Regression

On the same build: DCVC-RT TensorRT streams and reconstructions are
byte-identical to `main` (65-frame Akiyo QCIF GOP 8 on two QCIF bundles, in
performance and low-memory modes, and 100-frame BasketballDrive 1080p at
48.2/49.7 fps against main's 48.3/50.2); MLVC and DCVC-RT ONNX Runtime CPU
streams match their recorded hashes. The feasibility branch had emptied the
default codec, provider and model identities in `RuntimeConfiguration`; that
public-API change is not carried, so the defaults stay as on `main`.

## G1: encoder-internal reconstruction equals decoder output — 2026-11-07

Run under roadmap C13 (DCVC-UF qualification), on branch
`test/dcvcuf-encoder-recon-identity` based on `main` `676efac4`. **G1 passes
on both paths**: 32/32 cases (2 paths × 4 sequences × 4 QPs, full lengths),
15,520 frames and 7,936 state tensors compared byte for byte, no difference.

### What is compared

Neither upstream's fused HT-S encoder nor NVCR's TensorRT port runs the
reconstruction head for a predicted group: the encoder rebuilds only the
decoder's state (`y_hat`, the decoder feature, and the memory and context used
for the next group). Only intra frames give the encoder a reconstructed
picture. The approved G1 definition (user, 2026-11-07) therefore compares,
per coding step and byte for byte:

| Step | Encoder | Decoder |
|---|---|---|
| Intra frame | reconstructed frame (YUV420P8, decoder's conversion) | decoded frame |
| HT-S group, state | memory and context used for the group, `y_hat`, decoder feature | the same four tensors |
| HT-S group, pixels | the decoder's reconstruction head run on a **copy** of the encoder's feature, converted to YUV420P8 | decoded frames; for a padded tail only the real frames |

The TensorRT path also records the intra step's `z_hat`, hyper-decoder outputs,
each spatial-prior step's input and output and the final `y_hat` (ten more
tensors per sequence), used to localize failures.

Mechanism, with no production CLI or public API change:

- `src/dcvcuf/diagnostics.{hpp,cpp}`: a test-only, process-wide observer. With
  no observer installed the sessions do no extra work.
- TensorRT session: device bytes of the tensors above; encoder-side group
  pixels from an extra `p_reconstruction` run on the encoder's feature.
- Fused session: `materialize_native_sources.py` adds read-only accessors and a
  reconstruction-head call to NVCR's generated copy of `DMCHTSProxy` (the
  pinned upstream checkout is untouched). Upstream's head also updates the
  instance's reset feature, so the encoder-side pixels come from a separate,
  diagnostic-only `DMCHTSProxy` instance fed a clone of the encoder's feature.
- `tests/dcvcuf_reconstruction_identity.cpp`: drives the registered `dcvc-uf`
  codec through `nvcr::Runtime`, decodes each access unit as soon as it is
  emitted, and compares the decoder step with the encoder step that produced
  it. A decode error is recorded with its step and the first differing tensor
  among those computed before the failure: the TensorRT path records each
  tensor as it is computed; the fused path decodes a group in one upstream call,
  so a failed fused group reports its step without tensors.
- `scripts/backends/dcvcuf/run_reconstruction_identity.py`: runs the matrix and
  then encodes and decodes the same input with the production CLI; the
  harness's `.nvcr` and decoded YUV must equal the CLI's byte for byte, so the
  hook is shown not to change coding.

Two comparisons are classed separately rather than as failures. The TensorRT
encoder's `round_z` writes −0 where the decoder rebuilds +0 from integers, so
`z_hat` differs only in the sign of zero (one tensor per TensorRT case, 16 in
total; values equal). The allowance applies to `z_hat` and to ±0 only; every
other tensor and every frame must match byte for byte, NaN payloads included.

Negative controls (Akiyo QCIF, 17 frames, both paths): flipping one byte of the
first encoder HT-S feature is reported at step 1, tensor `feature`; flipping
one byte of an encoder group frame is reported at step 1, frame 2. Both runs
exit as failures.

### Environment

- NVIDIA GeForce RTX 4070 (SM 8.9, 12,282 MiB, 33 MiB in use before and after
  the run, no other workload), driver 580.178.04, Linux 7.0.0-42-generic,
  GCC 13.3.0, CMake 3.28.3.
- Fused: CUDA 12.6.85, PyTorch/LibTorch 2.13.0+cu126, bundle
  `build-dcvcuf-native-cu126/artifacts/dcvcuf-cvpr2026-hts.pt`
  (`81fe0a7d…37ab`), pinned DCVC `cbdae87a`, CUTLASS v4.4.1.
- TensorRT: CUDA 12.8.93, TensorRT 10.9.0.34, the C11 1080p bundle
  `build/dcvcuf-trt-bundle` (builder level 5, `engine.sha256` `ec3ae9d1…6536`).
- Release builds `build-g1-native` and `build-g1-trt` of this branch.
- Recipe as in G3: intra frame only at frame 0 (`--gop-size 65528`), reset
  interval 32, skip threshold 0, CLI default timestamps (16,667 µs).

### Results

Every sequence ends in a partly filled group (tail of 3 or 7 frames) and passes
through 7-18 HT-S feature-memory resets. Inputs match the hashes in
`c8_reference_sequences.json`. ✓ means the harness file equals the CLI's.

| Path | Sequence | QP | Frames | Steps (I + groups) | Memory resets | Tail | Frames compared | Tensors compared | Stream `.nvcr` (= CLI) | Decoded YUV (= CLI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| fused | BasketballDrive | 0 | 500 | 1 + 63 | 15 | 3 | 500 | 252 | `9acb11c2…f6fd` ✓ | `d65685ad…8d5e` ✓ |
| fused | BasketballDrive | 21 | 500 | 1 + 63 | 15 | 3 | 500 | 252 | `0d45dc7c…ae6d` ✓ | `58c6e483…468c` ✓ |
| fused | BasketballDrive | 42 | 500 | 1 + 63 | 15 | 3 | 500 | 252 | `cf9d65e9…0b25` ✓ | `922507ca…3a5b` ✓ |
| fused | BasketballDrive | 63 | 500 | 1 + 63 | 15 | 3 | 500 | 252 | `f4bf20a5…f1b5` ✓ | `dd4c6d9c…5fb4` ✓ |
| fused | Kimono | 0 | 240 | 1 + 30 | 7 | 7 | 240 | 120 | `a2c244b6…f91b` ✓ | `de610d50…c1f3` ✓ |
| fused | Kimono | 21 | 240 | 1 + 30 | 7 | 7 | 240 | 120 | `ba3f4521…7353` ✓ | `d1f07d9d…6447` ✓ |
| fused | Kimono | 42 | 240 | 1 + 30 | 7 | 7 | 240 | 120 | `f396f9e1…44ec` ✓ | `7553d656…09d3` ✓ |
| fused | Kimono | 63 | 240 | 1 + 30 | 7 | 7 | 240 | 120 | `f4b7dd9d…8e55` ✓ | `a469e3b5…fb38` ✓ |
| fused | Jockey | 0 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `17d49950…4df9` ✓ | `332d877c…c2ca` ✓ |
| fused | Jockey | 21 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `c2221ae3…eb3f` ✓ | `9f1d9b75…2339` ✓ |
| fused | Jockey | 42 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `00e258bd…5859` ✓ | `e1e937f5…3e66` ✓ |
| fused | Jockey | 63 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `07d656b6…a909` ✓ | `84d611e2…7fb0` ✓ |
| fused | HoneyBee | 0 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `3420c1e9…9802` ✓ | `c571ca99…494d` ✓ |
| fused | HoneyBee | 21 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `404b33db…4da3` ✓ | `33125649…3c73` ✓ |
| fused | HoneyBee | 42 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `5d750fa9…3ad0` ✓ | `63323120…2322` ✓ |
| fused | HoneyBee | 63 | 600 | 1 + 75 | 18 | 7 | 600 | 300 | `1ef93157…6453` ✓ | `1edda239…5d69` ✓ |
| TensorRT | BasketballDrive | 0 | 500 | 1 + 63 | 15 | 3 | 500 | 262 | `599cae26…3fb0` ✓ | `3c8f3595…ec88` ✓ |
| TensorRT | BasketballDrive | 21 | 500 | 1 + 63 | 15 | 3 | 500 | 262 | `df34aec7…f58c` ✓ | `5102470a…260b` ✓ |
| TensorRT | BasketballDrive | 42 | 500 | 1 + 63 | 15 | 3 | 500 | 262 | `49b84b4f…a689` ✓ | `fd111051…3890` ✓ |
| TensorRT | BasketballDrive | 63 | 500 | 1 + 63 | 15 | 3 | 500 | 262 | `90d67790…e5aa` ✓ | `33c21d14…637b` ✓ |
| TensorRT | Kimono | 0 | 240 | 1 + 30 | 7 | 7 | 240 | 130 | `f16e3bd4…d339` ✓ | `a743bdf8…6be5` ✓ |
| TensorRT | Kimono | 21 | 240 | 1 + 30 | 7 | 7 | 240 | 130 | `8f2d4080…d1f6` ✓ | `72516e3b…2a72` ✓ |
| TensorRT | Kimono | 42 | 240 | 1 + 30 | 7 | 7 | 240 | 130 | `925fe984…79f2` ✓ | `dfdb8d58…14bf` ✓ |
| TensorRT | Kimono | 63 | 240 | 1 + 30 | 7 | 7 | 240 | 130 | `558cb6df…c613` ✓ | `b94f08a6…669d` ✓ |
| TensorRT | Jockey | 0 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `5802424c…622c` ✓ | `be281a79…3848` ✓ |
| TensorRT | Jockey | 21 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `01e870ca…3ede` ✓ | `a07f5f36…b688` ✓ |
| TensorRT | Jockey | 42 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `f5165325…ba81` ✓ | `fe6b7367…26f5` ✓ |
| TensorRT | Jockey | 63 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `cf501f71…0114` ✓ | `6ab28600…96e8` ✓ |
| TensorRT | HoneyBee | 0 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `ff09449e…0fb6` ✓ | `feeab6b5…f1f0` ✓ |
| TensorRT | HoneyBee | 21 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `ba66c89a…512c` ✓ | `4a26e723…855a` ✓ |
| TensorRT | HoneyBee | 42 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `ab64ba83…07e8` ✓ | `a08d7a6a…bb97` ✓ |
| TensorRT | HoneyBee | 63 | 600 | 1 + 75 | 18 | 7 | 600 | 310 | `6bc029bd…d744` ✓ | `deaf12c6…c50a` ✓ |

The HoneyBee QP 63 streams of both paths are also byte-identical to the
retained G3 streams of 2026-11-03 (`build/dcvcuf-c11-gates-20261103/`).

### Reset and drain with a partly filled group (#194)

`nvcr_dcvcuf_reconstruction_identity partial-group` on Akiyo QCIF
(`e1efee0e…9108`), QP 32. Pass on both paths; the TensorRT path uses the
`build/dcvcuf-trt-bundle-256` bundle (`engine.sha256` `8009e965…17b3`) because
of the QCIF finding below. It is also the `nvcr_dcvcuf_partial_group_lifecycle`
CTest case when `NVCR_DCVCUF_TEST_BUNDLE` and `NVCR_DCVCUF_TEST_QCIF` are set.

| Check | Fused | TensorRT |
|---|---|---|
| Drain: intra frame + 5 frames, flush. Two access units; exactly the 5 real frames with their timestamps; G1 identity on both steps | Pass | Pass |
| Encoder reset with 5 frames pending (different content), then 17 frames: no output after reset; stream equals a fresh session's | Pass | Pass |
| Decoder reset with 9 decoded frames queued: no output after reset; the next stream decodes to the fresh decoder's 17 frames | Pass | Pass |
| Encoder reset after a flushed partial group, then reuse: stream equals a fresh session's | Pass | Pass |

### Finding outside the G1 matrix: TensorRT 1080p bundle at QCIF

The G1 matrix is 1080p. On smoke runs, Akiyo QCIF with the 1080p bundle does
not decode: `dcvcuf.rans: malformed bitstream: UF entropy stream ended early`
at step 0 (the intra frame), with the production CLI as well as the harness,
and with a build of the branch before this change. The decoder fails cleanly
with an error (no frames are returned for the unit); whether any wrong frames
reach a file in other cases is a C13 cross-decode item.

- Localization (harness): `z_hat` (sign of zero only), hyper-decoder `hyper_params`
  and `common`, and the accumulated `y_hat` input of spatial-prior step 1 are
  identical between encoder and decoder; the output of `i_spatial_prior_1`
  differs in 3,688 of 50,688 elements (max 0.0156) in one run and 6,656 in
  another. The scales then select different entropy tables.
- Run-to-run: encoding one intra frame three times gives three different
  streams. `compute-sanitizer --tool initcheck` (CUDA 12.6) reports 0 errors.
- Stage level (`nvcr_tensorrt_stage_runner`, fixed random FP16 inputs
  `4d08d9fa…981d`/`dfae1837…0c3a`, 5 runs in each of 2 processes):
  `i_spatial_prior_1.plan` of the 1080p bundle (`5dd49524…4773`) gives 9
  distinct outputs out of 10 at latent 9x11 (max 0.047), and one output at
  16x16 and 68x120 (1080p). The 256 bundle's plan (`451afe18…7227`) gives one
  output at 9x11 and 16x16.
- Both bundles were built at builder optimization level 5; they differ in the
  profile (`frame_max` 1088x1920 against 256x256). The TensorRT tactic chosen
  for that stage in the 1080p profile is nondeterministic at that shape; the
  tactic itself was not identified.

Not fixed here. It bears on C13's cross-decode item (Akiyo QCIF on TensorRT)
and on the configurable profile in C13's 4K item. Remedies to choose between:
per-resolution profiles, a determinism gate per stage and shape at bundle
build, or restricting tactics for the affected stages.

### Regression

Origin/main `676efac4` against this branch, both Release with TensorRT and
ONNX Runtime 1.26.0 (`build-reg-main`, `build-reg-branch`), same inputs and
bundles; stream and decoded-YUV SHA-256:

| Case | `main` = branch |
|---|---|
| DCVC-RT TensorRT, Akiyo QCIF, 65 frames, GOP 8, QP 32, installed profile | `.nvcr` `74fc3cc5…3b56`, YUV `10e02e46…3008` (= B7 reconstruction) |
| DCVC-RT TensorRT, BasketballDrive 1080p, 100 frames, QP 32, `engines-rtx4070-exact-20260805/1080p` | `.nvcr` `6dc0dfbb…87ab`, YUV `c82a6734…22fa` |
| DCVC-RT ONNX Runtime CPU, Akiyo QCIF, 17 frames, GOP 8, `build/models/dcvcrt-qcif` | `.nvcr` `4b9442e2…b4bb`, YUV `c15808e9…c7e9` |
| MLVC ONNX Runtime CPU, Akiyo QCIF, 17 frames, `mlvc-bundles/generic` | `.nvcr` `296ecede…280d`, YUV `6577badf…5d57` |
| MLVC TensorRT, Akiyo QCIF, 17 frames, SM 8.9 MLVC bundle | `.nvcr` `1b3bfd0a…bf01`, YUV `9239b960…fb10` |
| DCVC-UF TensorRT, BasketballDrive 1080p, 33 frames, QP 32 | `.nvcr` `f548f2be…7e3e`, YUV `01a6bba0…541c` |
| DCVC-UF fused, Akiyo QCIF, 9 frames, QP 32, GOP 32 (C7 CLI vector) | `.nvcr` `39da424d…f7bb` (1,450 bytes), YUV `aee81727…8352`: equal to the C7 record |

CTest: `build-g1-trt` 31/31 and `build-g1-native` 30/30, including
`nvcr_dcvcuf_architecture_boundary` and the new lifecycle case.

The regression cases and CTest were repeated on the final build after the
final matrix run; all hashes above are unchanged. No timing is reported.

### Retained local outputs

`build-g1-runs/g1-20261107-final/` in the worktree (ignored), run
20:53-21:15 UTC on the committed code with the GPU otherwise idle:
`cases.jsonl` (`a1a0c524…df46`), `lifecycle.jsonl` (`1cdf678f…80f9`),
`summary.json` (`ea21a42e…8314`), per-case reports, both `.nvcr` streams per
case, and `commands.log`. Decoded YUV files were hashed and deleted. An earlier
run (`build-g1-runs/g1-20261107/`, before review fixes that only change when
decoder tensors are recorded and how the driver handles an empty selection)
gave the same 32 stream and YUV hashes and the same counts.

### Reproduction

```sh
python3 scripts/backends/dcvcuf/run_reconstruction_identity.py \
  --harness-native build-g1-native/tests/nvcr_dcvcuf_reconstruction_identity \
  --cli-native build-g1-native/cli/nvcr \
  --native-bundle <dcvcuf-cvpr2026-hts.pt> \
  --harness-tensorrt build-g1-trt/tests/nvcr_dcvcuf_reconstruction_identity \
  --cli-tensorrt build-g1-trt/cli/nvcr \
  --tensorrt-bundle <1080p stage bundle> \
  --dataset-root <datasets> \
  --lifecycle-qcif <akiyo_qcif.yuv> --lifecycle-tensorrt-bundle <256 stage bundle> \
  --output-dir <new directory>
```

### Not verified

- G1 on any GPU other than the RTX 4070, and on resolutions other than 1080p
  and QCIF.
- G1 at QCIF with the 1080p TensorRT bundle (fails to decode; finding above).
- The upstream README recipe (reset 128, skip 0.15); NVCR cannot express it yet.
- G2 (format) and G4 (upstream decoding NVCR TensorRT streams): open in C13.
- Run-to-run determinism of every TensorRT stage at every shape; only
  `i_spatial_prior_1` was swept, at three shapes.

## Cross-decode of identical bytes (C13, C11 G4) — 2026-11-08

Run under roadmap C13 (DCVC-UF qualification) on branch
`test/dcvcuf-cross-decode`, based on `main` `07424a6c`: code `388c45e5`
(hardening and instrumentation), `7ef5f8c8` (G4, malformed-stream and
determinism tools) and `3defed53` (entropy-parallel validation, found by the
malformed-stream runs). Every result below was produced by `3defed53` builds
unless it says otherwise. DCVC-UF exact numerics (#217) are not started here.

**Pass rule (fixed):** every pair decodes bit-exactly or fails cleanly
(defined NVCR error, no out-of-bounds read, no wrong frames written).
Bit-exactness is reported, not required.

**Result on the RTX 4070:** every NVCR decoder pair passes. Same-path decodes
are bit-exact (the TensorRT 1080p bundle at QCIF excepted, which fails
cleanly); every cross-path pair fails cleanly at the intra frame with zero
frames written. Before this change the same pairs wrote 17 and 25 wrong
frames silently. **G4 does not meet the pass rule:** pinned upstream decodes
NVCR fused streams bit-exactly, but completes NVCR TensorRT streams with wrong
frames (18-21 dB against the source), because upstream has no entropy
desynchronization check. **Across GPUs (RTX 4070 ↔ RTX 5060), both directions:** every pair fails
cleanly at the intra frame with zero frames written; none is bit-exact.

### Changes

- `third_party/dcvcuf_rans`: reads were already bounded (`ByteReader`). Added,
  as #221 did for `dcvc_rans`: initial-state validation (`[2^23, 2^31)`), a
  bypass group cap of 16, rejection of decoded values outside `int8` (the
  `int32` arithmetic that could overflow on such values is now
  `uint32`/`int64`), and `at_initial_state()`/`consumed_bytes()`
  (`UPSTREAM.md`).
- `UfRansCodec::finish_decode`: a frame's stream must end at the encoder's
  initial state with all bytes consumed (coder pairs: consumed bytes within the
  8-byte overlap the encoder merges). Called after the last y decode of every
  intra frame and HT-S group, on both paths. Missing before this change.
- Fused path: upstream calls the rANS decoder on its entropy worker thread, so
  a stream error threw there and **terminated the process** (`std::terminate`,
  observed). The native rANS shim now records the error, returns zero symbols
  of the requested count, and the session returns it as `malformed_bitstream`
  after `decompress` returns (accessor added by `materialize_native_sources.py`;
  the pinned checkout is untouched).
- Adapter: after a failed unit, HT-S groups are rejected until an intra unit
  decodes, so continuing past an error cannot emit frames from a broken
  reference. The payload parser rejects an entropy-parallel field above 8; on
  the fused path such a unit had reached the rANS shim's
  `set_entropy_coder_parallel`, which threw on upstream's worker thread and
  terminated the process (6 variants per sequence; the TensorRT session already
  rejected it).
- Test-only instrumentation: the UF coder reports every decode call (entropy
  indexes and symbols) and the end check to the diagnostics observer;
  `nvcr_dcvcuf_entropy_trace` decodes a `.nvcr` file like the CLI and writes
  the trace; `run_cross_decode.py` encodes, decodes and compares (first
  divergent index, first divergent symbol, completion, PSNR, drift).
- `export_tensorrt_stages.py --from-export` and `add_tensorrt_tables.py
  --tables`: another GPU builds plans from byte-identical ONNX graphs.

Codec/provider separation (`AGENTS.md`): the check lives in the codec-owned UF
coder and is called by both execution sessions; no provider branch was added to
codec semantics and no GPU identity enters codec code.

### Environment

- NVIDIA GeForce RTX 4070 (SM 8.9, 12,282 MiB, no other GPU workload), driver
  580.178.04, Linux 7.0.0-42-generic, GCC 13.3.0.
- Fused: `build-g1-native` rebuilt at this branch, CUDA 12.6.85, LibTorch
  2.13.0+cu126, bundle `dcvcuf-cvpr2026-hts.pt` (`81fe0a7d…37ab`).
- TensorRT: `build-g1-trt` rebuilt at this branch, CUDA 12.8.93, TensorRT
  10.9.0.34, bundle `build/dcvcuf-trt-bundle` (1080p profile; manifest
  `74597106…0a82`, model profile `77f06cf2…f675`), built from the ONNX export
  `build/dcvcuf-c8-tensorrt-20261102/onnx` (all 20 plan digests match its
  report; `build/dcvcuf-onnx-1080p` is a different re-export and does not
  reproduce the model profile). Supplementary: `build/dcvcuf-trt-bundle-256`,
  same export, 256x256 maximum profile.
- Upstream (G4): pinned DCVC `cbdae87a` (clean before and after), Python
  3.12.3, PyTorch 2.13.0+cu126, its built `inference_extensions_cuda`.

### Streams

Encoded once on the RTX 4070 and kept byte for byte: 100 frames, intra only at
frame 0 (`--gop-size 65528`), feature-memory reset every 32 frames (3 resets),
1 intra unit + 12 groups + a 3-frame tail. Inputs match
`c8_reference_sequences.json`. Encoded with the `388c45e5` builds (no
encoder change since `main`; `3defed53` touches only payload parsing on
decode).

| Encoder | Sequence | QP | Bytes | `.nvcr` SHA-256 |
|---|---|---:|---:|---|
| fused | BasketballDrive 1080p | 21 | 146,931 | `6bed076d…ea23` |
| fused | BasketballDrive 1080p | 42 | 480,898 | `ee195b6f…6153` |
| TensorRT 1080p | BasketballDrive 1080p | 21 | 146,683 | `8abb46ae…f82a` |
| TensorRT 1080p | BasketballDrive 1080p | 42 | 480,936 | `97b959ce…5888` |
| fused | Akiyo QCIF | 21 | 4,323 | `5defb573…0646` |
| fused | Akiyo QCIF | 42 | 7,744 | `ec11e44f…90b0` |
| TensorRT 1080p | Akiyo QCIF | 21 | 4,362 | `4d782bdf…4305` |
| TensorRT 1080p | Akiyo QCIF | 42 | 7,781 | `389fb937…efb5` |
| TensorRT 256 (supplementary) | Akiyo QCIF | 21 | 4,371 | `a2605b4e…a884` |
| TensorRT 256 (supplementary) | Akiyo QCIF | 42 | 7,781 | `bec5f0d3…64d4` |

### Same-GPU matrix (RTX 4070)

Each stream decoded with the production CLI and the trace harness on each path;
harness and CLI YUV are equal in all 30 decodes. Reference = the encoder's own
path. "Entropy" = first divergent entropy index as (unit, call, position):
call `y0`-`y3` are the intra frame's four spatial steps; the share of differing
indexes in that call and their largest difference follow.

| Encoder → decoder | Sequence, QP | Outcome | Frames written | First divergent index | Differing in that call |
|---|---|---|---:|---|---|
| fused → fused | BD 21 / 42 | bit-exact (YUV `b1fd4fe4…70a5` / `bd6f8cae…819b`) | 100 / 100 | — | — |
| TensorRT → TensorRT | BD 21 / 42 | bit-exact (`12d5e45c…4d25` / `2795f20d…1c73`) | 100 / 100 | — | — |
| fused → fused | Akiyo 21 / 42 | bit-exact (`8eb053a7…3391` / `76df8193…9a8c`) | 100 / 100 | — | — |
| TensorRT 256 → TensorRT 256 | Akiyo 21 / 42 | bit-exact (`6331e3a9…ce06` / `196f08ca…417b`) | 100 / 100 | — | — |
| TensorRT 1080p → TensorRT 1080p | Akiyo 21 / 42 | clean failure, unit 0: end check (QP 21), stream ended early (QP 42) | 0 / 0 | not comparable (the reference itself fails) | — |
| fused → TensorRT | BD 21 | clean failure, unit 0 | 0 | (0, y0, 1) | 1,557 of 522,240, ±1 |
| fused → TensorRT | BD 42 | clean failure, unit 0 | 0 | (0, y0, 129) | 3,101 of 522,240, ±1 |
| TensorRT → fused | BD 21 | clean failure, unit 0 | 0 | (0, y0, 3,680) | 1,566 of 522,240, ±1 |
| TensorRT → fused | BD 42 | clean failure, unit 0 | 0 | (0, y0, 78) | 3,117 of 522,240, ±1 |
| fused → TensorRT 1080p / 256 | Akiyo 21 | clean failure, unit 0 | 0 | (0, y0, 61) | 65 / 51 of 6,336, ±1 |
| fused → TensorRT 1080p / 256 | Akiyo 42 | clean failure, unit 0 | 0 | (0, y0, 3) | 114 / 109 of 6,336, ±1 |
| TensorRT 1080p → fused / 256 | Akiyo 21 | clean failure, unit 0 | 0 | (0, y0, 8) | 52 / 49 of 6,336, ±1 |
| TensorRT 1080p → fused / 256 | Akiyo 42 | clean failure, unit 0 | 0 | (0, y0, 100) | 117 / 114 of 6,336, ±1 |
| TensorRT 256 → fused / 1080p | Akiyo 21 | clean failure, unit 0 | 0 | (0, y0, 9) / (0, y0, 27) | 71 / 57 of 6,336, ±1 |
| TensorRT 256 → fused / 1080p | Akiyo 42 | clean failure, unit 0 | 0 | (0, y0, 8) | 104 / 96 of 6,336, ±1 |

Every failure is `malformed_bitstream` (`UF entropy stream ended early`, or
`UF entropy decoding desynchronized` from the end check), CLI exit 1, an empty
output file. The first symbol divergence follows the first index divergence
within the same call (positions 35-4,552). No cross-path pair completes, so
there is no PSNR or drift to report for NVCR decoders on this GPU. (The 256
bundle decoding a 1080p stream fails before entropy decoding with a TensorRT
profile error, `backend_error`; not a planned pair.)

**Before this change** (merged G1 code, `5dd5c934`, equal to `main`
`07424a6c` for decoding; G1 BasketballDrive QP 21, 500-frame streams): the
TensorRT stream on the fused decoder wrote 17 frames at 20.8-26.0 dB PSNR-Y
against the TensorRT path's own decode of it (20.6-25.5 dB against the source,
where the correct decode gives 30.9-35.4 dB), then terminated the process
(`std::terminate`, exit 134). The fused stream on the TensorRT decoder wrote 25
frames at 19.9-25.8 dB against the fused path's own decode, then failed with
`UF entropy stream ended early`. Those were silent wrong frames; the end check now
rejects unit 0.

### Entropy-deciding stages (for #217)

z is coded with fixed CDF rows (QP × channel); no network output selects them.
y entropy indexes come from predicted scales through `build_index_dec` (scale
table, skip threshold 0), the step mask (`single_part_for_reading_4x`) and the
skip-condition compaction (`conditional_index_part1`), so the scales decide
both each symbol's CDF row and which symbols are coded.

| Unit | Index source (TensorRT stage / fused module) | Upstream inputs that reach it |
|---|---|---|
| Intra, step 0 | `i_hyper_decoder` (hyper decoder, prior fusion, spatial-prior reduction) + crop | z_hat |
| Intra, steps 1-3 | `i_spatial_prior_1..3` (adaptor k + spatial prior) | y_hat so far (decoded symbols + means of earlier steps) and `common` from `i_hyper_decoder` |
| HT-S group | `p_hyper_decoder` + crop + `p_prior` (scales; one y call) | z_hat, `memory`, `q_feature` |
| HT-S `memory` | `p_feature_adaptor_i` (after the intra frame or a reset group) or `p_feature_adaptor_m` | intra: `i_decoder` output (pixel unshuffle); later: the previous group's `p_decoder` feature (and `p_reset` after a reset) |

So every intra-frame decoder stage except `i_decoder` decides intra indexes,
and an HT-S group's indexes depend on nearly the whole decoder of every earlier
unit: `i_decoder`, `p_feature_adaptor_i/m`, `p_feature_extractor` (context into
`p_decoder`), `p_spatial_prior_1..3` (means into `y_hat`), `p_decoder` and
`p_reset`. Only the encoder stages and `p_reconstruction` (pixels) never reach
an entropy index. On this GPU every cross-path divergence starts at intra
step 0, i.e. in `i_hyper_decoder`'s scales.

### What the end check cannot detect

A wrong CDF row goes unnoticed when it holds the coded slot with the same start
and frequency as the right row: the rANS state then evolves identically and at
most that symbol's value changes. `check_uf_cdf_aliasing.py` on the bundle's
`tables.bin` (image and video y tables are identical): 424 such
(row, wrong row, slot) cases over all row pairs, symbol probability at most
48/65,536; for wrong rows within ±4: pairs 26-29↔30, 42↔46, 76↔79; within ±1
(every divergence observed here): 2 cases, rows 29↔30, probability 1/65,536.
Corruption of the bytes themselves is also undetectable where it lands in raw
bypass bits; the UF format has no integrity field.

### Bounded decoding and sanitizers

- CPU conformance test (`nvcr_dcvcuf_rans_conformance`), Debug, ASan + UBSan on
  every target including `third_party/dcvcuf_rans`
  (`-fsanitize=address,undefined -fno-sanitize-recover=all`), coder counts
  1-8: every truncation length rejected; trailing bytes and under-read
  rejected by the end check; 50/50 random streams rejected per count; a
  single-index divergence every 37 positions rejected (81-82 per count; 3
  skipped as aliased slots); one-bit flips at every third byte: 82-89%
  rejected, the rest undetectable by construction (bypass-heavy fixture). No
  sanitizer report.
- End-to-end, TensorRT path, RelWithDebInfo ASan + UBSan build
  (`build-t3-asan-trt`), `make_malformed_streams.py` variants (file
  truncation, UF stream cut with consistent framing, bit flips at the initial
  state/middle/end, random and zeroed UF streams, entropy-parallel and flag
  fields, dropped and swapped units) of the intra unit and groups 1 and 6:
  - Akiyo QCIF on the 256 bundle: 71/71 rejected by the CLI with a defined
    error, no sanitizer report, every written frame equal to the clean
    decode's (only frames before the damaged unit; at most 97).
  - BasketballDrive 1080p QP 42 on the 1080p bundle: 68/68 rejected, no
    sanitizer report, every written frame equal to the clean decode's.
- Fused path, Release (LibTorch is not sanitizer-instrumented), same variants
  of fused Akiyo QP 42 and BasketballDrive QP 21: 71/71 and 68/68 rejected,
  no crash, every written frame equal to the clean decode's. The first run
  (`388c45e5`) crashed on the 6 entropy-parallel 9/255 variants per sequence;
  fixed in `3defed53`, and all four sets were rerun on it.
- The trace harness agrees except where its own minimal file reader is
  involved (truncated records end the harness with a container error; an
  8-byte header-only file decodes zero frames); those cases are the CLI's.
- Every error is `malformed_bitstream` (entropy stream ended early, end check,
  invalid payload fields or entropy offsets, truncated unit) or the CLI's
  record/sequence errors.

### G4: pinned upstream decoding NVCR streams

`decode_with_upstream.py` passes each unit's UF payload, QP, entropy-parallel
count and reset bit to upstream's `DMCI.decompress` / `DMC.decompress` in
test_video.py's order and writes test_video.py's YUV conversion (the fused
image proxy is created with the lines upstream runs inside `compress`).

| NVCR encoder | Sequence, QP | Upstream outcome | Against NVCR reference decode | PSNR-Y against source (mean) |
|---|---|---|---|---|
| fused | BD 21 / 42, Akiyo 21 / 42 | completes, 100 frames | bit-exact (4/4) | 31.04 / 34.13 / 34.75 / 39.26 dB |
| TensorRT 1080p | BD 21 | completes silently | min 17.8 dB | 19.00 dB (25.5 at frame 0 → 18.0 at frame 99) |
| TensorRT 1080p | BD 42 | completes silently | min 16.2 dB | 18.22 dB (25.9 → 16.5) |
| TensorRT 1080p | Akiyo 21 / 42 | completes silently | NVCR reference fails | 20.13 / 20.43 dB |
| TensorRT 256 | Akiyo 21 / 42 | completes silently | min 17.1 / 20.7 dB | 17.59 / 21.25 dB |

Upstream decodes NVCR fused streams exactly (C7 already showed the payloads are
upstream's). On NVCR TensorRT streams upstream's rANS desynchronizes at the
intra frame like NVCR's fused decoder, but it has no end check, so it emits
wrong frames, drifting further down over the 100 frames. This is a G4 result
(reported only) and a pass-rule failure on the upstream side; NVCR cannot
change it. NVCR TensorRT streams are not decodable by upstream, and neither
are streams encoded on another GPU (see Cross-GPU).

### Stage determinism at the QCIF latent

`check_stage_determinism.py`, fixed random FP16 inputs (seed 217), 10 runs in
each of 2 processes (distinct outputs of 20):

| Bundle (RTX 4070) | Latent | `i_spatial_prior_1` | `_2` | `_3` | `p_spatial_prior_1` | `_2` | `_3` |
|---|---|---:|---:|---:|---:|---:|---:|
| 1080p | 9x11 (QCIF) | 9 | 16 | 17 | 14 | 18 | 20 |
| 1080p | 16x16 and 68x120 | 1 | 1 | 1 | 1 | 1 | 1 |
| 256 | 9x11 and 16x16 | 1 | 1 | 1 | 1 | not run | not run |

RTX 5060 (relayed from its session, same stage list except `p_spatial_prior_2/3`):
1080p bundle at 9x11: 6 / 13 / 9 / 4 of 20; one output at 16x16, 68x120 and on
its 256 bundle. The 1080p plans' tactic at the QCIF shape is nondeterministic on
both GPUs and in all spatial-prior stages, not only `i_spatial_prior_1`. Not
fixed in T3.

### Regression

- `main` `07424a6c` (`build-reg-main`) against this branch (`build-reg-branch`),
  TensorRT and ONNX Runtime 1.26.0, same inputs; stream and YUV identical, and
  equal to the G1 record: DCVC-RT TensorRT Akiyo QCIF 65 frames
  (`74fc3cc5…3b56` / `10e02e46…3008`), DCVC-RT TensorRT BasketballDrive 100
  frames (`6dc0dfbb…87ab` / `c82a6734…22fa`), DCVC-RT ONNX Runtime CPU
  (`4b9442e2…b4bb` / `c15808e9…c7e9`), MLVC ONNX Runtime CPU
  (`296ecede…280d` / `6577badf…5d57`), MLVC TensorRT (`1b3bfd0a…bf01` /
  `9239b960…fb10`), DCVC-UF TensorRT 33 frames (`f548f2be…7e3e` /
  `01a6bba0…541c`).
- Valid DCVC-UF streams on both paths equal the G1/C7 records: fused
  BasketballDrive QP 21 500 frames `0d45dc7c…ae6d` / `58c6e483…468c`, fused
  C7 vector `39da424d…f7bb` / `aee81727…8352`, TensorRT BasketballDrive QP 21
  500 frames `df34aec7…f58c` / `5102470a…260b`.
- CTest: `build-g1-trt` 31/31, `build-g1-native` 30/30 (on `3defed53`).
- The regression table above was repeated on the `3defed53` branch build; all
  hashes unchanged.

### Cross-GPU (RTX 4070 ↔ RTX 5060)

The RTX 5060 Laptop side was run by a separate session on that machine; its
package `/home/oelghati/nvcr-t3-exchange/from-rtx5060/` (156 files,
`SHA256SUMS` `7dac1044…1e6f`) verified here with `sha256sum -c`. The rows from
the 5060 are relayed from its `compare.jsonl` and `run-metadata.json` in that
package, not re-run here.

5060 environment (from its metadata): RTX 5060 Laptop (SM 12.0, 26 SMs,
8,151 MiB), driver 591.86 (Windows 11 host, Ubuntu 24.04 container under WSL2),
built from `git archive` of `388c45e5`, Release, SM 120. TensorRT path: CUDA
12.8.93, TensorRT 10.9.0.34 (the same versions as here); its 1080p bundle
(manifest `96d364a3…1021`) and 256 bundle (`40bf28d2…2657`) built with
`--from-export` from the 4070's export, model profile `77f06cf2…f675`; every
plan digest differs from the 4070's, and two 1080p builds on the 5060 differ
from each other. Fused path: the same `.pt` bundle (`81fe0a7d…37ab`) but
LibTorch 2.11.0+cu128 with CUDA 12.8.93 (LibTorch 2.13.0+cu126 has no SM 12.0
code), so a fused cross-GPU result is not a GPU-only comparison.

| Direction | Pairs | Outcome | First divergent index |
|---|---:|---|---|
| 4070 → 5060 (relayed) | 30 (fused, TensorRT 1080p, TensorRT 256 decoders × 10 streams) | 24 clean failures at the intra unit, zero frames written; 6 clean failures of the 256-bundle decoder on 1080p streams (TensorRT profile error, not a planned pair); none bit-exact | intra `y0`, ±1, in every entropy failure |
| 5060 → 4070 (here, `3defed53` builds) | 30 (same layout) | 24 clean failures at the intra unit (`malformed_bitstream`: stream ended early, or the end check), zero frames written; 6 clean failures of the 256-bundle decoder on 1080p streams; none bit-exact | intra `y0`, ±1, in every entropy failure (positions 1-3,168) |
| 5060 → 5060 (relayed) | 30 | bit-exact on the encoder's own path for fused 4/4, TensorRT 1080p on BasketballDrive 2/2, TensorRT 256 on Akiyo 2/2; every cross-path pair fails cleanly at the intra unit | intra `y0`, ±1 |

Repeats on the 5060 (4 runs each): BasketballDrive and fused decodes gave
identical traces each time; Akiyo on the 1080p-bundle decoder always failed
at the intra unit, but in a different call from run to run (the QCIF
nondeterminism below).

One 5060 same-path row needs attention: its own Akiyo QCIF QP 21 stream on its
1080p bundle wrote 25 frames before failing cleanly at a later unit. Those
units passed the end-of-stream check, so their entropy symbols were in sync
(apart from aliased slots), but with nondeterministic spatial priors the means,
and so the pixels, need not equal the encoder's; they were not compared with
the encoder's reconstruction. It is the known QCIF finding below, kept in the
matrix as required; on this GPU the same case fails at unit 0.

**Pass rule across GPUs:** every NVCR decoder pair in both directions decodes
bit-exactly (same GPU, same path) or fails cleanly; no pair wrote a frame
after its first divergence. No cross-GPU pair is bit-exact: cross-GPU decode of
DCVC-UF streams is not supported on either path, and fails cleanly.

G4 across GPUs: pinned upstream on the 4070 decoding the ten 5060 streams
completes all of them with wrong frames (17.9-20.5 dB mean PSNR-Y against the
source), fused streams included (LibTorch and GPU both differ).

### Retained local outputs

- `/home/oelghati/nvcr-t3-exchange/from-rtx4070/` (`SHA256SUMS`
  `00f3a684…7aa6`): streams, manifest, run metadata, the ONNX export, 4070
  reference traces and YUV.
- `build-t3-runs/rtx4070-final/` in the worktree (ignored): `decodes.jsonl`,
  `compare.jsonl`, traces, reference YUV of the final run (`3defed53`). Its 26
  planned pairs equal the earlier run `build-t3-runs/rtx4070-20261107/`
  (`388c45e5` builds; verdicts, frames written, YUV digests and first
  divergent indexes), which also holds `upstream/` (G4).
- `build-t3-runs/rtx5060-to-4070/`: decodes of the 5060 streams here,
  `compare.jsonl`, upstream decodes of them.
- `build-t3-runs/malformed/final/` (final malformed-stream results),
  `build-t3-runs/determinism-*`, `build-t3-runs/regression/`.
- `/home/oelghati/nvcr-t3-exchange/from-rtx5060/` (5060 package).

### Reproduction

```sh
python3 scripts/backends/dcvcuf/run_cross_decode.py encode --cli-native <fused cli> \
  --cli-tensorrt <tensorrt cli> --native-bundle <.pt> --tensorrt-bundle <1080p bundle> \
  --dataset-root <datasets> --gpu-label rtx4070 --output-dir <new dir>
python3 scripts/backends/dcvcuf/run_cross_decode.py decode --streams <dir> --label tensorrt \
  --cli <cli> --harness <build>/tests/nvcr_dcvcuf_entropy_trace --bundle <bundle> \
  --gpu-label rtx4070 --output-dir <run>          # and --label native
python3 scripts/backends/dcvcuf/run_cross_decode.py compare --decodes <run>/decodes.jsonl \
  --dataset-root <datasets> --output <run>/compare.jsonl
python3 scripts/backends/dcvcuf/decode_with_upstream.py --reference-root <dcvc-uf> \
  --image-checkpoint <image> --video-checkpoint <video> --streams <dir> --output-dir <run>/upstream
```

### Not verified

- G2 (format and malformed-unit rejection as a recorded gate); the malformed
  runs above cover rejection on two sequences only.
- Fused path under a sanitizer (LibTorch); only its Release behaviour was run.
- Cross-path decode of the README recipe, at 4K, or on GPUs other than the
  RTX 4070 and RTX 5060; the 5060 rows are relayed, not re-run here.
- Whether the frames a nondeterministic QCIF decode on the 1080p bundle writes
  before failing equal the encoder's reconstruction (5060, 25 frames).
- That no cross-path pair ever completes with wrong frames: the end check
  misses the aliased slots listed above.
