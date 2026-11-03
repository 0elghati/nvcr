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

Not yet run: the 1080p bundle, gates G1-G3 at 1080p on the reference set,
upstream decoding NVCR streams (G4), and speed. They wait for a free GPU.
