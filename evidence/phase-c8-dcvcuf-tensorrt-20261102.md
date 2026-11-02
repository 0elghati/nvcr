# Phase C8 DCVC-UF on the TensorRT provider — 2026-11-02

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

A second capture with upstream's README recipe (`--reset_interval 128`,
`--skip_thres 0.15`) was still running at commit time; NVCR cannot express
that recipe yet (C7 remaining gate), so it is context for later, not a C8 gate
input. It will be recorded when complete.

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

## Next

Phase 1: manifest-driven TensorRT stage loading, with DCVC-RT byte parity.
