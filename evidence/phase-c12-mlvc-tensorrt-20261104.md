# C12 evidence: MLVC on the TensorRT provider (2026-11-04)

Status: Phases 0 and 1 pass; Phase 2 partial; Phase 3 open. Diagnostic numbers
only: not the `docs/performance.md` protocol.

## Environment

- RTX 5060 Laptop GPU (SM 12.0, 26 SMs), driver 591.86, Windows 11 host.
- Docker image `nvcr-trt-ort-toolchain:local`: CUDA 12.8.1, TensorRT 10.9.0.34,
  ONNX Runtime 1.26.0 (`/opt/onnxruntime`), GCC, Ninja.
- NVCR Release build, `-DCMAKE_CUDA_ARCHITECTURES=120 -DNVCR_ENABLE_TENSORRT=ON
  -DNVCR_ENABLE_ONNXRUNTIME=ON`; clean, no warnings.
- Model: libmlvc `dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1.tar`,
  SHA-256 `c37ee2f5aa418c1fbdd44e4bd95483faad84cbea592367b305e787042f845457`
  (the C10 bundle), canvases 320x192, 432x240, 640x368, 960x544.

## Phase 0: parser and operators

`trtexec --fp16` parses and builds `MLVCEncoder` and `MLVCDecoder` for every
canvas with no unsupported operator or fallback. Tensor I/O stays as declared:
FP16 image/feature/latent tensors and the rank-1 INT32 `q_index_shifted`.

## Phase 1: engine bundle, session and artifacts

`scripts/backends/mlvc/build_tensorrt_bundle.py --target-profile
configs/targets/rtx5060-laptop-ubuntu2404.json --hardware-compatibility
same_compute_capability` writes the unchanged ONNX bundle plus eight plans
(builder level 3, 1 GiB workspace) and an `nvcr.engine-bundle.v2` manifest:
model `mlvc`, engine profile `s-generic`, `shape_profile` `fixed`.

Code changes:

- The TensorRT session binds INT32, FP16 and FP32 tensors of rank 1 to 4, and
  accepts pinned host buffers (zero copy) besides device buffers.
- MLVC names TensorRT stages by plan file, uses pinned buffers, and loads and
  finalizes every canvas at session creation.
- `nvcr_artifacts.py`, `package_engine_bundle.sh` and the catalog merge accept
  backend `mlvc`. MLVC archives may carry the ONNX bundle; DCVC-RT archives still
  may not. Each backend has its own rolling release (`engine-assets-mlvc`), and
  the merge refuses mixed catalogs, because released installers reject unknown
  profiles. Installing MLVC never moves `profiles/default`.

Admission dry run (`build-c12/admit_local.sh`, the workflow's own
`download_engine_release_assets.py` with `file://` URLs): packaging, archive and
bundle validation, target-profile digest, catalog merge and
`install_catalog_assets` on the detected GPU all pass.

| Item | Value |
|---|---|
| Archive | `nvcr-engines-linux-amd64-sm120-mlvc-s-generic.tar.gz` |
| Size | 94,060,655 bytes |
| SHA-256 | `af32f837dd939490ae1992e05ee10a245adf1147daceb55f6970ced37eed8a5d` |
| Staging | `s3://nvcr-release-assets-<account>-eu-west-1/engine-assets-mlvc/` |

Tests: `tests/engine_catalog_tests.py` 25/25 (three new MLVC cases), and
`tests/artifact_tool_tests.py` passes from an LF copy. In the Windows checkout,
CTest's `nvcr_artifact_profiles` and `nvcr_native_installer` fail on CRLF shell
scripts and `nvcr_measurement_semantics` on missing numpy in the image; all
three are environment failures that predate C12.

## Phase 2: streams across providers (partial)

65 frames, QP 32, the bundle's I period, run from the installed archive.
PSNR is pooled Y against the source.

| Input | Stream | Decoded on ORT CPU | Decoded on TensorRT | Stream bytes |
|---|---|---|---|---|
| akiyo 176x144 | ORT CPU | 33.707 | 33.706 | 11,648 |
| | TensorRT | 33.639 | 33.637 | 11,680 |
| BasketballDrive 640x360 | ORT CPU | 30.983 | 30.940 | 78,661 |
| | TensorRT | 30.938 | 30.988 | 78,682 |
| BasketballDrive 960x540 | ORT CPU | 32.077 | 31.878 | 162,166 |
| | TensorRT | 31.948 | 32.109 | 162,384 |

Decoder-to-decoder drift on one stream (lowest Y PSNR between ORT CPU and
TensorRT decodes, then at the frame-64 LTR recovery): 57.5/58.9 dB at 176x144,
44.4-44.8/58.4-58.6 dB at 640x360, 39.1-39.7/58.3-58.5 dB at 960x540. No
frame is byte-identical, as expected across providers. Not yet compared with
pinned upstream reference vectors.

## Phase 3: speed (diagnostic)

Steady rate over frames 2-65 from the CLI, one run each. The CLI's codec time
excludes TensorRT start-up, which loads and warms all eight plans when the
session is created (about 3 s of process time).

| Canvas | Encode ORT CPU | Encode TensorRT | Decode ORT CPU | Decode TensorRT | TensorRT model ms (enc/dec) |
|---|---|---|---|---|---|
| 320x192 | 66.6 | 555.0 | 52.0-65.6 | 642-645 | 1.63 / 1.43 |
| 640x368 | 20.4 | 304.2 | 12.8-15.3 | 337-367 | 2.63 / 2.15-2.39 |
| 960x544 | 6.5 | 160.2 | 8.7-8.8 | 174-176 | 3.79 / 4.03-4.06 |

Known costs still inside these numbers: per-frame pinned allocation of the
feature tensor (`cudaMallocHost`/`cudaFreeHost`) and zero-copy engine I/O over
PCIe.

Publication (2026-11-05): `upload-engine-assets.yml` run 41997497504 from
`feat/mlvc-tensorrt` published the archive to `engine-assets-mlvc`; an
anonymous `nvcr-artifacts install --backend mlvc` from GitHub then encoded and
decoded 640x360 on TensorRT (275/349 fps, PSNR-Y 30.988 dB). The
`engine-assets` DCVC-RT catalog was unchanged (42 entries).

## Variants (2026-11-05)

Profiles are now one per MLVC model and weights: `s-perceptual` (the profile
above, renamed from `s-generic`), `perceptual` and `psnr`. Installers and the
catalog merge skip catalog entries of profiles they do not register, so the
published `s-generic` entry drops out when the three are published and later
profiles do not break older installers.

The standard MLVC models have no published ONNX bundle.
`scripts/backends/mlvc/export_onnx_bundle.sh` exports them with MLVC's
`convert.py` at the pinned checkout `1425880`, from the published checkpoints
`mlvc-perceptual-v1.ckpt` (SHA-256 `0ba78176…4723e`) and `mlvc-psnr-v1.ckpt`
(`834adaab…5cca1`), 73,428,306 bytes each. Exporter environment: CPU PyTorch
2.10.0 and MLVC's `uv.lock` versions (onnx 1.22.0, onnxscript 0.7.1, onnx-ir
0.2.1, torchao 0.17.0, numpy 2.3.2), with ONNX Runtime 1.26.0 and
`packages/msrtc_rans`. Upstream validates each canvas on VCD clips (15.8 and
35.6 GB, with no 240p/180p sets published); the export used akiyo (320x192,
432x240) and BasketballDrive (640x368, 960x544) instead, so the metrics in the
bundles' metadata describe those clips.

The exporter's PyTorch-against-ONNX check on those clips (Q index 21 and 63):

| Canvas | `perceptual` PSNR-Y ONNX − PyTorch | bpp difference |
|---|---|---|
| 320x192 | +0.03 / +0.05 dB | −0.0004 |
| 432x240 | −0.07 / +0.04 dB | −0.0005 |
| 640x368 | +0.04 / 0.00 dB | 0.0000 |
| 960x544 | −0.01 / −0.01 dB | 0.0000 |

NVCR runs the standard model unchanged on ONNX Runtime CPU (first run of that
model in NVCR) and on TensorRT from the `perceptual` engine bundle. 65 akiyo
frames, QP 32; the standard model's I period is 64, so frame 64 is an I frame:

| Stream | Decoded on ORT CPU | Decoded on TensorRT | Bytes | Decoder drift (min Y) |
|---|---|---|---|---|
| ORT CPU | 34.565 | 34.549 | 12,655 | 54.63 dB |
| TensorRT | 34.624 | 34.612 | 12,565 | 55.02 dB |

Steady rates: 267 fps encode and 287-312 fps decode on TensorRT against 11-14
fps on ONNX Runtime CPU (measured while another plan build used the GPU).

## Open

- Phase 2 against pinned upstream vectors (none exist for the standard
  models); Phase 3 under the protocol.
- MLVC-S psnr is not exported.
- Other GPUs and TensorRT versions need their own bundles.
