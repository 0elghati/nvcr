# Phase C8 ONNX Runtime provider evidence — 2026-11-01

## Scope

- Branch: `feat/onnxruntime-provider`
- Base: `3ecdddefd26efbda72cb81be903dc749a2ed28ac` (`main`)
- Priority: C8 was moved ahead of C7 on 2026-11-01 so one provider covers CPU
  development, CUDA comparison, and a later Snapdragon NPU target (C9).
- No TensorRT backend, CUDA kernel, NVAU, payload, or entropy code changed.

## Change

- `src/dcvcrt/host_ops.*`: host versions of the 25 DCVC-RT inter-stage CUDA
  kernels, computing in float and rounding to FP16 at the same points.
- `src/dcvcrt/host_backend.*`: a DCVC-RT `CodecBackend` that drives the
  existing `IntraOrchestration`/`PredictedOrchestration` through the generic
  `IProviderSession` contract. Stage inputs and outputs use provider buffers;
  stage shapes mirror the TensorRT engine specifications.
- `IProviderBuffer::host_bytes()`: non-pure accessor, empty by default, so
  host-domain providers can expose storage. Existing buffers are unchanged.
- `src/provider/onnxruntime/session.*`: ONNX Runtime provider session with
  synchronous completions, owner/shape/byte validation, optional SHA-256 stage
  checks, and component-id path validation. Registers `onnxruntime-cpu`,
  `onnxruntime-cuda`, and `onnxruntime-qnn` only when the linked runtime
  reports that execution provider.
- `src/dcvcrt/adapter.cpp`: the TensorRT provider keeps the TensorRT backend;
  every other provider gets the host backend.
- CLI: the `engine_manifest.json` bundle check now applies only to TensorRT.
- Build: `NVCR_ENABLE_ONNXRUNTIME`, `cmake/FindOnnxRuntime.cmake`, package
  config dependency, and `NVCR_ONNXRUNTIME_MODEL_DIR` for model tests.

## Environment

- Host: Intel Core Ultra 9 275HX, Windows 11, Docker Desktop (WSL 2).
- Container: `mcr.microsoft.com/devcontainers/base:noble` plus CMake 3.28.3,
  GCC 13.3.0, Ninja, and `onnxruntime-linux-x64-1.26.0.tgz` (CPU) from the
  `microsoft/onnxruntime` v1.26.0 GitHub release.
- Models: local FP16 opset-18 export in `build/models/dcvcrt` (fourteen stages
  plus `i_/p_entropy.bin` and `i_/p_quant.bin`), torch 2.9.1+cu128 exporter.
- Execution provider: ONNX Runtime CPU.

## Results

| Gate | Result |
|---|---|
| Release build, `NVCR_ENABLE_ONNXRUNTIME=ON` | Pass; no warnings in new or changed files |
| `nvcr_dcvcrt_host_ops` (FP16 conversion over all finite halves, kernel vectors, encoder/decoder index and restoration symmetry) | Pass |
| `nvcr_onnxruntime_session_contract` | Pass |
| Host backend I/P self-conformance, 176x144 | Pass: encoder and decoder reconstructions byte-identical; I 165 B, P 58 B |
| Same, 200x120 (non-multiple-of-16 padding) | Pass: I 154 B, P 65 B |
| Same, 1280x720 (two-coder entropy path) | Pass: I 1,718 B, P 264 B |
| Fresh-session determinism | Pass: identical I payload |
| Runtime GOP-2, four frames, reset to intra | Pass |
| Malformed payloads (truncated, bit-flipped, width ≥ 4096) | Pass: no crash; oversized width rejected as `malformed_bitstream` |
| ASan/UBSan with leak detection: host ops and 176x144/200x120 round trips | Pass; same payload sizes as Release |
| Full CTest, Release | 23/26 pass; see environment failures below |

### 65-frame CLI fixture

The B7 TensorRT parity fixture, run through ONNX Runtime CPU: first 65 frames of
`akiyo_qcif.yuv`, 176x144, QP 32, GOP 8, 30 fps.

| Measure | ONNX Runtime CPU | TensorRT (B7 record) |
|---|---:|---:|
| NVAU payload bytes | 14,618 | 14,669 |
| PSNR-Y / U / V | 36.658 / 39.103 / 43.815 dB | not recorded |
| `.nvcr` SHA-256 | `6e507123d21cf69f21e54baf72755d75e9e028325a33534669595df36a2a361d` | `9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a` |
| Decoded YUV SHA-256 | `94ebdcff3e019d9a7b54f45cbf21e51c225892f21db4d942276dbdc427ffc243` | `10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008` |

A second encode produced a byte-identical `.nvcr`. Payload size differs from
TensorRT by -0.35%. The bitstreams differ, as expected for a different FP16
execution path; byte interchange between providers is not claimed.

CPU timings (encode 1.75 fps, decode 2.54 fps) overlapped another build and are
not performance evidence.

## TensorRT regression and CUDA parity

- Container: `docker/Dockerfile.x86_64` toolchain stage with the Blackwell
  arguments (`nvidia/cuda:12.8.1-devel-ubuntu24.04`, TensorRT 10.9.0.34,
  `CMAKE_CUDA_ARCHITECTURES=120`), plus the same ONNX Runtime 1.26.0 CPU package.
- GPU: RTX 5060 Laptop (SM 12.0), driver 591.86, no other GPU workload.
- Engines: exact `build/engines/sm120-family-20260805` bundles (TensorRT 10.9.0,
  CUDA runtime 12080) for qcif, cif, 360p, 540p, 720p, and 1080p.
- Baseline: `main` at `3ecddde` built in a separate worktree with the same image.

| Gate | Result |
|---|---|
| Branch Release build with TensorRT and ONNX Runtime | Pass; 0 compiler warnings |
| `nvcr_cuda_ops` including `host_parity_test` | Pass; 0 differing elements in every compared output (below) |
| Six exact-profile engine contracts and I/P round trips | Pass (12 tests) |
| ONNX Runtime CPU round trip in the same build | Pass |
| Full CTest | 36/39 pass; the same three environment failures |
| 65-frame Akiyo fixture, `main` vs branch, TensorRT | `.nvcr` and decoded YUV byte-identical |

Host-vs-CUDA parity on 4,096–16,384 random elements per output (four mask
stages for the I-frame path): `round_to_int8` (symbols and values),
`make_four_way_mask`, `process_with_mask` (symbols and reconstruction),
`reduce_masked_quarters`, `build_encode_indexes`, `build_decode_indexes`,
`apply_image_prior_and_quantize` (params and latent), `clamp_reciprocal_with_quant`,
`add_and_multiply`, `yuv420p8_to_ycbcr_padded`, `ycbcr_padded_to_yuv420p8`, and
`rgb24_to_ycbcr_padded` all reported 0 differences, including the kernels that
use `expf`/`logf` or FMA-eligible arithmetic.

TensorRT fixture (both trees, three runs each): 14,667 payload bytes;
PSNR-Y/U/V 36.670/39.155/43.837 dB.

- `.nvcr` SHA-256: `ed193fd4af1d27e0e3480456c5e3da8093225d92a2ddd1e5908319cfc833808d`
- Decoded YUV SHA-256: `f55cec04740179d17d9687abcee12867740696de51f929f108d0581fdd8776a4`

These differ from the B7 RTX 4070 hashes because the plans are GPU-specific.
Diagnostic codec-interval throughput (not the `docs/performance.md` protocol):
`main` encode 124.9–125.2 fps and decode 130.1–147.5 fps; branch encode
120.0–127.1 fps and decode 103.7–152.8 fps. The TensorRT hot path is
unchanged; the spread is run-to-run noise.

## Cross-provider comparison

| Same fixture | TensorRT (RTX 5060) | ONNX Runtime CPU |
|---|---:|---:|
| Payload bytes | 14,667 | 14,618 (-0.33%) |
| PSNR-Y | 36.670 dB | 36.658 dB |

Cross-provider decode completes without error but reconstructs the wrong
frames: the TensorRT stream decoded on ONNX Runtime CPU gives PSNR-Y 19.63 dB,
and the ONNX Runtime stream decoded on TensorRT gives 20.02 dB. NVAU carries no
provider or numeric-path identity, so a decoder cannot detect this mismatch.
The ONNX Runtime stream decoded identically in the CPU and CUDA toolchain
images.

### Environment failures

`nvcr_artifact_profiles` and `nvcr_native_installer` fail because
`core.autocrlf=true` checks shell scripts out with CRLF, which Linux bash
rejects (`set: pipefail: invalid option name`). `nvcr_measurement_semantics`
fails because the container has no NumPy. None of these files changed.

## Not verified

- The pinned Python/native I-frame golden; it needs the DCVC-RT Python
  reference environment, which the containers do not include.
- `onnxruntime-cuda` (needs the GPU ONNX Runtime package) and `onnxruntime-qnn`.
- Windows/MSVC and ARM64 builds.
- Protocol-grade performance measurements.
