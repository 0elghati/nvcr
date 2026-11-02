# Phase C9 Snapdragon QNN bring-up evidence — 2026-11-01

## Scope

- Branch: `feat/snapdragon-qnn`, based on `main` at `c214669` (the ONNX Runtime
  provider, PR #206).
- Goal: run the host DCVC-RT backend added with the ONNX Runtime provider on a
  Snapdragon NPU through the ONNX Runtime QNN execution provider. No
  TensorRT, CUDA, NVAU, payload, or entropy code changed.

## Change

- `src/provider/onnxruntime/session.cpp`: `onnxruntime-qnn` sessions set
  `session.disable_cpu_ep_fallback=1`, so session creation fails unless every
  node is placed on the QNN HTP (Hexagon Tensor Processor). This is the NPU
  partition-coverage gate.
- `scripts/backends/dcvcrt/fix_onnx_shapes.py`: writes a fixed-shape copy of
  the fourteen-stage bundle for one frame size (QNN HTP compiles static shapes
  only), with strict ONNX shape inference and the checker, and copies the
  entropy, quant, and frame-manifest assets.
- `src/dcvcrt/host_ops.cpp`, `src/dcvcrt/host_backend.cpp`,
  `tests/dcvcrt_host_ops_tests.cpp`: explicit size conversions so Android
  Clang builds with `-Wsign-conversion` cleanly. No behavior change.

## Environment

- Device: Samsung Galaxy S25 `SM-S931U`, `SM8750` (Snapdragon 8 Elite,
  Hexagon v79), Android 17 / API 37, build `CP2A.260605.016.S931USQUCDZIF`,
  reached over wireless adb.
- Build host: Windows 11, Android NDK 26.1.10909125, `arm64-v8a`,
  `android-33`, `c++_shared`, Release, `NVCR_ENABLE_ONNXRUNTIME=ON`,
  `NVCR_ENABLE_CUDA=OFF`, `NVCR_ENABLE_TENSORRT=OFF`. 0 compiler warnings.
- Runtime libraries, not committed: `onnxruntime-android-qnn-1.26.0.aar`
  (`libonnxruntime.so`) and `qnn-runtime-2.42.0.aar` (`libQnnHtp.so`,
  `libQnnHtpPrepare.so`, `libQnnHtpV79Skel.so`, `libQnnHtpV79Stub.so`,
  `libQnnSystem.so`).
- Models: the FP16 opset-18 export from the ONNX Runtime provider evidence,
  fixed to 176x144 with `fix_onnx_shapes.py --width 176 --height 144`. SHA-256
  over the sorted per-stage `.onnx` digests:
  `51e75c6e39ee82ae3ffe11aa6a0203c43ba17ea07767762a0847a8c4b2c9a86c`.
- Input: 65-frame 176x144 Akiyo YUV420p8,
  `3cb18778f1677c97bf8ce9a45761268c370209a35e948f969a2fe349b0b2f40e`.

| Artifact | SHA-256 |
| --- | --- |
| `nvcr` | `1d407c4ffb1dfee7f6a1dce40e64f8067e40d9d341c1f481f6c1a022aeb4f106` |
| `nvcr_onnxruntime_roundtrip_tests` | `65bb8c695d84509e35571a7af95ffe52b2a07fc8941daf0b6789a49cba9ecff0` |
| `nvcr_dcvcrt_host_ops_tests` | `ad36565cb273a853491905731a47bcd176a4573b7bb52b4cd78fa65451cd0b7d` |
| `libonnxruntime.so` | `2f5a17fe2a547ffadcfd1ba8f6945fce7a928cb637e06798ab60b33e01a98be7` |
| `libQnnHtp.so` | `429b74453f3dc95c6b94ca20894b4e5f08fbf8604bf339e37ef30fc17819e542` |
| `libQnnHtpV79Skel.so` | `b378e9dc83ab255058ff962b4ed334eaa791869c031e9db1c6ec2c780029ddd0` |

Files run from `/data/local/tmp/nvcr` with `LD_LIBRARY_PATH` set to that
directory and `ADSP_LIBRARY_PATH` listing it first.

## Results

| Gate | `onnxruntime-qnn` | `onnxruntime-cpu` (same phone) |
| --- | --- | --- |
| Host operators (`nvcr_dcvcrt_host_ops_tests`) | pass | pass |
| Session contract, I/P self-conformance, malformed payloads, fresh-session determinism, GOP-2 runtime run and reset | pass; I 165 B (luma MAE 0.716), P 60 B (0.541) | pass; I 163 B (0.752), P 60 B (0.610) |
| CLI encode and decode, 65 frames, GOP 32, QP 32 | pass; 8418 payload bytes | pass; 8400 payload bytes |
| PSNR-Y / U / V / YUV against the source | 37.049 / 40.104 / 44.048 / 38.306 dB | 36.968 / 39.992 / 44.166 / 38.246 dB |

- NPU placement: with CPU fallback disabled, all fourteen stages created QNN
  sessions. During the round trip, `logcat` shows the test process opening
  FastRPC on domain 3 (CDSP), and `cdsprpcd` loading
  `fastrpc_shell_unsigned_3`.
- QNN and CPU streams differ (`2e47562d…` vs `4929603f…`), as the ONNX Runtime
  cross-provider finding predicts: each provider decodes its own stream, and
  the HTP rounds FP16 differently from the CPU kernels.
- ONNX Runtime warns once per QNN session that file-mapped weights are
  Windows-only. This has no functional effect.

## NPU throughput per resolution

Diagnostic run with `scripts/measurement_android_qnn.sh`, not a controlled
result: the commit was not clean, there are no pinned Python rows, and only
one device was measured. Phone on battery (38% to 32%), power saving off,
thermal status 0 throughout; each case started at or below 34.0 °C battery
temperature. 33 frames per run (one 32-frame GOP plus the next intra frame),
QP 32, CLI codec-loop boundary, mean of three measured runs after one
warm-up. Inputs are the first 33 frames of the benchmark set used by
`scripts/benchmark_resolution_matrix.sh`; models were fixed per size with
`fix_onnx_shapes.py`.

| Resolution | Input | Encode fps | Decode fps | Encode ms/frame | Decode ms/frame | Spread enc / dec | Payload bpp | PSNR-Y dB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QCIF 176x144 | `akiyo_qcif.yuv` | 22.73 | 30.74 | 44.0 | 32.5 | 3.2% / 10.1% | 0.0436 | 37.00 |
| CIF 352x288 | `waterfall_cif.yuv` | 8.82 | 10.74 | 113.4 | 93.1 | 2.9% / 4.7% | 0.0471 | 33.37 |
| 360p 640x360 | `BasketballDrive_640x360_50.yuv` | 3.95 | 4.91 | 253.2 | 203.7 | 3.5% / 5.0% | 0.0286 | 34.46 |

- Spread is (fastest − slowest) / mean over the three runs. Payload bpp
  counts NVAU access-unit bytes. PSNR-Y is ffmpeg's `psnr` filter against
  the 33 source frames.
- From CIF to 360p, throughput falls about in proportion to pixel count
  (2.27x the pixels, 2.2x slower).
- The phone dozed during the CIF runs and the start of 360p. NPU codec time
  was unaffected (CIF runs 1 and 3 agree within 3%), but process time rose
  from about 31 s to 153–243 s, because graph preparation runs on the
  throttled CPU. Process time is therefore not reported.
- Stream digests (`.nvcr` / decoded `.yuv`):
  QCIF `3c61869e…` / `75c78264…`, CIF `403150d6…` / `d0a35359…`,
  360p `668831dc…` / `cb3ed584…`.

## Not measured

- 540p, 720p and 1080p. The bundles were generated and pushed but not run,
  by choice.
- The CPU baseline. Its runs dropped from warm-up to measured repetitions by
  up to 55% on a hot or dozing phone, and the awake re-run was stopped before
  it finished. The earlier low-battery figures (QNN about 22 fps encode and
  29 fps decode at 176x144) agree with the NPU table above.
- `--measurement-json` is CUDA-only. In a non-TensorRT build it fails only
  after the encode completes, leaving an 8-byte output. The option should be
  rejected before encoding.
- QNN context binaries (EP-context caching). Every run compiles the HTP
  graphs, so session setup is long and grows with the stage count.
- Profiles other than 176x144; the pinned Python golden; stream-level
  provider identity.
