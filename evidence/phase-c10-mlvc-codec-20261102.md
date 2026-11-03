# Phase C10 MLVC codec integration evidence — 2026-11-02

## Scope

C10 adds Microsoft MLVC as NVCR's second codec. It runs Microsoft's published
ONNX model bundles unchanged through the existing ONNX Runtime providers. The
work was prioritized by user direction on 2026-11-02.

It does not reopen C5/C6. Those milestones tested whether an exported MLVC
graph matches MLVC's PyTorch model, which is a property of the export. C10's
gate is that NVCR reproduces MLVC's own deployment runtime, given the same
bundle and the same execution provider. The user chose this gate on
2026-11-02, and it was declared before any NVCR MLVC code ran.

## Acceptance rule

For every reference case, all of the following must hold exactly:

1. each NVCR access-unit payload equals the corresponding record of the
   upstream `.mlvc` file;
2. decoding NVCR's stream reproduces the upstream 8-bit YUV byte for byte;
3. decoding the upstream records, wrapped in `NVAU`, reproduces the same YUV;
4. encoder and decoder reset restart the sequence identically; and
5. malformed access units fail without crashing.

For streams NVCR cannot encode, such as rate-controlled streams, only (3)
applies. No tolerance is allowed.

## Pinned inputs

- MLVC runtime reference: `https://github.com/microsoft/MLVC.git` at
  `1425880bf679bbafdaffb7be3fb61f80448fb0b6`, the commit the bundles were
  exported from (recorded in each `metadata_*.json`). The tree was clean.
- MLVC HEAD audited: `7a85fee68545c47688aaece9f38e018c2cf0798f` (2026-11-02).
  Between the two, `video/conversion` changed only in `64b10d4`, which adds
  NHWC I/O for qualcomm exports and the `{name, layout}` field schema.
  `packages/msrtc_rans` is byte-identical.
- Bundles, from `https://github.com/microsoft/libmlvc.git` at
  `c504aaa65d4f463255b065c6a1cfc194a06f5d3b`, `data/model_bundles/`:
  - `dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1.tar`,
    SHA-256 `c37ee2f5aa418c1fbdd44e4bd95483faad84cbea592367b305e787042f845457`
  - `dmc61sbr_mini_reglu-perceptual_v1-onnx-qualcomm-v0.1.tar`,
    SHA-256 `d724f148039e19f2bbd298f2dee22bd7406f20a2c6eda18c897789b261f83c25`
  - Both are MLVC-S perceptual, FP16 and NCHW, with canvases 960x544, 640x368,
    432x240 and 320x192. NVCR verified every part's registry digest before
    loading it.
- Inputs:
  - `akiyo_qcif.yuv`, SHA-256
    `e1efee0e95c6d27aefe2294a727768334db8466e67bf4a49cf8f5b2fb8b49108`
  - `foreman_qcif.yuv`, SHA-256
    `607a744279bb916e7e03f95bb1cf3605b7218b00a94a6135563d5868ce041a7f`
  - the first 12 frames of `BasketballDrive_640x360_50.yuv` (SHA-256
    `23f7de744d5e05d620629d2e6b5a443514d92ef4c3d7b463f65a9a900ef3fb9d`) as
    640x360 (`2703ab78…5de76`) and transposed to 360x640 (`2265bcc6…d78103`)

## Environment

- Host: AMD Ryzen 7 5800X (16 threads), Ubuntu 24.04, Linux 7.0.0-38, GCC 13.3.0
  and Clang, CMake 3.28.3, Ninja.
- ONNX Runtime 1.26.0, CPU execution provider: the C++ release package
  `onnxruntime-linux-x64-1.26.0.tgz` (SHA-256
  `1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57`) for
  NVCR, and the `onnxruntime==1.26.0` wheel for the upstream reference.
- Upstream reference environment: Python 3.12.3, numpy 2.3.2, torch
  2.10.0+cu128 (imported by the conversion package, not used for inference),
  `msrtc-rans` 0.1.0 built from the pinned checkout. It ran with `uv sync
  --frozen` against the pinned lock, after which ONNX Runtime was pinned to
  1.26.0.

## Results

| Case | Bundle / canvas | Frames | Rate | Upstream `.mlvc` SHA-256 | Upstream YUV SHA-256 | Result |
|---|---|---:|---|---|---|---|
| akiyo QCIF, model loop (LTR 8, recovery 64) | generic 320x192 | 66 | Q 32 | `d95ccd57…a6d9fc` | `04ab6394…31b9f0` | Pass: 66/66 records, both decodes identical |
| foreman QCIF, I period 16 | generic 320x192 | 34 | Q 63 | `12156183…5b65e8` | `0a33506e…619b48` | Pass: 34/34, identical |
| akiyo QCIF, qualcomm graph on CPU | qualcomm 320x192 | 66 | Q 32 | `accdbfe9…d6b420e` | `038ad2a3…961166` | Pass: 66/66, identical |
| BasketballDrive 640x360, edge padding | generic 640x368 | 12 | Q 21 | `f34fbd33…625856` | `5936be88…19c674b` | Pass: 12/12, identical |
| BasketballDrive 360x640, transposed canvas | generic 640x368 | 12 | Q 21 | `f34fbd33…625856` | `7086b809…189534b` | Pass: 12/12, identical |
| foreman QCIF, MLVC rate control 20 kbps, 16 dropped frames | generic 320x192 | 90 | variable Q | `dcb42117…c3cdbf8` | `a1068022…f7fffa` | Pass: decode-only, identical |

- Repeating the akiyo reference produced identical upstream hashes, so the
  upstream CPU path is deterministic for these inputs.
- The portrait clip produces the same records as its landscape source, as
  expected: MLVC transposes the portrait frame onto the same canvas. Only the
  decoded YUV differs.
- The NVCR CLI (`nvcr encode/decode --codec mlvc`) on the akiyo case decodes
  to the upstream YUV hash `04ab6394…`. At QCIF it encodes 66 frames at
  53.97 fps and decodes at 51.18 fps on ONNX Runtime CPU. These are single
  diagnostic runs, not a controlled measurement.
- On akiyo, the 66 MLVC payloads total 4,207 bytes. NVCR's `.nvcr` packets
  total 9,553 bytes, because `NVAU` v1 adds about 81 bytes per access unit.

## Regression and robustness gates

| Gate | Result |
|---|---|
| Clean GCC Release build, `NVCR_ENABLE_ONNXRUNTIME=ON` | Pass; no warnings in new or changed files |
| Clang Release build (`-Wconversion` includes sign conversion) | Pass; no warnings in new or changed files |
| Full CTest, Release | 26/26 pass. The new `nvcr_mlvc_unit` test runs everywhere; parity cases are registered only when references are configured |
| ASan/UBSan with leak detection: MLVC unit tests, parity (QCIF, I period 16, transposed, rate-controlled decode-only), and malformed-unit checks | Pass |
| DCVC-RT through ONNX Runtime CPU: 9-frame GOP-4 QCIF CLI encode/decode, `main` (`d53f11ae`) versus this branch | Identical `.nvcr` (`9754aeb0…ab13f7a5`) and decoded YUV (`a811794f…30284f2c`) |
| DCVC-RT ONNX Runtime I/P round trip, 176x144 | Pass: I 165 bytes, P 58 bytes, unchanged from C8 |
| Malformed MLVC units: truncated record, wrong length, unknown model identity, predicted unit first, Q mismatch, flipped payload bytes | Rejected or decoded without crashing |

## Commands

The upstream reference, run with the pinned checkout's environment:

```text
<mlvc>/.venv/bin/python -I scripts/backends/mlvc/reference_vectors.py \
  --mlvc-video-dir <mlvc>/video --bundle <generic-bundle> --model-id 320x192 \
  --input akiyo_qcif.yuv --width 176 --height 144 --frames 66 --q-index 32 \
  --output <reference>
```

The NVCR gate:

```text
nvcr_mlvc_parity_tests onnxruntime-cpu <generic-bundle> <reference>/q_index=32 \
  akiyo_qcif.yuv 176 144 66 model
```

`--bitrate 20000` with `... 90 model decode-only` reproduces the
rate-controlled case.

## Snapdragon device run — 2026-11-03

Samsung Galaxy S25 (SM-S931U, SM8750, Hexagon v79), `ro.build.version.release` 17. NVCR was built for
arm64-v8a with NDK r26b (`android-33`, `c++_shared`) against
`onnxruntime-android-qnn-1.26.0.aar` and `qnn-runtime-2.42.0.aar`, the same
packages as C9. The run used the qualcomm bundle, Q 32, akiyo QCIF (66 frames)
and BasketballDrive 640x360 (65 frames).
[`scripts/backends/mlvc/run_device_checks.sh`](../scripts/backends/mlvc/run_device_checks.sh)
ran from `/data/local/tmp/nvcr-mlvc` and expects the binaries, ONNX Runtime and QNN
libraries, both bundles, the inputs, the x86 upstream references under
`golden/` and NVCR's x86 CPU encodes of the same clips under `x86/`. The
x86 encodes used the same build and flags (`.nvcr` SHA-256 `7fa456e5…` for
QCIF and `9844e1f1…` for 640x360).

**Load gate.** Both MLVC parts load and run on `onnxruntime-qnn` with CPU
fallback disabled, so every node runs on the HTP.

**Parity on the phone CPU fails against the x86 references.** On
`onnxruntime-cpu` the phone produces 2/66 (generic) and 1/66 (qualcomm)
records identical to the x86 upstream references, and decoded YUV differs from
frame 0. The phone's CPU encode of both clips also differs from NVCR's x86 CPU
encode. NVCR runs the same code on both hosts, and the encoder reset check (which
compares the phone with itself) passes. The difference is therefore in ONNX
Runtime's arm64 CPU kernels for these FP16 graphs, not in NVCR. The decoder
reset check reports a failure only because it compares against the x86
reference YUV. The upstream-runtime parity gate is established on x86 only.
Running it on arm64 needs the upstream runtime on the same platform, which has
not been done.

**Same provider, same device.** The QNN stream decodes on the QNN provider
with PSNR-Y 30.97 dB (640x360) against the source, and the CPU stream decodes
on the CPU provider with 31.00 dB. Encoder reconstructions are not exported, so
closed-loop equality between encoder and decoder on the device is not measured
directly.

**Upstream QNN session options change speed, not output.** With
`NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION=disable`,
`NVCR_QNN_HTP_PERFORMANCE_MODE=burst` and `NVCR_QNN_HTP_FINALIZATION_MODE=3`
(upstream's settings), the `.nvcr` streams and decoded YUV are byte-identical
to NVCR's default QNN settings. The three options were not ablated
separately.

Mean time per frame from the `--verbose` phase timers (one run each; not a
controlled measurement):

| Clip | Provider / options | Encode host | Encode model | Encode entropy | Decode host | Decode model | Decode entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| 176x144 | CPU | 0.40 ms | 19.95 ms | 0.08 ms | 0.18 ms | 23.17 ms | 0.08 ms |
| 176x144 | QNN, NVCR defaults | 1.29 ms | 4.75 ms | 0.22 ms | 0.58 ms | 5.04 ms | 0.24 ms |
| 176x144 | QNN, upstream options | 0.40 ms | 0.76 ms | 0.07 ms | 0.28 ms | 0.92 ms | 0.13 ms |
| 640x360 | CPU | 1.94 ms | 82.10 ms | 0.52 ms | 1.54 ms | 75.89 ms | 0.37 ms |
| 640x360 | QNN, NVCR defaults | 9.78 ms | 13.82 ms | 2.09 ms | 2.57 ms | 15.33 ms | 0.50 ms |
| 640x360 | QNN, upstream options | 1.51 ms | 2.39 ms | 0.36 ms | 2.59 ms | 3.22 ms | 0.64 ms |

At 640x360 with upstream options the phases sum to about 4.3 ms per encoded
frame and 6.5 ms per decoded frame. The CLI's codec time is still 5.1 s
(encode) and 6.1 s (decode) for 65 frames. The remaining ~4.8 s is outside the
per-frame phases. It is most likely session creation and HTP graph
finalization, which NVCR does not time separately and does not cache between
runs.

**Cross-backend drift.** Each row decodes one stream on two backends and
compares the two reconstructions (Y PSNR between them, not against the
source). Frame 64 is an LTR-recovery frame.

| Stream | Backends compared | QCIF frame 0 / 32 / 63 / last | 640x360 frame 0 / 32 / 63 / 64 |
|---|---|---|---|
| x86 CPU encode | phone CPU vs x86 CPU | 53.4 / 53.0 / 52.2 / 53.2 dB | 53.3 / 44.9 / 39.7 / 52.6 dB |
| x86 CPU encode | phone QNN vs x86 CPU | 62.3 / 60.4 / 58.9 / 60.8 dB | 64.6 / 50.5 / 43.4 / 60.6 dB |
| phone QNN encode | phone CPU vs phone QNN | 53.3 / 53.0 / 52.3 / 53.2 dB | 53.3 / 43.5 / 34.4 / 52.7 dB |

No frame is bit-identical across backends. Every cross-backend decode
completed without an entropy error. At 640x360 the mismatch accumulates
through the P chain to 34–43 dB by frame 63 and drops back at the frame-64
LTR recovery, as the design note in the
[MLVC integration](../docs/mlvc-integration.md) anticipated. Against the source,
a QNN stream decoded on the phone CPU loses 0.44 dB PSNR-Y at 640x360 (30.53
versus 30.97 dB) and 0.02 dB at QCIF.

## Not tested

- Upstream-runtime parity on arm64. The phone CPU does not reproduce the x86
  references, and no upstream reference was generated on the device.
- Closed-loop equality between encoder and decoder reconstructions on the
  device, and which of the three upstream QNN options gives the speedup.
- QNN context caching. Session creation dominates short CLI runs on the
  device.
- ONNX Runtime CUDA beyond a diagnostic run.
- The standard MLVC model (`dmc61sbr_reglu`), PSNR checkpoints, FP32 bundles
  and HEAD-exported NHWC bundles. NHWC bundles fail closed.
- Encoder rate control, Q-index overrides and frame dropping on encode.
- Controlled performance measurements and long sequences beyond 90 frames.
