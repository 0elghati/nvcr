# Codec/provider integration status

Reviewed 2026-11-07. This table describes implementation status in the current
repository; a check means the path has been exercised, not that it is a
production-supported combination.

| Codec | Provider | Status | Notes |
|---|---|---|---|
| DCVC-RT | TensorRT | ✅ Supported | The current production end-to-end path on Linux/NVIDIA. See [DCVC-RT integration](dcvcrt-integration.md). |
| DCVC-RT | ONNX Runtime CPU | ✅ Experimental | CPU self-conformance passes; development and conformance path. See [C8 in the roadmap](../ROADMAP.md#active-v2-work). |
| DCVC-RT | ONNX Runtime CUDA | Planned | The CUDA execution provider remains an open roadmap item. |
| DCVC-RT | ONNX Runtime QNN (Snapdragon HTP) | ✅ Experimental | On-device self-conformance passes on Galaxy S25; measurements and stream-level provider identity remain open. See [C9 evidence](../evidence/phase-c9-snapdragon-qnn-20261101.md). |
| MLVC | ONNX Runtime CPU | ✅ Experimental | Upstream-runtime parity passes on x86 with the published MLVC-S bundles. See [MLVC integration](mlvc-integration.md) and [C10 evidence](../evidence/phase-c10-mlvc-codec-20261102.md). |
| MLVC | ONNX Runtime CUDA | Planned | Not validated for MLVC; listed as open in the roadmap. |
| MLVC | ONNX Runtime QNN (Snapdragon HTP) | ✅ Experimental | Runs on Galaxy S25 with CPU fallback disabled; arm64 parity against an upstream reference is not established. See [C10 evidence](../evidence/phase-c10-mlvc-codec-20261102.md). |
| MLVC | TensorRT | Planned | C12 is active; initial adapter/session wiring is in progress, with model-bundle and runtime validation still open. See [C12 in the roadmap](../ROADMAP.md#active-v2-work). |
| DCVC-UF | Fused CUDA (`dcvcuf-upstream-cuda`) | ✅ Experimental | Opt-in native path; encoder-internal reconstruction equals decoder output (C13 G1). See [C7 evidence](../evidence/phase-c7-dcvcuf-export-20261031.md). |
| DCVC-UF | TensorRT | ✅ Experimental | C11's 1080p fidelity gate passes, and encoder-internal reconstruction equals decoder output byte for byte at 1080p (C13 G1). At QCIF the 1080p engine bundle has a nondeterministic stage and its streams fail to decode; a bundle with a 256x256 profile works. A stream decodes only on the path and GPU that encoded it: fused-to-TensorRT, TensorRT-to-fused and RTX 4070 ↔ RTX 5060 decodes fail cleanly with `malformed_bitstream`, and pinned upstream decodes NVCR TensorRT and other-GPU streams to wrong frames (C13 cross-decode). Measured speed is about 71% of the fused CUDA path. See [C11 evidence](../evidence/phase-c11-dcvcuf-tensorrt-20261102.md). |

## Status key

- ✅ **Supported**: the production-supported end-to-end path.
- ✅ **Experimental**: the code path exists and has recorded execution evidence; support or validation gates remain open.
- **Planned**: identified as open work in the current roadmap; not a claim that it is usable now.
- **Not integrated**: no implementation or current roadmap commitment for that pairing.

Only DCVC-RT with TensorRT is currently production-supported. Planned and
experimental entries do not imply artifact availability in a released package.
