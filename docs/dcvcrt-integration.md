# DCVC-RT integration contract

Return to the [docs index](README.md) or the [project overview](../README.md).

## Current native status

NVCR now runs configured DCVC-RT I/P GOPs in C++, TensorRT, CUDA, and the
vendored rANS implementation. `TensorRTBackend` loads seven I-frame plans and
seven P-frame plans, stores the configured QP/GOP state, and serves encode/decode
through CUDA-backed TensorRT execution.

The `nvcr` CLI accepts planar YUV420P8 input and writes planar YUV420P8 output.
Its `encode` and `decode` commands run independently and exchange a persistent
NVCR sequence file. The current path is correctness-first; GPU-resident state and
performance parity remain active roadmap work.

Engine artifacts are generated outside the repository source tree. The normal
pipeline is `scripts/prepare_dcvcrt_artifacts.sh`: it verifies the public
Microsoft DCVC-RT checkout and pretrained checkpoints, exports ONNX/runtime
assets into `build/models/dcvcrt`, then calls `scripts/build_dcvcrt_tensorrt.sh`
to build target-local plans in the chosen engine directory. The CLI consumes that
directory via `--engine-dir`, and the same path can be provided to
`NVCR_TENSORRT_ENGINE_DIR` for native test runs.

TensorRT `.plan` files are not portable across platforms or TensorRT runtimes.
If deserialization reports a platform tag mismatch, keep the exported ONNX/assets
and rebuild the engine directory on the target machine. Engine directories also
carry `engine_manifest.json`; initialization rejects bundles whose recorded GPU
model, compute capability, multiprocessor count, or TensorRT version does not
match the selected runtime device.

This is an internal NVCR encode/decode path, not a declaration of byte-for-byte
compatibility with the upstream DCVC-RT Python payloads or containers. Current
correctness is verified at the encoder/decoder reconstruction level and by the
vendored rANS conformance test.

## Native entropy coder

The pinned upstream rANS primitive remains authoritative. NVCR wraps it with
codec-owned tables and the native I-frame payload wrapper. The current payload
layout is:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 bytes | ASCII `NVI1` |
| Width | `u32` | Encoded frame width |
| Height | `u32` | Encoded frame height |
| QP | `u32` | Intra quantizer parameter |
| Flags | `u32` | `0` = one coder, `1` = two coders |
| Payload | variable | Native rANS stream |

The outer NVCR `Packet` envelope is still owned by NVCR and carries timestamps and
metadata around that codec payload.

## TensorRT backend

The current TensorRT backend owns the seven I-frame plans:

- `i_analysis`
- `i_hyper_analysis`
- `i_hyper_synthesis`
- `i_spatial_prior_1`
- `i_spatial_prior_2`
- `i_spatial_prior_3`
- `i_synthesis`

and seven P-frame plans:

- `p_reference_frame`
- `p_reference_feature`
- `p_analysis`
- `p_hyper_analysis`
- `p_prior`
- `p_spatial_prior`
- `p_synthesis`

During `initialize`, the backend validates the plan directory, loads
`engine_manifest.json`, checks it against `--device-id`, loads `i_entropy.bin`,
`i_quant.bin`, `i_frame_manifest.json`, `p_entropy.bin`, `p_quant.bin`, and
`p_frame_manifest.json`, creates execution contexts, creates a CUDA stream, warms
the engines, and stores the configured QP.

Encode/decode are correctness-first staged pipelines. They run analysis,
hyper-analysis, prior, spatial-prior, synthesis, and rANS stages while preserving
explicit sequence state for references and GOP rules.

No TensorRT type, CUDA type, or exception crosses the public `CodecBackend`
boundary.

## Current conformance

Current checks in the tree cover:

- native I-frame encode/decode reconstruction;
- native I/P frame encode/decode reconstruction;
- engine directory validation and warm-up;
- vendored rANS synthetic vectors;
- packet framing and malformed-packet rejection;
- repeated sequence reuse through the runtime and pool layers.

Deferred until later roadmap gates:

- upstream golden I/P payloads;
- at least two complete GOPs;
- reset before and after P frames;
- flush with and without pending work;
- engine/binding mismatch diagnostics for the predicted graph.

## See also

- [Architecture](architecture.md)
- [NVCR packet envelope](bitstream.md)
- [Performance](performance.md)
