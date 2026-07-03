# DCVC-RT integration contract

Return to the [docs index](README.md) or the [project overview](../README.md).

## Current native status

NVCR now runs the DCVC-RT I-frame path entirely in C++, TensorRT, CUDA, and the
vendored rANS implementation. `TensorRTBackend` loads seven I-frame plans from
`intra_engine_path`, stores `intra_qp` during initialization, and serves encode/
decode through one nonblocking CUDA stream.

The native codec path accepts RGB24 frames only. The `nvcr` CLI converts planar
YUV420p8 input to RGB24 when encoding and converts decoded RGB24 frames back to
YUV420p8. Its `encode` and `decode` commands run independently and exchange a
persistent NVCR sequence file. Predicted frames remain
`not_implemented`, so `gop_size=1` is required.

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

During `initialize`, the backend validates the plan directory, loads
`i_entropy.bin`, `i_quant.bin`, and `i_frame_manifest.json`, creates execution
contexts, creates a single nonblocking CUDA stream, warms the engines, and stores
the configured intra QP.

`encode_intra` and `decode_intra` are correctness-first host-staged pipelines.
They convert RGB24 to YCbCr and back, run the analysis/hyper-analysis/hyper-
synthesis stages, iterate the four spatial-prior passes, and then hand the result
to the vendored rANS wrapper. Predicted-frame encode/decode are still reported as
`not_implemented`.

No TensorRT type, CUDA type, or exception crosses the public `CodecBackend`
boundary.

## Current conformance

Current checks in the tree cover:

- native I-frame encode/decode reconstruction;
- engine directory validation and warm-up;
- vendored rANS synthetic vectors;
- packet framing and malformed-packet rejection;
- repeated sequence reuse through the runtime and pool layers.

Deferred until P-frames land:

- upstream golden I/P payloads;
- at least two complete GOPs;
- reset before and after P frames;
- flush with and without pending work;
- engine/binding mismatch diagnostics for the predicted graph.

## See also

- [Architecture](architecture.md)
- [NVCR packet envelope](bitstream.md)
- [Performance](performance.md)
