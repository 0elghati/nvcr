# Phase C6 MLVC CUDA equivalence diagnosis — 2026-09-14

## Scope

This evidence follows the rejected
[C5 reference/export experiment](phase-c5-mlvc-reference-export-20260914.md).
It locates the first differing state between the pinned MLVC in-memory PyTorch
reference and the upstream-default generic FP16 ONNX export loaded through ONNX
Runtime CUDA. It does not modify upstream source or add NVCR code.

The source, checkpoint, input, environment, provider, and acceptance identities
are unchanged from C5.

## Reference-device finding

The conversion CLI accepted `--torch-device cuda`, but that setting was not
applied to the in-memory PyTorch model used to generate the conversion result.
At the pinned MLVC revision:

- `full_model_factory` loads the checkpoint on CPU;
- `prepare_model_parts` wraps the model without moving it to
  `runtime_params.torch_device`; and
- `ModelWrapper._transform_input` creates CPU tensors for an in-memory Torch
  model.

A direct check requested CUDA and reported:

```text
requested=cuda
parameter_device=cpu
```

C5 therefore compared the upstream PyTorch CPU reference with the ONNX Runtime
CUDA candidate. The ONNX encoder session reported
`CUDAExecutionProvider, CPUExecutionProvider`; its CUDA provider loaded without
the CUDA-13 library failure seen with ONNX Runtime 1.29.0.

## Named-state comparison

Both paths ran Q index 21 over the same two QCIF frames with
`FrameLoop(..., output_model_data=True)`. Arrays were compared at the named
encoder outputs, decoder inputs, and decoder outputs exposed by the official
split model.

| Frame | Boundary | Field | Different values | Elements | Max absolute difference |
|---:|---|---|---:|---:|---:|
| 0 | Encoder output | `feature` | 38,016 | 38,016 | 0.580142 |
| 0 | Encoder output | `z_raw` | 0 | 432 | 0 |
| 0 | Encoder output | `y_raw_0` | 2 | 2,376 | 1 |
| 0 | Encoder output | `y_raw_1` | 7 | 2,376 | 1 |
| 1 | Encoder output | `feature` | 38,016 | 38,016 | 1.292068 |
| 1 | Encoder output | `z_raw` | 2 | 432 | 1 |
| 1 | Encoder output | `y_raw_0` | 15 | 2,376 | 1 |
| 1 | Encoder output | `y_raw_1` | 16 | 2,376 | 1 |

Frame 0 begins with identical `ref_feature` and `q_index_shifted` inputs. Its
`z_raw` is also identical. The first stream-relevant divergence is therefore
inside the FP16 encoder output: nine quantized `y_raw` symbols differ by one
before the entropy coder receives them. The entropy coder is not the first
cause.

The frame-0 bitstreams are both 503 bytes but are byte-different. On frame 1,
the candidate consumes its differing FP16 reference feature, the raw-symbol
differences grow, and payload length changes from 199 reference bytes to 188
candidate bytes. Decoder-input symbol differences match the encoder-output
differences. Decoder reconstruction and feature differences are downstream
effects.

## Decision

C6 is complete. The pinned default FP16 MLVC/ORT-CUDA pair is rejected as
NVCR's next production codec/provider axis because its encoder changes
stream-bearing quantized symbols before entropy coding and the C5 numerical
limits fail. No NVAU, entropy, session, or generic-runtime change can correct
that model-export divergence.

This does not permanently exclude MLVC or ONNX Runtime. Reconsideration requires
an upstream-fixed, hash-pinned artifact/reference path that meets the unchanged
C5 limits. NVCR should not build integration scaffolding in anticipation of
that result.

The next bounded Phase C gate is C7: test one DCVC-UF fused operator and one
frame-specific high-throughput branch against a pinned upstream vector before
deciding whether its CUDA-specific execution path is portable enough for an
NVCR codec or provider experiment.
