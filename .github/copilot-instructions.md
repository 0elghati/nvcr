# NVCR Copilot Instructions

NVCR is an extensible native runtime architecture for neural video codecs.

The first complete vertical implementation is:
- Codec adapter: DCVC-RT
- Execution provider: TensorRT
- Target platform: NVIDIA CUDA/TensorRT
- Language: C++20 with CUDA/TensorRT integration
- Build system: CMake

## Architectural invariants

Do not treat DCVC-RT and TensorRT as one generic backend.

Keep these concerns separate:
- Codec adapter: owns codec semantics, temporal state, entropy coding, codec payload syntax, and model-component meaning.
- Execution provider: owns graph/engine loading, tensors, device execution, synchronization, and provider-specific errors.
- Artifact resolver: maps codec/model/component/provider/target/precision to executable artifacts.
- Stream layer: owns codec-neutral stream headers and access-unit framing.
- Codec payload: opaque to the NVCR stream layer.

A `.nvcr` stream must identify codec, codec profile, payload syntax, and model set.
It must not require TensorRT engine filenames, CUDA device IDs, binding indices, or provider-specific serialized artifacts to be decoded.

## Dependency rules

Core public headers must not expose:
- TensorRT types
- CUDA runtime types
- DCVC-RT tensor names
- DCVC-RT model-component internals

TensorRT-specific code belongs under the TensorRT execution provider.
DCVC-RT-specific code belongs under the DCVC-RT codec adapter.

## API rules

Prefer send/receive session APIs:
- encoder send_frame / receive_access_unit / flush / reset
- decoder send_access_unit / receive_frame / flush / reset

Do not design the generic API around one-frame-in/one-packet-out assumptions.

## Testing rules

Before large refactors:
1. inspect existing architecture and tests;
2. run available build and test commands;
3. preserve current DCVC-RT/TensorRT behavior;
4. add contract tests for codec plugins, providers, stream parsing, artifact resolution, and external CMake consumption.

## Paper framing

The SoftwareX framing is:
NVCR is an extensible neural video codec runtime architecture whose first complete reference implementation integrates DCVC-RT with TensorRT.

Do not claim:
- new neural codec model;
- production-ready universal codec runtime;
- multi-codec support before a second codec works;
- multi-provider support before a second provider works;
- cross-provider bitstream interoperability before conformance tests prove it;
- energy efficiency without controlled energy measurements.
