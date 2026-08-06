# Compatibility

This page is the short support matrix. The detailed product boundary is in [Scope and support](scope-and-support.md).

## Current target profiles

| Profile | Platform | Engine requirement | Status |
|---|---|---|---|
| `rtx4070-ubuntu2404` | Linux x86_64, RTX 4070, SM 8.9 | Exact TensorRT version and target-local plans | Validation pending after refactor rerun |
| `orin-nano-l4t3647` | Linux aarch64, Jetson Orin Nano, SM 8.7 | Exact TensorRT and CUDA runtime, target-local plans | Validation pending after refactor rerun |

Other NVIDIA Linux systems may build and run, but they are not support targets until they have a target profile and clean evidence.

## What must match

An engine bundle is accepted only when its manifest, model set, engine profile, precision, CUDA/TensorRT versions, GPU identity, required file set, and SHA-256 values match the request. A successful TensorRT deserialization is not enough.

Desktop compatibility bundles may match by compute capability when explicitly cataloged. Jetson bundles remain target-local. NVCR never builds a missing engine during install or decode.

## Format compatibility

- `NVAU` is the versioned codec access unit.
- `NVCR` and `NVCS` are development wrappers with no stable pre-v1 promise.
- The inner DCVC-RT payload is an NVCR implementation format, not an upstream Python bitstream contract.
- No cross-runtime byte or payload interchangeability claim is made until bidirectional golden tests pass.

## Feature status

| Feature | Status |
|---|---|
| DCVC-RT I/P FP16 encode/decode | Implemented, target-gated |
| YUV420P8 raw I/O | Implemented, target-gated |
| Bounded `NVAU` v1/v2 parsing | Implemented and tested |
| Static codec/provider registry | Implemented; test implementations exercise the boundary |
| Artifact resolver and catalog loading | Implemented and tested |
| INT8 | Experimental, not a release profile |
| Additional codecs/providers | Future work |
| C ABI, FFmpeg, standard containers | Future work |
