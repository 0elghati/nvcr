# Scope and support

This page defines what NVCR currently is. Everything else should stay within this boundary.

## Supported path

NVCR is a C++20 runtime architecture for neural video codecs. The complete implemented path is:

```text
DCVC-RT model profile -> target-local TensorRT FP16 engines -> native C++ I/P runtime
```

The current product boundary is:

- Linux on NVIDIA hardware.
- DCVC-RT only, using model profile `dcvcrt-cvpr2025`.
- TensorRT FP16 execution.
- Planar 8-bit YUV420 input and output.
- Stateful I/P GOP operation, reset, flush, and bounded access units.
- Offline checkpoint/model export and target-local engine preparation.
- Separate public-catalog installation of validated compatible engine bundles.

The pinned upstream source and checkpoint identities are maintained in [Model and engine preparation](dcvcrt-artifacts.md) and `configs/models/dcvcrt-cvpr2025.json`.

## What is implemented

The repository contains the codec adapter, TensorRT provider, static codec/provider registry, artifact resolver, `NVAU` parser and serializer, native rANS, CLI, and CPU/GPU contract tests.

The architecture is extensible, but only DCVC-RT and TensorRT are production implementations today. The test codec and CPU provider exist to exercise the boundaries; they are not additional supported products.

## What is not supported yet

- A second production codec or execution provider.
- INT8 as a release profile.
- Windows, macOS, CPU inference, or non-NVIDIA accelerators.
- A stable C ABI.
- FFmpeg or standard container integration.
- Universal TensorRT-plan portability.
- Checkpoints or exported model assets in generic NVCR packages.
- TensorRT plans embedded in generic cross-target packages.

`NVCR` and `NVCS` are development file wrappers. `NVAU` is the codec access-unit contract; it is not an MP4, Matroska, or upstream DCVC-RT container.

## Support rule

A target is supported only after a clean Release build, artifact validation,
I/P round trip, malformed-input checks, reference comparison, and
target-specific performance run have passed.

Until then, describe the code as implemented and tested, not as a supported v1 release.
