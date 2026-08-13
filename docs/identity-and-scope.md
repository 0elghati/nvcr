# What NVCR is

NVCR is a native C++ runtime for deploying neural video codecs. It is the
systems layer around a learned compression method: it runs the model on a GPU,
manages encoder and decoder state, selects compatible engines, and packages
compressed video for an application.

NVCR is not a new learned compression model and it is not a C++ copy of
DCVC-RT. DCVC-RT is the first codec integration. TensorRT is the current GPU
execution provider. NVCR keeps those roles separate so future codecs and
providers can use the same runtime.

## What you get

- A C++ library and the `nvcr` command for encoding and decoding.
- Stateful video sessions with flush and reset support.
- Compatible engine selection for the detected GPU, CUDA, and TensorRT runtime.
- `NVAU`, a bounded, versioned access-unit format used inside NVCR streams.
- Native packages and containers for the supported Linux and Jetson paths.
- Tests and result tools for checking an installation and recording execution.

## Current integration

The current production path is DCVC-RT with TensorRT FP16 on NVIDIA Linux.
It supports native x86_64 packages and containers, native Jetson packages, and
Linux containers hosted through Windows Docker Desktop/WSL 2.

CPU builds are useful for runtime and format tests, but they do not run neural
video inference.

## Current limits

NVCR is pre-v1, so its C++ API and ABI may change. Native Windows and macOS
execution, CPU neural inference, INT8 release support, FFmpeg integration,
standard multimedia containers, and additional production codec/provider pairs
are not part of the current release.

See [Architecture](architecture.md) to understand the component boundaries,
[Installation](installation.md) to choose a delivery path, and
[Compatibility](compatibility.md) for the supported target matrix.
