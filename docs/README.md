# Documentation

NVCR is a development snapshot of a neural video codec runtime architecture,
progressing through v0.3 foundation gates toward a first scoped v1 release. The
current release path supports DCVC-RT only. Read the scope and evidence status
before treating an implemented path as supported.

## Reading order

1. [Project overview](../README.md)
2. [Authoritative scope and support](scope-and-support.md)
3. [Compatibility matrix](compatibility.md)
4. [Getting started and builds](getting-started.md)
5. [Model and engine preparation](dcvcrt-artifacts.md)
6. [Architecture](architecture.md)
7. [C++ API status](reference.md)
8. [Bitstream and access units](bitstream.md)
9. [CLI](cli.md)
10. [DCVC-RT integration](dcvcrt-integration.md)
11. [Performance protocol and historical results](performance.md)
12. [Experimental mlvc-fast codec](mlvc-fast.md)
13. [Release gates](releasing.md)
14. [Roadmap and evidence](../ROADMAP.md)

## Policy shortcuts

- Checkpoints, ONNX/runtime assets, and TensorRT engines are built locally and
  are not shipped in NVCR releases.
- RTX 4070 and Jetson Orin Nano are the only v1 reference targets.
- DCVC-RT is the only implemented and supported v1 codec backend.
- FP16 and YUV420P8 are the v1 product profiles; INT8 is experimental.
- `NVAU` is the candidate codec access-unit contract. `NVCR`/`NVCS` remain
  development application framing.
- C ABI, FFmpeg, and standard container mapping are post-v1.
- Historical binary-install instructions, where retained, describe old snapshots
  and are not current product-support guidance.
