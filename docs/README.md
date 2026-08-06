# Documentation

These pages describe the software that exists in this repository. The short version is on the [project overview](../README.md); this page is the map.

## Start here

1. [Scope and support](scope-and-support.md)
2. [Getting started](getting-started.md)
3. [Architecture](architecture.md)
4. [Model and engine preparation](dcvcrt-artifacts.md)
5. [CLI](cli.md)
6. [Bitstream and access units](bitstream.md)
7. [C++ API](reference.md)
8. [Compatibility](compatibility.md)
9. [Performance protocol](performance.md)
10. [SoftwareX experiment protocol](experiments/README.md)
11. [Docker](docker.md)
12. [Release policy](releasing.md)

## Contracts

- [DCVC-RT integration](dcvcrt-integration.md)
- [Neural codec envelope direction](neural-bitstream-envelope.md)
- [Elementary stream specification](spec/nvcr-elementary-stream-v1.md)
- [Architecture decisions](adr/ADR-001-codec-provider-separation.md)
- [SoftwareX experiment protocol](experiments/README.md)

## Current boundary

NVCR currently supports one codec adapter, DCVC-RT, and one execution provider, TensorRT. The static registry and artifact resolver prove the extension boundary; they do not mean that a second production codec or provider already exists.

The release is Linux-only, FP16-only, and target-local for TensorRT engines. Checkpoints, exported model assets, and plans are excluded from generic packages.

The latest machine-readable performance matrix is kept under `../evidence/`. It is diagnostic until a clean post-refactor run replaces it. Old run dumps and working notes are intentionally not part of the active documentation set.
