# Documentation

These pages describe the software that exists in this repository. The short version is on the [project overview](../README.md); this page is the map.

## Start here

1. [Project overview](../README.md)
2. [Getting started](getting-started.md)
3. [Command line](cli.md)
4. [Docker](docker.md)
5. [Compatibility](compatibility.md)
6. [Scope and support](scope-and-support.md)
7. [Model and engine preparation](dcvcrt-artifacts.md)
8. [Architecture](architecture.md)
9. [C++ API](reference.md)
10. [Bitstream and access units](bitstream.md)
11. [Performance protocol](performance.md)
12. [Release policy](releasing.md)

## Contracts

- [DCVC-RT integration](dcvcrt-integration.md)
- [Neural codec envelope direction](neural-bitstream-envelope.md)
- [Elementary stream specification](spec/nvcr-elementary-stream-v1.md)
- [Architecture decisions](adr/ADR-001-codec-provider-separation.md)
- [SoftwareX experiment protocol](experiments/README.md)
- [SoftwareX executable runbook](experiments/runbook.md)

## Current boundary

NVCR currently has one production execution path: DCVC-RT with the TensorRT
backend. The static registry, provider API, and artifact resolver support that
path, but no second production codec or provider exists.

The release is Linux-only, FP16-only, and target-local for TensorRT engines. Checkpoints, exported model assets, and plans are excluded from generic packages.
