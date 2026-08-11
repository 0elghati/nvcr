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
11. [Docker overview](docker.md)
    - [Jetson Orin](docker-jetson.md)
    - [x86_64 NVIDIA](docker-x86_64.md)
12. [Release policy](releasing.md)

## Distribution and licensing

- [Asset distribution policy](../ASSET_DISTRIBUTION_POLICY.md)
- [Model and checkpoint licensing](../MODEL_LICENSES.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Contracts

- [DCVC-RT integration](dcvcrt-integration.md)
- [Neural codec envelope direction](neural-bitstream-envelope.md)
- [Elementary stream specification](spec/nvcr-elementary-stream-v1.md)
- [Architecture decisions](adr/ADR-001-codec-provider-separation.md)
- [SoftwareX experiment protocol](experiments/README.md)
- [SoftwareX executable runbook](experiments/runbook.md)

## Current boundary

NVCR currently has one production execution path: DCVC-RT with the TensorRT
backend. Production component creation is provider-mediated through
`RuntimeServices`; TensorRT remains the only production provider, and its
component implementation remains monolithic. No second production codec or
provider exists.

The release is Linux-only and FP16-only. Checkpoints, exported model assets,
and plans are excluded from generic packages; validated compatible engines are
installed separately from the public rolling catalog or built on the target.

The latest machine-readable performance matrix is kept under `../evidence/`.
Its recorded completion status determines whether it is publication evidence
or a diagnostic result.
