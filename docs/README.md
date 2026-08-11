# Documentation map

NVCR is a native runtime architecture for neural video codecs. The first
complete production vertical is DCVC-RT with TensorRT FP16 on Linux/NVIDIA
targets. Start with the canonical [identity and scope](identity-and-scope.md)
when a document or claim needs architectural context.

## New users

- [Getting started](getting-started.md)
- [CLI](cli.md)
- [Docker overview](docker.md)
  - [x86_64 NVIDIA](docker-x86_64.md)
  - [Jetson Orin](docker-jetson.md)
  - [Windows host (WSL 2 or native PowerShell)](docker-windows.md)
- [Compatibility and target support](compatibility.md)

## Runtime and architecture

- [Identity and scope](identity-and-scope.md)
- [Architecture](architecture.md)
- [C++ API reference](reference.md)
- [Scope and support](scope-and-support.md)
- [Architecture decisions](adr/ADR-001-codec-provider-separation.md)

## Codec and provider developers

- [Extending NVCR](extending-nvcr.md)
- [DCVC-RT integration](dcvcrt-integration.md)
- [Conformance levels](adr/ADR-005-conformance-levels.md)

## Artifacts and deployment

- [DCVC-RT model and engine preparation](dcvcrt-artifacts.md)
- [Compatibility classes](experiments/compatibility-levels.md)
- [Release policy](releasing.md)
- [Docker overview](docker.md)

## Streams and access units

- [Bitstream and access-unit contract](bitstream.md)
- [Elementary stream specification](spec/nvcr-elementary-stream-v1.md)
- [Neural access-unit envelope direction](neural-bitstream-envelope.md)

## Validation and reproducibility

- [Performance protocol](performance.md)
- [Reproducible experiments](experiments/README.md)
- [Evaluation protocol](experiments/evaluation-protocol.md)
- [Runbook](experiments/runbook.md)
- [Result schema](experiments/result-schema.md)
- [Missing-data checklist](experiments/missing-data-checklist.md)

## Release and governance

- [Roadmap](../ROADMAP.md)
- [Contributing](../CONTRIBUTING.md)
- [Support](../SUPPORT.md)
- [Security](../SECURITY.md)
- [Citation](../CITATION.cff)
- [Asset distribution policy](../ASSET_DISTRIBUTION_POLICY.md)
- [Model and checkpoint licensing](../MODEL_LICENSES.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)

## Current production integration

The only production codec/provider pair is DCVC-RT/TensorRT. The test codec
and CPU provider are deterministic conformance fixtures. TensorRT's public
provider contract is implemented, but its production path currently creates a
provider-owned monolithic DCVC-RT backend; independent loading of every model
stage through `IExecutable` remains transitional work.

## Reproducibility and review path

1. [Identity and scope](identity-and-scope.md)
2. [Architecture](architecture.md)
3. [Getting started](getting-started.md)
4. [Extension contracts](extending-nvcr.md)
5. [Stream specification](spec/nvcr-elementary-stream-v1.md)
6. [Artifact and release model](dcvcrt-artifacts.md)
7. [Evaluation protocol](experiments/evaluation-protocol.md)
8. [Evidence package](reproducibility/claims-and-evidence.md)
9. [License and citation](reproducibility/code-metadata.md)
