# NVCR documentation

Use [First run](first-run.md) for the canonical installation and validation
workflow. The pages below provide deeper technical or platform-specific
material without repeating that workflow.

## Start here

- [First run](first-run.md): Linux Docker, Windows Docker Desktop, Jetson
  native, and CPU contract validation.
- [Installation](installation.md): select a release delivery method and record
  its resolved identity.
- [Troubleshooting](troubleshooting.md): diagnose GPU, runtime, artifact, input,
  and container failures.
- [Scope and support](scope-and-support.md): current implementation and support
  boundaries.

## Use NVCR

- [Command line](cli.md)
- [Docker overview](docker.md)
- [Docker on Linux x86_64](docker-x86_64.md)
- [Docker from Windows](docker-windows.md)
- [Jetson](docker-jetson.md)
- [Compatibility](compatibility.md)
- [Model and engine artifacts](dcvcrt-artifacts.md)

## Understand and integrate NVCR

- [Identity and scope](identity-and-scope.md)
- [Architecture](architecture.md)
- [C++ API](reference.md)
- [DCVC-RT integration](dcvcrt-integration.md)
- [Bitstreams and access units](bitstream.md)
- [Elementary-stream specification](spec/nvcr-elementary-stream-v1.md)
- [Extension guide](extending-nvcr.md)
- [Architecture decisions](adr/ADR-001-codec-provider-separation.md)

The public C++ API/ABI is transitional. Integrations should retain the NVCR
revision, codec/provider IDs, stream version, and artifact manifest identity.

## Build and contribute

- [Building from source](building-from-source.md)
- [Contributing](../CONTRIBUTING.md)
- [Packaging and releases](releasing.md)
- [Roadmap](../ROADMAP.md)

## Benchmark and validate

- [Performance and benchmarking](performance.md)
- [Experiment protocol](experiments/README.md)
- [Retained results](../results/README.md)

## Project policies

- [Support](../SUPPORT.md)
- [Security](../SECURITY.md)
- [Citation](../CITATION.cff)
- [Asset distribution](../ASSET_DISTRIBUTION_POLICY.md)
- [Model terms](../MODEL_LICENSES.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md)
