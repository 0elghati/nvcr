# NVCR

NVCR is a native C++ runtime architecture for neural video codecs.

It provides stateful encoder and decoder sessions, separates codec semantics
from execution-provider mechanics, resolves target-specific artifacts, and
carries codec output in bounded, versioned access units. NVCR is neither a
compression model nor only a C++ port of one model implementation.

The current supported end-to-end integration is DCVC-RT with TensorRT FP16 on
NVIDIA Linux systems. The deterministic codec and CPU provider are conformance
fixtures used to test the runtime contracts.

## Contents

- [Choose a path](#choose-a-path)
- [Quick start](#quick-start)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Current support](#current-support)
- [Documentation](#documentation)
- [Current limitations](#current-limitations)

## Choose a path

| Environment or goal | Recommended path | Scope |
|---|---|---|
| Linux x86_64 with an NVIDIA GPU | [Docker on Linux](docs/docker-x86_64.md) | Released DCVC-RT/TensorRT runtime |
| Windows 11 with an NVIDIA GPU | [Docker Desktop and WSL 2](docs/docker-windows.md) | Linux container execution from Windows |
| Jetson Orin | [Native Jetson installation](docs/docker-jetson.md) | AArch64 package and exact-target engines |
| Linux without a supported GPU | [CPU contract validation](docs/first-run.md#cpu-only-contract-validation) | Runtime and format contracts; no neural inference |
| Build or modify NVCR | [Contributing](CONTRIBUTING.md) | Source build and contributor checks |

The [first-run guide](docs/first-run.md) contains the canonical installation,
runtime inspection, encode, decode, and validation commands for every supported
entry path.

## Quick start

For Linux x86_64, use the architecture-qualified rolling image. Resolve it once
to an immutable digest before running NVCR:

```bash
export NVCR_IMAGE="omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9"
docker pull "$NVCR_IMAGE"

export NVCR_IMAGE_REF="$(
  docker image inspect "$NVCR_IMAGE" --format '{{ index .RepoDigests 0 }}'
)"
docker image inspect "$NVCR_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

`NVCR_IMAGE_REF` is immutable for the resolved image. Record the printed
version, source revision, and digest in bug reports or results.

Install only the QCIF engine required by the reference workflow, then inspect
the registered runtime:

```bash
docker volume create nvcr-engines

docker run --rm --gpus all \
  --volume nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE_REF" \
  install --profile qcif --engine-root /opt/nvcr/engines

docker run --rm --gpus all "$NVCR_IMAGE_REF" codec list
docker run --rm --gpus all "$NVCR_IMAGE_REF" codec describe dcvc-rt
docker run --rm --gpus all "$NVCR_IMAGE_REF" provider list
docker run --rm --gpus all "$NVCR_IMAGE_REF" provider describe tensorrt
docker run --rm --gpus all "$NVCR_IMAGE_REF" \
  compatibility check --codec dcvc-rt --provider tensorrt
```

Continue with [First run](docs/first-run.md#linux-x86_64-with-an-nvidia-gpu-and-docker)
to generate a deterministic input and complete the encode/decode validation.

There is intentionally no architecture-neutral `latest` image. NVIDIA
desktop and Jetson images have different architectures and userspace
requirements.

## Capabilities

- C++20 runtime and stateful encoder/decoder session contracts.
- Explicit codec-adapter and execution-provider boundaries.
- Static codec and provider discovery.
- Target-aware artifact catalogs, compatibility ranking, hashes, and policy.
- Bounded `NVAU` v1 and sectioned v2 access units.
- DCVC-RT integration with native entropy coding and TensorRT FP16 execution.
- CLI, CMake package, native archives, containers, and validation tooling.

## Architecture

```text
Application / CLI
        |
        v
NVCR runtime and session contracts
        |
        +-- codec adapter: DCVC-RT semantics and state
        |
        +-- execution provider: TensorRT and CUDA execution
        |
        +-- artifact resolver: model, profile, target, and digest identity
        |
        +-- access units: bounded NVAU framing
```

The runtime architecture is codec-extensible. DCVC-RT and TensorRT are the
first supported codec/provider pair, not the long-term boundary of NVCR. See
[Architecture](docs/architecture.md) and [Identity and scope](docs/identity-and-scope.md).

## Current support

- Linux x86_64 containers and native packages for NVIDIA discrete GPUs.
- Native AArch64 packages for Jetson Orin on the documented JetPack/L4T family.
- Windows as a host for the Linux x86_64 container through Docker Desktop and
  WSL 2.
- CPU-only builds for portable contract and conformance validation.

A target profile or successful compilation is not by itself a support claim.
The selected engine must match the GPU compatibility class, CUDA runtime,
TensorRT runtime, model profile, and manifest identities. See
[Compatibility](docs/compatibility.md) and
[Scope and support](docs/scope-and-support.md).

The [latest stable release](https://github.com/0elghati/nvcr/releases/latest)
is the default for general installation. Reproducible reports must retain the
resolved release tag or container digest.

## Documentation

- [First run](docs/first-run.md)
- [Documentation map](docs/README.md)
- [Command line](docs/cli.md)
- [C++ API](docs/reference.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Performance and benchmarking](docs/performance.md)
- [Retained results](results/README.md)
- [Roadmap](ROADMAP.md)

## Current limitations

NVCR is pre-v1. The C++ API/ABI and application wrappers are not frozen. The
current supported boundary does not include native Windows or macOS
execution, CPU neural inference, INT8 release support, FFmpeg integration,
standard multimedia containers, arbitrary TensorRT-plan portability, or
additional supported codec/provider integrations.

`.nvcr`, `NVCR`, and `NVCS` are development or application wrappers, not
standard multimedia containers.

## License, citation, and support

NVCR source is MIT-licensed. Model checkpoints, exported model assets,
TensorRT plans, and datasets have separate terms and are excluded from generic
packages. Review the [asset policy](ASSET_DISTRIBUTION_POLICY.md),
[model terms](MODEL_LICENSES.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

Use [CITATION.cff](CITATION.cff) for citation metadata,
[SUPPORT.md](SUPPORT.md) for usage and bug-report guidance, and
[SECURITY.md](SECURITY.md) for security reporting.
