# Docker Hub image contract

NVCR publishes separate runtime image families for Linux/amd64 discrete GPUs
and Linux/arm64 Jetson. There is intentionally no architecture-neutral
`latest` tag.

## Public tags

| Family | Rolling latest | Rolling family | Immutable release pattern |
|---|---|---|---|
| Linux/amd64, CUDA 12.8, TensorRT 10.9 | `latest-amd64-cuda12.8-trt10.9` | `amd64-cuda12.8-trt10.9` | `<version>-amd64-cuda12.8-trt10.9` |
| Jetson, L4T 36.4 | `latest-jetson-l4t36.4` | `jetson-l4t36.4` | `<version>-jetson-l4t36.4` |

Rolling tags are delivery pointers. They can move independently and the Jetson
image can temporarily trail the latest native GitHub release. Use the native
AArch64 package for the default Jetson installation.

## Consumer workflow

A general Linux/amd64 installation may begin with the rolling tag. Resolve and
record the immutable identity before running NVCR:

```bash
export NVCR_IMAGE="omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9"
docker pull "$NVCR_IMAGE"

export NVCR_IMAGE_REF="$(
  docker image inspect "$NVCR_IMAGE" --format '{{ index .RepoDigests 0 }}'
)"
docker image inspect "$NVCR_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Use `NVCR_IMAGE_REF` for subsequent commands. For reproduction of a recorded
run, use its saved digest or immutable versioned tag.

The image entrypoint is `/opt/nvcr/bin/nvcr`. Invoke the artifact client by
overriding that entrypoint:

```bash
docker volume create nvcr-engines
docker run --rm --gpus all \
  --volume nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE_REF" \
  install --profile qcif --engine-root /opt/nvcr/engines
```

See [First run](../docs/first-run.md) for the canonical functional procedure.

## Image contents and metadata

Runtime images contain:

- the NVCR runtime and CLI;
- the public artifact client;
- the platform's CUDA/TensorRT userspace; and
- OCI source, revision, version, creation-time, license, and description
  labels.

They do not contain checkpoints, ONNX exports, entropy/model assets, TensorRT
plans, datasets, input video, or result files. Engine bundles remain in the
separate rolling catalog and are selected for the active target.

The Linux/amd64 image compiles CUDA helper kernels for the supported
compiler architecture set. This host-binary property does not make TensorRT
plans portable.

## Publication behavior

`docker/publish.sh --push` accepts only a clean checkout at the exact release
tag matching `version.txt`. For each family it publishes:

1. the immutable versioned tag;
2. the runtime-family alias; and
3. the architecture-qualified `latest-*` alias.

Publication also attaches provenance and an SBOM. The workflow builds
Linux/amd64 on the x86_64 release runner and Jetson natively on the AArch64
Jetson runner.

If one architecture fails or is delayed, report that family independently.
Do not imply that a rolling tag contains a particular release without
inspecting its labels and digest.

Maintainer procedures are documented in
[Packaging and releases](../docs/releasing.md). Platform guidance is in
[Docker](../docs/docker.md).
