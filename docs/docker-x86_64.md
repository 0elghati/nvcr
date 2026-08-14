# Docker on Linux x86_64

This guide covers the host, image, storage, and engine requirements for running
NVCR on a Linux x86_64 system with a discrete NVIDIA GPU. The complete
encode/decode procedure is in
[First run](first-run.md#linux-x86_64-with-an-nvidia-gpu-and-docker).

## Host requirements

- Linux x86_64.
- Docker Engine.
- NVIDIA Container Toolkit with the `nvidia` runtime configured.
- A working NVIDIA driver compatible with CUDA 12.8 userspace.
- A discrete NVIDIA GPU supported by a published engine compatibility class.

Check the host and container independently:

```bash
nvidia-smi
docker version
docker info --format '{{ json .Runtimes }}'
docker run --rm --runtime=nvidia \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Both NVIDIA queries must name the intended GPU. Resolve driver or container
runtime failures before diagnosing NVCR.

These Linux commands use the configured NVIDIA runtime because that is the
NVCR container path validated on the documented host. Docker's `--gpus all`
form is an alternative; use one form consistently while diagnosing a failure.

## Select and identify the image

Select the published Linux/amd64 release version, then retain its immutable
digest:

```bash
export NVCR_VERSION="<version>"
export NVCR_IMAGE="omarelghati/nvcr:${NVCR_VERSION}-amd64-cuda12.8-trt10.9"
docker pull "$NVCR_IMAGE"

export NVCR_IMAGE_REF="$(
  docker image inspect "$NVCR_IMAGE" --format '{{ index .RepoDigests 0 }}'
)"
docker image inspect "$NVCR_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Use `NVCR_IMAGE_REF` for the run. `latest-amd64-cuda12.8-trt10.9` is a
rolling convenience alias; there is no unqualified NVCR `latest` image.

The entrypoint is already `nvcr`:

```bash
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" codec list
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" provider list
```

## Engine storage and selection

Runtime images do not include TensorRT plans. Install the required profile into
one persistent volume:

```bash
docker volume create nvcr-engines
docker run --rm --runtime=nvidia \
  --volume nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE_REF" \
  install --profile qcif --engine-root /opt/nvcr/engines
```

The public catalog is anonymous. The installer validates the detected
operating system, architecture, GPU, CUDA runtime, TensorRT version, archive
hash, and bundle manifest.

For a desktop GPU, selection is automatic:

1. exact device;
2. same compute capability; and
3. Ampere-plus, if an applicable broad bundle has been published.

Same-compute bundles are not interchangeable across SM classes. An SM 8.9
bundle applies only to SM 8.9 devices; an SM 12.0 bundle applies only to
SM 12.0 devices. Ampere-plus is a broader discrete-GPU fallback, not an
optimized or universal plan. Jetson does not participate in either desktop
fallback class.

If installation reports no compatible bundle, retain the complete detected
identity. The catalog has already considered every published accepted class;
do not rename another GPU's plan or bypass its manifest. Follow
[Model and engine preparation](dcvcrt-artifacts.md) to build and validate an
engine for the target.

Mount `nvcr-engines` read-only during execution:

```bash
docker run --rm --runtime=nvidia \
  --volume nvcr-engines:/opt/nvcr/engines:ro \
  "$NVCR_IMAGE_REF" codec list
```

## Input and output mounts

NVCR reads headerless planar YUV420P8. Use absolute host paths and distinct
read/write intent:

```text
host input directory  -> /input   (read-only)
host output directory -> /output  (writable)
engine volume         -> /opt/nvcr/engines (read-only at runtime)
```

The first-run workflow uses one work directory and creates its output files on
the host before the container writes them. This avoids broad ownership changes.
Do not add an arbitrary `--user` mapping unless that user has access to the
NVIDIA capability nodes.

## Compose and development images

The checked-in Compose file and Dockerfile are source-tree tooling for
contributors and controlled local builds. They are not required to run the
published image. If Compose is used, set `NVCR_X86_64_IMAGE` to the resolved
image reference and invoke one-shot commands with `docker compose run --rm`;
do not use `docker compose up` for the CLI.

A source-built image must not be described as a released image merely because
its build argument contains a release-looking version. Retain its source
revision and local tag separately.

## Next step

Run the canonical
[Linux functional validation](first-run.md#linux-x86_64-with-an-nvidia-gpu-and-docker).
For CUDA, artifact, mount, or output failures, use
[Troubleshooting](troubleshooting.md).
