# Docker

NVCR runtime images provide a fixed Linux userspace and application build.
TensorRT engine bundles remain separate because they are selected for the
detected GPU, CUDA runtime, TensorRT runtime, and profile.

## Platform selection

| Host | Delivery | Guide |
|---|---|---|
| Linux x86_64 with an NVIDIA discrete GPU | Linux/amd64 runtime image | [Docker on Linux](docker-x86_64.md) |
| Windows 11 with Docker Desktop and WSL 2 | The same Linux/amd64 image | [Docker on Windows](docker-windows.md) |
| Jetson Orin | Native AArch64 package recommended | [Jetson](docker-jetson.md) |
| Contributor environment | Development container or native source build | [Building from source](building-from-source.md) |

The complete functional workflow is maintained once in
[First run](first-run.md).

## Image tags

Published tags are architecture and runtime qualified:

| Purpose | Linux/amd64 pattern | Jetson pattern |
|---|---|---|
| Latest delivery for that family | `latest-amd64-cuda12.8-trt10.9` | `latest-jetson-l4t36.4` |
| Immutable release | `<version>-amd64-cuda12.8-trt10.9` | `<version>-jetson-l4t36.4` |
| Family alias | `amd64-cuda12.8-trt10.9` | `jetson-l4t36.4` |

There is no unqualified `latest` tag. A family alias is mutable and can move
when a release is published. The Jetson rolling alias can temporarily trail
the latest native GitHub release.

General installation may begin with a rolling alias, but execution and reports
should use its resolved digest:

```bash
export NVCR_IMAGE="omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9"
docker pull "$NVCR_IMAGE"

export NVCR_IMAGE_REF="$(
  docker image inspect "$NVCR_IMAGE" --format '{{ index .RepoDigests 0 }}'
)"
docker image inspect "$NVCR_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Use `NVCR_IMAGE_REF` for subsequent `docker run` commands. An immutable
versioned tag is also appropriate when reproducing an earlier run.

## Runtime image contract

The runtime image:

- uses `/opt/nvcr/bin/nvcr` as its entrypoint;
- uses `/work` as its working directory;
- searches `/opt/nvcr/engines` as its default engine collection;
- contains the NVCR runtime and artifact client; and
- contains no model checkpoints, ONNX exports, TensorRT plans, input data, or
  output data.

Commands after the image reference therefore begin with `codec`,
`provider`, `compatibility`, `encode`, or `decode`, not a second
`nvcr`. Override the entrypoint to invoke `nvcr-artifacts`.

Use one persistent engine volume:

```text
nvcr-engines -> /opt/nvcr/engines
```

It is writable during artifact installation and should be mounted read-only
for encode/decode. Input and output remain explicit host bind mounts.

## What Docker establishes

Docker fixes the NVCR executable and its Linux CUDA/TensorRT userspace. It does
not normalize the host driver, GPU, clocks, power, thermals, storage, Docker
Desktop/WSL overhead, or TensorRT-plan compatibility.

A successful image pull establishes only that the image is available. A
successful catalog installation establishes that a published engine matches
the detected identity under the accepted compatibility rules. A completed
encode/decode validation establishes functionality for that recorded
environment.

Dev Containers are contributor environments. Plain `docker run` is the
user-facing distribution path and does not depend on editor integration or a
source checkout.

See [Docker Hub image contract](../docker/docker-hub.md) for publication
details and [Troubleshooting](troubleshooting.md) for failures.
