# Docker

Choose one complete platform guide:

- [Jetson Orin / L4T 36.4](docker-jetson.md)
- [x86_64 Linux / NVIDIA GPU](docker-x86_64.md)

There is no shared image because the CPU architecture and NVIDIA userspace
differ. TensorRT engines also remain target-specific.

| Platform | Image tag | Dockerfile |
|---|---|---|
| Jetson Orin | `latest-jetson-l4t36.4` | `docker/Dockerfile.jetson` |
| x86_64 NVIDIA | `latest-amd64-cuda12.8-trt10.9` | `docker/Dockerfile.x86_64` |

Both guides use the same simple storage layout:

| Data | Container path | Access |
|---|---|---|
| managed engine volume | `/opt/nvcr/engines` | install: writable; runtime: read-only |
| host YUV directory | `/input` | read-only |
| host output directory | `/output` | writable |

Images contain the runtime and catalog installer, but no engines, models,
datasets, or output. The platform guide installs engines once into the managed
volume and uses Xiph's `akiyo_qcif.yuv` example.

The `latest-*` tags are architecture-qualified rolling aliases. Pin an
immutable image tag such as `0.18.0-amd64-cuda12.8-trt10.9` when reproducing a
specific release. There is intentionally no shared unqualified `latest` tag.
