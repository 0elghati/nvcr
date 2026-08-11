# NVCR runtime images

NVCR provides native CUDA/TensorRT execution images for two distinct NVIDIA
platform families:

- `amd64-cuda12.8-trt10.9`: linux/amd64, CUDA 12.8, TensorRT 10.9,
  CUDA helper kernels for the compiler-supported GPU architecture set; engines
  are selected from the catalog.
- `jetson-l4t36.4`: linux/arm64 Jetson, JetPack 6.1/L4T 36.4, SM 8.7
  runtime code.

There is no shared `latest` tag because the CPU architecture, NVIDIA
userspace, and TensorRT engine compatibility differ. Prefer immutable tags such
as `0.18.0-amd64-cuda12.8-trt10.9`.

The images contain the native Release runtime and rolling-catalog client, but no
TensorRT plans, models, input video, or generated output. The architecture-
specific Compose `engine-install` service downloads every exact-compatible
bundle from the non-semver `engine-assets` release into a persistent volume.
The runtime mounts that collection read-only and automatically selects a profile
from input dimensions when encoding and from bitstream metadata when decoding.

Documentation:

- https://github.com/0elghati/nvcr/blob/main/docs/docker-jetson.md
- https://github.com/0elghati/nvcr/blob/main/docs/docker-x86_64.md
