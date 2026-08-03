# NVCR runtime images

NVCR provides native CUDA/TensorRT execution images for two distinct NVIDIA
platform families:

- `x86_64-cuda12.6-trt10.7`: linux/amd64, CUDA 12.6, TensorRT 10.7, SM 8.9
  reference target.
- `jetson-l4t36.4`: linux/arm64 Jetson, JetPack 6.1/L4T 36.4, SM 8.7
  reference target.

There is no shared `latest` tag because the CPU architecture, NVIDIA
userspace, and TensorRT engine compatibility differ. Prefer immutable tags such
as `0.5.1-x86_64-cuda12.6-trt10.7`.

The images contain the native Release runtime only. TensorRT engine bundles,
models, input video, and generated output are mounted at execution time. The
runtime automatically selects a mounted engine profile from input dimensions
when encoding and from bitstream metadata when decoding.

Documentation and Compose examples:
https://github.com/0elghati/nvcr/blob/main/docs/docker.md
