#!/usr/bin/env bash
# Validate the x86_64-hosted Jetson cross-compilation environment.
set -euo pipefail

sysroot="${NVCR_JETSON_SYSROOT:-}"
cross_prefix="${NVCR_JETSON_CROSS_PREFIX:-aarch64-linux-gnu-}"
cuda_root="${NVCR_CUDA_CROSS_ROOT:-/usr/local/cuda}"

if [[ -z "$sysroot" || ! -d "$sysroot" ]]; then
    echo "Jetson cross build requires NVCR_JETSON_SYSROOT pointing to an extracted JetPack/L4T rootfs" >&2
    exit 1
fi
if [[ ! -x "${cross_prefix}gcc" || ! -x "${cross_prefix}g++" ]]; then
    echo "Jetson cross compiler not found for prefix: $cross_prefix" >&2
    exit 1
fi
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    echo "CUDA cross compiler not found: $cuda_root/bin/nvcc" >&2
    exit 1
fi
if [[ ! -d "$cuda_root/targets/aarch64-linux" ]]; then
    echo "CUDA AArch64 target libraries not found below: $cuda_root/targets/aarch64-linux" >&2
    exit 1
fi
if [[ ! -f "$sysroot/usr/include/NvInfer.h" && ! -f "$sysroot/usr/include/aarch64-linux-gnu/NvInfer.h" ]]; then
    echo "TensorRT headers are missing from the Jetson sysroot: $sysroot" >&2
    exit 1
fi

echo "Jetson cross toolchain:"
echo "  sysroot=$sysroot"
echo "  compiler=${cross_prefix}gcc"
echo "  nvcc=$cuda_root/bin/nvcc"
echo "  target=aarch64-linux, CUDA SM 87"

