#!/usr/bin/env bash
# Installs CUDA toolkit and TensorRT development packages on a GitHub-hosted
# Ubuntu x86_64 runner, for compile-only validation of the CUDA/TensorRT
# backend. There is no GPU on these runners: this enables configuring and
# building NVCR with -DNVCR_ENABLE_TENSORRT=ON (including the portable
# multi-GPU-architecture path), but not running CUDA-executing tests.
#
# Used by .github/workflows/ci.yml (PR compile validation) and
# .github/workflows/release-assets.yml (portable release build).
set -euo pipefail

cuda_toolkit_package="${NVCR_CI_CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-6}"
# Pin TensorRT to the project's validated 10.x baseline (see ROADMAP.md,
# TensorRT 10.3.0.30 on Jetson / 10.7.0 on discrete dev hosts). Leaving this
# unpinned lets apt silently install whatever NVIDIA currently ships as
# "latest" libnvinfer-dev (for example a TensorRT 11.x ABI break), producing
# portable release binaries that fail to load anywhere still running
# TensorRT 10.x: `libnvinfer.so.11: cannot open shared object file`.
tensorrt_package_version="${NVCR_CI_TENSORRT_VERSION:-10.7.0.23-1+cuda12.6}"

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "install_cuda_tensorrt.sh only supports x86_64 hosted runners" >&2
    exit 1
fi

# shellcheck source=/dev/null
. /etc/os-release
case "$VERSION_ID" in
24.04) distro=ubuntu2404 ;;
22.04) distro=ubuntu2204 ;;
*)
    echo "unsupported Ubuntu version for hosted CUDA install: $VERSION_ID" >&2
    exit 1
    ;;
esac

keyring_deb="cuda-keyring_1.1-1_all.deb"
keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/${keyring_deb}"

curl -fsSL -o "/tmp/${keyring_deb}" "$keyring_url"
sudo dpkg -i "/tmp/${keyring_deb}"
sudo apt-get update

# CUDA toolkit provides nvcc; TensorRT dev packages provide headers/libs used
# to configure and compile (not execute) the TensorRT backend. Pin the exact
# TensorRT version so the compiled ABI (libnvinfer.so.<N>) matches what
# release consumers actually have installed.
sudo apt-get install -y --no-install-recommends \
    "$cuda_toolkit_package" \
    "libnvinfer-dev=${tensorrt_package_version}" \
    "libnvinfer-plugin-dev=${tensorrt_package_version}" \
    "libnvonnxparsers-dev=${tensorrt_package_version}"

cuda_root="$(compgen -G '/usr/local/cuda-*' | sort -V | tail -n1)"
if [[ -z "$cuda_root" ]]; then
    echo "CUDA toolkit install did not create /usr/local/cuda-*" >&2
    exit 1
fi

{
    echo "CUDA_ROOT=$cuda_root"
    echo "PATH=$cuda_root/bin:$PATH"
} >>"$GITHUB_ENV"

echo "Installed CUDA toolkit at $cuda_root and TensorRT development packages"
