#!/usr/bin/env bash
# Installs the minimal CUDA and TensorRT packages needed on a GitHub-hosted
# Ubuntu x86_64 runner for compile-only validation of the CUDA/TensorRT backend.
# There is no GPU on these runners: this enables configuring and building NVCR
# with -DNVCR_ENABLE_TENSORRT=ON (including the portable multi-GPU-architecture
# path), but not running CUDA-executing tests.
#
# Used by .github/workflows/ci.yml (PR compile validation) and
# .github/workflows/release-assets.yml (portable release build).
set -euo pipefail

if [[ -n "${NVCR_CI_CUDA_PACKAGES:-}" ]]; then
    read -r -a cuda_packages <<<"${NVCR_CI_CUDA_PACKAGES}"
elif [[ -n "${NVCR_CI_CUDA_TOOLKIT_PACKAGE:-}" ]]; then
    cuda_packages=("${NVCR_CI_CUDA_TOOLKIT_PACKAGE}")
else
    cuda_packages=(cuda-nvcc-12-6 cuda-cudart-dev-12-6)
fi
# Pin TensorRT to the project's validated 10.x baseline (see ROADMAP.md,
# TensorRT 10.3.0.30 on Jetson / 10.7.0 on discrete dev hosts). Leaving this
# unpinned lets apt silently install whatever NVIDIA currently ships as "latest"
# libnvinfer packages (for example a TensorRT 11.x ABI break), producing
# portable release binaries that fail to load anywhere still running TensorRT
# 10.x: `libnvinfer.so.11: cannot open shared object file`.
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

# Hosted runners do not have enough free disk for the full CUDA toolkit and
# TensorRT dev packages. Install only nvcc, cudart headers/libs, TensorRT
# headers, and the shared TensorRT runtime. Avoid libnvinfer-dev because it
# carries the large static archive that overflowed hosted-runner disk space.
sudo apt-get install -y --no-install-recommends \
    "${cuda_packages[@]}" \
    "libnvinfer-headers-dev=${tensorrt_package_version}" \
    "libnvinfer10=${tensorrt_package_version}"

tensorrt_soname="$(find /usr/lib/x86_64-linux-gnu -maxdepth 1 -name 'libnvinfer.so.*' -print | sort -V | tail -n1)"
if [[ -z "$tensorrt_soname" ]]; then
    echo "TensorRT runtime install did not create libnvinfer.so.*" >&2
    exit 1
fi
sudo mkdir -p /usr/local/lib
sudo ln -sf "$tensorrt_soname" /usr/local/lib/libnvinfer.so
sudo ldconfig

cuda_root="$(compgen -G '/usr/local/cuda-*' | sort -V | tail -n1)"
if [[ -z "$cuda_root" ]]; then
    echo "CUDA toolkit install did not create /usr/local/cuda-*" >&2
    exit 1
fi

{
    echo "CUDA_ROOT=$cuda_root"
    echo "PATH=$cuda_root/bin:$PATH"
} >>"$GITHUB_ENV"

echo "Installed CUDA compile dependencies at $cuda_root and TensorRT runtime headers"
