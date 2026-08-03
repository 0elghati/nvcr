#!/usr/bin/env bash
set -euo pipefail

mode="${1:-gpu}"
source_dir="${NVCR_TEST_SOURCE_DIR:-/workspace/nvcr}"
engine_dir="${NVCR_ENGINE_DIR:-/opt/nvcr/engines}"
expected_arch="${NVCR_CONTAINER_ARCH:-}"
cuda_arch="${NVCR_CUDA_ARCHITECTURES:-}"

if [[ -n "$expected_arch" && "$(uname -m)" != "$expected_arch" ]]; then
    echo "nvcr-test: image expects $expected_arch but is running on $(uname -m)" >&2
    exit 2
fi

case "$mode" in
shell)
    exec /bin/bash
    ;;
cpu)
    enable_tensorrt=OFF
    build_dir="${NVCR_TEST_BUILD_ROOT:-/test-build}/cpu"
    ;;
gpu)
    enable_tensorrt=ON
    build_dir="${NVCR_TEST_BUILD_ROOT:-/test-build}/gpu"
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        cat >&2 <<EOF
nvcr-test: no NVIDIA GPU is visible in the container
Install/configure NVIDIA Container Toolkit on x86_64, or the NVIDIA container
runtime on Jetson, then launch the container with the GPU options in the
architecture-specific Compose file. Use 'cpu' for the non-GPU test suite.
EOF
        exit 2
    fi
    if [[ ! -f "$engine_dir/engine_manifest.json" ]]; then
        cat >&2 <<EOF
nvcr-test: no TensorRT engine bundle is mounted at $engine_dir
Set NVCR_ENGINE_DIR on the host for Compose, or mount a compatible target-local
bundle at /opt/nvcr/engines. Use 'cpu' to run the non-GPU test suite.
EOF
        exit 2
    fi
    echo "Validating mounted engine bundle: $engine_dir"
    python3 "$source_dir/scripts/nvcr_artifacts.py" validate "$engine_dir" --json
    ;;
*)
    echo "usage: nvcr-test [cpu|gpu|shell]" >&2
    exit 2
    ;;
esac

cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DNVCR_ENABLE_TENSORRT="$enable_tensorrt"
)
if [[ "$mode" == gpu ]]; then
    cmake_args+=(
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
        -DNVCR_TENSORRT_ENGINE_DIR="$engine_dir"
    )
fi

echo "Configuring NVCR test suite ($mode)"
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "${NVCR_BUILD_JOBS:-$(nproc)}"

echo "Registered tests"
ctest --test-dir "$build_dir" -N

echo "Running tests"
ctest --test-dir "$build_dir" --output-on-failure
