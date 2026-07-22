#!/usr/bin/env bash
# One-command configure/build/install for NVCR.
#
# Auto-detects the platform (Jetson vs. discrete GPU host), CUDA compiler,
# CUDA architecture, and TensorRT location via scripts/detect_platform.sh, and
# bootstraps a modern CMake (>= 3.24) via pip if the system one is too old.
# All auto-detected choices can be overridden with explicit flags.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

build_dir=""
prefix=""
build_type="Release"
enable_tensorrt=1
run_tests=0
jobs=""
arch_set="auto"
cuda_arch=""
cuda_compiler=""
tensorrt_root=""
extra_cmake_args=()

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --build-dir DIR     CMake build directory (default: build-<platform>)
  --prefix DIR        Install prefix (default: <repo>/install-<platform>)
  --build-type TYPE   CMake build type (default: Release)
  --no-tensorrt       Disable the TensorRT/CUDA backend (CPU-only build)
  --arch-set MODE     auto (default): detect this machine's single GPU for
                      the fastest local build. portable: build a
                      redistributable fat binary covering common Jetson and
                      discrete RTX/datacenter GPU architectures in one build
                      (slower, larger; intended for release packaging).
  --cuda-arch ARCH    Override CMAKE_CUDA_ARCHITECTURES explicitly; takes
                      precedence over --arch-set
  --cuda-compiler BIN Override auto-detected nvcc path
  --tensorrt-root DIR Override auto-detected TensorRT_ROOT
  --run-tests         Run ctest after building
  --jobs N            Parallel build jobs (default: nproc)
  --cmake-arg ARG     Extra literal argument forwarded to cmake configure
                      (repeatable)
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
    --build-dir) build_dir="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --build-type) build_type="$2"; shift 2 ;;
    --no-tensorrt) enable_tensorrt=0; shift ;;
    --arch-set) arch_set="$2"; shift 2 ;;
    --cuda-arch) cuda_arch="$2"; shift 2 ;;
    --cuda-compiler) cuda_compiler="$2"; shift 2 ;;
    --tensorrt-root) tensorrt_root="$2"; shift 2 ;;
    --run-tests) run_tests=1; shift ;;
    --jobs) jobs="$2"; shift 2 ;;
    --cmake-arg) extra_cmake_args+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ "$arch_set" != "auto" && "$arch_set" != "portable" ]]; then
    echo "invalid --arch-set: $arch_set (expected auto or portable)" >&2
    exit 2
fi

# shellcheck source=scripts/detect_platform.sh
source "$script_dir/detect_platform.sh"
nvcr_detect_platform_report

[[ -z "$build_dir" ]] && build_dir="build-${NVCR_DETECT_PLATFORM}"
[[ -z "$prefix" ]] && prefix="$repo_root/install-${NVCR_DETECT_PLATFORM}"
[[ -z "$jobs" ]] && jobs="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
# Only auto-fill --cuda-arch from single-machine detection in "auto" mode;
# "portable" mode must reach the arch_set branch below untouched so it uses
# the curated multi-arch list instead of this build machine's single GPU.
if [[ -z "$cuda_arch" && "$arch_set" == "auto" ]]; then
    cuda_arch="$NVCR_DETECT_ARCH"
fi
[[ -z "$cuda_compiler" ]] && cuda_compiler="$NVCR_DETECT_NVCC"
[[ -z "$tensorrt_root" ]] && tensorrt_root="$NVCR_DETECT_TENSORRT_ROOT"

# Bootstrap a modern-enough CMake (>= 3.24) if the system one is too old, or
# missing, using a pip-installed wheel (works for both x86_64 and aarch64).
cmake_bin="$(command -v cmake || true)"
need_bootstrap=0
if [[ -z "$cmake_bin" ]]; then
    need_bootstrap=1
else
    cmake_version="$("$cmake_bin" --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
    cmake_major="${cmake_version%%.*}"
    cmake_minor="$(echo "$cmake_version" | cut -d. -f2)"
    if [[ -z "$cmake_major" ]] || (( cmake_major < 3 )) || { (( cmake_major == 3 )) && (( cmake_minor < 24 )); }; then
        need_bootstrap=1
    fi
fi
if ((need_bootstrap)); then
    echo "System CMake is missing or older than 3.24 (required); installing a local copy with pip." >&2
    python3 -m pip install --user --upgrade "cmake>=3.24" >&2
    user_base="$(python3 -m site --user-base)"
    export PATH="$user_base/bin:$PATH"
    cmake_bin="$(command -v cmake)"
fi
echo "Using cmake: $cmake_bin ($("$cmake_bin" --version | head -1))"

cmake_args=(-S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type")
if ((enable_tensorrt)); then
    cmake_args+=(-DNVCR_ENABLE_TENSORRT=ON)
    [[ -n "$cuda_compiler" ]] && cmake_args+=(-DCMAKE_CUDA_COMPILER="$cuda_compiler")
    if [[ -n "$cuda_arch" ]]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_arch")
    elif [[ "$arch_set" == "portable" ]]; then
        # Let cmake/NVCRAutodetect.cmake pick its curated multi-arch list;
        # do not force a single build-machine architecture here.
        cmake_args+=(-DNVCR_CUDA_ARCH_SET=portable)
    else
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${NVCR_DETECT_ARCH:-native}")
    fi
    [[ -n "$tensorrt_root" ]] && cmake_args+=(-DTensorRT_ROOT="$tensorrt_root")
else
    cmake_args+=(-DNVCR_ENABLE_TENSORRT=OFF)
fi
cmake_args+=("${extra_cmake_args[@]}")

echo "Configuring: $cmake_bin ${cmake_args[*]}"
"$cmake_bin" "${cmake_args[@]}"
"$cmake_bin" --build "$build_dir" --parallel "$jobs"
"$cmake_bin" --install "$build_dir" --prefix "$prefix"

if ((run_tests)); then
    ctest_bin="$(command -v ctest || true)"
    if [[ -z "$ctest_bin" ]]; then
        ctest_bin="$(dirname "$cmake_bin")/ctest"
    fi
    "$ctest_bin" --test-dir "$build_dir" --output-on-failure
fi

cat <<EOF

NVCR installed to: $prefix
Add it to PATH for this shell:
  export PATH="$prefix/bin:\$PATH"
EOF
