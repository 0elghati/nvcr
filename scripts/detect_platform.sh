#!/usr/bin/env bash
# Detects the NVIDIA platform (Jetson vs. discrete GPU host), CUDA
# architecture, CUDA/TensorRT install locations, and a conservative TensorRT
# engine-build memory budget (workspace size + builder optimization level).
#
# Usage:
#   source scripts/detect_platform.sh   # sets NVCR_DETECT_* variables, silent
#   ./scripts/detect_platform.sh        # runs the same detection and prints
#                                        # a human-readable report
#
# This script intentionally avoids `set -e`/`set -u` so it is safe to source
# from callers with different shell option expectations.

nvcr_detect_platform() {
    NVCR_DETECT_PLATFORM="unknown"
    NVCR_DETECT_ARCH=""
    NVCR_DETECT_NVCC=""
    NVCR_DETECT_CUDA_ROOT=""
    NVCR_DETECT_TENSORRT_ROOT=""
    NVCR_DETECT_TRTEXEC=""
    NVCR_DETECT_DEVICE_ID=0
    NVCR_DETECT_GPU_FREE_MIB=""
    NVCR_DETECT_MEM_AVAILABLE_MIB=""
    NVCR_DETECT_WORKSPACE_MIB=512
    NVCR_DETECT_BUILDER_LEVEL=1

    # --- Platform: Jetson (Tegra SoC, unified CPU/GPU memory) vs. a host with
    # a discrete GPU. ---
    if [[ -f /etc/nv_tegra_release ]]; then
        NVCR_DETECT_PLATFORM="jetson"
    elif [[ -f /proc/device-tree/model ]] && grep -qiE "jetson|tegra|orin|xavier" /proc/device-tree/model 2>/dev/null; then
        NVCR_DETECT_PLATFORM="jetson"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NVCR_DETECT_PLATFORM="discrete"
    fi

    # --- CUDA architecture and best device id via nvidia-smi. This works on
    # both Jetson and discrete GPUs; nvidia-smi reports compute_cap correctly
    # on Jetson even though memory.total/memory.free report N/A there. ---
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_csv best_index best_free idx cap free
        gpu_csv="$(nvidia-smi --query-gpu=index,compute_cap,memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
        best_index=""
        best_free=-1
        if [[ -n "$gpu_csv" ]]; then
            while IFS=',' read -r idx cap free; do
                idx="$(echo "${idx:-}" | tr -d '[:space:]')"
                cap="$(echo "${cap:-}" | tr -d '[:space:]')"
                free="$(echo "${free:-}" | tr -d '[:space:]')"
                if [[ -z "$NVCR_DETECT_ARCH" && "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
                    NVCR_DETECT_ARCH="${cap//./}"
                fi
                if [[ "$free" =~ ^[0-9]+$ ]] && (( free > best_free )); then
                    best_free="$free"
                    best_index="$idx"
                fi
            done <<<"$gpu_csv"
        fi
        [[ -n "$best_index" ]] && NVCR_DETECT_DEVICE_ID="$best_index"
        (( best_free >= 0 )) && NVCR_DETECT_GPU_FREE_MIB="$best_free"
    fi

    # --- CUDA compiler. ---
    NVCR_DETECT_NVCC="$(command -v nvcc || true)"
    if [[ -z "$NVCR_DETECT_NVCC" ]]; then
        local candidate
        for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
            [[ -x "$candidate" ]] && NVCR_DETECT_NVCC="$candidate"
        done
    fi
    [[ -n "$NVCR_DETECT_NVCC" ]] && NVCR_DETECT_CUDA_ROOT="$(cd "$(dirname "$(dirname "$NVCR_DETECT_NVCC")")" && pwd)"

    # --- TensorRT root and trtexec. ---
    NVCR_DETECT_TRTEXEC="$(command -v trtexec || true)"
    if [[ -z "$NVCR_DETECT_TRTEXEC" ]]; then
        local candidate
        for candidate in /usr/src/tensorrt/bin/trtexec /usr/src/tensorrt/targets/*/bin/trtexec; do
            [[ -x "$candidate" ]] && NVCR_DETECT_TRTEXEC="$candidate"
        done
    fi
    if [[ -f /usr/include/NvInfer.h ]]; then
        NVCR_DETECT_TENSORRT_ROOT="/usr"
    else
        local hdr
        for hdr in /usr/include/*-linux-gnu/NvInfer.h; do
            [[ -f "$hdr" ]] && NVCR_DETECT_TENSORRT_ROOT="/usr"
        done
    fi
    if [[ -z "$NVCR_DETECT_TENSORRT_ROOT" && -d /usr/src/tensorrt ]]; then
        NVCR_DETECT_TENSORRT_ROOT="/usr/src/tensorrt"
    fi
    if [[ -z "$NVCR_DETECT_TENSORRT_ROOT" ]] && command -v python3 >/dev/null 2>&1; then
        local py_root
        py_root="$(python3 -c 'import tensorrt, os; print(os.path.dirname(tensorrt.__file__))' 2>/dev/null || true)"
        [[ -n "$py_root" ]] && NVCR_DETECT_TENSORRT_ROOT="$py_root"
    fi

    # --- Memory budget for TensorRT engine building. Jetson has unified
    # CPU/GPU memory, so use /proc/meminfo; discrete GPUs use nvidia-smi's
    # reported free VRAM. ---
    NVCR_DETECT_MEM_AVAILABLE_MIB="$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || true)"

    local budget_mib=""
    if [[ "$NVCR_DETECT_PLATFORM" == "jetson" ]]; then
        budget_mib="$NVCR_DETECT_MEM_AVAILABLE_MIB"
    elif [[ -n "$NVCR_DETECT_GPU_FREE_MIB" ]]; then
        budget_mib="$NVCR_DETECT_GPU_FREE_MIB"
    else
        budget_mib="$NVCR_DETECT_MEM_AVAILABLE_MIB"
    fi

    if [[ "$budget_mib" =~ ^[0-9]+$ ]]; then
        if (( budget_mib >= 16000 )); then
            NVCR_DETECT_WORKSPACE_MIB=2048
            NVCR_DETECT_BUILDER_LEVEL=3
        elif (( budget_mib >= 8000 )); then
            NVCR_DETECT_WORKSPACE_MIB=1024
            NVCR_DETECT_BUILDER_LEVEL=2
        elif (( budget_mib >= 3000 )); then
            NVCR_DETECT_WORKSPACE_MIB=512
            NVCR_DETECT_BUILDER_LEVEL=1
        else
            NVCR_DETECT_WORKSPACE_MIB=256
            NVCR_DETECT_BUILDER_LEVEL=0
        fi
    fi
}

nvcr_detect_platform_report() {
    cat <<EOF
NVCR platform auto-detection
  platform              : ${NVCR_DETECT_PLATFORM}
  CUDA architecture     : ${NVCR_DETECT_ARCH:-unknown (cmake will fall back to CMAKE_CUDA_ARCHITECTURES=native)}
  nvcc                  : ${NVCR_DETECT_NVCC:-not found}
  CUDA root             : ${NVCR_DETECT_CUDA_ROOT:-not found}
  TensorRT root         : ${NVCR_DETECT_TENSORRT_ROOT:-not found}
  trtexec               : ${NVCR_DETECT_TRTEXEC:-not found}
  selected device id    : ${NVCR_DETECT_DEVICE_ID}
  GPU free memory (MiB) : ${NVCR_DETECT_GPU_FREE_MIB:-N/A (unified memory)}
  MemAvailable (MiB)    : ${NVCR_DETECT_MEM_AVAILABLE_MIB:-unknown}
  TensorRT workspace    : ${NVCR_DETECT_WORKSPACE_MIB} MiB
  builder opt level     : ${NVCR_DETECT_BUILDER_LEVEL}
EOF
}

nvcr_detect_platform

# Print a report only when this script is executed directly, not sourced.
if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
    nvcr_detect_platform_report
fi
