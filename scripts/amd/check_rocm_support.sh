#!/usr/bin/env bash
# Reports the current state of ROCm, the AMD GPU driver, and ONNX Runtime
# with the MIGraphX execution provider on this machine. Read-only: installs
# nothing, changes no system state. Prints a human-readable summary and
# sets NVCR_AMD_* variables when sourced.
#
# Usage:
#   source scripts/amd/check_rocm_support.sh   # silent, sets NVCR_AMD_*
#   ./scripts/amd/check_rocm_support.sh        # prints a report
#
# Covers the acceptance criteria of issue #224: OS, ROCm, driver versions,
# detected GPU, kernel module state, user group membership, and whether the
# ONNX Runtime MIGraphX execution provider is importable.

nvcr_amd_check() {
    NVCR_AMD_OS=""
    NVCR_AMD_KERNEL=""
    NVCR_AMD_GPU_PCI=""
    NVCR_AMD_GPU_NAME=""
    NVCR_AMD_GFX_TARGET=""
    NVCR_AMD_AMDGPU_LOADED="no"
    NVCR_AMD_ROCM_VERSION=""
    NVCR_AMD_ROCMINFO=""
    NVCR_AMD_USER_RENDER="no"
    NVCR_AMD_USER_VIDEO="no"
    NVCR_AMD_ORT_VERSION=""
    NVCR_AMD_ORT_PROVIDERS=""
    NVCR_AMD_HSA_OVERRIDE="${HSA_OVERRIDE_GFX_VERSION:-}"

    if [[ -r /etc/os-release ]]; then
        NVCR_AMD_OS="$(. /etc/os-release && printf '%s %s' "${NAME:-unknown}" "${VERSION:-unknown}")"
    fi
    NVCR_AMD_KERNEL="$(uname -r 2>/dev/null)"

    if command -v lspci >/dev/null 2>&1; then
        NVCR_AMD_GPU_PCI="$(lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' | grep -i amd | head -n1)"
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        NVCR_AMD_ROCMINFO="present"
        NVCR_AMD_GPU_NAME="$(rocminfo 2>/dev/null | awk -F: '/Marketing Name/ {sub(/^[ \t]+/, "", $2); print $2; exit}')"
        NVCR_AMD_GFX_TARGET="$(rocminfo 2>/dev/null | awk -F: '/gfx[0-9]+/ {for (i=1;i<=NF;i++) if ($i ~ /gfx[0-9]+/) {print $i; exit}}' | tr -d ' ')"
    else
        NVCR_AMD_ROCMINFO="missing"
    fi

    if [[ -r /sys/module/amdgpu/version ]] || lsmod 2>/dev/null | grep -q '^amdgpu '; then
        NVCR_AMD_AMDGPU_LOADED="yes"
    fi

    if command -v dpkg >/dev/null 2>&1; then
        NVCR_AMD_ROCM_VERSION="$(dpkg -l 'rocm-core' 2>/dev/null | awk '/^ii/ {print $3; exit}')"
    fi
    if [[ -z "${NVCR_AMD_ROCM_VERSION}" && -r /opt/rocm/.info/version ]]; then
        NVCR_AMD_ROCM_VERSION="$(cat /opt/rocm/.info/version 2>/dev/null)"
    fi

    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx render; then NVCR_AMD_USER_RENDER="yes"; fi
    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx video;  then NVCR_AMD_USER_VIDEO="yes"; fi

    local python
    for python in python3 python; do
        if command -v "$python" >/dev/null 2>&1; then
            NVCR_AMD_ORT_VERSION="$("$python" -c 'import onnxruntime as ort; print(ort.__version__)' 2>/dev/null)"
            NVCR_AMD_ORT_PROVIDERS="$("$python" -c 'import onnxruntime as ort; print(",".join(ort.get_available_providers()))' 2>/dev/null)"
            if [[ -n "${NVCR_AMD_ORT_VERSION}" ]]; then break; fi
        fi
    done
}

nvcr_amd_report() {
    nvcr_amd_check
    printf 'OS:               %s\n' "${NVCR_AMD_OS:-unknown}"
    printf 'Kernel:           %s\n' "${NVCR_AMD_KERNEL:-unknown}"
    printf 'GPU (PCI):        %s\n' "${NVCR_AMD_GPU_PCI:-not detected}"
    printf 'GPU (rocminfo):   %s\n' "${NVCR_AMD_GPU_NAME:-not detected}"
    printf 'GFX target:       %s\n' "${NVCR_AMD_GFX_TARGET:-unknown}"
    printf 'amdgpu loaded:    %s\n' "${NVCR_AMD_AMDGPU_LOADED}"
    printf 'rocminfo:         %s\n' "${NVCR_AMD_ROCMINFO}"
    printf 'ROCm version:     %s\n' "${NVCR_AMD_ROCM_VERSION:-not installed}"
    printf 'User in render:   %s\n' "${NVCR_AMD_USER_RENDER}"
    printf 'User in video:    %s\n' "${NVCR_AMD_USER_VIDEO}"
    printf 'HSA_OVERRIDE:     %s\n' "${NVCR_AMD_HSA_OVERRIDE:-not set}"
    printf 'onnxruntime:      %s\n' "${NVCR_AMD_ORT_VERSION:-not importable}"
    printf 'ORT providers:    %s\n' "${NVCR_AMD_ORT_PROVIDERS:-unknown}"

    if [[ "${NVCR_AMD_GFX_TARGET}" == gfx1031 ]] && [[ -z "${NVCR_AMD_HSA_OVERRIDE}" ]]; then
        printf '\nNote: gfx1031 (RX 6700 / 6700 XT) is outside the official ROCm support\n'
        printf '      matrix. Export HSA_OVERRIDE_GFX_VERSION=10.3.0 before running ROCm\n'
        printf '      workloads; expect best-effort behaviour and track as a risk.\n'
    fi
    if [[ "${NVCR_AMD_ORT_PROVIDERS}" != *MIGraphXExecutionProvider* ]] && [[ -n "${NVCR_AMD_ORT_PROVIDERS}" ]]; then
        printf '\nNote: MIGraphXExecutionProvider is not in the ORT providers list. The\n'
        printf '      installed onnxruntime wheel does not include the MIGraphX EP.\n'
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    nvcr_amd_report
fi
