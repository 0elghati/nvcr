#!/usr/bin/env bash
# Enforce that boundary-module public headers contain no forbidden symbols.
# Exits non-zero and prints each violation on first failure.
# Usage: scripts/ci/check_header_deps.sh [REPO_ROOT]
set -euo pipefail

REPO_ROOT="${1:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
INCLUDE_ROOT="${REPO_ROOT}/include"

# Headers belonging to the generic boundary modules must not pull in
# TensorRT, CUDA runtime, or DCVC-RT implementation internals.
BOUNDARY_DIRS=(
    "${INCLUDE_ROOT}/nvcr/common"
    "${INCLUDE_ROOT}/nvcr/runtime"
    "${INCLUDE_ROOT}/nvcr/codec"
    "${INCLUDE_ROOT}/nvcr/bitstream"
    "${INCLUDE_ROOT}/nvcr/provider"
    "${INCLUDE_ROOT}/nvcr/memory"
    "${INCLUDE_ROOT}/nvcr/logging"
    "${INCLUDE_ROOT}/nvcr/statistics"
    "${INCLUDE_ROOT}/nvcr/configuration"
)

# Symbols whose presence in boundary headers is a violation.
FORBIDDEN_PATTERNS=(
    'NvInfer'
    'nvinfer1'
    'cuda_runtime'
    'cuda_runtime_api'
    'cudaStream_t'
    'cudaError_t'
    'CUDAToolkit'
    'tensorrt_backend\.hpp'
    'dcvcrt/backend'
    'engine_specs\.hpp'
    'cuda_ops\.hpp'
)

VIOLATIONS=0

for dir in "${BOUNDARY_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' header; do
        for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
            if grep -qP "$pattern" "$header" 2>/dev/null; then
                echo "VIOLATION: ${header#"${REPO_ROOT}/"} matches forbidden pattern '${pattern}'"
                VIOLATIONS=$((VIOLATIONS + 1))
            fi
        done
    done < <(find "$dir" -name '*.hpp' -print0)
done

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "check_header_deps: $VIOLATIONS violation(s) found." >&2
    exit 1
fi

echo "check_header_deps: all boundary headers are clean."
