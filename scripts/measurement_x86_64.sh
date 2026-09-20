#!/usr/bin/env bash
# User-operated NVCR-only campaign for the exact local RTX 4070 target.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NVCR_MEASUREMENT_MANIFEST="${NVCR_MEASUREMENT_MANIFEST:-$PWD/docs/experiments/measurement-campaign-rtx4070.json}"
export NVCR_MEASUREMENT_ENGINE_ROOT="${NVCR_MEASUREMENT_ENGINE_ROOT:-$PWD/build/engines-measurement-published}"
export NVCR_MEASUREMENT_OUTPUT="${NVCR_MEASUREMENT_OUTPUT:-evidence/measurement-rtx4070}"
exec bash scripts/measurement_nvcr.sh "$@"
