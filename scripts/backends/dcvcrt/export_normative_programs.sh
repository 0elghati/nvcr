#!/usr/bin/env bash
# Exports the DCVC-RT dcvc-rt.exact.v1 normative programs (issue #216): every
# decoder-side stage lowered from the exported ONNX stages, with the
# validated nvcr.det-int.v1 bit-width map. The programs are target independent;
# encoders and decoders on every provider must use the same files (their
# digests are written into every exact stream).
#
# Artifact bundles carry them in a normative/ subdirectory, where the runtime
# looks by default (prepare_artifacts.sh and build_tensorrt.sh do this).
#
# Usage: export_normative_programs.sh ONNX_MODEL_DIR OUTPUT_DIR
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 ONNX_MODEL_DIR OUTPUT_DIR" >&2
    exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
"${PYTHON:-python3}" "${root}/scripts/backends/dcvcrt/export_deterministic_program.py" \
    --model-dir "$1" --output-dir "$2" \
    --integer-map "${root}/configs/numerics/dcvcrt-det-int-v1.json" \
    i_hyper_synthesis i_spatial_prior_1 i_spatial_prior_2 i_spatial_prior_3 \
    i_synthesis p_reference_frame p_reference_feature p_prior p_spatial_prior p_synthesis
