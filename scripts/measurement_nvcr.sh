#!/usr/bin/env bash
# User-operated NVCR-only campaign using the already downloaded release bundles.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-run}"
case "$action" in run|resume|check|smoke|analyze) ;; *) echo 'Usage: bash scripts/measurement_nvcr.sh [run|resume|check|smoke|analyze] [output-directory]'; exit 2 ;; esac
output="${2:-${NVCR_MEASUREMENT_OUTPUT:-evidence/measurement-nvcr}}"
manifest="${NVCR_MEASUREMENT_MANIFEST:-$PWD/docs/experiments/measurement-campaign.json}"
engine_root="${NVCR_MEASUREMENT_ENGINE_ROOT:-$PWD/build/engines-measurement-published}"
mkdir -p evidence/measurement-assets/logs
log_path="$(mktemp "$PWD/evidence/measurement-assets/logs/nvcr-$action.XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
report_failure() {
  local status=$?
  if (( status != 0 )); then echo "Stopped (exit $status). Log: $log_path"; fi
}
trap report_failure EXIT
echo "Log: $log_path"
common=(--manifest "$manifest" --implementation nvcr --engine-root "$engine_root")
case "$action" in
  run)
    if [[ -d "$output" ]] && [[ -n "$(ls -A "$output")" ]]; then
      echo 'Output already contains data. Use resume, or choose a new output-directory.'
      exit 2
    fi
    smoke_output="$(mktemp -d "$PWD/evidence/measurement-nvcr-smoke.XXXXXX")"
    python3 scripts/measurement_campaign.py smoke "${common[@]}" --output "$smoke_output"
    # Save beside the campaign, without making its output non-empty before launch.
    mkdir -p "$(dirname "$output")"
    printf '%s\n' "$smoke_output" > "${output}.smoke-path.txt"
    python3 scripts/measurement_campaign.py run "${common[@]}" \
      --smoke-evidence "$smoke_output" --output "$output"
    ;;
  resume)
    [[ -f "${output}.smoke-path.txt" ]] || { echo 'Missing smoke path for this campaign.'; exit 2; }
    python3 scripts/measurement_campaign.py run "${common[@]}" --resume \
      --smoke-evidence "$(cat "${output}.smoke-path.txt")" --output "$output"
    ;;
  check)
    check_output="$(mktemp -d "$PWD/evidence/measurement-nvcr-check.XXXXXX")"
    python3 scripts/measurement_campaign.py run "${common[@]}" --dry-run --output "$check_output"
    ;;
  smoke)
    smoke_output="$(mktemp -d "$PWD/evidence/measurement-nvcr-smoke.XXXXXX")"
    python3 scripts/measurement_campaign.py smoke "${common[@]}" --output "$smoke_output"
    printf '%s\n' "$smoke_output" > "${output}.smoke-path.txt"
    echo "NVCR smoke passed: $smoke_output"
    ;;
  analyze)
    python3 scripts/measurement_campaign.py analyze --output "$output"
    ;;
esac
