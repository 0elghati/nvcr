#!/usr/bin/env bash
# User-operated NVCR-only campaign using the already downloaded release bundles.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-run}"
case "$action" in run|resume|check|analyze) ;; *) echo 'Usage: bash scripts/measurement_nvcr.sh [run|resume|check|analyze] [output-directory]'; exit 2 ;; esac
output="${2:-evidence/measurement-nvcr}"
mkdir -p evidence/measurement-assets/logs
log_path="$(mktemp "$PWD/evidence/measurement-assets/logs/nvcr-$action.XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
trap 'result=$?; if (( result != 0 )); then echo "Stopped (exit $result). Log: $log_path"; fi' EXIT
echo "Log: $log_path"
common=(--implementation nvcr --engine-root "$PWD/build/engines-measurement-published")
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
  analyze)
    python3 scripts/measurement_campaign.py analyze --output "$output"
    ;;
esac
