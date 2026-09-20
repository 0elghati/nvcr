#!/usr/bin/env bash
# User-operated convenience entry point; never changes services or power settings.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
  prepare|check|smoke|run|resume|analyze) action="$1" ;;
  *) echo "Usage: bash scripts/measurement_jetson.sh {prepare|check|smoke|run|resume|analyze}"; exit 2 ;;
esac
assets_dir="$PWD/evidence/measurement-assets"
mkdir -p "$assets_dir/logs"
log_path="$(mktemp "$assets_dir/logs/$action.XXXXXX.log")"
exec > >(tee -a "$log_path") 2>&1
trap 'result=$?; if (( result != 0 )); then echo "Stopped (exit $result). Keep this log: $log_path"; fi' EXIT
echo "Log: $log_path"
manifest="$assets_dir/campaign.json"

if [[ "$action" == prepare ]]; then
  cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
  cmake --build build-release --target nvcr_cli nvcr_measurement_wire_fixture -j 2
  ctest --test-dir build-release --output-on-failure \
    -R 'nvcr_measurement_semantics|nvcr_legacy_consolidation|nvcr_softwarex_driver|nvcr_contract_tests|nvcr_format_contract_tests|nvcr_dcvcrt_payloads|nvcr_cli_accepts_inter_gop'
  python3 - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
base = root/'evidence/measurement-assets'
m = json.loads((root/'docs/experiments/measurement-campaign.json').read_text())
m['engine_root'] = str(root/'build/engines-measurement-published')
(base/'campaign.json').write_text(json.dumps(m, indent=2)+'\n')
PY
  python3 scripts/nvcr_artifacts.py install \
    --repo 0elghati/nvcr --asset-release engine-assets \
    --profile qcif cif 360p 540p 720p 1080p \
    --engine-root build/engines-measurement-published
  echo 'Preparation finished. Next: bash scripts/measurement_jetson.sh smoke'
  exit 0
fi

if [[ ! -f "$manifest" ]]; then
  echo 'Local artifacts are not prepared. First: bash scripts/measurement_jetson.sh prepare'
  exit 2
fi
case "$action" in
  check)
    output="$(mktemp -d "$PWD/evidence/measurement-check.XXXXXX")"
    python3 scripts/measurement_campaign.py run --dry-run --manifest "$manifest" --output "$output"
    ;;
  smoke)
    output="$(mktemp -d "$PWD/evidence/measurement-smoke.XXXXXX")"
    python3 scripts/measurement_campaign.py smoke --manifest "$manifest" --output "$output"
    printf '%s\n' "$output" > "$assets_dir/passed-smoke-path.txt"
    echo 'Smoke passed. Next: bash scripts/measurement_jetson.sh run'
    ;;
  run|resume)
    [[ -f "$assets_dir/passed-smoke-path.txt" ]] || { echo 'A passing smoke is required first.'; exit 2; }
    smoke_path="$(cat "$assets_dir/passed-smoke-path.txt")"
    extra=()
    [[ "$action" != resume ]] || extra+=(--resume)
    python3 scripts/measurement_campaign.py run --manifest "$manifest" \
      --smoke-evidence "$smoke_path" --output evidence/measurement-campaign "${extra[@]}"
    ;;
  analyze)
    python3 scripts/measurement_campaign.py analyze --output evidence/measurement-campaign
    ;;
esac
