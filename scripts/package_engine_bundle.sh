#!/usr/bin/env bash
set -euo pipefail

engine_dir=""
output_dir="dist"

usage() {
    cat <<EOF_USAGE
Usage: $0 --engine-dir DIR [--output-dir DIR]

Packages one validated target-local TensorRT bundle for the rolling
'engine-assets' release. The stable archive name contains no NVCR application
version and no user-facing precision suffix.
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --engine-dir) engine_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$engine_dir" || ! -d "$engine_dir" ]]; then
    usage >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$engine_dir" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
"$script_dir/nvcr_artifacts.py" validate "$engine_dir" --json >/dev/null

asset_name="$(python3 - "$engine_dir/engine_manifest.json" <<'PY_NAME'
import json
import re
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
safe = re.compile(r"^[0-9A-Za-z._-]+$")
values = []
for key in ("target_profile_id", "model_profile_id", "engine_profile_id"):
    value = manifest.get(key)
    if not isinstance(value, str) or not safe.fullmatch(value):
        raise SystemExit(f"invalid {key}: {value!r}")
    values.append(value)
target, model, profile = values
if profile.endswith("-fp16"):
    profile = profile[:-5]
if profile != manifest.get("optimization_point"):
    raise SystemExit("engine profile does not match optimization point")
compatibility = manifest.get("hardware_compatibility", "exact")
if compatibility not in ("exact", "same_compute_capability", "ampere_plus"):
    raise SystemExit(f"invalid hardware_compatibility: {compatibility!r}")
if compatibility == "same_compute_capability":
    target = f"linux-amd64-sm{manifest['compute_capability_major']}{manifest['compute_capability_minor']}"
elif compatibility == "ampere_plus":
    target = "linux-amd64-ampere-plus"
print(f"nvcr-engines-{target}-{model}-{profile}")
PY_NAME
)"

staging_root="$(mktemp -d "${TMPDIR:-/tmp}/nvcr-engine-package.XXXXXX")"
trap 'rm -rf -- "$staging_root"' EXIT
bundle_root="$staging_root/$asset_name/dcvcrt"
mkdir -p "$bundle_root"
cp -p "$engine_dir/engine_manifest.json" "$bundle_root/engine_manifest.json"
cp -p "$engine_dir/engine.sha256" "$bundle_root/engine.sha256"

while read -r digest filename extra || [[ -n "${digest:-}${filename:-}${extra:-}" ]]; do
    [[ -z "${digest:-}${filename:-}${extra:-}" ]] && continue
    if [[ -n "${extra:-}" || ! "$digest" =~ ^[0-9a-f]{64}$ ||
          "$filename" == /* || "$filename" == *"/"* || "$filename" == *"\\"* ||
          "$filename" == "." || "$filename" == ".." ]]; then
        echo "invalid engine checksum manifest entry: ${digest:-} ${filename:-} ${extra:-}" >&2
        exit 1
    fi
    source_file="$engine_dir/$filename"
    if [[ ! -f "$source_file" || -L "$source_file" ]]; then
        echo "engine checksum entry is not a regular file: $filename" >&2
        exit 1
    fi
    cp -p "$source_file" "$bundle_root/$filename"
done <"$engine_dir/engine.sha256"
"$script_dir/nvcr_artifacts.py" validate "$bundle_root" --json >/dev/null

manifest_tmp="$staging_root/ENGINE-ASSET-MANIFEST.sha256.tmp"
(
    cd "$staging_root/$asset_name"
    find . -type f ! -name ENGINE-ASSET-MANIFEST.sha256 -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum >"$manifest_tmp"
)
mv "$manifest_tmp" "$staging_root/$asset_name/ENGINE-ASSET-MANIFEST.sha256"

archive="$output_dir/$asset_name.tar.gz"
tar -C "$staging_root" --sort=name --mtime="UTC 1970-01-01" \
    --owner=0 --group=0 --numeric-owner -czf "$archive" "$asset_name"
(
    cd "$output_dir"
    sha256sum "$(basename "$archive")" >"$(basename "$archive").sha256"
)
size_bytes="$(stat -c '%s' "$archive")"
if ((size_bytes >= 2147483648)); then
    echo "engine release asset is >= 2 GiB: $archive" >&2
    exit 1
fi
echo "Wrote $archive and $archive.sha256"
