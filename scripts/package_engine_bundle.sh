#!/usr/bin/env bash
set -euo pipefail

version=""
engine_dir=""
output_dir="dist"

usage() {
    cat <<EOF_USAGE
Usage: $0 --version X.Y.Z --engine-dir DIR [options]

Packages one validated NVCR TensorRT engine bundle as a separate reviewer
convenience release asset. This is not the public binary package; it contains
target-specific plans and runtime assets and must be downloaded separately from
the GitHub Release.

Options:
  --version X.Y.Z      Release version without leading v
  --engine-dir DIR     Validated engine bundle directory
  --output-dir DIR     Archive output directory (default: dist)
  -h, --help           Show this help
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --version) version="$2"; shift 2 ;;
    --engine-dir) engine_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ -z "$version" || -z "$engine_dir" ]]; then
    usage >&2
    exit 2
fi
if [[ "$version" == v* || "$version" =~ [^0-9A-Za-z.+-] ]]; then
    echo "invalid version label: $version" >&2
    exit 2
fi
if [[ ! -d "$engine_dir" ]]; then
    echo "engine directory does not exist: $engine_dir" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$(cd "$engine_dir" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

"$script_dir/nvcr_artifacts.py" validate "$engine_dir" --json >/dev/null

manifest_fields="$(
    python3 - "$engine_dir/engine_manifest.json" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
safe = re.compile(r"^[0-9A-Za-z._-]+$")
keys = ("model_profile_id", "target_profile_id", "engine_profile_id")
values = []
for key in keys:
    value = manifest.get(key)
    if not isinstance(value, str) or not safe.fullmatch(value):
        raise SystemExit(f"{manifest_path}: invalid {key}: {value!r}")
    values.append(value)
print("\t".join(values))
PY
)"
IFS=$'\t' read -r model_profile_id target_profile_id engine_profile_id <<<"$manifest_fields"

asset_name="nvcr-v$version-$model_profile_id-$target_profile_id-$engine_profile_id-engines"
staging_root="$(mktemp -d "${TMPDIR:-/tmp}/nvcr-engine-package.XXXXXX")"
trap 'rm -rf -- "$staging_root"' EXIT
bundle_root="$staging_root/$asset_name/dcvcrt"
mkdir -p "$bundle_root"

cp -p "$engine_dir/engine_manifest.json" "$bundle_root/engine_manifest.json"
cp -p "$engine_dir/engine.sha256" "$bundle_root/engine.sha256"

while read -r digest filename extra || [[ -n "${digest:-}${filename:-}${extra:-}" ]]; do
    if [[ -z "${digest:-}" && -z "${filename:-}" && -z "${extra:-}" ]]; then
        continue
    fi
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

(
    cd "$staging_root/$asset_name"
    manifest_tmp="ENGINE-ASSET-MANIFEST.sha256.tmp"
    find . -type f ! -name ENGINE-ASSET-MANIFEST.sha256 ! -name "$manifest_tmp" -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum >"$manifest_tmp"
    mv "$manifest_tmp" ENGINE-ASSET-MANIFEST.sha256
)

archive="$output_dir/$asset_name.tar.gz"
tar -C "$staging_root" \
    --sort=name \
    --mtime="UTC 1970-01-01" \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -czf "$archive" "$asset_name"

(
    cd "$output_dir"
    sha256sum "$(basename "$archive")" >"$(basename "$archive").sha256"
)

size_bytes="$(stat -c '%s' "$archive")"
if ((size_bytes >= 2147483648)); then
    echo "engine release asset is >= 2 GiB and cannot be uploaded to GitHub Releases as one file: $archive" >&2
    exit 1
fi

echo "Wrote $archive and $archive.sha256"
