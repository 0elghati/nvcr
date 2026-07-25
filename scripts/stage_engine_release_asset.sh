#!/usr/bin/env bash
set -euo pipefail

version=""
engine_dir=""
output_dir="dist"
copy_to=""
download_url=""
public_url_base=""
asset_manifest=""
append=0

usage() {
    cat <<EOF_USAGE
Usage: $0 --version X.Y.Z --engine-dir DIR [options]

Package a validated TensorRT engine bundle and generate the engine_assets.txt row
consumed by upload-engine-assets.yml. Optionally copy the archive and checksum to
a local staging folder such as a OneDrive-synced directory.

Options:
  --version X.Y.Z        Release version without leading v
  --engine-dir DIR       Validated engine bundle directory
  --output-dir DIR       Local archive output directory (default: dist)
  --copy-to DIR          Also copy .tar.gz and .sha256 to this staging directory
  --download-url URL     Direct HTTPS download URL for the .tar.gz archive
  --public-url-base URL  Build URL as URL/asset-file-name; use only for hosts
                         where that form is a direct archive download
  --asset-manifest FILE  Output engine asset input file
                         (default: OUTPUT_DIR/nvcr-engine-assets.txt)
  --append               Append to asset manifest instead of replacing it
  -h, --help             Show this help
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --version) version="$2"; shift 2 ;;
    --engine-dir) engine_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --copy-to) copy_to="$2"; shift 2 ;;
    --download-url) download_url="$2"; shift 2 ;;
    --public-url-base) public_url_base="$2"; shift 2 ;;
    --asset-manifest) asset_manifest="$2"; shift 2 ;;
    --append) append=1; shift ;;
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
if [[ -n "$download_url" && -n "$public_url_base" ]]; then
    echo "use either --download-url or --public-url-base, not both" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
if [[ -z "$asset_manifest" ]]; then
    asset_manifest="$output_dir/nvcr-engine-assets.txt"
fi

before_list="$(mktemp "${TMPDIR:-/tmp}/nvcr-engine-before.XXXXXX")"
after_list="$(mktemp "${TMPDIR:-/tmp}/nvcr-engine-after.XXXXXX")"
trap 'rm -f -- "$before_list" "$after_list"' EXIT
find "$output_dir" -maxdepth 1 -type f -name 'nvcr-v*-engines.tar.gz' -print | sort >"$before_list"

"$script_dir/package_engine_bundle.sh" \
    --version "$version" \
    --engine-dir "$engine_dir" \
    --output-dir "$output_dir"

find "$output_dir" -maxdepth 1 -type f -name 'nvcr-v*-engines.tar.gz' -print | sort >"$after_list"
archive="$(comm -13 "$before_list" "$after_list" | tail -n 1)"
if [[ -z "$archive" ]]; then
    archive="$(find "$output_dir" -maxdepth 1 -type f -name "nvcr-v$version-*-engines.tar.gz" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d ' ' -f2-)"
fi
if [[ -z "$archive" || ! -f "$archive" ]]; then
    echo "could not identify generated engine archive in $output_dir" >&2
    exit 1
fi
checksum="$archive.sha256"
if [[ ! -f "$checksum" ]]; then
    echo "missing checksum file: $checksum" >&2
    exit 1
fi
(
    cd "$(dirname "$archive")"
    sha256sum -c "$(basename "$checksum")" >/dev/null
)
archive_name="$(basename "$archive")"
archive_sha256="$(awk '{print $1}' "$checksum")"

if [[ -n "$copy_to" ]]; then
    mkdir -p "$copy_to"
    copy_to="$(cd "$copy_to" && pwd)"
    cp -p "$archive" "$checksum" "$copy_to/"
    echo "Copied $archive_name and $archive_name.sha256 to $copy_to"
fi

if [[ -n "$public_url_base" ]]; then
    download_url="$(python3 - "$public_url_base" "$archive_name" <<'NVCR_URL_PY'
from urllib.parse import quote
import sys
base = sys.argv[1].rstrip('/')
name = quote(sys.argv[2])
print(f"{base}/{name}")
NVCR_URL_PY
)"
fi
if [[ -z "$download_url" ]]; then
    download_url="<staging-download-url-for-$archive_name>"
fi
if [[ "$download_url" != https://* && "$download_url" != \<staging-download-url-for-* ]]; then
    echo "download URL must be HTTPS for the GitHub workflow: $download_url" >&2
    exit 1
fi

manifest_dir="$(dirname "$asset_manifest")"
mkdir -p "$manifest_dir"
row="$archive_name $archive_sha256 $download_url"
if ((append)); then
    printf '%s\n' "$row" >>"$asset_manifest"
else
    printf '%s\n' "$row" >"$asset_manifest"
fi

echo "Wrote $asset_manifest"
echo "$row"
