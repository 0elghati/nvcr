#!/usr/bin/env bash
set -euo pipefail

version=""
engine_dir=""
output_dir="dist"
copy_to=""
download_url=""
public_url_base=""
s3_uri=""
aws_region=""
presign_expires="604800"
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
  --s3-uri URI           Upload .tar.gz and .sha256 under this s3://bucket/prefix
                         and generate a presigned HTTPS download URL
  --aws-region REGION    AWS region for S3 upload/presign commands
  --presign-expires SEC  Presigned URL lifetime in seconds (default: 604800)
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
    --s3-uri) s3_uri="$2"; shift 2 ;;
    --aws-region) aws_region="$2"; shift 2 ;;
    --presign-expires) presign_expires="$2"; shift 2 ;;
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
url_source_count=0
[[ -n "$download_url" ]] && ((url_source_count += 1))
[[ -n "$public_url_base" ]] && ((url_source_count += 1))
[[ -n "$s3_uri" ]] && ((url_source_count += 1))
if ((url_source_count > 1)); then
    echo "use only one of --download-url, --public-url-base, or --s3-uri" >&2
    exit 2
fi
if [[ ! "$presign_expires" =~ ^[0-9]+$ || "$presign_expires" -lt 60 || "$presign_expires" -gt 604800 ]]; then
    echo "--presign-expires must be an integer between 60 and 604800 seconds" >&2
    exit 2
fi
if [[ -n "$s3_uri" && "$s3_uri" != s3://* ]]; then
    echo "--s3-uri must start with s3://" >&2
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

if [[ -n "$s3_uri" ]]; then
    if ! command -v aws >/dev/null; then
        echo "aws CLI is required when --s3-uri is used" >&2
        exit 1
    fi
    s3_prefix="${s3_uri%/}"
    aws_args=()
    if [[ -n "$aws_region" ]]; then
        aws_args+=(--region "$aws_region")
    fi
    aws "${aws_args[@]}" s3 cp "$archive" "$s3_prefix/$archive_name"
    aws "${aws_args[@]}" s3 cp "$checksum" "$s3_prefix/$archive_name.sha256"
    download_url="$(aws "${aws_args[@]}" s3 presign "$s3_prefix/$archive_name" --expires-in "$presign_expires")"
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
