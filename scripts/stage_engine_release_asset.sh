#!/usr/bin/env bash
set -euo pipefail

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
Usage: $0 --engine-dir DIR [options]

Package a stable, unversioned engine archive and write one staging row consumed
by upload-engine-assets.yml.

Options:
  --engine-dir DIR       Validated engine bundle directory
  --output-dir DIR       Local output directory (default: dist)
  --copy-to DIR          Copy the archive and checksum to another directory
  --s3-uri URI           Upload under this exact s3://bucket/prefix
  --aws-region REGION    AWS region
  --presign-expires SEC  Presigned URL lifetime, 60..604800
  --download-url URL     Direct archive URL
  --public-url-base URL  Direct-download base URL
  --asset-manifest FILE  Staging row file
  --append               Append instead of replacing the row file
EOF_USAGE
}

while (($#)); do
    case "$1" in
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
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$engine_dir" ]]; then usage >&2; exit 2; fi
url_source_count=0
[[ -n "$download_url" ]] && ((url_source_count += 1))
[[ -n "$public_url_base" ]] && ((url_source_count += 1))
[[ -n "$s3_uri" ]] && ((url_source_count += 1))
if ((url_source_count > 1)); then
    echo "use only one URL/staging source" >&2
    exit 2
fi
if [[ ! "$presign_expires" =~ ^[0-9]+$ || "$presign_expires" -lt 60 || "$presign_expires" -gt 604800 ]]; then
    echo "--presign-expires must be 60..604800" >&2
    exit 2
fi
if [[ -n "$s3_uri" && "$s3_uri" != s3://* ]]; then
    echo "--s3-uri must start with s3://" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
[[ -n "$asset_manifest" ]] || asset_manifest="$output_dir/nvcr-engine-assets.txt"
"$script_dir/package_engine_bundle.sh" --engine-dir "$engine_dir" --output-dir "$output_dir"

archive_name="$(python3 - "$engine_dir/engine_manifest.json" <<'PY_NAME'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
profile = manifest["engine_profile_id"].removesuffix("-fp16")
compatibility = manifest.get("hardware_compatibility", "exact")
target = manifest["target_profile_id"]
if compatibility == "same_compute_capability":
    target = f"linux-amd64-sm{manifest['compute_capability_major']}{manifest['compute_capability_minor']}"
elif compatibility == "ampere_plus":
    target = "linux-amd64-ampere-plus"
print(f"nvcr-engines-{target}-{manifest['model_profile_id']}-{profile}.tar.gz")
PY_NAME
)"
archive="$output_dir/$archive_name"
checksum="$archive.sha256"
test -f "$archive" && test -f "$checksum"
(
    cd "$output_dir"
    sha256sum -c "$archive_name.sha256" >/dev/null
)
archive_sha256="$(awk '{print $1}' "$checksum")"

if [[ -n "$copy_to" ]]; then
    mkdir -p "$copy_to"
    cp -p "$archive" "$checksum" "$copy_to/"
fi
if [[ -n "$s3_uri" ]]; then
    command -v aws >/dev/null || { echo "aws CLI is required" >&2; exit 1; }
    aws_args=()
    [[ -z "$aws_region" ]] || aws_args+=(--region "$aws_region")
    s3_prefix="${s3_uri%/}"
    aws "${aws_args[@]}" s3 cp "$archive" "$s3_prefix/$archive_name"
    aws "${aws_args[@]}" s3 cp "$checksum" "$s3_prefix/$archive_name.sha256"
    download_url="$(aws "${aws_args[@]}" s3 presign "$s3_prefix/$archive_name" --expires-in "$presign_expires")"
elif [[ -n "$public_url_base" ]]; then
    download_url="${public_url_base%/}/$archive_name"
fi
[[ -n "$download_url" ]] || download_url="<staging-download-url-for-$archive_name>"
if [[ "$download_url" != https://* && "$download_url" != \<staging-download-url-for-* ]]; then
    echo "download URL must use HTTPS" >&2
    exit 1
fi
mkdir -p "$(dirname "$asset_manifest")"
row="$archive_name $archive_sha256 $download_url"
if ((append)); then printf '%s\n' "$row" >>"$asset_manifest"; else printf '%s\n' "$row" >"$asset_manifest"; fi
echo "$row"
