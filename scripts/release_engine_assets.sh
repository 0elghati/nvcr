#!/usr/bin/env bash
set -euo pipefail

repo=""
workflow_ref="main"
asset_release="engine-assets"
output_dir="dist"
asset_manifest=""
s3_uri=""
s3_prefix=""
aws_region=""
presign_expires="604800"
dry_run=0
skip_dispatch=0
engine_dirs=()

usage() {
    cat <<EOF_USAGE
Usage: $0 --engine-dir DIR (--s3-prefix URI | --s3-uri URI) [options]

Stage target-local engines and dispatch the rolling engine-assets workflow.
Application versions and Release Please tags are intentionally not involved.

Options:
  --engine-dir DIR      Validated bundle; repeat for multiple profiles
  --s3-prefix URI       Base S3 prefix; asset-release tag is appended
  --s3-uri URI          Exact S3 staging prefix
  --asset-release TAG   Rolling GitHub release tag (default: engine-assets)
  --repo OWNER/REPO     GitHub repository
  --workflow-ref REF    Workflow ref (default: main)
  --output-dir DIR      Local archive directory (default: dist)
  --asset-manifest FILE Staging row file
  --aws-region REGION   AWS region
  --presign-expires SEC Presigned URL lifetime
  --skip-dispatch       Stage only
  --dry-run             Validate and print the dispatch command
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --engine-dir) engine_dirs+=("$2"); shift 2 ;;
    --s3-uri) s3_uri="$2"; shift 2 ;;
    --s3-prefix) s3_prefix="$2"; shift 2 ;;
    --asset-release) asset_release="$2"; shift 2 ;;
    --aws-region) aws_region="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --workflow-ref) workflow_ref="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --asset-manifest) asset_manifest="$2"; shift 2 ;;
    --presign-expires) presign_expires="$2"; shift 2 ;;
    --skip-dispatch) skip_dispatch=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
if [[ "${#engine_dirs[@]}" -eq 0 || ( -z "$s3_uri" && -z "$s3_prefix" ) ||
      ( -n "$s3_uri" && -n "$s3_prefix" ) ]]; then
    usage >&2
    exit 2
fi
if [[ -z "$repo" ]]; then
    origin_url="$(git -C "$repo_root" config --get remote.origin.url 2>/dev/null || true)"
    repo="$(printf '%s\n' "$origin_url" | sed -E 's#^git@github.com:##; s#^https://github.com/##; s#[.]git$##')"
fi
[[ "$repo" == */* ]] || { echo "pass --repo OWNER/REPO" >&2; exit 2; }
[[ "$asset_release" =~ ^[0-9A-Za-z._-]+$ ]] || { echo "invalid asset release" >&2; exit 2; }
if [[ -n "$s3_prefix" ]]; then s3_uri="${s3_prefix%/}/$asset_release"; fi
[[ "$s3_uri" == s3://* ]] || { echo "S3 URI must start with s3://" >&2; exit 2; }
if ((dry_run == 0)); then command -v aws >/dev/null || { echo "aws CLI is required" >&2; exit 1; }; fi
if ((skip_dispatch == 0)); then command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 1; }; fi

for engine_dir in "${engine_dirs[@]}"; do
    "$script_dir/nvcr_artifacts.py" validate "$engine_dir" --json >/dev/null
done
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
[[ -n "$asset_manifest" ]] || asset_manifest="$output_dir/nvcr-engine-assets.txt"

if ((dry_run == 0)); then
    rm -f -- "$asset_manifest"
    append=()
    for engine_dir in "${engine_dirs[@]}"; do
        "$script_dir/stage_engine_release_asset.sh" \
            --engine-dir "$engine_dir" --output-dir "$output_dir" \
            --s3-uri "${s3_uri%/}" --aws-region "$aws_region" \
            --presign-expires "$presign_expires" --asset-manifest "$asset_manifest" \
            "${append[@]}"
        append=(--append)
    done
fi
if ((skip_dispatch)); then
    echo "Staged rolling engine assets in $asset_manifest"
    exit 0
fi
workflow_args=(workflow run upload-engine-assets.yml --repo "$repo" --ref "$workflow_ref" \
    -f "asset_release=$asset_release" -F "engine_assets=@$asset_manifest")
if ((dry_run)); then
    printf 'gh'; printf ' %q' "${workflow_args[@]}"; printf '\n'
else
    gh "${workflow_args[@]}"
    echo "Dispatched rolling $asset_release engine asset update"
fi
