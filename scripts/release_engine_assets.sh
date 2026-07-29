#!/usr/bin/env bash
set -euo pipefail

tag=""
repo=""
workflow_ref="main"
output_dir="dist"
asset_manifest=""
s3_uri=""
aws_region=""
presign_expires="604800"
publish_release=0
publish_confirmation=""
dry_run=0
skip_dispatch=0
engine_dirs=()

usage() {
    cat <<EOF_USAGE
Usage: $0 --tag vX.Y.Z --s3-uri s3://bucket/prefix --engine-dir DIR [options]

Stage validated TensorRT engine bundles to S3 and dispatch the GitHub workflow
that copies them into the matching Release Please draft release.

Options:
  --tag vX.Y.Z          Exact Release Please tag and draft release
  --engine-dir DIR      Validated engine bundle directory; repeat for many
  --s3-uri URI          S3 staging prefix, usually s3://bucket/releases/vX.Y.Z
  --aws-region REGION   AWS region for S3 upload/presign commands
  --repo OWNER/REPO     GitHub repository for workflow dispatch
  --workflow-ref REF    Ref containing upload-engine-assets.yml (default: main)
  --output-dir DIR      Local archive output directory (default: dist)
  --asset-manifest FILE Manifest file to write (default: OUTPUT_DIR/nvcr-engine-assets.txt)
  --presign-expires SEC Presigned URL lifetime, 60..604800 (default: 604800)
  --publish-release     Ask upload workflow to publish after upload
  --publish-confirmation TAG
                         Required with --publish-release; must equal --tag
  --skip-dispatch       Stage assets only; do not call GitHub Actions
  --dry-run             Validate inputs and print the workflow command only
  -h, --help            Show this help
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --tag) tag="$2"; shift 2 ;;
    --engine-dir) engine_dirs+=("$2"); shift 2 ;;
    --s3-uri) s3_uri="$2"; shift 2 ;;
    --aws-region) aws_region="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --workflow-ref) workflow_ref="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --asset-manifest) asset_manifest="$2"; shift 2 ;;
    --presign-expires) presign_expires="$2"; shift 2 ;;
    --publish-release) publish_release=1; shift ;;
    --publish-confirmation) publish_confirmation="$2"; shift 2 ;;
    --skip-dispatch) skip_dispatch=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ -z "$tag" || -z "$s3_uri" || "${#engine_dirs[@]}" -eq 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$tag" =~ ^v[0-9]+[.][0-9]+[.][0-9]+([.+-][0-9A-Za-z.+-]+)?$ ]]; then
    echo "--tag must be a Release Please style tag such as v0.4.0: $tag" >&2
    exit 2
fi
if [[ "$s3_uri" != s3://* ]]; then
    echo "--s3-uri must start with s3://: $s3_uri" >&2
    exit 2
fi
if [[ ! "$presign_expires" =~ ^[0-9]+$ || "$presign_expires" -lt 60 || "$presign_expires" -gt 604800 ]]; then
    echo "--presign-expires must be an integer between 60 and 604800 seconds" >&2
    exit 2
fi
if ((publish_release)) && [[ "$publish_confirmation" != "$tag" ]]; then
    echo "--publish-confirmation must exactly match --tag when publishing" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
version="${tag#v}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
if [[ -z "$asset_manifest" ]]; then
    asset_manifest="$output_dir/nvcr-engine-assets.txt"
fi

if [[ -z "$repo" ]]; then
    if git -C "$repo_root" config --get remote.origin.url >/dev/null; then
        origin_url="$(git -C "$repo_root" config --get remote.origin.url)"
        repo="$(printf '%s\n' "$origin_url" | sed -E 's#^git@github.com:##; s#^https://github.com/##; s#[.]git$##')"
    fi
fi
if ((skip_dispatch == 0)) && [[ -z "$repo" ]]; then
    echo "could not infer GitHub repo; pass --repo OWNER/REPO" >&2
    exit 2
fi

if ((dry_run == 0)) && ! command -v aws >/dev/null; then
    echo "aws CLI is required to stage engine assets" >&2
    exit 1
fi
if ((skip_dispatch == 0 && dry_run == 0)) && ! command -v gh >/dev/null; then
    echo "gh CLI is required unless --skip-dispatch is used" >&2
    exit 1
fi

for engine_dir in "${engine_dirs[@]}"; do
    if [[ ! -d "$engine_dir" ]]; then
        echo "engine directory does not exist: $engine_dir" >&2
        exit 1
    fi
    "$script_dir/nvcr_artifacts.py" validate "$engine_dir" --json >/dev/null
done

if ((skip_dispatch == 0 && dry_run == 0)); then
    gh release view "$tag" --repo "$repo" >/dev/null
fi

if ((dry_run == 0)); then
    rm -f -- "$asset_manifest"
    append_flag=()
    for engine_dir in "${engine_dirs[@]}"; do
        "$script_dir/stage_engine_release_asset.sh" \
            --version "$version" \
            --engine-dir "$engine_dir" \
            --output-dir "$output_dir" \
            --s3-uri "${s3_uri%/}" \
            --aws-region "$aws_region" \
            --presign-expires "$presign_expires" \
            --asset-manifest "$asset_manifest" \
            "${append_flag[@]}"
        append_flag=(--append)
    done
fi

if ((skip_dispatch)); then
    echo "Staged engine assets in $asset_manifest"
    exit 0
fi

workflow_args=(
    workflow run upload-engine-assets.yml
    --repo "$repo"
    --ref "$workflow_ref"
    -f "tag=$tag"
    -F "engine_assets=@$asset_manifest"
)
if ((publish_release)); then
    workflow_args+=(
        -f "publish_release=true"
        -f "publish_confirmation=$publish_confirmation"
    )
fi

if ((dry_run)); then
    printf 'gh'
    printf ' %q' "${workflow_args[@]}"
    printf '\n'
else
    gh "${workflow_args[@]}"
    echo "Dispatched upload-engine-assets.yml for $tag"
fi
