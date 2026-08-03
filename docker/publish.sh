#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: docker/publish.sh [--load|--push] <x86_64|jetson>

Builds one architecture-specific NVCR runtime image. --load imports it into the
local Docker daemon (default). --push publishes both immutable-version and
runtime-family tags to Docker Hub.

Environment:
  NVCR_DOCKERHUB_REPOSITORY  Docker Hub repository (default: 0elghati/nvcr)
  NVCR_IMAGE_VERSION         Image version (default: version.txt)
  NVCR_IMAGE_SOURCE          OCI source URL (default: GitHub repository)
EOF
}

action=load
case "${1:-}" in
--load)
    shift
    ;;
--push)
    action=push
    shift
    ;;
-h|--help)
    usage
    exit 0
    ;;
esac

architecture="${1:-}"
if [[ "$architecture" != x86_64 && "$architecture" != jetson ]]; then
    usage >&2
    exit 2
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
repository="${NVCR_DOCKERHUB_REPOSITORY:-0elghati/nvcr}"
version="${NVCR_IMAGE_VERSION:-$(tr -d '[:space:]' <"$repository_root/version.txt")}"
source_url="${NVCR_IMAGE_SOURCE:-https://github.com/0elghati/nvcr}"
revision="$(git -C "$repository_root" rev-parse HEAD)"
created="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! "$repository" =~ ^[a-z0-9]+([._-][a-z0-9]+)*/[a-z0-9]+([._-][a-z0-9]+)*$ ]]; then
    echo "publish: invalid Docker Hub repository: $repository" >&2
    exit 2
fi
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([._-][A-Za-z0-9.-]+)?$ ]]; then
    echo "publish: invalid image version: $version" >&2
    exit 2
fi

if [[ "$action" == push ]]; then
    if [[ -n "$(git -C "$repository_root" status --porcelain)" ]]; then
        echo "publish: refusing to push from a dirty worktree" >&2
        exit 2
    fi
    expected_tag="v$version"
    actual_tag="$(git -C "$repository_root" describe --tags --exact-match HEAD 2>/dev/null || true)"
    if [[ "$actual_tag" != "$expected_tag" ]]; then
        echo "publish: HEAD must be the exact $expected_tag release tag (found: ${actual_tag:-none})" >&2
        exit 2
    fi
fi

case "$architecture" in
x86_64)
    platform=linux/amd64
    dockerfile=docker/Dockerfile.x86_64
    immutable_tag="$repository:$version-x86_64-cuda12.6-trt10.7"
    family_tag="$repository:x86_64-cuda12.6-trt10.7"
    ;;
jetson)
    if [[ "$(uname -m)" != aarch64 ]]; then
        echo "publish: Jetson images must be built on the native aarch64 Jetson target" >&2
        exit 2
    fi
    platform=linux/arm64
    dockerfile=docker/Dockerfile.jetson
    immutable_tag="$repository:$version-jetson-l4t36.4"
    family_tag="$repository:jetson-l4t36.4"
    ;;
esac

output_flag=--load
attestation_args=()
if [[ "$action" == push ]]; then
    output_flag=--push
    attestation_args=(--provenance=mode=max --sbom=true)
fi

echo "Building $immutable_tag"
docker buildx build \
    --platform "$platform" \
    --target runtime \
    --file "$dockerfile" \
    --build-arg "NVCR_VERSION=$version" \
    --build-arg "NVCR_REVISION=$revision" \
    --build-arg "NVCR_SOURCE=$source_url" \
    --build-arg "NVCR_CREATED=$created" \
    --tag "$immutable_tag" \
    --tag "$family_tag" \
    "${attestation_args[@]}" \
    "$output_flag" \
    "$repository_root"

if [[ "$action" == load ]]; then
    docker image inspect "$immutable_tag" \
        --format 'Loaded {{.RepoTags}} ({{.Architecture}}); version={{index .Config.Labels "org.opencontainers.image.version"}}'
else
    echo "Published $immutable_tag"
    echo "Published $family_tag"
fi
