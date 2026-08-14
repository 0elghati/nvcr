#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: docker/publish.sh [--load|--push] <amd64-cuda12.8-trt10.9|jetson>

Builds one architecture-specific NVCR runtime image. --load imports it into the
local Docker daemon (default). --push publishes the immutable versioned tag as
the primary release tag, plus runtime-family and architecture-qualified latest
aliases to Docker Hub.

Environment:
  NVCR_DOCKERHUB_REPOSITORY  Docker Hub repository (default: omarelghati/nvcr)
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

image_family="${1:-}"
if [[ "$image_family" != amd64-cuda12.8-trt10.9 \
    && "$image_family" != jetson ]]; then
    usage >&2
    exit 2
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
repository="${NVCR_DOCKERHUB_REPOSITORY:-omarelghati/nvcr}"
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

build_args=()
case "$image_family" in
amd64-cuda12.8-trt10.9)
    platform=linux/amd64
    dockerfile=docker/Dockerfile.x86_64
    tag_suffix=amd64-cuda12.8-trt10.9
    build_args=(
        "CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04"
        "CUDA_RUNTIME_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04"
        "TENSORRT_PACKAGE_VERSION=10.9.0.34-1+cuda12.8"
        "NVCR_CUDA_ARCHITECTURES=all"
        "NVCR_TARGET_PROFILE=desktop-amd64-cuda12.8-trt10.9"
        "NVCR_CUDA_VERSION=12.8"
        "NVCR_TENSORRT_VERSION=10.9"
        "NVCR_IMAGE_DESCRIPTION=NVCR runtime for linux/amd64 NVIDIA desktop GPUs using CUDA 12.8 and TensorRT 10.9 with CUDA helper kernels for the compiler-supported GPU architecture set; TensorRT engines are installed separately from the rolling catalog"
    )
    ;;
jetson)
    if [[ "$(uname -m)" != aarch64 ]]; then
        echo "publish: Jetson images must be built on the native aarch64 Jetson target" >&2
        exit 2
    fi
    platform=linux/arm64
    dockerfile=docker/Dockerfile.jetson
    tag_suffix=jetson-l4t36.4
    ;;
esac

immutable_tag="$repository:$version-$tag_suffix"
family_tag="$repository:$tag_suffix"
latest_tag="$repository:latest-$tag_suffix"
docker_tags=("$immutable_tag" "$family_tag" "$latest_tag")

tag_args=()
for tag in "${docker_tags[@]}"; do
    tag_args+=(--tag "$tag")
done

build_arg_args=()
for arg in "${build_args[@]}"; do
    build_arg_args+=(--build-arg "$arg")
done

output_flag=--load
attestation_args=()
if [[ "$action" == push ]]; then
    output_flag=--push
    attestation_args=(--provenance=mode=max --sbom=true)
fi

echo "Building immutable release tag $immutable_tag"
echo "Adding moving aliases $family_tag and $latest_tag"
docker buildx build \
    --platform "$platform" \
    --target runtime \
    --file "$dockerfile" \
    "${build_arg_args[@]}" \
    --build-arg "NVCR_VERSION=$version" \
    --build-arg "NVCR_REVISION=$revision" \
    --build-arg "NVCR_SOURCE=$source_url" \
    --build-arg "NVCR_CREATED=$created" \
    "${tag_args[@]}" \
    "${attestation_args[@]}" \
    "$output_flag" \
    "$repository_root"

if [[ "$action" == load ]]; then
    docker image inspect "${docker_tags[0]}" \
        --format 'Loaded {{.RepoTags}} ({{.Architecture}}); version={{index .Config.Labels "org.opencontainers.image.version"}}'
else
    for tag in "${docker_tags[@]}"; do
        echo "Published $tag"
    done
fi
