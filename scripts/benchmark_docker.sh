#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
nvcr_version="$(tr -d '[:space:]' < "$repo_root/version.txt")"
image="${NVCR_DOCKER_IMAGE:-omarelghati/nvcr:${nvcr_version}-amd64-cuda12.8-trt10.9}"
engine_volume="${NVCR_DOCKER_ENGINE_VOLUME:-nvcr-engines}"
input_dir="${NVCR_DOCKER_INPUT_DIR:-$repo_root/datasets}"
results_dir="${NVCR_DOCKER_RESULTS_DIR:-$repo_root/evidence/performance/docker-$timestamp}"
output_dir="${NVCR_DOCKER_OUTPUT_DIR:-/tmp/nvcr-docker-benchmark-$timestamp}"
hardware="${NVCR_DOCKER_HARDWARE:-}"
container_user="${NVCR_DOCKER_USER:-0:0}"
install_profiles=""
pull_image=true
matrix_args=()

usage() {
    cat <<'USAGE'
Usage: benchmark_docker.sh [launcher-options] -- [matrix-options]

Pulls an NVCR runtime image and runs the diagnostic resolution matrix with
docker run. The benchmark uses the same native matrix script and records both
codec-reported throughput and full process wall-time throughput.

Launcher options:
  --image IMAGE              Image to pull and run
  --engine-volume NAME|DIR   Named Docker volume or host engine directory
  --input-dir DIR            Host directory containing benchmark YUV inputs
  --results-dir DIR          Host directory for JSONL, CSV, and Markdown output
  --output-dir DIR           Host directory for temporary encoded streams
  --hardware LABEL           Stable hardware label in result rows
  --user UID:GID             Container user (default: 0:0 for NVIDIA access)
  --install-profiles LIST    Install profiles into the engine volume before running
                            (use 'all' or a space-separated list)
  --skip-pull                Use the local image without pulling it
  -h, --help                 Show this help

All options after '--' are passed to scripts/benchmark_resolution_matrix.sh,
for example:

  --frames 300 --qp 32 --gops "1 8" --resolutions "qcif 720p" --repetitions 3

The engine volume must already contain target-local dcvcrt-<profile> bundles,
unless --install-profiles is supplied. Pull an immutable image tag for a
reproducible run; the resolved image ID is recorded in the result metadata.
USAGE
}

while (($#)); do
    case "$1" in
        --image) image="$2"; shift 2 ;;
        --engine-volume) engine_volume="$2"; shift 2 ;;
        --input-dir) input_dir="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --hardware) hardware="$2"; shift 2 ;;
        --user) container_user="$2"; shift 2 ;;
        --install-profiles) install_profiles="$2"; shift 2 ;;
        --skip-pull) pull_image=false; shift ;;
        --) shift; matrix_args+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) matrix_args+=("$1"); shift ;;
    esac
done

command -v docker >/dev/null 2>&1 || {
    echo "benchmark_docker: docker is required" >&2
    exit 2
}
[[ -d "$input_dir" ]] || {
    echo "benchmark_docker: missing input directory: $input_dir" >&2
    exit 2
}
mkdir -p "$results_dir" "$output_dir"
input_dir="$(cd "$input_dir" && pwd)"
results_dir="$(cd "$results_dir" && pwd)"
output_dir="$(cd "$output_dir" && pwd)"

commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
[[ -n "$commit" ]] || commit=unknown
dirty=false
if [[ -n "$(git -C "$repo_root" status --short 2>/dev/null || true)" ]]; then
    dirty=true
fi

if [[ "$pull_image" == true ]]; then
    echo "Pulling Docker image: $image" >&2
    docker pull "$image"
fi

image_digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null || true)"
if [[ -z "$image_digest" ]]; then
    image_digest="$(docker image inspect --format '{{.Id}}' "$image")"
fi
[[ -n "$image_digest" ]] || {
    echo "benchmark_docker: cannot resolve image identity: $image" >&2
    exit 2
}

gpu_args=(--gpus all)
for device in /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    if [[ -e "$device" ]]; then
        gpu_args+=(--device "$device")
    fi
done

if [[ -n "$install_profiles" ]]; then
    install_command=(docker run --rm "${gpu_args[@]}" \
        -v "$engine_volume:/opt/nvcr/engines" \
        --entrypoint /opt/nvcr/bin/nvcr-artifacts)
    if [[ -n "${GH_TOKEN:-}" ]]; then
        install_command+=(-e "GH_TOKEN=$GH_TOKEN")
    fi
    install_command+=("$image" install --engine-root /opt/nvcr/engines)
    if [[ "$install_profiles" == all ]]; then
        install_command+=(--all)
    else
        for profile in $install_profiles; do
            install_command+=(--profile "$profile")
        done
    fi
    echo "Installing Docker engine profiles: $install_profiles" >&2
    "${install_command[@]}"
fi

container_env=(
    -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVCR_BIN=/opt/nvcr/bin/nvcr
    -e NVCR_BENCH_DATA_ROOT=/input
    -e NVCR_BENCH_ENGINE_ROOT=/opt/nvcr/engines
    -e NVCR_BENCH_OUTPUT_DIR=/output
    -e NVCR_BENCH_RESULTS_DIR=/results
    -e NVCR_BENCH_EXECUTION_MODE=docker
    -e "NVCR_BENCH_CONTAINER_IMAGE=$image"
    -e "NVCR_BENCH_CONTAINER_DIGEST=$image_digest"
    -e "NVCR_BENCH_COMMIT=$commit"
    -e "NVCR_BENCH_DIRTY=$dirty"
    -e "NVCR_BENCH_HARDWARE=$hardware"
    -e NVCR_BENCH_QCIF_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-qcif
    -e NVCR_BENCH_CIF_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-cif
    -e NVCR_BENCH_360P_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-360p
    -e NVCR_BENCH_540P_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-540p
    -e NVCR_BENCH_720P_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-720p
    -e NVCR_BENCH_1080P_ENGINE_DIR=/opt/nvcr/engines/dcvcrt-1080p
)

echo "Running Docker benchmark: image=$image digest=$image_digest" >&2
echo "Results: $results_dir" >&2
echo "Streams: $output_dir" >&2
exec docker run --rm "${gpu_args[@]}" \
    --user "$container_user" \
    --entrypoint /bin/bash \
    "${container_env[@]}" \
    -v "$engine_volume:/opt/nvcr/engines:ro" \
    -v "$input_dir:/input:ro" \
    -v "$results_dir:/results" \
    -v "$output_dir:/output" \
    -v "$repo_root:/workspace/nvcr:ro" \
    -w /workspace/nvcr \
    "$image" scripts/benchmark_resolution_matrix.sh "${matrix_args[@]}"
