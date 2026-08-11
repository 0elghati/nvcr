#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
nvcr_version="$(tr -d '[:space:]' < "$repo_root/version.txt")"
image="${NVCR_DOCKER_IMAGE:-omarelghati/nvcr:${nvcr_version}-amd64-cuda12.8-trt10.9}"
engine_volume="${NVCR_DOCKER_ENGINE_VOLUME:-nvcr-engines}"
input_dir="${NVCR_DOCKER_INPUT_DIR:-$repo_root/datasets}"
results_dir="${NVCR_DOCKER_RESULTS_DIR:-$repo_root/evidence/performance/docker-$timestamp}"
output_dir="${NVCR_DOCKER_OUTPUT_DIR:-$repo_root/evidence/performance/docker-$timestamp/streams}"
host_repo_dir="${NVCR_DOCKER_HOST_REPO_DIR:-}"
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
    --host-repo-dir DIR        Docker-daemon-visible repository path (auto-detected)
  --hardware LABEL           Stable hardware label in result rows
  --user UID:GID             Container user (default: 0:0 for NVIDIA access)
  --install-profiles LIST    Install profiles into the engine volume before running
                            (use 'all' or a space-separated list)
  --skip-pull                Use the local image without pulling it
  -h, --help                 Show this help

All options after '--' are passed to scripts/benchmark_resolution_matrix.sh,
for example:

  --frames 300 --qp 32 --gops "1 8" --resolutions "qcif 720p" --repetitions 3

The launcher always writes matrix results to the mounted results directory and
temporary streams to the mounted output directory. Use the launcher options
above for host paths; matrix output paths after '--' are overridden.

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
        --host-repo-dir) host_repo_dir="$2"; shift 2 ;;
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

if [[ -z "$host_repo_dir" ]]; then
    host_repo_dir="$(docker inspect "$(hostname)" \
        --format '{{range .Mounts}}{{if eq .Destination "/workspace/nvcr"}}{{.Source}}{{end}}{{end}}' \
        2>/dev/null || true)"
fi
[[ -n "$host_repo_dir" ]] || host_repo_dir="$repo_root"

docker_host_path() {
    local path="$1"
    if [[ "$path" == "$repo_root" ]]; then
        printf '%s' "$host_repo_dir"
    elif [[ "$path" == "$repo_root/"* ]]; then
        printf '%s/%s' "$host_repo_dir" "${path#"$repo_root/"}"
    else
        printf '%s' "$path"
    fi
}

docker_input_dir="$(docker_host_path "$input_dir")"
docker_results_dir="$(docker_host_path "$results_dir")"
docker_output_dir="$(docker_host_path "$output_dir")"
docker_engine_volume="$(docker_host_path "$engine_volume")"
if [[ "$host_repo_dir" != "$repo_root" ]]; then
    echo "Docker daemon repository path: $host_repo_dir" >&2
fi
if [[ "$docker_output_dir" == "$output_dir" && "$host_repo_dir" != "$repo_root" ]]; then
    echo "benchmark_docker: output directory is not visible to the Docker daemon: $output_dir" >&2
    echo "Set --output-dir under $repo_root or provide --host-repo-dir for the daemon host." >&2
    exit 2
fi

input_mount=(--mount "type=bind,source=$docker_input_dir,target=/input,readonly")
results_mount=(--mount "type=bind,source=$docker_results_dir,target=/results")
output_mount=(--mount "type=bind,source=$docker_output_dir,target=/output")
repo_mount=(--mount "type=bind,source=$host_repo_dir,target=/workspace/nvcr,readonly")
if [[ "$docker_engine_volume" == "$engine_volume" ]]; then
    engine_install_mount=(-v "$engine_volume:/opt/nvcr/engines")
    engine_runtime_mount=(-v "$engine_volume:/opt/nvcr/engines:ro")
else
    engine_install_mount=(--mount "type=bind,source=$docker_engine_volume,target=/opt/nvcr/engines")
    engine_runtime_mount=(--mount "type=bind,source=$docker_engine_volume,target=/opt/nvcr/engines,readonly")
fi
engine_container_root=/opt/nvcr/engines
if [[ -d "$engine_volume" && ! -d "$engine_volume/dcvcrt-qcif" ]]; then
    nested_engine_dirs=("$engine_volume"/*)
    if (( ${#nested_engine_dirs[@]} == 1 )) && [[ -d "${nested_engine_dirs[0]}/dcvcrt-qcif" ]]; then
        engine_container_root="/opt/nvcr/engines/$(basename "${nested_engine_dirs[0]}")"
    fi
fi

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
        "${engine_install_mount[@]}" \
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
    -e "NVCR_BENCH_QCIF_ENGINE_DIR=$engine_container_root/dcvcrt-qcif"
    -e "NVCR_BENCH_CIF_ENGINE_DIR=$engine_container_root/dcvcrt-cif"
    -e "NVCR_BENCH_360P_ENGINE_DIR=$engine_container_root/dcvcrt-360p"
    -e "NVCR_BENCH_540P_ENGINE_DIR=$engine_container_root/dcvcrt-540p"
    -e "NVCR_BENCH_720P_ENGINE_DIR=$engine_container_root/dcvcrt-720p"
    -e "NVCR_BENCH_1080P_ENGINE_DIR=$engine_container_root/dcvcrt-1080p"
)

echo "Running Docker benchmark: image=$image digest=$image_digest" >&2
echo "Results: $results_dir" >&2
echo "Streams: $output_dir" >&2
exec docker run --rm "${gpu_args[@]}" \
    --user "$container_user" \
    --entrypoint /bin/bash \
    "${container_env[@]}" \
    "${engine_runtime_mount[@]}" \
    "${input_mount[@]}" \
    "${results_mount[@]}" \
    "${output_mount[@]}" \
    "${repo_mount[@]}" \
    -w /workspace/nvcr \
    "$image" scripts/benchmark_resolution_matrix.sh "${matrix_args[@]}" \
    --output-dir /output \
    --results-dir /results \
    --jsonl /results/results.jsonl \
    --csv /results/results.csv \
    --report /results/summary.md
