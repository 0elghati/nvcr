#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
data_root=/home/oelghati/datasets
engine_root="$repo_root/build/engines/dcvcrt-cvpr2025/orin-nano-l4t3647"
evidence_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scratch_dir=/tmp/nvcr-orin-container-gop-sequence-2026-08-05
results="$evidence_dir/results.jsonl"

commit="$(git -C "$repo_root" rev-parse HEAD)"
short_commit="$(git -C "$repo_root" rev-parse --short=12 HEAD)"
image="nvcr:orin-benchmark-$short_commit"

if [[ "$(git -C "$repo_root" branch --show-current)" != main ]]; then
    echo "benchmark must run from main" >&2
    exit 2
fi
if [[ -e "$results" ]]; then
    echo "refusing to append to existing evidence: $results" >&2
    exit 2
fi
image_revision="$(docker image inspect "$image" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
if [[ "$image_revision" != "$commit" ]]; then
    echo "image revision $image_revision does not match source $commit" >&2
    exit 2
fi

mkdir -p "$scratch_dir" "$evidence_dir"
exec > >(tee "$evidence_dir/run.log") 2>&1

source_dirty=false
if [[ -n "$(git -C "$repo_root" status --short)" ]]; then
    source_dirty=true
fi

echo "STARTED $(date --iso-8601=seconds)"
echo "SOURCE commit=$commit dirty=$source_dirty"
echo "PROTOCOL one initial I-frame followed by inter frames; GOP equals sequence length"
echo "REUSE revision-matched image=$image"

git -C "$repo_root" status --porcelain=v2 >"$evidence_dir/source-status.txt"
git -C "$repo_root" diff --binary >"$evidence_dir/source.patch"
git -C "$repo_root" diff --check >"$evidence_dir/source-diff-check.txt"
docker image inspect "$image" >"$evidence_dir/image-inspect.json"
{
    printf 'reused_image=%s\n' "$image"
    docker image inspect "$image" --format 'id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}} created={{index .Config.Labels "org.opencontainers.image.created"}}'
} >"$evidence_dir/validation-reference.txt"

sha256sum \
    "$data_root/qcif/akiyo_qcif.yuv" \
    "$data_root/cif/paris_cif.yuv" \
    "$data_root/360p/BasketballDrive_640x360_50.yuv" \
    "$data_root/540p/BasketballDrive_960x540_50.yuv" \
    "$data_root/720p/FourPeople_1280x720_60.yuv" \
    "$data_root/hd/BasketballDrive_1920x1080_50.yuv" \
    >"$evidence_dir/input-sha256.txt"
sha256sum \
    "$engine_root/qcif-fp16/engine_manifest.json" \
    "$engine_root/cif-fp16/engine_manifest.json" \
    "$engine_root/360p-fixed-canonical-fp16/engine_manifest.json" \
    "$engine_root/540p-fixed-canonical-fp16/engine_manifest.json" \
    "$engine_root/720p-fp16/engine_manifest.json" \
    "$engine_root/1080p-fp16/engine_manifest.json" \
    >"$evidence_dir/engine-manifest-sha256.txt"

record_state() {
    local suffix="$1"
    {
        date --iso-8601=seconds
        uname -a
        cat /etc/nv_tegra_release
        nvpmodel -q
        /usr/local/cuda/bin/nvcc --version
        dpkg-query -W -f='${Version}\n' libnvinfer10
        for node in \
            /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq \
            /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq \
            /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq \
            /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq \
            /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq \
            /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq; do
            printf '%s=' "$node"
            sed -n '1p' "$node"
        done
    } >"$evidence_dir/platform-$suffix.txt" 2>&1
    timeout 2 tegrastats --interval 1000 \
        >"$evidence_dir/tegrastats-$suffix.txt" 2>&1 || true
}

run_case() {
    local label="$1"
    local frames="$2"
    local input_rel="$3"
    local engine_profile="$4"
    local input_option="--${label}-input"
    local engine_option="--${label}-engine-dir"

    echo "START resolution=$label frames=$frames gop=$frames qp=32"
    docker run --rm \
        --name "nvcr-gop-sequence-$label-$short_commit" \
        --runtime=nvidia \
        --network=host \
        --user "$(id -u):$(id -g)" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e "NVCR_BENCH_COMMIT=$commit" \
        -e "NVCR_BENCH_DIRTY=$source_dirty" \
        -v "$repo_root:/workspace/nvcr:ro" \
        -v "$data_root:/input:ro" \
        -v "$engine_root:/opt/nvcr/engines:ro" \
        -v "$scratch_dir:/scratch" \
        -v "$evidence_dir:/evidence" \
        --entrypoint /workspace/nvcr/scripts/benchmark_resolution_matrix.sh \
        "$image" \
        --nvcr /opt/nvcr/bin/nvcr \
        --frames "$frames" \
        --qp 32 \
        --gops "$frames" \
        --resolutions "$label" \
        --repetitions 3 \
        --warmup-frames 10 \
        "$input_option" "/input/$input_rel" \
        "$engine_option" "/opt/nvcr/engines/$engine_profile" \
        --output-dir /scratch \
        --jsonl /evidence/results.jsonl
}

record_state before

echo "Running lightweight QCIF container I/P smoke"
docker run --rm \
    --runtime=nvidia \
    --network=host \
    --user "$(id -u):$(id -g)" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$data_root:/input:ro" \
    -v "$engine_root:/opt/nvcr/engines:ro" \
    -v "$scratch_dir:/scratch" \
    "$image" encode \
    -i /input/qcif/akiyo_qcif.yuv \
    -o /scratch/validation.nvcr \
    -s 176x144 -r 30 --frames 2 --gop-size 2 --qp 32 \
    --engine-dir /opt/nvcr/engines/qcif-fp16 \
    >"$evidence_dir/validation-smoke.log" 2>&1
docker run --rm \
    --runtime=nvidia \
    --network=host \
    --user "$(id -u):$(id -g)" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$data_root:/input:ro" \
    -v "$engine_root:/opt/nvcr/engines:ro" \
    -v "$scratch_dir:/scratch" \
    "$image" decode \
    -i /scratch/validation.nvcr \
    -o /dev/null --frames 2 \
    --quality-metrics /input/qcif/akiyo_qcif.yuv \
    --engine-dir /opt/nvcr/engines/qcif-fp16 \
    >>"$evidence_dir/validation-smoke.log" 2>&1

status=0
run_case qcif 300 qcif/akiyo_qcif.yuv qcif-fp16 || status=1
run_case cif 266 cif/paris_cif.yuv cif-fp16 || status=1
run_case 360p 500 360p/BasketballDrive_640x360_50.yuv 360p-fixed-canonical-fp16 || status=1
run_case 540p 500 540p/BasketballDrive_960x540_50.yuv 540p-fixed-canonical-fp16 || status=1
run_case 720p 601 720p/FourPeople_1280x720_60.yuv 720p-fp16 || status=1
run_case 1080p 500 hd/BasketballDrive_1920x1080_50.yuv 1080p-fp16 || status=1
record_state after

if [[ -f "$results" ]]; then
    python3 - "$results" "$evidence_dir/summary.md" "$commit" "$source_dirty" "$status" <<'PY'
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
commit = sys.argv[3]
dirty = sys.argv[4]
status = int(sys.argv[5])
rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
averages = {
    (row["resolution"], row["operation"]): row
    for row in rows
    if row.get("run_index") == "average"
}

lines = [
    "# Orin Docker sequence-length GOP benchmark",
    "",
    f"Commit: `{commit}` (dirty: `{dirty}`)",
    "",
    f"Runner status: `{'complete' if status == 0 else 'partial'}`",
    "",
    "Each sequence has one initial I-frame followed by inter frames; GOP equals frame count.",
    "",
    "| Resolution | Frames/GOP | Payload bytes | Encode fps | Decode fps | PSNR-YUV |",
    "|---|---:|---:|---:|---:|---:|",
]
for resolution in ("qcif", "cif", "360p", "540p", "720p", "1080p"):
    encode = averages.get((resolution, "encode"))
    decode = averages.get((resolution, "decode"))
    if not encode or not decode:
        lines.append(f"| {resolution} | — | — | failed/incomplete | failed/incomplete | — |")
        continue
    lines.append(
        f"| {resolution} | {encode['frames']} | {encode['payload_bytes']} | "
        f"{encode['throughput_fps']:.6f} | {decode['throughput_fps']:.6f} | "
        f"{decode['psnr_yuv']:.6f} dB |"
    )
summary_path.write_text("\n".join(lines) + "\n")
PY
fi

echo "COMPLETED $(date --iso-8601=seconds) status=$status"
echo "Evidence: $evidence_dir"
exit "$status"
