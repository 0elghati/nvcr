#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcr_bin="${NVCR_BIN:-$repo_root/build-release/cli/nvcr}"
output_dir="${NVCR_BENCH_OUTPUT_DIR:-/tmp}"
frames="${NVCR_BENCH_FRAMES:-97}"
qp="${NVCR_BENCH_QP:-32}"
gops="${NVCR_BENCH_GOPS:-1 97}"
repetitions="${NVCR_BENCH_REPETITIONS:-3}"
warmup_frames="${NVCR_BENCH_WARMUP_FRAMES:-10}"
jsonl=""

input_720="${NVCR_BENCH_720P_INPUT:-/home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv}"
input_1080="${NVCR_BENCH_1080P_INPUT:-/home/oelghati/DCVC/datasets/misc/BQTerrace_1920x1080_60.yuv}"

usage() {
    cat <<'USAGE'
Usage: benchmark_resolution_pair.sh [options]

Runs the same encode benchmark matrix at 720p and 1080p so performance changes
cannot be accepted from a single-resolution win.
Use repeated runs and optional warm-up passes to support the warmed-protocol gate.

Options:
  --nvcr PATH          nvcr binary (default: build-release/cli/nvcr)
  --frames N           Frames per run (default: 97)
  --qp N               Base QP (default: 32)
  --gops "A B"         GOP sizes to run (default: "1 97")
  --repetitions N      Number of measured runs per case (default: 3)
  --warmup-frames N    Additional warm-up run frames (default: 10)
  --720p-input FILE    1280x720 YUV420P8 input
  --1080p-input FILE   1920x1080 YUV420P8 input
  --output-dir DIR     Encoded output directory (default: /tmp)
  --jsonl FILE         Also append machine-readable JSONL rows
  -h, --help           Show this help

Environment overrides use the NVCR_BENCH_* names shown in the script.
USAGE
}

while (($#)); do
    case "$1" in
        --nvcr) nvcr_bin="$2"; shift 2 ;;
        --frames) frames="$2"; shift 2 ;;
        --qp) qp="$2"; shift 2 ;;
        --gops) gops="$2"; shift 2 ;;
        --repetitions) repetitions="$2"; shift 2 ;;
        --warmup-frames) warmup_frames="$2"; shift 2 ;;
        --720p-input) input_720="$2"; shift 2 ;;
        --1080p-input) input_1080="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --jsonl) jsonl="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$frames" =~ ^[1-9][0-9]*$ ]]; then
    echo "--frames must be a positive integer" >&2
    exit 2
fi
if [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repetitions must be a positive integer" >&2
    exit 2
fi
if [[ ! "$warmup_frames" =~ ^[0-9]+$ ]]; then
    echo "--warmup-frames must be a non-negative integer" >&2
    exit 2
fi

if [[ ! -x "$nvcr_bin" ]]; then
    echo "missing executable nvcr binary: $nvcr_bin" >&2
    exit 2
fi
for input in "$input_720" "$input_1080"; do
    if [[ ! -f "$input" ]]; then
        echo "missing benchmark input: $input" >&2
        exit 2
    fi
done
mkdir -p "$output_dir"

commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
dirty="false"
if [[ -n "$(git -C "$repo_root" status --short 2>/dev/null || true)" ]]; then
    dirty="true"
fi

run_once() {
    local label="$1"
    local size="$2"
    local fps="$3"
    local profile="$4"
    local input="$5"
    local gop="$6"
    local run_index="$7"
    local output="$output_dir/nvcr_${label}_qp${qp}_gop${gop}_run${run_index}.nvcr"
    local log
    local summary
    local parsed
    local frames_done
    local payload
    local seconds
    local throughput

    echo "== $label run=$run_index qp=$qp gop=$gop profile=$profile ==" >&2
    log="$("$nvcr_bin" encode \
        -i "$input" -o "$output" -s "$size" -r "$fps" \
        --frames "$frames" --gop-size "$gop" --qp "$qp" \
        --engine-profile "$profile")"
    printf '%s\n' "$log" | tail -n 2 >&2

    summary="$(printf '%s\n' "$log" | grep '^Encoded ' | tail -n 1)"
    parsed="$(printf '%s\n' "$summary" | sed -n 's/^Encoded \([0-9][0-9]*\) frame(s), \([0-9][0-9]*\) payload bytes, codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2 \3 \4/p')"
    if [[ -z "$parsed" ]]; then
        echo "could not parse nvcr summary for $label run=$run_index gop=$gop" >&2
        return 1
    fi
    read -r frames_done payload seconds throughput <<< "$parsed"

    if [[ -n "$jsonl" ]]; then
        mkdir -p "$(dirname "$jsonl")"
        printf '{"schema":"nvcr.benchmark.pair.v1","nvcr_commit":"%s","nvcr_dirty":%s,"resolution":"%s","input":"%s","size":"%s","fps":%s,"frames":%s,"warmup_frames":%s,"run_index":%s,"runs_planned":%s,"qp":%s,"gop_size":%s,"engine_profile":"%s","payload_bytes":%s,"codec_time_seconds":%s,"throughput_fps":%s,"output":"%s","aggregate":false}\n' \
            "$commit" "$dirty" "$label" "$input" "$size" "$fps" "$frames_done" "$warmup_frames" "$run_index" "$repetitions" \
            "$qp" "$gop" "$profile" "$payload" "$seconds" "$throughput" "$output" >> "$jsonl"
    fi

    printf '%s %s %s\n' "$payload" "$seconds" "$throughput"
}

run_case() {
    local label="$1"
    local size="$2"
    local fps="$3"
    local profile="$4"
    local input="$5"
    local gop="$6"
    local run_index
    local payload
    local seconds
    local throughput
    local result
    local aggregate_payload=0
    local aggregate_seconds=0.0
    local aggregate_fps=0.0

    if ((warmup_frames > 0)); then
        "$nvcr_bin" encode \
            -i "$input" -o "$output_dir/nvcr_${label}_qp${qp}_gop${gop}_warmup.nvcr" -s "$size" -r "$fps" \
            --frames "$warmup_frames" --gop-size "$gop" --qp "$qp" \
            --engine-profile "$profile" >/dev/null
    fi

    for run_index in $(seq 1 "$repetitions"); do
        result="$(run_once "$label" "$size" "$fps" "$profile" "$input" "$gop" "$run_index")"
        read -r payload seconds throughput <<< "$result"
        aggregate_payload=$((aggregate_payload + payload))
        aggregate_seconds="$(awk "BEGIN {printf \"%.6f\", $aggregate_seconds + $seconds}")"
        aggregate_fps="$(awk "BEGIN {printf \"%.6f\", $aggregate_fps + $throughput}")"
    done

    local avg_payload avg_seconds avg_fps
    avg_payload="$(awk "BEGIN {printf \"%.0f\", $aggregate_payload / $repetitions}")"
    avg_seconds="$(awk "BEGIN {printf \"%.6f\", $aggregate_seconds / $repetitions}")"
    avg_fps="$(awk "BEGIN {printf \"%.6f\", $aggregate_fps / $repetitions}")"
    if [[ -n "$jsonl" ]]; then
        printf '{"schema":"nvcr.benchmark.pair.v1","nvcr_commit":"%s","nvcr_dirty":%s,"resolution":"%s","input":"%s","size":"%s","fps":%s,"frames":%s,"warmup_frames":%s,"run_index":"average","runs_planned":%s,"qp":%s,"gop_size":%s,"engine_profile":"%s","payload_bytes":%s,"codec_time_seconds":%s,"throughput_fps":%s,"output":null,"aggregate":true}\n' \
            "$commit" "$dirty" "$label" "$input" "$size" "$fps" "$frames" "$warmup_frames" "$repetitions" \
            "$qp" "$gop" "$profile" "$avg_payload" "$avg_seconds" "$avg_fps" >> "$jsonl"
    fi

    printf 'case=%s gop=%s runs=%s avg_payload_bytes=%s avg_codec_time_s=%s avg_fps=%s\n' \
        "$label" "$gop" "$repetitions" "$avg_payload" "$avg_seconds" "$avg_fps"
}

for gop in $gops; do
    run_case "720p" "1280x720" "30" "720p-fp16" "$input_720" "$gop"
    run_case "1080p" "1920x1080" "60" "1080p-fp16" "$input_1080" "$gop"
done
