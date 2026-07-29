#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcr_bin="${NVCR_BIN:-$repo_root/build-release/cli/nvcr}"
output_dir="${NVCR_BENCH_OUTPUT_DIR:-/tmp}"
frames="${NVCR_BENCH_FRAMES:-97}"
qp="${NVCR_BENCH_QP:-32}"
gops="${NVCR_BENCH_GOPS:-1 97}"
backends="${NVCR_BENCH_BACKENDS:-dcvcrt}"
jsonl=""

input_720="${NVCR_BENCH_720P_INPUT:-/home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv}"
input_1080="${NVCR_BENCH_1080P_INPUT:-/home/oelghati/DCVC/datasets/misc/BQTerrace_1920x1080_60.yuv}"

usage() {
    cat <<'EOF'
Usage: benchmark_resolution_pair.sh [options]

Runs the same encode benchmark matrix at 720p and 1080p so performance changes
cannot be accepted from a single-resolution win.

Options:
  --nvcr PATH          nvcr binary (default: build-release/cli/nvcr)
  --frames N           Frames per run (default: 97)
  --qp N               Base QP (default: 32)
  --gops "A B"         GOP sizes to run (default: "1 97")
  --backends "A B"     Backends to compare (default: "dcvcrt")
  --720p-input FILE    1280x720 YUV420P8 input
  --1080p-input FILE   1920x1080 YUV420P8 input
  --output-dir DIR     Encoded output directory (default: /tmp)
  --jsonl FILE         Also append machine-readable JSONL rows
  -h, --help           Show this help

Environment overrides use the NVCR_BENCH_* names shown in the script.
EOF
}

while (($#)); do
    case "$1" in
        --nvcr) nvcr_bin="$2"; shift 2 ;;
        --frames) frames="$2"; shift 2 ;;
        --qp) qp="$2"; shift 2 ;;
        --gops) gops="$2"; shift 2 ;;
        --backends) backends="$2"; shift 2 ;;
        --720p-input) input_720="$2"; shift 2 ;;
        --1080p-input) input_1080="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --jsonl) jsonl="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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

run_case() {
    local label="$1"
    local size="$2"
    local fps="$3"
    local profile="$4"
    local input="$5"
    local gop="$6"
    local backend="$7"
    local safe_backend="${backend//[^A-Za-z0-9_.-]/_}"
    local output="$output_dir/nvcr_${safe_backend}_${label}_qp${qp}_gop${gop}.nvcr"
    local log
    local summary
    local payload
    local seconds
    local throughput

    echo "== backend=$backend $label qp=$qp gop=$gop profile=$profile =="
    log="$("$nvcr_bin" encode \
        -i "$input" -o "$output" -s "$size" -r "$fps" \
        --frames "$frames" --gop-size "$gop" --qp "$qp" \
        --backend "$backend" --engine-profile "$profile")"
    printf '%s\n' "$log" | tail -n 2

    summary="$(printf '%s\n' "$log" | grep '^Encoded ' | tail -n 1)"
    parsed="$(printf '%s\n' "$summary" | sed -n 's/^Encoded \([0-9][0-9]*\) frame(s), \([0-9][0-9]*\) payload bytes, codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2 \3 \4/p')"
    if [[ -z "$parsed" ]]; then
        echo "could not parse nvcr summary for $label gop=$gop" >&2
        return 1
    fi
    read -r frames_done payload seconds throughput <<< "$parsed"

    if [[ -n "$jsonl" ]]; then
        mkdir -p "$(dirname "$jsonl")"
        printf '{"schema":"nvcr.benchmark.pair.v1","nvcr_commit":"%s","nvcr_dirty":%s,"backend":"%s","resolution":"%s","input":"%s","size":"%s","fps":%s,"frames":%s,"qp":%s,"gop_size":%s,"engine_profile":"%s","payload_bytes":%s,"codec_time_seconds":%s,"throughput_fps":%s,"output":"%s"}\n' \
            "$commit" "$dirty" "$backend" "$label" "$input" "$size" "$fps" "$frames_done" "$qp" "$gop" \
            "$profile" "$payload" "$seconds" "$throughput" "$output" >> "$jsonl"
    fi
}

for backend in $backends; do
    for gop in $gops; do
        run_case "720p" "1280x720" "30" "720p-fp16" "$input_720" "$gop" "$backend"
        run_case "1080p" "1920x1080" "60" "1080p-fp16" "$input_1080" "$gop" "$backend"
    done
done
