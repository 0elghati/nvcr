#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcr_bin="${NVCR_BIN:-$repo_root/build-release/cli/nvcr}"
output_dir="${NVCR_BENCH_OUTPUT_DIR:-/tmp}"
frames="${NVCR_BENCH_FRAMES:-97}"
qp="${NVCR_BENCH_QP:-32}"
gops="${NVCR_BENCH_GOPS:-1 97}"
resolutions="${NVCR_BENCH_RESOLUTIONS:-qcif cif 720p 1080p}"
repetitions="${NVCR_BENCH_REPETITIONS:-3}"
warmup_frames="${NVCR_BENCH_WARMUP_FRAMES:-10}"
jsonl=""

input_qcif="${NVCR_BENCH_QCIF_INPUT:-/home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv}"
input_cif="${NVCR_BENCH_CIF_INPUT:-/home/oelghati/DCVC/datasets/cif/paris_cif.yuv}"
input_720="${NVCR_BENCH_720P_INPUT:-/home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv}"
input_1080="${NVCR_BENCH_1080P_INPUT:-/home/oelghati/DCVC/datasets/misc/BQTerrace_1920x1080_60.yuv}"
engine_qcif="${NVCR_BENCH_QCIF_ENGINE_DIR:-}"
engine_cif="${NVCR_BENCH_CIF_ENGINE_DIR:-}"
engine_720="${NVCR_BENCH_720P_ENGINE_DIR:-}"
engine_1080="${NVCR_BENCH_1080P_ENGINE_DIR:-}"
engine_args=()

usage() {
    cat <<'USAGE'
Usage: benchmark_resolution_matrix.sh [options]

Runs warmed encode and decode measurements at QCIF, CIF, 720p, and 1080p.
Decode measurements report weighted YUV PSNR against the source outside codec
timing. Every measured point is appended to JSONL when --jsonl is supplied.

Options:
  --nvcr PATH               nvcr binary (default: build-release/cli/nvcr)
  --frames N                Frames per run (default: 97)
  --qp N                    Base QP (default: 32)
  --gops "A B"              GOP sizes to run (default: "1 97")
  --resolutions "A B"       Resolution labels (default: "qcif cif 720p 1080p")
  --repetitions N           Measured runs per case (default: 3)
  --warmup-frames N         Encode/decode warm-up frames (default: 10)
  --qcif-input FILE         176x144 YUV420P8 input
  --cif-input FILE          352x288 YUV420P8 input
  --720p-input FILE         1280x720 YUV420P8 input
  --1080p-input FILE        1920x1080 YUV420P8 input
  --qcif-engine-dir DIR     Use a local QCIF bundle instead of --engine-profile
  --cif-engine-dir DIR      Use a local CIF bundle instead of --engine-profile
  --720p-engine-dir DIR     Use a local 720p bundle instead of --engine-profile
  --1080p-engine-dir DIR    Use a local 1080p bundle instead of --engine-profile
  --output-dir DIR          Encoded output directory (default: /tmp)
  --jsonl FILE              Append machine-readable per-run and aggregate rows
  -h, --help                Show this help
USAGE
}

while (($#)); do
    case "$1" in
        --nvcr) nvcr_bin="$2"; shift 2 ;;
        --frames) frames="$2"; shift 2 ;;
        --qp) qp="$2"; shift 2 ;;
        --gops) gops="$2"; shift 2 ;;
        --resolutions) resolutions="$2"; shift 2 ;;
        --repetitions) repetitions="$2"; shift 2 ;;
        --warmup-frames) warmup_frames="$2"; shift 2 ;;
        --qcif-input) input_qcif="$2"; shift 2 ;;
        --cif-input) input_cif="$2"; shift 2 ;;
        --720p-input) input_720="$2"; shift 2 ;;
        --1080p-input) input_1080="$2"; shift 2 ;;
        --qcif-engine-dir) engine_qcif="$2"; shift 2 ;;
        --cif-engine-dir) engine_cif="$2"; shift 2 ;;
        --720p-engine-dir) engine_720="$2"; shift 2 ;;
        --1080p-engine-dir) engine_1080="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --jsonl) jsonl="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$frames" "$repetitions"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "frame and repetition counts must be positive integers" >&2
        exit 2
    fi
done
if [[ ! "$warmup_frames" =~ ^[0-9]+$ ]]; then
    echo "--warmup-frames must be a non-negative integer" >&2
    exit 2
fi
if [[ ! -x "$nvcr_bin" ]]; then
    echo "missing executable nvcr binary: $nvcr_bin" >&2
    exit 2
fi
mkdir -p "$output_dir"
if [[ -n "$jsonl" ]]; then
    mkdir -p "$(dirname "$jsonl")"
fi

commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
dirty="false"
if [[ -n "$(git -C "$repo_root" status --short 2>/dev/null || true)" ]]; then
    dirty="true"
fi

select_engine() {
    local label="$1"
    local profile="$2"
    local directory=""
    case "$label" in
        qcif) directory="$engine_qcif" ;;
        cif) directory="$engine_cif" ;;
        720p) directory="$engine_720" ;;
        1080p) directory="$engine_1080" ;;
    esac
    if [[ -n "$directory" ]]; then
        engine_args=(--engine-dir "$directory")
    else
        engine_args=(--engine-profile "$profile")
    fi
}

append_row() {
    local operation="$1" label="$2" input="$3" size="$4" fps="$5" gop="$6"
    local profile="$7" run_index="$8" payload="$9" seconds="${10}" throughput="${11}"
    local psnr_yuv="${12}"
    [[ -z "$jsonl" ]] && return
    printf '{"schema":"nvcr.benchmark.resolution-matrix.v1","nvcr_commit":"%s","nvcr_dirty":%s,"operation":"%s","resolution":"%s","input":"%s","size":"%s","fps":%s,"frames":%s,"warmup_frames":%s,"run_index":"%s","runs_planned":%s,"qp":%s,"gop_size":%s,"engine_profile":"%s","payload_bytes":%s,"codec_time_seconds":%s,"throughput_fps":%s,"psnr_yuv":%s}\n' \
        "$commit" "$dirty" "$operation" "$label" "$input" "$size" "$fps" "$frames" \
        "$warmup_frames" "$run_index" "$repetitions" "$qp" "$gop" "$profile" \
        "$payload" "$seconds" "$throughput" "$psnr_yuv" >>"$jsonl"
}

run_once() {
    local label="$1" size="$2" fps="$3" profile="$4" input="$5" gop="$6" run_index="$7"
    local stream="$output_dir/nvcr_${label}_qp${qp}_gop${gop}_run${run_index}.nvcr"
    local encode_log decode_log parsed frames_done payload encode_seconds encode_fps
    local decode_seconds decode_fps psnr_yuv
    select_engine "$label" "$profile"
    echo "== $label run=$run_index qp=$qp gop=$gop profile=$profile ==" >&2
    encode_log="$("$nvcr_bin" encode -i "$input" -o "$stream" -s "$size" -r "$fps" \
        --frames "$frames" --gop-size "$gop" --qp "$qp" "${engine_args[@]}")"
    parsed="$(printf '%s\n' "$encode_log" | sed -n 's/^Encoded \([0-9][0-9]*\) frame(s), \([0-9][0-9]*\) payload bytes, codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2 \3 \4/p')"
    [[ -n "$parsed" ]] || { echo "could not parse encode summary" >&2; return 1; }
    read -r frames_done payload encode_seconds encode_fps <<<"$parsed"
    decode_log="$("$nvcr_bin" decode -i "$stream" -o /dev/null --frames "$frames" \
        --quality-metrics "$input" "${engine_args[@]}")"
    parsed="$(printf '%s\n' "$decode_log" | sed -n 's/^Decoded [0-9][0-9]* frame(s), codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2/p')"
    [[ -n "$parsed" ]] || { echo "could not parse decode summary" >&2; return 1; }
    read -r decode_seconds decode_fps <<<"$parsed"
    psnr_yuv="$(printf '%s\n' "$decode_log" | sed -n 's/^Quality [0-9][0-9]* frame(s):.*PSNR-YUV \([0-9.][0-9.]*\) dB$/\1/p')"
    [[ -n "$psnr_yuv" ]] || { echo "could not parse quality summary" >&2; return 1; }
    append_row encode "$label" "$input" "$size" "$fps" "$gop" "$profile" "$run_index" \
        "$payload" "$encode_seconds" "$encode_fps" null
    append_row decode "$label" "$input" "$size" "$fps" "$gop" "$profile" "$run_index" \
        "$payload" "$decode_seconds" "$decode_fps" "$psnr_yuv"
    printf '%s %s %s %s %s %s\n' "$payload" "$encode_seconds" "$encode_fps" \
        "$decode_seconds" "$decode_fps" "$psnr_yuv"
}

run_case() {
    local label="$1" size="$2" fps="$3" profile="$4" input="$5" gop="$6"
    local run_index result payload encode_seconds encode_fps decode_seconds decode_fps psnr_yuv
    local payload_sum=0 encode_seconds_sum=0 encode_fps_sum=0 decode_seconds_sum=0 decode_fps_sum=0 psnr_sum=0
    [[ -f "$input" ]] || { echo "missing benchmark input: $input" >&2; return 2; }
    select_engine "$label" "$profile"
    if ((warmup_frames > 0)); then
        local warmup_stream="$output_dir/nvcr_${label}_qp${qp}_gop${gop}_warmup.nvcr"
        "$nvcr_bin" encode -i "$input" -o "$warmup_stream" -s "$size" -r "$fps" \
            --frames "$warmup_frames" --gop-size "$gop" --qp "$qp" "${engine_args[@]}" >/dev/null
        "$nvcr_bin" decode -i "$warmup_stream" -o /dev/null --frames "$warmup_frames" \
            "${engine_args[@]}" >/dev/null
    fi
    for run_index in $(seq 1 "$repetitions"); do
        result="$(run_once "$label" "$size" "$fps" "$profile" "$input" "$gop" "$run_index")"
        read -r payload encode_seconds encode_fps decode_seconds decode_fps psnr_yuv <<<"$result"
        payload_sum=$((payload_sum + payload))
        encode_seconds_sum="$(awk "BEGIN {printf \"%.6f\", $encode_seconds_sum + $encode_seconds}")"
        encode_fps_sum="$(awk "BEGIN {printf \"%.6f\", $encode_fps_sum + $encode_fps}")"
        decode_seconds_sum="$(awk "BEGIN {printf \"%.6f\", $decode_seconds_sum + $decode_seconds}")"
        decode_fps_sum="$(awk "BEGIN {printf \"%.6f\", $decode_fps_sum + $decode_fps}")"
        psnr_sum="$(awk "BEGIN {printf \"%.6f\", $psnr_sum + $psnr_yuv}")"
    done
    local avg_payload avg_encode_seconds avg_encode_fps avg_decode_seconds avg_decode_fps avg_psnr
    avg_payload="$(awk "BEGIN {printf \"%.0f\", $payload_sum / $repetitions}")"
    avg_encode_seconds="$(awk "BEGIN {printf \"%.6f\", $encode_seconds_sum / $repetitions}")"
    avg_encode_fps="$(awk "BEGIN {printf \"%.6f\", $encode_fps_sum / $repetitions}")"
    avg_decode_seconds="$(awk "BEGIN {printf \"%.6f\", $decode_seconds_sum / $repetitions}")"
    avg_decode_fps="$(awk "BEGIN {printf \"%.6f\", $decode_fps_sum / $repetitions}")"
    avg_psnr="$(awk "BEGIN {printf \"%.6f\", $psnr_sum / $repetitions}")"
    append_row encode "$label" "$input" "$size" "$fps" "$gop" "$profile" average \
        "$avg_payload" "$avg_encode_seconds" "$avg_encode_fps" null
    append_row decode "$label" "$input" "$size" "$fps" "$gop" "$profile" average \
        "$avg_payload" "$avg_decode_seconds" "$avg_decode_fps" "$avg_psnr"
    printf 'case=%s gop=%s runs=%s payload=%s encode_fps=%s decode_fps=%s psnr_yuv=%s\n' \
        "$label" "$gop" "$repetitions" "$avg_payload" "$avg_encode_fps" "$avg_decode_fps" "$avg_psnr"
}

run_resolution() {
    local label="$1" gop="$2"
    case "$label" in
        qcif) run_case qcif 176x144 30 qcif-fp16 "$input_qcif" "$gop" ;;
        cif) run_case cif 352x288 30 cif-fp16 "$input_cif" "$gop" ;;
        720p) run_case 720p 1280x720 30 720p-fp16 "$input_720" "$gop" ;;
        1080p) run_case 1080p 1920x1080 60 1080p-fp16 "$input_1080" "$gop" ;;
        *) echo "unsupported resolution label: $label" >&2; return 2 ;;
    esac
}

for gop in $gops; do
    for resolution in $resolutions; do
        run_resolution "$resolution" "$gop"
    done
done
