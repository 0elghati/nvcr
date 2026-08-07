#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcr_bin="${NVCR_BIN:-$repo_root/build-release/cli/nvcr}"
output_dir="${NVCR_BENCH_OUTPUT_DIR:-/tmp}"
results_dir="${NVCR_BENCH_RESULTS_DIR:-}"
hardware="${NVCR_BENCH_HARDWARE:-}"
engine_root="${NVCR_BENCH_ENGINE_ROOT:-}"
frames="${NVCR_BENCH_FRAMES:-97}"
qp="${NVCR_BENCH_QP:-63}"
gops="${NVCR_BENCH_GOPS:-1 97}"
resolutions="${NVCR_BENCH_RESOLUTIONS:-qcif cif 720p 1080p}"
repetitions="${NVCR_BENCH_REPETITIONS:-3}"
warmup_frames="${NVCR_BENCH_WARMUP_FRAMES:-10}"
jsonl=""
csv=""
report=""

input_qcif="${NVCR_BENCH_QCIF_INPUT:-/workspace/nvcr/dataset/qcif/akiyo_qcif.yuv}"
input_cif="${NVCR_BENCH_CIF_INPUT:-/workspace/nvcr/dataset/cif/paris_cif.yuv}"
input_360="${NVCR_BENCH_360P_INPUT:-/workspace/nvcr/dataset/360p/BasketballDrive_640x360_50.yuv}"
input_540="${NVCR_BENCH_540P_INPUT:-/workspace/nvcr/dataset/540p/BasketballDrive_960x540_50.yuv}"
input_720="${NVCR_BENCH_720P_INPUT:-/workspace/nvcr/dataset/720p/FourPeople_1280x720_60.yuv}"
input_1080="${NVCR_BENCH_1080P_INPUT:-/workspace/nvcr/dataset/hd/BasketballDrive_1920x1080_50.yuv}"
engine_qcif="${NVCR_BENCH_QCIF_ENGINE_DIR:-}"
engine_cif="${NVCR_BENCH_CIF_ENGINE_DIR:-}"
engine_360="${NVCR_BENCH_360P_ENGINE_DIR:-}"
engine_540="${NVCR_BENCH_540P_ENGINE_DIR:-}"
engine_720="${NVCR_BENCH_720P_ENGINE_DIR:-}"
engine_1080="${NVCR_BENCH_1080P_ENGINE_DIR:-}"
engine_args=()

usage() {
    cat <<'USAGE'
Usage: benchmark_resolution_matrix.sh [options]

Runs warmed encode and decode measurements at QCIF, CIF, 360p, 540p, 720p, and 1080p.
Decode measurements report weighted YUV PSNR against the source outside codec
timing. Every measured point is written as JSONL, CSV, and Markdown by default.

Options:
  --nvcr PATH               nvcr binary (default: build-release/cli/nvcr)
  --frames N                Frames per run (default: 97)
    --qp N                    Base QP (default: 63)
  --gops "A B"              GOP sizes to run (default: "1 97")
  --resolutions "A B"       Resolution labels (default: "qcif cif 720p 1080p")
  --repetitions N           Measured runs per case (default: 3)
  --warmup-frames N         Encode/decode warm-up frames (default: 10)
  --qcif-input FILE         176x144 YUV420P8 input
  --cif-input FILE          352x288 YUV420P8 input
  --360p-input FILE         640x360 YUV420P8 input
  --540p-input FILE         960x540 YUV420P8 input
  --720p-input FILE         1280x720 YUV420P8 input
  --1080p-input FILE        1920x1080 YUV420P8 input
    --engine-root DIR         Sibling engine bundles named dcvcrt-<resolution>
  --qcif-engine-dir DIR     Use a local QCIF bundle instead of --engine-profile
  --cif-engine-dir DIR      Use a local CIF bundle instead of --engine-profile
  --360p-engine-dir DIR     Use a local 360p bundle instead of --engine-profile
  --540p-engine-dir DIR     Use a local 540p bundle instead of --engine-profile
  --720p-engine-dir DIR     Use a local 720p bundle instead of --engine-profile
  --1080p-engine-dir DIR    Use a local 1080p bundle instead of --engine-profile
  --output-dir DIR          Encoded output directory (default: /tmp)
    --hardware LABEL          Hardware label for aggregation (auto-detected by default)
    --results-dir DIR         Result directory (default: evidence/performance/<UTC timestamp>)
    --jsonl FILE              Machine-readable per-run and aggregate JSONL
    --csv FILE                Machine-readable per-run and aggregate CSV
    --report FILE             Human-readable Markdown report
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
        --360p-input) input_360="$2"; shift 2 ;;
        --540p-input) input_540="$2"; shift 2 ;;
        --720p-input) input_720="$2"; shift 2 ;;
        --1080p-input) input_1080="$2"; shift 2 ;;
        --engine-root) engine_root="$2"; shift 2 ;;
        --qcif-engine-dir) engine_qcif="$2"; shift 2 ;;
        --cif-engine-dir) engine_cif="$2"; shift 2 ;;
        --360p-engine-dir) engine_360="$2"; shift 2 ;;
        --540p-engine-dir) engine_540="$2"; shift 2 ;;
        --720p-engine-dir) engine_720="$2"; shift 2 ;;
        --1080p-engine-dir) engine_1080="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --hardware) hardware="$2"; shift 2 ;;
        --jsonl) jsonl="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        --csv) csv="$2"; shift 2 ;;
        --report) report="$2"; shift 2 ;;
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

commit="${NVCR_BENCH_COMMIT:-}"
if [[ -z "$commit" ]]; then
    commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)"
fi
dirty="${NVCR_BENCH_DIRTY:-}"
if [[ -z "$dirty" ]]; then
    dirty="false"
    if [[ -n "$(git -C "$repo_root" status --short 2>/dev/null || true)" ]]; then
        dirty="true"
    fi
fi
if [[ "$dirty" != true && "$dirty" != false ]]; then
    echo "NVCR_BENCH_DIRTY must be true or false" >&2
    exit 2
fi
if [[ -z "$hardware" ]]; then
    hardware="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/[[:space:]]*$//' || true)"
fi
[[ -n "$hardware" ]] || hardware="$(hostname)"

if [[ -z "$results_dir" ]]; then
    results_dir="$repo_root/evidence/performance/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$results_dir"
[[ -n "$jsonl" ]] || jsonl="$results_dir/results.jsonl"
[[ -n "$csv" ]] || csv="$results_dir/results.csv"
[[ -n "$report" ]] || report="$results_dir/summary.md"
mkdir -p "$(dirname "$jsonl")" "$(dirname "$csv")" "$(dirname "$report")"
printf '%s\n' 'schema,hardware,operation,resolution,input,size,fps,frames,warmup_frames,run_index,runs_planned,qp,gop_size,engine_profile,payload_bytes,payload_bpp,codec_time_seconds,throughput_fps,psnr_yuv' >"$csv"
cat >"$report" <<EOF
# NVCR Performance Run

- Commit: $commit
- Dirty: $dirty
- Hardware: $hardware
- Resolutions: $resolutions
- QP: $qp
- GOPs: $gops
- Frames: $frames
- Warm-up frames: $warmup_frames
- Measured repetitions: $repetitions

The CSV and JSONL files in this directory contain individual repetitions and
average rows. Throughput is the codec-reported encode/decode FPS. Payload BPP
is codec payload bits divided by width * height * frames.

## Results

| Operation | Resolution | Run | GOP | Payload bytes | Payload BPP | Codec seconds | Throughput FPS | PSNR-YUV |
|---|---|---|---:|---:|---:|---:|---:|---:|
EOF

select_engine() {
    local label="$1"
    local profile="$2"
    local directory=""
    case "$label" in
        qcif) directory="$engine_qcif" ;;
        cif) directory="$engine_cif" ;;
        360p) directory="$engine_360" ;;
        540p) directory="$engine_540" ;;
        720p) directory="$engine_720" ;;
        1080p) directory="$engine_1080" ;;
    esac
    if [[ -z "$directory" && -n "$engine_root" ]]; then
        directory="$engine_root/$label"
    fi
    if [[ -n "$directory" ]]; then
        [[ -d "$directory" ]] || {
            echo "missing engine bundle for $label: $directory" >&2
            return 2
        }
        engine_args=(--engine-dir "$directory")
    else
        engine_args=(--engine-profile "$profile")
    fi
}

csv_escape() {
    local value="${1//\"/\"\"}"
    printf '"%s"' "$value"
}

append_row() {
    local operation="$1" label="$2" input="$3" size="$4" fps="$5" gop="$6"
    local profile="$7" run_index="$8" payload="$9" seconds="${10}" throughput="${11}"
    local psnr_yuv="${12}" payload_bpp="${13}"
    printf '{"schema":"nvcr.benchmark.resolution-matrix.v1","hardware":"%s","nvcr_commit":"%s","nvcr_dirty":%s,"operation":"%s","resolution":"%s","input":"%s","size":"%s","fps":%s,"frames":%s,"warmup_frames":%s,"run_index":"%s","runs_planned":%s,"qp":%s,"gop_size":%s,"engine_profile":"%s","payload_bytes":%s,"payload_bpp":%s,"codec_time_seconds":%s,"throughput_fps":%s,"psnr_yuv":%s}\n' \
        "$hardware" "$commit" "$dirty" "$operation" "$label" "$input" "$size" "$fps" "$frames" \
        "$warmup_frames" "$run_index" "$repetitions" "$qp" "$gop" "$profile" \
        "$payload" "$payload_bpp" "$seconds" "$throughput" "$psnr_yuv" >>"$jsonl"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape resolution-matrix.v1)" "$(csv_escape "$hardware")" "$(csv_escape "$operation")" \
        "$(csv_escape "$label")" "$(csv_escape "$input")" "$(csv_escape "$size")" \
        "$(csv_escape "$fps")" "$(csv_escape "$frames")" "$(csv_escape "$warmup_frames")" \
        "$(csv_escape "$run_index")" "$(csv_escape "$repetitions")" "$(csv_escape "$qp")" \
        "$(csv_escape "$gop")" "$(csv_escape "$profile")" "$(csv_escape "$payload")" \
        "$(csv_escape "$payload_bpp")" "$(csv_escape "$seconds")" "$(csv_escape "$throughput")" \
        "$(csv_escape "$psnr_yuv")" >>"$csv"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$operation" "$label" "$run_index" "$gop" "$payload" "$payload_bpp" "$seconds" "$throughput" "$psnr_yuv" >>"$report"
}

run_once() {
    local label="$1" size="$2" fps="$3" profile="$4" input="$5" gop="$6" run_index="$7"
    local stream="$output_dir/nvcr_${label}_qp${qp}_gop${gop}_run${run_index}.nvcr"
    local encode_log decode_log parsed frames_done payload encode_seconds encode_fps
    local decode_seconds decode_fps psnr_yuv width height payload_bpp
    select_engine "$label" "$profile"
    echo "== $label run=$run_index qp=$qp gop=$gop profile=$profile ==" >&2
    encode_log="$("$nvcr_bin" encode -i "$input" -o "$stream" -s "$size" -r "$fps" \
        --frames "$frames" --gop-size "$gop" --qp "$qp" "${engine_args[@]}")"
    parsed="$(printf '%s\n' "$encode_log" | sed -n 's/^Encoded \([0-9][0-9]*\) frame(s), \([0-9][0-9]*\) payload bytes, codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2 \3 \4/p')"
    [[ -n "$parsed" ]] || { echo "could not parse encode summary" >&2; return 1; }
    read -r frames_done payload encode_seconds encode_fps <<<"$parsed"
    if [[ "$frames_done" != "$frames" ]]; then
        echo "encoded frame count mismatch: expected $frames, got $frames_done" >&2
        return 1
    fi
    IFS=x read -r width height <<<"$size"
    payload_bpp="$(awk "BEGIN {printf \"%.9f\", ($payload * 8) / ($width * $height * $frames)}")"
    decode_log="$("$nvcr_bin" decode -i "$stream" -o /dev/null --frames "$frames" \
        --quality-metrics "$input" "${engine_args[@]}")"
    parsed="$(printf '%s\n' "$decode_log" | sed -n 's/^Decoded [0-9][0-9]* frame(s), codec time \([0-9.][0-9.]*\) s (\([0-9.][0-9.]*\) fps)$/\1 \2/p')"
    [[ -n "$parsed" ]] || { echo "could not parse decode summary" >&2; return 1; }
    read -r decode_seconds decode_fps <<<"$parsed"
    psnr_yuv="$(printf '%s\n' "$decode_log" | sed -n 's/^Quality [0-9][0-9]* frame(s):.*PSNR-YUV \([0-9.][0-9.]*\) dB$/\1/p')"
    [[ -n "$psnr_yuv" ]] || { echo "could not parse quality summary" >&2; return 1; }
    append_row encode "$label" "$input" "$size" "$fps" "$gop" "$profile" "$run_index" \
        "$payload" "$encode_seconds" "$encode_fps" null "$payload_bpp"
    append_row decode "$label" "$input" "$size" "$fps" "$gop" "$profile" "$run_index" \
        "$payload" "$decode_seconds" "$decode_fps" "$psnr_yuv" "$payload_bpp"
    printf '%s %s %s %s %s %s %s\n' "$payload" "$encode_seconds" "$encode_fps" \
        "$decode_seconds" "$decode_fps" "$psnr_yuv" "$payload_bpp"
}

run_case() {
    local label="$1" size="$2" fps="$3" profile="$4" input="$5" gop="$6"
    local run_index result payload encode_seconds encode_fps decode_seconds decode_fps psnr_yuv payload_bpp
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
        read -r payload encode_seconds encode_fps decode_seconds decode_fps psnr_yuv payload_bpp <<<"$result"
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
    local avg_payload_bpp
    IFS=x read -r width height <<<"$size"
    avg_payload_bpp="$(awk "BEGIN {printf \"%.9f\", ($avg_payload * 8) / ($width * $height * $frames)}")"
    append_row encode "$label" "$input" "$size" "$fps" "$gop" "$profile" average \
        "$avg_payload" "$avg_encode_seconds" "$avg_encode_fps" null "$avg_payload_bpp"
    append_row decode "$label" "$input" "$size" "$fps" "$gop" "$profile" average \
        "$avg_payload" "$avg_decode_seconds" "$avg_decode_fps" "$avg_psnr" "$avg_payload_bpp"
    printf 'case=%s gop=%s runs=%s payload=%s payload_bpp=%s encode_fps=%s decode_fps=%s psnr_yuv=%s\n' \
        "$label" "$gop" "$repetitions" "$avg_payload" "$avg_payload_bpp" "$avg_encode_fps" "$avg_decode_fps" "$avg_psnr"
}

run_resolution() {
    local label="$1" gop="$2"
    case "$label" in
        qcif) run_case qcif 176x144 30 qcif "$input_qcif" "$gop" ;;
        cif) run_case cif 352x288 30 cif "$input_cif" "$gop" ;;
        360p) run_case 360p 640x360 50 360p "$input_360" "$gop" ;;
        540p) run_case 540p 960x540 50 540p "$input_540" "$gop" ;;
        720p) run_case 720p 1280x720 30 720p "$input_720" "$gop" ;;
        1080p) run_case 1080p 1920x1080 60 1080p "$input_1080" "$gop" ;;
        *) echo "unsupported resolution label: $label" >&2; return 2 ;;
    esac
}

for gop in $gops; do
    for resolution in $resolutions; do
        run_resolution "$resolution" "$gop"
    done
done
