#!/usr/bin/env bash
set -euo pipefail

model_dir="build/models/dcvcrt"
engine_dir="build/engines/dcvcrt"
trtexec_bin="${TRTEXEC:-}"
run_smoke=1
optimization_point="1080p"

while (($#)); do
    case "$1" in
    --models)
        model_dir="$2"
        shift 2
        ;;
    --engines)
        engine_dir="$2"
        shift 2
        ;;
    --trtexec)
        trtexec_bin="$2"
        shift 2
        ;;
    --optimization-point)
        optimization_point="$2"
        shift 2
        ;;
    --skip-smoke)
        run_smoke=0
        shift
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
    esac
done

case "$optimization_point" in
qcif)
    i_frame_opt=1x3x144x176
    i_y_opt=1x256x12x12
    i_z_opt=1x128x3x3
    i_spatial_opt=1x512x9x11
    i_synthesis_opt=1x256x9x11
    p_frame_opt=1x3x192x192
    p_feature_opt=1x256x24x24
    p_y_opt=1x128x12x12
    p_z_opt=1x128x3x3
    p_spatial_opt=1x512x12x12
    ;;
1080p)
    i_frame_opt=1x3x1088x1920
    i_y_opt=1x256x68x120
    i_z_opt=1x128x17x30
    i_spatial_opt=1x512x68x120
    i_synthesis_opt=1x256x68x120
    p_frame_opt=1x3x1088x1920
    p_feature_opt=1x256x136x240
    p_y_opt=1x128x68x120
    p_z_opt=1x128x17x30
    p_spatial_opt=1x512x68x120
    ;;
*)
    echo "unsupported optimization point: $optimization_point (expected qcif or 1080p)" >&2
    exit 2
    ;;
esac

if [[ -z "$trtexec_bin" ]]; then
    trtexec_bin="$(command -v trtexec || true)"
fi
if [[ -z "$trtexec_bin" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
if [[ ! -x "$trtexec_bin" ]]; then
    echo "trtexec not found; install libnvinfer-bin or pass --trtexec" >&2
    exit 1
fi

mkdir -p "$engine_dir"
for asset in i_entropy.bin i_quant.bin i_frame_manifest.json \
    p_entropy.bin p_quant.bin p_frame_manifest.json; do
    if [[ ! -f "$model_dir/$asset" ]]; then
        echo "missing runtime asset: $model_dir/$asset" >&2
        exit 1
    fi
    cp -f "$model_dir/$asset" "$engine_dir/$asset"
done
timing_cache="$engine_dir/timing.cache"
common_build=(
    --fp16
    --memPoolSize=workspace:2048
    --builderOptimizationLevel=3
    "--timingCacheFile=$timing_cache"
    --skipInference
)

build_engine() {
    local name="$1"
    local min_shapes="$2"
    local opt_shapes="$3"
    local max_shapes="$4"
    "$trtexec_bin" \
        "--onnx=$model_dir/$name.onnx" \
        "--saveEngine=$engine_dir/$name.plan" \
        "--minShapes=$min_shapes" \
        "--optShapes=$opt_shapes" \
        "--maxShapes=$max_shapes" \
        "${common_build[@]}"
}

build_engine i_analysis \
    frame:1x3x64x64 "frame:$i_frame_opt" frame:1x3x1088x1920
build_engine i_hyper_analysis \
    y_padded:1x256x4x4 "y_padded:$i_y_opt" y_padded:1x256x68x120
build_engine i_hyper_synthesis \
    z_hat:1x128x1x1 "z_hat:$i_z_opt" z_hat:1x128x17x30
for stage in 1 2 3; do
    build_engine "i_spatial_prior_$stage" \
        context:1x512x4x4 "context:$i_spatial_opt" context:1x512x68x120
done
build_engine i_synthesis \
    y_hat:1x256x4x4 "y_hat:$i_synthesis_opt" y_hat:1x256x68x120

build_engine p_reference_frame \
    reference_frame:1x3x64x64 "reference_frame:$p_frame_opt" reference_frame:1x3x1088x1920
build_engine p_reference_feature \
    reference_feature:1x256x8x8 "reference_feature:$p_feature_opt" reference_feature:1x256x136x240
build_engine p_analysis \
    frame:1x3x64x64,context:1x256x8x8 \
    "frame:$p_frame_opt,context:$p_feature_opt" \
    frame:1x3x1088x1920,context:1x256x136x240
build_engine p_hyper_analysis \
    y_padded:1x128x4x4 "y_padded:$p_y_opt" y_padded:1x128x68x120
build_engine p_prior \
    z_hat:1x128x1x1,temporal_context:1x256x8x8 \
    "z_hat:$p_z_opt,temporal_context:$p_feature_opt" \
    z_hat:1x128x17x30,temporal_context:1x256x136x240
build_engine p_spatial_prior \
    context:1x512x4x4 "context:$p_spatial_opt" context:1x512x68x120
build_engine p_synthesis \
    y_hat:1x128x4x4,context:1x256x8x8 \
    "y_hat:$p_y_opt,context:$p_feature_opt" \
    y_hat:1x128x68x120,context:1x256x136x240

if ((run_smoke)); then
    smoke_engine() {
        local name="$1"
        local shapes="$2"
        "$trtexec_bin" \
            "--loadEngine=$engine_dir/$name.plan" \
            "--shapes=$shapes" \
            --iterations=1 \
            --warmUp=0 \
            --duration=0 \
            --useSpinWait
    }

    smoke_engine i_analysis frame:1x3x144x176
    smoke_engine i_hyper_analysis y_padded:1x256x12x12
    smoke_engine i_hyper_synthesis z_hat:1x128x3x3
    for stage in 1 2 3; do
        smoke_engine "i_spatial_prior_$stage" context:1x512x9x11
    done
    smoke_engine i_synthesis y_hat:1x256x9x11

    smoke_engine p_reference_frame reference_frame:1x3x192x192
    smoke_engine p_reference_feature reference_feature:1x256x24x24
    smoke_engine p_analysis frame:1x3x192x192,context:1x256x24x24
    smoke_engine p_hyper_analysis y_padded:1x128x12x12
    smoke_engine p_prior z_hat:1x128x3x3,temporal_context:1x256x24x24
    smoke_engine p_spatial_prior context:1x512x12x12
    smoke_engine p_synthesis y_hat:1x128x12x12,context:1x256x24x24
fi

sha256sum "$engine_dir"/*.plan >"$engine_dir/engine.sha256"
echo "TensorRT engines are ready in $engine_dir"
