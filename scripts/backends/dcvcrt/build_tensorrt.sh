#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
model_dir="build/models/dcvcrt"
engine_dir="build/engines/dcvcrt"
trtexec_bin="${TRTEXEC:-}"
run_smoke=1
stamp_only=0
optimization_point="1080p"
workspace_mib=2048
builder_optimization_level=3
device_id=0
python_bin="${PYTHON:-python3}"
enable_int8=0
model_profile_id=dcvcrt-cvpr2025
target_profile_id=local-auto
model_profile_path="$repo_root/configs/models/dcvcrt-cvpr2025.json"
engine_profile_path="$repo_root/configs/engine-profiles/1080p-fp16.json"
target_profile_path=""
engine_profile_explicit=0

while (($#)); do
    case "$1" in
    --model-profile-id)
        model_profile_id="$2"
        shift 2
        ;;
    --model-profile-path)
        model_profile_path="$2"
        shift 2
        ;;
    --target-profile-id)
        target_profile_id="$2"
        shift 2
        ;;
    --target-profile-path)
        target_profile_path="$2"
        shift 2
        ;;
    --engine-profile-path)
        engine_profile_path="$2"
        engine_profile_explicit=1
        shift 2
        ;;
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
    --workspace-mib)
        workspace_mib="$2"
        shift 2
        ;;
    --builder-optimization-level)
        builder_optimization_level="$2"
        shift 2
        ;;
    --device-id)
        device_id="$2"
        shift 2
        ;;
    --python)
        python_bin="$2"
        shift 2
        ;;
    --skip-smoke)
        run_smoke=0
        shift
        ;;
    --stamp-only)
        stamp_only=1
        shift
        ;;
    --enable-int8)
        enable_int8=1
        shift
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
    esac
done

if ((engine_profile_explicit == 0 && optimization_point == "qcif")); then
    engine_profile_path="$(cd "$script_dir/.." && pwd)/configs/engine-profiles/qcif-fp16.json"
fi

for profile_path in "$model_profile_path" "$engine_profile_path"; do
    if [[ ! -f "$profile_path" ]]; then
        echo "missing profile: $profile_path" >&2
        exit 1
    fi
done
if [[ -n "$target_profile_path" && ! -f "$target_profile_path" ]]; then
    echo "missing target profile: $target_profile_path" >&2
    exit 1
fi

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
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
720p)
    i_frame_opt=1x3x720x1280
    i_y_opt=1x256x48x80
    i_z_opt=1x128x12x20
    i_spatial_opt=1x512x45x80
    i_synthesis_opt=1x256x45x80
    p_frame_opt=1x3x768x1280
    p_feature_opt=1x256x96x160
    p_y_opt=1x128x48x80
    p_z_opt=1x128x12x20
    p_spatial_opt=1x512x48x80
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
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
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
*)
    echo "unsupported optimization point: $optimization_point (expected qcif, 720p, or 1080p)" >&2
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
if ((stamp_only == 0)); then
    for asset in i_entropy.bin i_quant.bin i_frame_manifest.json \
        p_entropy.bin p_quant.bin p_frame_manifest.json; do
        if [[ ! -f "$model_dir/$asset" ]]; then
            echo "missing runtime asset: $model_dir/$asset" >&2
            exit 1
        fi
        cp -f "$model_dir/$asset" "$engine_dir/$asset"
    done
fi
timing_cache="$engine_dir/timing.cache"
common_build=(
    --fp16
    "--device=$device_id"
    "--memPoolSize=workspace:$workspace_mib"
    "--builderOptimizationLevel=$builder_optimization_level"
    "--timingCacheFile=$timing_cache"
    --skipInference
)
if ((enable_int8)); then
    common_build+=(--int8)
fi

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

if ((stamp_only == 0)); then
    build_engine i_analysis \
        frame:1x3x64x64 "frame:$i_frame_opt" "frame:$i_frame_max"
    build_engine i_hyper_analysis \
        y_padded:1x256x4x4 "y_padded:$i_y_opt" "y_padded:$i_y_max"
    build_engine i_hyper_synthesis \
        z_hat:1x128x1x1 "z_hat:$i_z_opt" "z_hat:$i_z_max"
    for stage in 1 2 3; do
        build_engine "i_spatial_prior_$stage" \
            context:1x512x4x4 "context:$i_spatial_opt" "context:$i_spatial_max"
    done
    build_engine i_synthesis \
        y_hat:1x256x4x4 "y_hat:$i_synthesis_opt" "y_hat:$i_synthesis_max"

    build_engine p_reference_frame \
        reference_frame:1x3x64x64 "reference_frame:$p_frame_opt" "reference_frame:$p_frame_max"
    build_engine p_reference_feature \
        reference_feature:1x256x8x8 "reference_feature:$p_feature_opt" "reference_feature:$p_feature_max"
    build_engine p_analysis \
        frame:1x3x64x64,context:1x256x8x8 \
        "frame:$p_frame_opt,context:$p_feature_opt" \
        "frame:$p_frame_max,context:$p_feature_max"
    build_engine p_hyper_analysis \
        y_padded:1x128x4x4 "y_padded:$p_y_opt" "y_padded:$p_y_max"
    build_engine p_prior \
        z_hat:1x128x1x1,temporal_context:1x256x8x8 \
        "z_hat:$p_z_opt,temporal_context:$p_feature_opt" \
        "z_hat:$p_z_max,temporal_context:$p_feature_max"
    build_engine p_spatial_prior \
        context:1x512x4x4 "context:$p_spatial_opt" "context:$p_spatial_max"
    build_engine p_synthesis \
        y_hat:1x128x4x4,context:1x256x8x8 \
        "y_hat:$p_y_opt,context:$p_feature_opt" \
        "y_hat:$p_y_max,context:$p_feature_max"
fi

if ((run_smoke && stamp_only == 0)); then
    smoke_engine() {
        local name="$1"
        local shapes="$2"
        "$trtexec_bin" \
            "--loadEngine=$engine_dir/$name.plan" \
            "--device=$device_id" \
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

for plan in i_analysis.plan i_hyper_analysis.plan i_hyper_synthesis.plan \
    i_spatial_prior_1.plan i_spatial_prior_2.plan i_spatial_prior_3.plan \
    i_synthesis.plan p_reference_frame.plan p_reference_feature.plan \
    p_analysis.plan p_hyper_analysis.plan p_prior.plan p_spatial_prior.plan \
    p_synthesis.plan; do
    if [[ ! -f "$engine_dir/$plan" ]]; then
        echo "missing TensorRT plan: $engine_dir/$plan" >&2
        exit 1
    fi
done

manifest_args=(
    --engines "$engine_dir"
    --trtexec "$trtexec_bin"
    --device-id "$device_id"
    --optimization-point "$optimization_point"
    --workspace-mib "$workspace_mib"
    --builder-optimization-level "$builder_optimization_level"
    --model-profile-id "$model_profile_id"
    --target-profile-id "$target_profile_id"
    --model-profile-path "$model_profile_path"
    --engine-profile-path "$engine_profile_path"
)
if [[ -n "$target_profile_path" ]]; then
    manifest_args+=(--target-profile-path "$target_profile_path")
fi
if ((stamp_only)); then
    manifest_args+=(--reject-device-warning)
fi
if ((enable_int8)); then
    manifest_args+=(--enable-int8)
fi
"$python_bin" "$script_dir/write_tensorrt_engine_manifest.py" "${manifest_args[@]}"
echo "TensorRT engines are ready in $engine_dir"
