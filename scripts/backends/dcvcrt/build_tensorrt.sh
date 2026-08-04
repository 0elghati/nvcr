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
engine_profile_path="$repo_root/configs/engine-profiles/1080p.json"
target_profile_path=""
engine_profile_explicit=0
hardware_compatibility=exact

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
    --hardware-compatibility)
        hardware_compatibility="$2"
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

case "$hardware_compatibility" in
exact|same_compute_capability|ampere_plus) ;;
*)
    echo "--hardware-compatibility must be exact, same_compute_capability, or ampere_plus" >&2
    exit 2
    ;;
esac

if ((engine_profile_explicit == 0)); then
    engine_profile_path="$repo_root/configs/engine-profiles/${optimization_point}.json"
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
if [[ "$hardware_compatibility" != exact ]]; then
    if [[ -z "$target_profile_path" ]]; then
        echo "hardware-compatible builds require --target-profile-path" >&2
        exit 2
    fi
    target_architecture="$("$python_bin" - "$target_profile_path" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream).get("host", {}).get("architecture", ""))
PY
)"
    if [[ "$target_architecture" != x86_64 ]]; then
        echo "hardware-compatible TensorRT plans are supported only on desktop x86_64; Jetson remains exact" >&2
        exit 2
    fi
fi

read -r declared_engine_id declared_optimization_point declared_precision \
    declared_workspace_mib declared_builder_level < <(
    "$python_bin" - "$engine_profile_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    profile = json.load(stream)
print(
    profile.get("id", ""),
    profile.get("optimization_point", ""),
    profile.get("precision", ""),
    profile.get("workspace_mib", ""),
    profile.get("builder_optimization_level", ""),
)
PY
)
if [[ ( "$declared_engine_id" != "$optimization_point" &&
        "$declared_engine_id" != "$optimization_point-fp16" ) ||
      "$declared_optimization_point" != "$optimization_point" ||
      "$declared_precision" != fp16 ||
      "$declared_workspace_mib" != "$workspace_mib" ||
      "$declared_builder_level" != "$builder_optimization_level" ]]; then
    echo "TensorRT build arguments do not match engine profile: $engine_profile_path" >&2
    exit 1
fi

i_frame_min=1x3x64x64
i_y_min=1x256x4x4
i_z_min=1x128x1x1
i_spatial_min=1x512x4x4
i_synthesis_min=1x256x4x4
p_frame_min=1x3x64x64
p_feature_min=1x256x8x8
p_y_min=1x128x4x4
p_y_padded_min=1x128x4x4
p_z_min=1x128x1x1
p_spatial_min=1x512x4x4

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
    p_y_padded_opt="$p_y_opt"
    p_z_opt=1x128x3x3
    p_spatial_opt=1x512x12x12
    i_frame_max=1x3x192x192
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max=1x512x12x12
    i_synthesis_max=1x256x12x12
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_y_padded_max="$p_y_padded_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
cif)
    i_frame_opt=1x3x288x352
    i_y_opt=1x256x20x24
    i_z_opt=1x128x5x6
    i_spatial_opt=1x512x18x22
    i_synthesis_opt=1x256x18x22
    p_frame_opt=1x3x320x384
    p_feature_opt=1x256x40x48
    p_y_opt=1x128x20x24
    p_y_padded_opt="$p_y_opt"
    p_z_opt=1x128x5x6
    p_spatial_opt=1x512x20x24
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_y_padded_max="$p_y_padded_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
360p)
    i_frame_opt=1x3x368x640
    i_y_opt=1x256x24x40
    i_z_opt=1x128x6x10
    i_spatial_opt=1x512x23x40
    i_synthesis_opt=1x256x23x40
    p_frame_opt=1x3x368x640
    p_feature_opt=1x256x46x80
    p_y_opt=1x128x23x40
    p_y_padded_opt=1x128x24x40
    p_z_opt=1x128x6x10
    p_spatial_opt=1x512x23x40
    i_frame_min="$i_frame_opt"
    i_y_min="$i_y_opt"
    i_z_min="$i_z_opt"
    i_spatial_min="$i_spatial_opt"
    i_synthesis_min="$i_synthesis_opt"
    p_frame_min="$p_frame_opt"
    p_feature_min="$p_feature_opt"
    p_y_min="$p_y_opt"
    p_y_padded_min="$p_y_padded_opt"
    p_z_min="$p_z_opt"
    p_spatial_min="$p_spatial_opt"
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_y_padded_max="$p_y_padded_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
540p)
    i_frame_opt=1x3x544x960
    i_y_opt=1x256x36x60
    i_z_opt=1x128x9x15
    i_spatial_opt=1x512x34x60
    i_synthesis_opt=1x256x34x60
    p_frame_opt=1x3x544x960
    p_feature_opt=1x256x68x120
    p_y_opt=1x128x34x60
    p_y_padded_opt=1x128x36x60
    p_z_opt=1x128x9x15
    p_spatial_opt=1x512x34x60
    i_frame_min="$i_frame_opt"
    i_y_min="$i_y_opt"
    i_z_min="$i_z_opt"
    i_spatial_min="$i_spatial_opt"
    i_synthesis_min="$i_synthesis_opt"
    p_frame_min="$p_frame_opt"
    p_feature_min="$p_feature_opt"
    p_y_min="$p_y_opt"
    p_y_padded_min="$p_y_padded_opt"
    p_z_min="$p_z_opt"
    p_spatial_min="$p_spatial_opt"
    i_frame_max="$i_frame_opt"
    i_y_max="$i_y_opt"
    i_z_max="$i_z_opt"
    i_spatial_max="$i_spatial_opt"
    i_synthesis_max="$i_synthesis_opt"
    p_frame_max="$p_frame_opt"
    p_feature_max="$p_feature_opt"
    p_y_max="$p_y_opt"
    p_y_padded_max="$p_y_padded_opt"
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
    p_y_padded_opt="$p_y_opt"
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
    p_y_padded_max="$p_y_padded_opt"
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
    p_y_padded_opt="$p_y_opt"
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
    p_y_padded_max="$p_y_padded_opt"
    p_z_max="$p_z_opt"
    p_spatial_max="$p_spatial_opt"
    ;;
*)
    echo "unsupported optimization point: $optimization_point (expected qcif, cif, 360p, 540p, 720p, or 1080p)" >&2
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
    if ! "$python_bin" "$repo_root/scripts/nvcr_artifacts.py" \
        validate "$model_dir" >/dev/null; then
        echo "refusing to build TensorRT plans from an invalid or stale model bundle: $model_dir" >&2
        exit 1
    fi
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
case "$hardware_compatibility" in
same_compute_capability)
    common_build+=(--hardwareCompatibilityLevel=sameComputeCapability)
    ;;
ampere_plus)
    common_build+=(--hardwareCompatibilityLevel=ampere+)
    ;;
esac
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
        "frame:$i_frame_min" "frame:$i_frame_opt" "frame:$i_frame_max"
    build_engine i_hyper_analysis \
        "y_padded:$i_y_min" "y_padded:$i_y_opt" "y_padded:$i_y_max"
    build_engine i_hyper_synthesis \
        "z_hat:$i_z_min" "z_hat:$i_z_opt" "z_hat:$i_z_max"
    for stage in 1 2 3; do
        build_engine "i_spatial_prior_$stage" \
            "context:$i_spatial_min" "context:$i_spatial_opt" "context:$i_spatial_max"
    done
    build_engine i_synthesis \
        "y_hat:$i_synthesis_min" "y_hat:$i_synthesis_opt" "y_hat:$i_synthesis_max"

    build_engine p_reference_frame \
        "reference_frame:$p_frame_min" "reference_frame:$p_frame_opt" "reference_frame:$p_frame_max"
    build_engine p_reference_feature \
        "reference_feature:$p_feature_min" "reference_feature:$p_feature_opt" "reference_feature:$p_feature_max"
    build_engine p_analysis \
        "frame:$p_frame_min,context:$p_feature_min" \
        "frame:$p_frame_opt,context:$p_feature_opt" \
        "frame:$p_frame_max,context:$p_feature_max"
    build_engine p_hyper_analysis \
        "y_padded:$p_y_padded_min" "y_padded:$p_y_padded_opt" "y_padded:$p_y_padded_max"
    build_engine p_prior \
        "z_hat:$p_z_min,temporal_context:$p_feature_min" \
        "z_hat:$p_z_opt,temporal_context:$p_feature_opt" \
        "z_hat:$p_z_max,temporal_context:$p_feature_max"
    build_engine p_spatial_prior \
        "context:$p_spatial_min" "context:$p_spatial_opt" "context:$p_spatial_max"
    build_engine p_synthesis \
        "y_hat:$p_y_min,context:$p_feature_min" \
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

    smoke_engine i_analysis "frame:$i_frame_opt"
    smoke_engine i_hyper_analysis "y_padded:$i_y_opt"
    smoke_engine i_hyper_synthesis "z_hat:$i_z_opt"
    for stage in 1 2 3; do
        smoke_engine "i_spatial_prior_$stage" "context:$i_spatial_opt"
    done
    smoke_engine i_synthesis "y_hat:$i_synthesis_opt"

    smoke_engine p_reference_frame "reference_frame:$p_frame_opt"
    smoke_engine p_reference_feature "reference_feature:$p_feature_opt"
    smoke_engine p_analysis "frame:$p_frame_opt,context:$p_feature_opt"
    smoke_engine p_hyper_analysis "y_padded:$p_y_padded_opt"
    smoke_engine p_prior "z_hat:$p_z_opt,temporal_context:$p_feature_opt"
    smoke_engine p_spatial_prior "context:$p_spatial_opt"
    smoke_engine p_synthesis "y_hat:$p_y_opt,context:$p_feature_opt"
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
    --hardware-compatibility "$hardware_compatibility"
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
