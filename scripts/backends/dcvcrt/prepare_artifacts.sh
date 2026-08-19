#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
generic_script_dir="$repo_root/scripts"

dcvcrt_root="${NVCR_DCVCRT_ROOT:-$repo_root/assets}"
dcvcrt_repo="https://github.com/microsoft/DCVC.git"
dcvcrt_ref="${NVCR_DCVCRT_REF:-1feb52a592a9ff2c4e4ba2e5122e2da49a211466}"
models_dir="build/models/dcvcrt"
engines_dir="build/engines/dcvcrt-1080p"
trtexec_bin="${TRTEXEC:-}"
python_bin="${PYTHON:-python3}"
optimization_point="1080p"
# Sentinels: empty means "not explicitly set by the caller"; the selected
# versioned engine profile supplies reproducible defaults below.
workspace_mib=""
builder_optimization_level=""
device_id=""
auto_tune=1
skip_clone=0
skip_export=0
skip_engine=0
skip_smoke=0
enable_int8=0
model_profile_id=dcvcrt-cvpr2025
target_profile_id=local-auto
model_profile_path="$repo_root/configs/models/dcvcrt-cvpr2025.json"
engine_profile_path="$repo_root/configs/engine-profiles/1080p.json"
target_profile_path=""
engine_profile_explicit=0
hardware_compatibility=exact

checkpoint_url="https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8"
checkpoint_backup_url="https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA"

usage() {
    cat <<EOF
Usage: $0 [options]

Prepare NVCR DCVC-RT artifacts:
  1. ensure the pinned DCVC-RT source checkout exists;
  2. verify the two cvpr2025 .pth.tar checkpoints;
  3. export ONNX graphs plus entropy/quant assets;
  4. build target-local TensorRT .plan files.

Options:
  --model-profile-id ID    Model profile identity (default: dcvcrt-cvpr2025)
  --model-profile-path FILE
                           Versioned model profile bound to the bundle
  --engine-profile-path FILE
                           Versioned engine profile bound to the bundle
  --target-profile-id ID   Target profile identity (default: local-auto)
  --target-profile-path FILE
                           Versioned target profile bound to the bundle
  --dcvcrt-root DIR        Upstream DCVC-RT checkout (default: $dcvcrt_root)
  --dcvcrt-repo URL        Upstream repo URL (default: $dcvcrt_repo)
  --dcvcrt-ref REF         Pinned git ref to checkout after clone/fetch
  --models DIR             ONNX/runtime asset output (default: $models_dir)
  --engines DIR            TensorRT engine output (default: $engines_dir)
  --trtexec PATH           TensorRT trtexec path
  --python PATH            Python used for ONNX export (default: $python_bin)
  --optimization-point X   TensorRT profile point: qcif, cif, 360p, 540p, 720p, or 1080p
                           (default: $optimization_point)
  --workspace-mib N        TensorRT workspace memory pool in MiB
                           (must match the selected engine profile)
  --builder-optimization-level N
                           TensorRT builder optimization level 0-5
                           (must match the selected engine profile)
  --hardware-compatibility CLASS
                           exact, same_compute_capability, or ampere_plus
                           (default: exact; compatibility modes are desktop-only)
  --device-id N            CUDA device used for TensorRT engine building (default: auto-selected)
  --no-auto-tune           Disable platform/tool/device auto-detection; use
                           device 0 unless overridden. Build settings still
                           come from the selected engine profile.
  --skip-clone             Do not clone/fetch the upstream repo
  --skip-export            Reuse existing --models directory
  --skip-engine            Stop after ONNX/runtime assets are ready
  --skip-smoke             Skip trtexec smoke checks while building engines
  -h, --help               Show this help

Checkpoint download page:
  $checkpoint_url

Backup checkpoint page:
  $checkpoint_backup_url
EOF
}

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
    --engine-profile-path)
        engine_profile_path="$2"
        engine_profile_explicit=1
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
    --dcvcrt-root)
        dcvcrt_root="$2"
        shift 2
        ;;
    --dcvcrt-repo)
        dcvcrt_repo="$2"
        shift 2
        ;;
    --dcvcrt-ref)
        dcvcrt_ref="$2"
        shift 2
        ;;
    --models)
        models_dir="$2"
        shift 2
        ;;
    --engines)
        engines_dir="$2"
        shift 2
        ;;
    --trtexec)
        trtexec_bin="$2"
        shift 2
        ;;
    --python)
        python_bin="$2"
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
    --no-auto-tune)
        auto_tune=0
        shift
        ;;
    --skip-clone)
        skip_clone=1
        shift
        ;;
    --skip-export)
        skip_export=1
        shift
        ;;
    --skip-engine)
        skip_engine=1
        shift
        ;;
    --skip-smoke)
        skip_smoke=1
        shift
        ;;
    --enable-int8)
        enable_int8=1
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if ((engine_profile_explicit == 0)); then
    engine_profile_path="$repo_root/configs/engine-profiles/${optimization_point}.json"
fi

for profile_path in "$model_profile_path" "$engine_profile_path"; do
    if [[ ! -f "$profile_path" ]]; then
        echo "missing profile: $profile_path" >&2
        exit 1
    fi
done
if [[ -z "$target_profile_path" ]]; then
    echo "--target-profile-path is required for reproducible engine bundles" >&2
    exit 1
fi
if [[ ! -f "$target_profile_path" ]]; then
    echo "missing target profile: $target_profile_path" >&2
    exit 1
fi

# The versioned engine profile is the reproducible source of truth for tactic
# search depth and workspace. CLI values are checked against that profile by
# the manifest writer; experiments must select a separate versioned profile.
if [[ -z "$workspace_mib" || -z "$builder_optimization_level" ]]; then
    read -r profile_workspace_mib profile_builder_level < <(
        "$python_bin" - "$engine_profile_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    profile = json.load(stream)
print(profile["workspace_mib"], profile["builder_optimization_level"])
PY
    )
    [[ -z "$workspace_mib" ]] && workspace_mib="$profile_workspace_mib"
    [[ -z "$builder_optimization_level" ]] && \
        builder_optimization_level="$profile_builder_level"
fi

if ((auto_tune)); then
    # shellcheck source=scripts/detect_platform.sh
    source "$generic_script_dir/detect_platform.sh"
    nvcr_detect_platform_report
    [[ -z "$trtexec_bin" ]] && trtexec_bin="$NVCR_DETECT_TRTEXEC"
    [[ -z "$device_id" ]] && device_id="$NVCR_DETECT_DEVICE_ID"
    [[ -z "$workspace_mib" ]] && workspace_mib="$NVCR_DETECT_WORKSPACE_MIB"
    [[ -z "$builder_optimization_level" ]] && builder_optimization_level="$NVCR_DETECT_BUILDER_LEVEL"
fi
# Fixed fallback defaults for any value auto-tuning did not resolve (or when
# --no-auto-tune was passed).
[[ -z "$device_id" ]] && device_id=0
[[ -z "$workspace_mib" ]] && workspace_mib=2048
[[ -z "$builder_optimization_level" ]] && builder_optimization_level=3

if [[ -z "$trtexec_bin" ]]; then
    trtexec_bin="$(command -v trtexec || true)"
fi
if [[ -z "$trtexec_bin" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
echo "Using device-id=$device_id workspace-mib=$workspace_mib builder-optimization-level=$builder_optimization_level trtexec=${trtexec_bin:-<not found>}"

if ((skip_clone == 0)); then
    if [[ -d "$dcvcrt_root/.git" ]]; then
        echo "Updating upstream DCVC-RT checkout: $dcvcrt_root"
        git -C "$dcvcrt_root" fetch --all --tags
    elif [[ -e "$dcvcrt_root" ]]; then
        echo "DCVC-RT root exists but is not a git checkout: $dcvcrt_root" >&2
        exit 1
    else
        echo "Cloning upstream DCVC-RT: $dcvcrt_repo -> $dcvcrt_root"
        mkdir -p "$(dirname "$dcvcrt_root")"
        git clone "$dcvcrt_repo" "$dcvcrt_root"
    fi
    if [[ -n "$dcvcrt_ref" ]]; then
        git -C "$dcvcrt_root" checkout "$dcvcrt_ref"
    fi
fi

checkpoint_dir="$dcvcrt_root/checkpoints"
image_checkpoint="$checkpoint_dir/cvpr2025_image.pth.tar"
video_checkpoint="$checkpoint_dir/cvpr2025_video.pth.tar"
if [[ ! -f "$image_checkpoint" || ! -f "$video_checkpoint" ]]; then
    mkdir -p "$checkpoint_dir"
    cat >&2 <<EOF
Missing DCVC-RT checkpoints.

Download the pinned DCVC-RT pretrained models and place them here:
  $image_checkpoint
  $video_checkpoint

Primary:
  $checkpoint_url

Backup:
  $checkpoint_backup_url
EOF
    exit 1
fi

if ((skip_export == 0)); then
    "$python_bin" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "onnx", "onnxscript") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python export modules: " + ", ".join(missing), file=sys.stderr)
    print("Create an export environment with PyTorch, ONNX, and ONNXScript before running this step.", file=sys.stderr)
    sys.exit(1)
PY

    echo "Exporting I-frame ONNX/runtime assets to $models_dir"
    "$python_bin" "$script_dir/export_i_onnx.py" \
        --dcvcrt-root "$dcvcrt_root" \
        --output-dir "$models_dir" \
        --height 144 \
        --width 176 \
        --qp 0

    echo "Exporting P-frame ONNX/runtime assets to $models_dir"
    "$python_bin" "$script_dir/export_p_onnx.py" \
        --dcvcrt-root "$dcvcrt_root" \
        --output-dir "$models_dir" \
        --height 192 \
        --width 192 \
        --qp 32
fi

required_assets=(
    i_analysis.onnx
    i_hyper_analysis.onnx
    i_hyper_synthesis.onnx
    i_spatial_prior_1.onnx
    i_spatial_prior_2.onnx
    i_spatial_prior_3.onnx
    i_synthesis.onnx
    i_entropy.bin
    i_quant.bin
    i_frame_manifest.json
    p_reference_frame.onnx
    p_reference_feature.onnx
    p_analysis.onnx
    p_hyper_analysis.onnx
    p_prior.onnx
    p_spatial_prior.onnx
    p_synthesis.onnx
    p_entropy.bin
    p_quant.bin
    p_frame_manifest.json
)
for asset in "${required_assets[@]}"; do
    if [[ ! -f "$models_dir/$asset" ]]; then
        echo "missing exported asset: $models_dir/$asset" >&2
        exit 1
    fi
done

if ((skip_engine == 0)); then
    if [[ -z "$trtexec_bin" || ! -x "$trtexec_bin" ]]; then
        echo "trtexec not found; pass --trtexec or set TRTEXEC" >&2
        exit 1
    fi
    engine_args=(
        --model-profile-id "$model_profile_id"
        --model-profile-path "$model_profile_path"
        --target-profile-id "$target_profile_id"
        --engine-profile-path "$engine_profile_path"
        --models "$models_dir"
        --engines "$engines_dir"
        --trtexec "$trtexec_bin"
        --optimization-point "$optimization_point"
        --workspace-mib "$workspace_mib"
        --builder-optimization-level "$builder_optimization_level"
        --hardware-compatibility "$hardware_compatibility"
        --device-id "$device_id"
        --python "$python_bin"
    )
    if [[ -n "$target_profile_path" ]]; then
        engine_args+=(--target-profile-path "$target_profile_path")
    fi
    if ((skip_smoke)); then
        engine_args+=(--skip-smoke)
    fi
    if ((enable_int8)); then
        engine_args+=(--enable-int8)
    fi
    echo "Building target-local TensorRT engines in $engines_dir"
    "$script_dir/build_tensorrt.sh" "${engine_args[@]}"
fi

cat <<EOF

DCVC-RT artifacts are ready.

Use this engine directory with NVCR:
  --engine-dir $engines_dir

Example:
  nvcr encode -i input.yuv -o output.nvcr -s 1920x1080 -r 50 -c dcvc-rt --frames 97 --qp 32 --engine-dir $engines_dir

Exact TensorRT plans are target-specific. Desktop compatibility plans may be
shared only according to their recorded hardware class. TensorRT versions remain
exact. Desktop CUDA runtimes may consume engines recorded on an older runtime
within the same CUDA major family; Jetson plans remain exact.
EOF
