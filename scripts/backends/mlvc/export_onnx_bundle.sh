#!/usr/bin/env bash
# Export an MLVC generic ONNX bundle with MLVC's own convert.py (C12).
#
# Usage: export_onnx_bundle.sh MLVC_ROOT PYTHON MODEL_VERSION CHECKPOINT WEIGHTS_VERSION OUT_DIR
#   MLVC_ROOT   pinned microsoft/MLVC checkout (1425880); copied, never written
#   PYTHON      interpreter with MLVC's uv.lock exporter versions and msrtc_rans
#   CHECKPOINT  absolute path to the published .ckpt (hash-checked by the caller)
#
# Upstream validates each canvas on a VCD clip (15.8-35.6 GB downloads). This
# uses NVCR's datasets instead, so the bundle's recorded conversion_loop metrics
# describe these clips: akiyo for 320x192 and 432x240, BasketballDrive for
# 640x368 and 960x544. The clips only drive tracing and the PyTorch/ONNX check.
set -euo pipefail
mlvc_root=$1 py=$2 model=$3 checkpoint=$4 version=$5 out=$6
datasets=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../datasets" && pwd)
work=$(mktemp -d)
cp -r "$mlvc_root/." "$work"
cd "$work/video"
cp conversion/_full_model/model_configs_example.yaml conversion/_full_model/model_configs.yaml
mkdir -p "$out" job/pretrained
ln -s "$checkpoint" "job/pretrained/$(basename "$checkpoint")"
common=(--model-version "$model" --weights-path "pretrained/$(basename "$checkpoint")"
        --weights-version "$version" --model-type onnx --target-device generic)
dirs=(--job-outputs-dir "$work/video/job" --test-data-dir "$datasets")
for canvas in "320x192 akiyo_qcif.yuv 176 144" "432x240 akiyo_qcif.yuv 176 144" \
              "640x368 360p/BasketballDrive_640x360_50.yuv 640 360" \
              "960x544 540p/BasketballDrive_960x540_50.yuv 960 540"; do
    set -- $canvas
    log="$out/export-$model-$version-$1.log"
    echo "[export] $model $version $1"
    "$py" -W ignore convert.py "${dirs[@]}" export "${common[@]}" \
        --model-width "${1%x*}" --model-height "${1#*x}" \
        --exporter-params-json "{\"test_video_path\": \"$datasets/$2\", \"image_width\": $3, \"image_height\": $4}" \
        >"$log" 2>&1 || { tail -30 "$log"; exit 1; }
done
echo "[bundle] $model $version"
"$py" -W ignore convert.py "${dirs[@]}" bundle "${common[@]}" --mlvc-version v0.1 --skip-if-exists \
    >"$out/bundle-$model-$version.log" 2>&1 || { tail -30 "$out/bundle-$model-$version.log"; exit 1; }
cp -r output/model_bundles/. "$out/"
