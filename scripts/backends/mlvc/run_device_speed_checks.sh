#!/system/bin/sh
# MLVC and DCVC-RT QNN checks on a Snapdragon phone. Runs from
# /data/local/tmp/nvcr-mlvc (see run_device_checks.sh) and reuses the DCVC-RT
# C9 folder /data/local/tmp/nvcr when present. Log: out-speed/log.txt.
#
# 1. MLVC reproduces the earlier S25 streams; cold and warm cache agree.
# 2. DCVC-RT reproduces the C9 digests, cold and warm.
# 3. DCVC-RT QCIF with one MLVC QNN setting at a time, cache off.
# 4. With SUSTAINED=1: longer clips, burst versus balanced.
M=/data/local/tmp/nvcr-mlvc
cd "$M" || exit 1
export LD_LIBRARY_PATH="$M"
export ADSP_LIBRARY_PATH="$M;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp"
export TMPDIR=/data/local/tmp
O=$M/out-speed
rm -rf "$O" && mkdir -p "$O"
exec > "$O/log.txt" 2>&1 < /dev/null
NVCR=$M/nvcr
B=bundles/qualcomm

show() { grep -E "Encoded|Decoded|First output|Quality|MLVC .* per frame|nvcr:|rror"; }
battery() { dumpsys battery | grep -E "^  $1:" | tr -dc '0-9'; }
state() { echo "state: temp=$(battery temperature) level=$(battery level) $(dumpsys thermalservice | grep 'Thermal Status')"; }
expect() { # file sha256
  [ -f "$1" ] || { echo "MISSING $1"; return; }
  actual=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$actual" = "$2" ]; then echo "MATCH $1"; else echo "DIFFER $1 $actual (expected $2)"; fi
}
same() {
  { [ -f "$1" ] && [ -f "$2" ]; } || { echo "MISSING $1 or $2"; return; }
  if cmp -s "$1" "$2"; then echo "IDENTICAL $1 $2"; else echo "DIFFER $1 $2"; fi
}
enc() { # out clip size frames [env...]
  out=$1 clip=$2 size=$3 frames=$4; shift 4
  echo "== encode $out ($*)"
  env "$@" "$NVCR" encode -v -i "data/$clip" -o "$O/$out.nvcr" -s "$size" -r 30 --frames "$frames" \
    --qp 32 --codec mlvc --provider onnxruntime-qnn --engine-dir "$B" 2>&1 | show
}
dec() { # in out clip [env...]
  in=$1 out=$2 clip=$3; shift 3
  echo "== decode $in -> $out ($*)"
  env "$@" "$NVCR" decode -v -i "$O/$in.nvcr" -o "$O/$out.yuv" --codec mlvc --provider onnxruntime-qnn \
    --engine-dir "$B" --quality-metrics "data/$clip" 2>&1 | show
}
dcvc() { # tag name size [env...]
  tag=$1 name=$2 size=$3; shift 3
  state
  echo "== dcvc-rt $name $tag ($*)"
  env "$@" "$NVCR" encode -i "inputs/$name.yuv" -o "$O/dcvc_${name}_$tag.nvcr" -s "$size" --frames 33 \
    --provider onnxruntime-qnn --engine-dir "models-$name" 2>&1 | show
  env "$@" "$NVCR" decode -i "$O/dcvc_${name}_$tag.nvcr" -o "$O/dcvc_${name}_$tag.yuv" \
    --provider onnxruntime-qnn --engine-dir "models-$name" 2>&1 | show
}

echo "== device"
getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release
./nvcr_mlvc_unit_tests 2>&1 | tail -1
state

echo "== 1. MLVC, cold then warm cache"
rm -rf "$B/.nvcr-cache"
for pass in cold warm; do
  enc "qcif_$pass" akiyo_qcif.yuv 176x144 66
  dec "qcif_$pass" "qcif_$pass" akiyo_qcif.yuv
  enc "360p_$pass" basketball_640x360_65.yuv 640x360 65
  dec "360p_$pass" "360p_$pass" basketball_640x360_65.yuv
done
# Streams from the first S25 run (2026-11-03).
expect "$O/qcif_cold.nvcr" 457beff130d2573e98e76490b01dcbf480afe8715ab4fbb334b8fcb3970676b3
expect "$O/qcif_cold.yuv" be56f28f233bd3b4ce5eacbc582cd5dd767950db1a45375331613877db8805b8
expect "$O/360p_cold.nvcr" 4c9e190d1305a16e03eaad6198ee193eab4f61ce03960081f8f524de8c9e1521
expect "$O/360p_cold.yuv" d2208bc7e426444f46bc35e7a6a3bbae20e747edb6847115d1f2a1aa2c81f079
for name in qcif 360p; do
  same "$O/${name}_cold.nvcr" "$O/${name}_warm.nvcr"
  same "$O/${name}_cold.yuv" "$O/${name}_warm.yuv"
done

if cd /data/local/tmp/nvcr 2>/dev/null; then
  echo "== 2. DCVC-RT, cold then warm cache"
  # name:size:stream:yuv, digests from C9 (2026-11-01).
  for case in \
    qcif:176x144:3c61869e8baf691531a82def8f4c0e88ae2385b56573ffd8c63aa76b9507f5f2:75c78264b479c947111ca82cb374a02c6872ea6dc7b143a814de133b2050466b \
    360p:640x360:668831dc04fb05dc07997f98df3aa3bef6bdfe09003f1c5decb39b796b25b212:cb3ed584360f0d24c3050f04aa0318e63f25cae318997f08245aab8abc9cb0ac; do
    name=${case%%:*} rest=${case#*:}; size=${rest%%:*} rest=${rest#*:}; stream=${rest%%:*} yuv=${rest#*:}
    if [ ! -d "models-$name" ] || [ ! -f "inputs/$name.yuv" ]; then echo "skip $name"; continue; fi
    rm -rf "models-$name/.nvcr-cache"
    for tag in cold warm; do
      dcvc "$tag" "$name" "$size"
      expect "$O/dcvc_${name}_$tag.nvcr" "$stream"
      expect "$O/dcvc_${name}_$tag.yuv" "$yuv"
    done
  done

  echo "== 3. DCVC-RT QCIF with one MLVC QNN setting at a time"
  if [ -d models-qcif ]; then
    dcvc graph-disable qcif 176x144 NVCR_ONNXRUNTIME_CACHE_DIR=off NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION=disable
    dcvc burst qcif 176x144 NVCR_ONNXRUNTIME_CACHE_DIR=off NVCR_QNN_HTP_PERFORMANCE_MODE=burst
    dcvc finalize-3 qcif 176x144 NVCR_ONNXRUNTIME_CACHE_DIR=off NVCR_QNN_HTP_FINALIZATION_MODE=3
    for tag in graph-disable burst finalize-3; do
      expect "$O/dcvc_qcif_$tag.nvcr" 3c61869e8baf691531a82def8f4c0e88ae2385b56573ffd8c63aa76b9507f5f2
    done
  fi
  cd "$M" || exit 1
else
  echo "skip DCVC-RT: /data/local/tmp/nvcr is missing"
fi

if [ "$SUSTAINED" = 1 ]; then
  echo "== 4. sustained speed"
  for mode in burst balanced; do
    for run in 1 2 3 4 5; do
      state; echo "run $run"
      enc "long360_$mode" basketball_640x360_300.yuv 640x360 300 "NVCR_QNN_HTP_PERFORMANCE_MODE=$mode"
      dec "long360_$mode" "long360_$mode" basketball_640x360_300.yuv "NVCR_QNN_HTP_PERFORMANCE_MODE=$mode"
    done
    for run in 1 2; do
      state; echo "run $run"
      enc "long540_$mode" basketball_960x540_192.yuv 960x540 192 "NVCR_QNN_HTP_PERFORMANCE_MODE=$mode"
      dec "long540_$mode" "long540_$mode" basketball_960x540_192.yuv "NVCR_QNN_HTP_PERFORMANCE_MODE=$mode"
    done
  done
  same "$O/long360_burst.nvcr" "$O/long360_balanced.nvcr"
  same "$O/long540_burst.yuv" "$O/long540_balanced.yuv"
fi
state
echo "== hashes"
(cd "$O" && sha256sum ./*.nvcr ./*.yuv)
echo "== done"
