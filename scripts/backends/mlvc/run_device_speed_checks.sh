#!/system/bin/sh
# MLVC and DCVC-RT speed checks on a Snapdragon phone after the QNN defaults
# and compiled-graph cache change. Runs from /data/local/tmp/nvcr-mlvc (see
# run_device_checks.sh for its contents) and reuses the DCVC-RT C9 folder
# /data/local/tmp/nvcr when present. The full log goes to out-speed/log.txt.
#
# 1. New QNN defaults reproduce the previous run's streams (byte-identical).
# 2. The compiled-graph cache: cold run writes it, warm run loads it; both
#    must produce identical output.
# 3. Speed on longer clips, burst versus balanced, five back-to-back 360p
#    runs and two 540p runs per mode, with battery temperature and thermal
#    status before each run.
# 4. DCVC-RT (C9 bundles): old QNN settings versus the new defaults.
cd /data/local/tmp/nvcr-mlvc || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nvcr-mlvc
export ADSP_LIBRARY_PATH="/data/local/tmp/nvcr-mlvc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp"
export TMPDIR=/data/local/tmp
O=/data/local/tmp/nvcr-mlvc/out-speed
rm -rf $O && mkdir -p $O
exec > $O/log.txt 2>&1 < /dev/null
NVCR=/data/local/tmp/nvcr-mlvc/nvcr
B=bundles/qualcomm

show() { grep -E "Encoded|Decoded|First output|Quality|MLVC .* per frame|nvcr:|rror"; }
battery() { dumpsys battery | grep -E "^  $1:" | tr -dc '0-9'; }
state() {
  echo "state: temp=$(battery temperature) level=$(battery level) $(dumpsys thermalservice | grep 'Thermal Status') $(dumpsys power | grep -m1 'mWakefulness=' | tr -d ' ')"
}
expect() { # file sha256
  actual=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$actual" = "$2" ]; then echo "MATCH $1"; else echo "DIFFER $1 $actual (expected $2)"; fi
}
same() { cmp -s "$1" "$2" && echo "IDENTICAL $1 $2" || echo "DIFFER $1 $2"; }
enc() { # out clip size frames [env...]
  out=$1 clip=$2 size=$3 frames=$4; shift 4
  echo "== encode $out ($*)"
  env "$@" $NVCR encode -v -i data/$clip -o $O/$out.nvcr -s $size -r 30 --frames $frames --qp 32 \
    --codec mlvc --provider onnxruntime-qnn --engine-dir $B 2>&1 | show
}
dec() { # in out clip [env...]
  in=$1 out=$2 clip=$3; shift 3
  echo "== decode $in -> $out ($*)"
  env "$@" $NVCR decode -v -i $O/$in.nvcr -o $O/$out.yuv --codec mlvc --provider onnxruntime-qnn \
    --engine-dir $B --quality-metrics data/$clip 2>&1 | show
}

echo "== device"
getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release
./nvcr_mlvc_unit_tests 2>&1 | tail -1
state

echo "== 1+2. new defaults and cache: cold, then warm"
rm -rf $B/.nvcr-cache
for pass in cold warm; do
  enc qcif_$pass akiyo_qcif.yuv 176x144 66
  dec qcif_$pass qcif_$pass akiyo_qcif.yuv
  enc 360p_$pass basketball_640x360_65.yuv 640x360 65
  dec 360p_$pass 360p_$pass basketball_640x360_65.yuv
  ls -la $B/.nvcr-cache
done
# Streams from the 2026-11-03 run (old defaults and upstream options agreed).
expect $O/qcif_cold.nvcr 457beff130d2573e98e76490b01dcbf480afe8715ab4fbb334b8fcb3970676b3
expect $O/qcif_cold.yuv be56f28f233bd3b4ce5eacbc582cd5dd767950db1a45375331613877db8805b8
expect $O/360p_cold.nvcr 4c9e190d1305a16e03eaad6198ee193eab4f61ce03960081f8f524de8c9e1521
expect $O/360p_cold.yuv d2208bc7e426444f46bc35e7a6a3bbae20e747edb6847115d1f2a1aa2c81f079
for f in qcif.nvcr qcif.yuv 360p.nvcr 360p.yuv; do
  same $O/${f%%.*}_cold.${f#*.} $O/${f%%.*}_warm.${f#*.}
done

echo "== 3. sustained speed"
for mode in burst balanced; do
  for rep in 1 2 3 4 5; do
    state
    enc long360_$mode basketball_640x360_300.yuv 640x360 300 NVCR_QNN_HTP_PERFORMANCE_MODE=$mode
    dec long360_$mode long360_$mode basketball_640x360_300.yuv NVCR_QNN_HTP_PERFORMANCE_MODE=$mode
  done
  for rep in 1 2; do
    state
    enc long540_$mode basketball_960x540_192.yuv 960x540 192 NVCR_QNN_HTP_PERFORMANCE_MODE=$mode
    dec long540_$mode long540_$mode basketball_960x540_192.yuv NVCR_QNN_HTP_PERFORMANCE_MODE=$mode
  done
done
same $O/long360_burst.nvcr $O/long360_balanced.nvcr
same $O/long540_burst.yuv $O/long540_balanced.yuv
state

echo "== 4. DCVC-RT C9 bundles"
OLD="NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION=all NVCR_QNN_HTP_PERFORMANCE_MODE=default NVCR_QNN_HTP_FINALIZATION_MODE=0 NVCR_ONNXRUNTIME_CACHE_DIR=off"
if cd /data/local/tmp/nvcr 2>/dev/null; then
  for case in qcif:176x144 360p:640x360; do
    name=${case%%:*}; size=${case#*:}
    if [ ! -d models-$name ] || [ ! -f inputs/$name.yuv ]; then echo "skip $name: no models-$name or inputs/$name.yuv"; continue; fi
    rm -rf models-$name/.nvcr-cache
    for run in old new-cold new-warm; do
      settings=""; [ $run = old ] && settings=$OLD
      state
      echo "== dcvc-rt $name $run encode"
      env $settings $NVCR encode -i inputs/$name.yuv -o $O/dcvc_${name}_$run.nvcr -s $size --frames 33 \
        --provider onnxruntime-qnn --engine-dir models-$name 2>&1 | show
      echo "== dcvc-rt $name $run decode"
      env $settings $NVCR decode -i $O/dcvc_${name}_$run.nvcr -o $O/dcvc_${name}_$run.yuv \
        --provider onnxruntime-qnn --engine-dir models-$name 2>&1 | show
    done
    for ext in nvcr yuv; do
      same $O/dcvc_${name}_old.$ext $O/dcvc_${name}_new-cold.$ext
      same $O/dcvc_${name}_new-cold.$ext $O/dcvc_${name}_new-warm.$ext
    done
  done
  cd /data/local/tmp/nvcr-mlvc
else
  echo "skip DCVC-RT: /data/local/tmp/nvcr is missing"
fi

echo "== hashes"
cd $O && sha256sum *.nvcr *.yuv
echo "== done"
