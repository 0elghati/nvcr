#!/system/bin/sh
# MLVC on-device checks. Run from anywhere on the phone; everything lives in
# /data/local/tmp/nvcr-mlvc and the full log goes to out/log.txt.
cd /data/local/tmp/nvcr-mlvc || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nvcr-mlvc
export ADSP_LIBRARY_PATH="/data/local/tmp/nvcr-mlvc;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp"
rm -rf out && mkdir -p out
exec > out/log.txt 2>&1 < /dev/null

show() { grep -E "Encoded|Decoded|Quality|MLVC .* per frame|nvcr:|error|Error"; }
B=bundles/qualcomm

encode() { # tag provider clip size frames [env...]
  tag=$1 provider=$2 clip=$3 size=$4 frames=$5; shift 5
  echo "== encode $tag $size"
  env "$@" ./nvcr encode -v -i "data/$clip" -o "out/${tag}_$size.nvcr" -s "$size" -r 30 \
    --frames "$frames" --qp 32 --codec mlvc --provider "$provider" --engine-dir "$B" 2>&1 | show
}
decode() { # stream outname provider clip [env...]
  stream=$1 name=$2 provider=$3 clip=$4; shift 4
  echo "== decode $stream -> $name ($provider)"
  env "$@" ./nvcr decode -v -i "$stream" -o "out/$name.yuv" --codec mlvc --provider "$provider" \
    --engine-dir "$B" --quality-metrics "data/$clip" 2>&1 | show
}

echo "== device"
getprop ro.product.model; getprop ro.soc.model; getprop ro.build.version.release
echo "== unit"
./nvcr_mlvc_unit_tests 2>&1 | tail -3

echo "== parity vs x86 upstream reference on the phone CPU"
./nvcr_mlvc_parity_tests onnxruntime-cpu bundles/generic golden/akiyo-320x192-q32 data/akiyo_qcif.yuv 176 144 66 model 2>&1 | grep -v "^\[nvcr"
./nvcr_mlvc_parity_tests onnxruntime-cpu bundles/qualcomm golden/akiyo-320x192-q32-qualcomm data/akiyo_qcif.yuv 176 144 66 model 2>&1 | grep -v "^\[nvcr"

for clip in akiyo_qcif.yuv:176x144:66 basketball_640x360_65.yuv:640x360:65; do
  c=${clip%%:*} rest=${clip#*:}; s=${rest%%:*} n=${rest#*:}
  encode cpu onnxruntime-cpu "$c" "$s" "$n"
  encode qnn onnxruntime-qnn "$c" "$s" "$n"
  decode "out/cpu_$s.nvcr" "cpu_$s" onnxruntime-cpu "$c"
  decode "out/qnn_$s.nvcr" "qnn_$s" onnxruntime-qnn "$c"
  # Cross-decodes show drift between providers.
  decode "out/qnn_$s.nvcr" "qnn_enc_cpu_dec_$s" onnxruntime-cpu "$c"
  decode "x86/cpu_$s.nvcr" "x86_enc_cpu_dec_$s" onnxruntime-cpu "$c"
  decode "x86/cpu_$s.nvcr" "x86_enc_qnn_dec_$s" onnxruntime-qnn "$c"
done

echo "== hashes (phone)"
(cd out && sha256sum ./*.nvcr ./*.yuv)
echo "== x86 reference hashes"
cat x86/SHA256SUMS
echo "== phone CPU vs x86 CPU (expect identical)"
for s in 176x144 640x360; do
  cmp "out/cpu_$s.nvcr" "x86/cpu_$s.nvcr" && echo "stream $s identical"
  cmp "out/x86_enc_cpu_dec_$s.yuv" "x86/cpu_$s.yuv" && echo "decode $s identical"
done
echo "== done"
