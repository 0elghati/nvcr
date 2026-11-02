#!/system/bin/sh
# On-device DCVC-RT resolution sweep for the ONNX Runtime providers (roadmap C9).
#
# Run on the phone from /data/local/tmp/nvcr, which holds the arm64 `nvcr`,
# libonnxruntime.so, libc++_shared.so and the QNN HTP libraries, plus
#   inputs/<name>.yuv   at least 33 YUV420p8 frames per case
#   models-<name>/      a fixed-shape bundle from
#                       scripts/backends/dcvcrt/fix_onnx_shapes.py
#
# Per case and provider: one warm-up plus three measured encode and decode
# runs of 33 frames (GOP 32, QP 32) at the CLI codec-loop boundary. Each result
# line records the wake state, because a dozing phone throttles the CPU and
# NPU graph preparation. CASES and PROVIDERS override the defaults, e.g.
#   CASES="qcif:176x144 cif:352x288" PROVIDERS=onnxruntime-qnn sh measurement_android_qnn.sh
cd /data/local/tmp/nvcr || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/nvcr
export ADSP_LIBRARY_PATH="/data/local/tmp/nvcr;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp"
export TMPDIR=/data/local/tmp
max_battery_temp=${MAX_BATTERY_TEMP:-340}  # tenths of a degree C

battery() { dumpsys battery | grep -E "^  $1:" | tr -dc '0-9'; }
state() {
    echo "state: low_power=$(settings get global low_power) ac=$(dumpsys battery | grep -c '^  AC powered: true') level=$(battery level) temp=$(battery temperature) $(dumpsys thermalservice | grep 'Thermal Status')"
}
cool_down() {
    waited=0
    while [ "$(battery temperature)" -gt "$max_battery_temp" ] && [ $waited -lt 900 ]; do
        sleep 15; waited=$((waited + 15))
    done
    echo "cooldown: waited ${waited}s"
}

for case in ${CASES:-qcif:176x144 cif:352x288 360p:640x360 540p:960x540 720p:1280x720 1080p:1920x1080}; do
    name=${case%%:*}; size=${case#*:}
    providers=onnxruntime-qnn
    case $name in qcif|cif|360p) providers="onnxruntime-qnn onnxruntime-cpu" ;; esac
    providers=${PROVIDERS:-$providers}
    for p in $providers; do
        cool_down; state
        for rep in warmup 1 2 3; do
            for dir in encode decode; do
                # Positional parameters keep each argument whole; plain sh has no arrays.
                if [ "$dir" = encode ]; then
                    set -- -i "inputs/$name.yuv" -o "s-$name-$p.nvcr" -s "$size"
                else
                    set -- -i "s-$name-$p.nvcr" -o "s-$name-$p.yuv"
                fi
                start=$(cut -d' ' -f1 /proc/uptime)
                ./nvcr "$dir" "$@" --frames 33 --provider "$p" --engine-dir "models-$name" > s.log 2>&1
                status=$?
                end=$(cut -d' ' -f1 /proc/uptime)
                line=$(grep -E "^(Encoded|Decoded) |^nvcr:" s.log | head -2 | tr '\n' ' ')
                wake=$(dumpsys power | grep -m1 "mWakefulness=" | tr -d ' ')
                echo "result $name $size $p $dir $rep rc=$status process_s=$(awk "BEGIN{print $end-$start}") $wake | $line"
            done
        done
        sha256sum "s-$name-$p.nvcr" "s-$name-$p.yuv" 2>/dev/null
    done
done
state
echo "sweep done"
