#!/usr/bin/env python3
"""Diagnostic native-vs-exact FPS A/B for DCVC-RT (issue #216).

Not the docs/performance.md protocol. Each observation is one independent nvcr
process; the script records process wall time (spawn to wait4), ru_maxrss, and
the CLI's measurement JSON (codec interval, per-frame synchronized). Native and
exact runs alternate order every repetition, decode and encode alike.

  python3 exact_fps_ab.py OUTPUT_DIR --nvcr build-release/cli/nvcr \\
      --engine-dir ENGINES/1080p --input BasketballDrive_1920x1080_50.yuv

Writes OUTPUT_DIR/observations.jsonl and prints per-condition statistics
(n, mean, median, sample SD, min, max, CV, Student-t 95% interval).
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
        10: 2.262, 11: 2.228, 12: 2.201, 15: 2.145, 20: 2.093, 30: 2.045}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("output", type=Path)
parser.add_argument("--nvcr", required=True)
parser.add_argument("--engine-dir", required=True, help="TensorRT bundle with normative/ programs")
parser.add_argument("--input", required=True, help="YUV420P8 input")
parser.add_argument("--size", default="1920x1080")
parser.add_argument("--frames", type=int, default=97)
parser.add_argument("--gop", type=int, default=32)
parser.add_argument("--qp", type=int, default=32)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--repetitions", type=int, default=10)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
MODES = {"native": [], "exact": ["--numerics", "nvcr.det-int.v1"]}


def run(command):
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, usage = os.wait4(process.pid, 0)
    seconds = time.perf_counter() - started
    assert process.stderr is not None
    stderr = process.stderr.read().decode(errors="replace")
    process.stderr.close()
    code = os.waitstatus_to_exitcode(status)
    if code != 0:
        raise SystemExit(f"failed ({code}): {' '.join(command)}\n{stderr}")
    return seconds, usage.ru_maxrss / 1024.0


def encode_command(mode, stream, measurement=None):
    command = [args.nvcr, "encode", "-i", args.input, "-o", str(stream), "-s", args.size,
               "--frames", str(args.frames), "--gop-size", str(args.gop), "--qp", str(args.qp),
               "--provider", "tensorrt", "--engine-dir", args.engine_dir, *MODES[mode]]
    if measurement:
        command += ["--measurement-json", str(measurement), "--warmup-frames", str(args.warmup)]
    return command


def decode_command(stream, output, measurement):
    return [args.nvcr, "decode", "-i", str(stream), "-o", str(output), "--frames", str(args.frames),
            "--provider", "tensorrt", "--engine-dir", args.engine_dir,
            "--measurement-json", str(measurement), "--warmup-frames", str(args.warmup)]


def describe(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else 0.0
    return (f"n={n} mean={mean:.2f} median={statistics.median(values):.2f} sd={sd:.3f} "
            f"min={min(values):.2f} max={max(values):.2f} cv={100 * sd / mean:.2f}% "
            f"ci95=[{mean - half:.2f}, {mean + half:.2f}]")


streams = {}
for mode in MODES:
    streams[mode] = args.output / f"{mode}.nvcr"
    run(encode_command(mode, streams[mode]))

records = []
for repetition in range(args.repetitions):
    order = list(MODES) if repetition % 2 == 0 else list(reversed(MODES))
    for operation in ("decode", "encode"):
        for mode in order:
            measurement = args.output / f"{operation}-{mode}-{repetition}.json"
            if operation == "decode":
                command = decode_command(streams[mode], args.output / "decoded.yuv", measurement)
            else:
                command = encode_command(mode, args.output / "timed.nvcr", measurement)
            seconds, rss = run(command)
            data = json.loads(measurement.read_text())
            records.append({"operation": operation, "mode": mode, "repetition": repetition,
                            "process_seconds": seconds, "process_fps": args.frames / seconds,
                            "max_rss_mib": rss, **data})
            print(f"{operation:6s} {mode:6s} rep {repetition}: process {args.frames / seconds:6.2f} fps, "
                  f"codec {data['throughput_fps']:6.2f} fps", flush=True)
(args.output / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

for operation in ("decode", "encode"):
    for mode in MODES:
        group = [r for r in records if r["operation"] == operation and r["mode"] == mode]
        print(f"{operation} {mode} process_fps {describe([r['process_fps'] for r in group])}")
        print(f"{operation} {mode} codec_fps   {describe([r['throughput_fps'] for r in group])}")
print("stream bytes:", {mode: stream.stat().st_size for mode, stream in streams.items()})
