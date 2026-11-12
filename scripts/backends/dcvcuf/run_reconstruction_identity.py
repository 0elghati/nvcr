#!/usr/bin/env python3
"""C13 G1: DCVC-UF encoder-internal reconstruction equals decoder output.

For every path x sequence x QP, runs ``nvcr_dcvcuf_reconstruction_identity g1``
(which pairs each encoder step with the decoder step that consumes its access
unit and compares tensors and frames byte for byte), then encodes and decodes
the same input with the production CLI. The harness stream and decoded YUV must
equal the CLI's byte for byte, so the diagnostic hook is shown not to change
coding. Optionally runs the #194 partly-filled-group lifecycle check per path.

Writes ``cases.jsonl`` (one line per case), ``lifecycle.jsonl`` and
``summary.json`` into a new output directory. Decoded YUV files are hashed and
deleted; streams are kept.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

GOP = 65528  # intra only at frame 0 (upstream intra_period -1), as in C11 G3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log):
    started = time.perf_counter()
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    log.write(f"$ {' '.join(str(part) for part in command)}\n{result.stdout}{result.stderr}\n")
    log.flush()
    return result, time.perf_counter() - started


def gpu_state():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True)
    return query.stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="C13 G1: DCVC-UF encoder reconstruction equals decoder output")
    parser.add_argument("--harness-native", type=Path)
    parser.add_argument("--harness-tensorrt", type=Path)
    parser.add_argument("--cli-native", type=Path)
    parser.add_argument("--cli-tensorrt", type=Path)
    parser.add_argument("--native-bundle", type=Path)
    parser.add_argument("--tensorrt-bundle", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, default=Path(__file__).with_name("c8_reference_sequences.json"))
    parser.add_argument("--only", nargs="*", help="sequence names to run (default: all)")
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 21, 42, 63])
    parser.add_argument("--paths", nargs="+", choices=("native", "tensorrt"), default=["native", "tensorrt"])
    parser.add_argument("--frames", type=int, help="override the frame count (smoke runs)")
    parser.add_argument("--lifecycle-qcif", type=Path, help="176x144 input for the #194 check")
    parser.add_argument("--lifecycle-tensorrt-bundle", type=Path,
                        help="TensorRT bundle whose profile admits 176x144 (default: --tensorrt-bundle)")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tools = {
        "native": (args.harness_native, args.cli_native, args.native_bundle),
        "tensorrt": (args.harness_tensorrt, args.cli_tensorrt, args.tensorrt_bundle),
    }
    for path in args.paths:
        if not all(tools[path]):
            sys.exit(f"--harness-{path}, --cli-{path} and --{path}-bundle are required")
    sequences = json.loads(args.sequences.read_text())["sequences"]
    if args.only:
        unknown = set(args.only) - {sequence["name"] for sequence in sequences}
        if unknown:
            sys.exit(f"unknown sequences: {sorted(unknown)}")
        sequences = [sequence for sequence in sequences if sequence["name"] in args.only]
    if not sequences or not args.qps:
        sys.exit("no cases selected")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    log = open(args.output_dir / "commands.log", "w")
    cases = open(args.output_dir / "cases.jsonl", "w")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gpu_before = gpu_state()

    input_digests = {}
    for sequence in sequences:
        source = args.dataset_root / sequence["path"]
        input_digests[sequence["name"]] = sha256(source)
        if input_digests[sequence["name"]] != sequence["sha256"]:
            sys.exit(f"{source}: SHA-256 differs from {args.sequences}")

    results = []
    for path in args.paths:
        harness, cli, bundle = tools[path]
        for sequence in sequences:
            source = args.dataset_root / sequence["path"]
            frames = args.frames or sequence["frames"]
            size = f"{sequence['width']}x{sequence['height']}"
            for qp in args.qps:
                work = args.output_dir / path / sequence["name"] / f"qp{qp}"
                work.mkdir(parents=True)
                g1, g1_seconds = run(
                    [harness, "g1", bundle, source, sequence["width"], sequence["height"], frames, qp,
                     GOP, work / "harness.nvcr", work / "harness.yuv", work / "g1.json"], log)
                common = ["--codec", "dcvc-uf", "--engine-dir", bundle]
                encoded, encode_seconds = run(
                    [cli, "encode", *common, "-i", source, "-o", work / "cli.nvcr", "-s", size,
                     "--frames", frames, "--qp", qp, "--gop-size", GOP], log)
                decoded, decode_seconds = run(
                    [cli, "decode", *common, "-i", work / "cli.nvcr", "-o", work / "cli.yuv"], log)
                record = {
                    "path": path, "sequence": sequence["name"], "qp": qp, "frames": frames,
                    "harness_exit": g1.returncode, "cli_encode_exit": encoded.returncode,
                    "cli_decode_exit": decoded.returncode,
                    "seconds": {"harness": round(g1_seconds, 2), "cli_encode": round(encode_seconds, 2),
                                "cli_decode": round(decode_seconds, 2)},
                }
                if (work / "g1.json").exists():
                    record["g1"] = json.loads((work / "g1.json").read_text())
                digests = {}
                for name in ("harness.nvcr", "harness.yuv", "cli.nvcr", "cli.yuv"):
                    if (work / name).exists():
                        digests[name] = sha256(work / name)
                record["sha256"] = digests
                record["stream_equals_cli"] = (
                    "harness.nvcr" in digests and digests.get("harness.nvcr") == digests.get("cli.nvcr"))
                record["decoded_equals_cli"] = (
                    "harness.yuv" in digests and digests.get("harness.yuv") == digests.get("cli.yuv"))
                record["pass"] = bool(
                    g1.returncode == 0 and record.get("g1", {}).get("pass")
                    and record["g1"]["frames_decoded"] == frames
                    and record["stream_equals_cli"] and record["decoded_equals_cli"])
                for name in ("harness.yuv", "cli.yuv"):
                    (work / name).unlink(missing_ok=True)
                cases.write(json.dumps(record) + "\n")
                cases.flush()
                results.append(record)
                print(f"{'PASS' if record['pass'] else 'FAIL'} {path} {sequence['name']} qp{qp} "
                      f"steps={record.get('g1', {}).get('steps')} "
                      f"resets={record.get('g1', {}).get('feature_memory_resets')} "
                      f"tail={record.get('g1', {}).get('last_group_frames')} "
                      f"cli_stream={record['stream_equals_cli']} cli_yuv={record['decoded_equals_cli']}",
                      flush=True)

    lifecycle = []
    if args.lifecycle_qcif:
        with open(args.output_dir / "lifecycle.jsonl", "w") as handle:
            for path in args.paths:
                harness, _, bundle = tools[path]
                if path == "tensorrt" and args.lifecycle_tensorrt_bundle:
                    bundle = args.lifecycle_tensorrt_bundle
                report = args.output_dir / f"lifecycle-{path}.json"
                result, seconds = run([harness, "partial-group", bundle, args.lifecycle_qcif, 176, 144,
                                       32, report], log)
                record = {"path": path, "bundle": str(bundle), "exit": result.returncode,
                          "seconds": round(seconds, 2),
                          "input_sha256": sha256(args.lifecycle_qcif)}
                if report.exists():
                    record["report"] = json.loads(report.read_text())
                record["pass"] = result.returncode == 0 and record.get("report", {}).get("pass", False)
                handle.write(json.dumps(record) + "\n")
                lifecycle.append(record)
                print(f"{'PASS' if record['pass'] else 'FAIL'} lifecycle {path}", flush=True)

    summary = {
        "schema": "nvcr.dcvcuf.g1-summary.v1",
        "started": started,
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_before": gpu_before,
        "gpu_after": gpu_state(),
        "gop": GOP,
        "input_sha256": input_digests,
        "bundles": {path: str(tools[path][2]) for path in args.paths},
        "cases": len(results),
        "cases_passed": sum(record["pass"] for record in results),
        "frames_compared": sum(record.get("g1", {}).get("frames_compared", 0) for record in results),
        "tensors_compared": sum(record.get("g1", {}).get("tensors_compared", 0) for record in results),
        "lifecycle_passed": [record["path"] for record in lifecycle if record["pass"]],
        "lifecycle_failed": [record["path"] for record in lifecycle if not record["pass"]],
        "pass": bool(results) and all(record["pass"] for record in results)
        and all(record["pass"] for record in lifecycle),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
