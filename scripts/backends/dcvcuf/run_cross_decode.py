#!/usr/bin/env python3
"""C13 cross-decode of identical DCVC-UF bytes across execution paths and GPUs.

Three steps, each usable on its own machine:

  encode   encode the cross-decode sequences once per execution path and keep
           the exact .nvcr bytes, with a SHA-256 manifest and run metadata.
  decode   decode every stream of a manifest on one path with the production
           CLI (exit status, stderr, frames written) and with the test-only
           nvcr_dcvcuf_entropy_trace harness (every UF entropy call).
  compare  compare each decode with the reference decode of the same bytes
           (normally the encoder's own path on the encoder's GPU): first
           divergent entropy index (unit, call, position), first divergent
           symbol, completion or failure, per-frame PSNR-Y and drift, and the
           C13 pass rule: bit-exact, or a clean failure (defined NVCR error,
           frames written before it equal the reference's, none after).

Bit-exactness is reported, not required. A decode that completes with frames
different from the reference is reported with its PSNR; "silent garbage" (a
completed decode or a written frame far below the reference) fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

SEQUENCES = (
    {"name": "BasketballDrive_1920x1080_50", "path": "hd/BasketballDrive_1920x1080_50.yuv",
     "sha256": "dc89d33e4910379dca548c5a807c2c83dde471014a09a852c370bd49fb354543",
     "width": 1920, "height": 1080},
    {"name": "akiyo_qcif", "path": "qcif/akiyo_qcif.yuv",
     "sha256": "e1efee0e95c6d27aefe2294a727768334db8466e67bf4a49cf8f5b2fb8b49108",
     "width": 176, "height": 144},
)
GOP = 65528  # intra frame only at frame 0 (upstream intra_period -1); reset interval 32
# A written frame this far below the reference decode is treated as garbage.
GARBAGE_PSNR_DB = 25.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log=None, **kwargs):
    if log is not None:
        log.write("$ " + " ".join(str(part) for part in command) + "\n")
    completed = subprocess.run([str(part) for part in command], capture_output=True, text=True,
                               **kwargs)
    if log is not None:
        log.write(completed.stdout + completed.stderr + f"[exit {completed.returncode}]\n\n")
        log.flush()
    return completed


def tool_versions():
    def query(*command):
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    gpu = query("nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader")
    return {"gpu": gpu, "platform": platform.platform(), "python": platform.python_version(),
            "nvcc_default": query("nvcc", "--version")}


def git_commit(root: Path):
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True).strip())
        return {"commit": commit, "dirty_tracked_files": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def bundle_identity(bundle: Path):
    """Digests that bind a bundle: the TensorRT manifest and plans, or the .pt file."""
    bundle = Path(bundle)
    manifest = bundle / "engine_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        return {"kind": "tensorrt", "path": str(bundle),
                "engine_manifest_sha256": sha256(manifest),
                "engine_sha256_sha256": sha256(bundle / "engine.sha256"),
                **{key: data.get(key) for key in (
                    "model_profile_id", "model_profile_sha256", "engine_profile_id",
                    "device_name", "compute_capability_major", "compute_capability_minor",
                    "cuda_runtime_version", "tensorrt_version_major", "tensorrt_version_minor",
                    "tensorrt_version_patch", "builder_optimization_level")},
                "files": data.get("files")}
    return {"kind": "native", "path": str(bundle), "sha256": sha256(bundle)}


# ---------------------------------------------------------------- encode

def encode(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {"native": (args.cli_native, args.native_bundle),
             "tensorrt": (args.cli_tensorrt, args.tensorrt_bundle)}
    for extra in args.extra_tensorrt or []:
        label, bundle = extra.split("=", 1)
        paths[label] = (args.cli_tensorrt, Path(bundle))
    streams = []
    with (output / "commands.log").open("w") as log:
        for sequence in SEQUENCES:
            source = args.dataset_root / sequence["path"]
            actual = sha256(source)
            if actual != sequence["sha256"]:
                raise SystemExit(f"{source}: SHA-256 {actual} != {sequence['sha256']}")
            for label, (cli, bundle) in paths.items():
                if args.only_qcif_for_extra and label not in ("native", "tensorrt") \
                        and sequence["width"] != 176:
                    continue
                for qp in args.qp:
                    relative = Path("streams") / label / f"{sequence['name']}_qp{qp}.nvcr"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    command = [cli, "encode", "--codec", "dcvc-uf", "--engine-dir", bundle,
                               "-i", source, "-o", target,
                               "-s", f"{sequence['width']}x{sequence['height']}",
                               "--frames", args.frames, "--qp", qp, "--gop-size", GOP]
                    print(f"[encode] {label} {sequence['name']} qp{qp}", flush=True)
                    completed = run(command, log)
                    if completed.returncode != 0:
                        raise SystemExit(f"encode failed: {completed.stderr}")
                    streams.append({
                        "stream": str(relative), "sha256": sha256(target),
                        "bytes": target.stat().st_size, "encoder_path": label,
                        "sequence": sequence["name"], "width": sequence["width"],
                        "height": sequence["height"], "qp": qp, "frames": args.frames,
                        "gop_size": GOP, "source_sha256": actual,
                        "command": [str(part) for part in command],
                        "cli_output": completed.stdout.strip()})
    manifest = {
        "schema": "nvcr.dcvcuf.cross_decode_streams.v1",
        "gpu_label": args.gpu_label,
        "nvcr": git_commit(args.repository),
        "environment": tool_versions(),
        "recipe": {"frames": args.frames, "qps": args.qp, "gop_size": GOP,
                   "reset_interval": 32, "skip_threshold": 0},
        "bundles": {label: bundle_identity(bundle) for label, (_cli, bundle) in paths.items()},
        "streams": streams,
    }
    (output / "streams.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "SHA256SUMS").open("w") as sums:
        for entry in streams:
            sums.write(f"{entry['sha256']}  {entry['stream']}\n")
        sums.write(f"{sha256(output / 'streams.json')}  streams.json\n")
    print(f"[encode] wrote {len(streams)} streams to {output}")
    return 0


def verify_manifest(root: Path):
    root = Path(root)
    lines = (root / "SHA256SUMS").read_text().split("\n")
    checked = 0
    for line in filter(None, lines):
        digest, name = line.split("  ", 1)
        actual = sha256(root / name)
        if actual != digest:
            raise SystemExit(f"{root / name}: SHA-256 {actual} != {digest}")
        checked += 1
    return json.loads((root / "streams.json").read_text()), checked


# ---------------------------------------------------------------- decode

def decode(args):
    manifest, checked = verify_manifest(args.streams)
    print(f"[decode] manifest verified: {checked} files", flush=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / f"commands-{args.label}.log").open("a") as log:
        for entry in manifest["streams"]:
            if args.encoder_path and entry["encoder_path"] not in args.encoder_path:
                continue
            stream = Path(args.streams) / entry["stream"]
            case = output / args.label / Path(entry["stream"]).with_suffix("")
            case.mkdir(parents=True, exist_ok=True)
            cli_yuv = case / "cli.yuv"
            print(f"[decode] {args.label} <- {entry['stream']}", flush=True)
            cli = run([args.cli, "decode", "--codec", "dcvc-uf", "--engine-dir", args.bundle,
                       "-i", stream, "-o", cli_yuv], log)
            trace_yuv = case / "trace.yuv"
            trace = case / "trace.bin"
            harness = run([args.harness, args.bundle, stream, trace_yuv, trace], log)
            frame_bytes = entry["width"] * entry["height"] * 3 // 2
            row = {
                "decoder": args.label, "decoder_gpu": args.gpu_label,
                "stream": entry["stream"], "stream_sha256": entry["sha256"],
                "encoder_path": entry["encoder_path"], "encoder_gpu": manifest.get("gpu_label"),
                "sequence": entry["sequence"], "qp": entry["qp"], "frames": entry["frames"],
                "width": entry["width"], "height": entry["height"],
                "bundle": bundle_identity(args.bundle),
                "cli": {"exit": cli.returncode, "stderr": cli.stderr.strip()[-2000:],
                        "stdout": cli.stdout.strip()[-1000:],
                        "yuv": str(cli_yuv.relative_to(output)) if cli_yuv.exists() else None,
                        "yuv_sha256": sha256(cli_yuv) if cli_yuv.exists() else None,
                        "frames_written": (cli_yuv.stat().st_size // frame_bytes
                                           if cli_yuv.exists() else 0),
                        "partial_frame_bytes": (cli_yuv.stat().st_size % frame_bytes
                                                if cli_yuv.exists() else 0)},
                "harness": {"exit": harness.returncode, "stderr": harness.stderr.strip()[-2000:],
                            "stdout": harness.stdout.strip(),
                            "yuv": str(trace_yuv.relative_to(output)),
                            "yuv_sha256": sha256(trace_yuv) if trace_yuv.exists() else None,
                            "trace": str(trace.relative_to(output)),
                            "trace_sha256": sha256(trace) if trace.exists() else None},
            }
            row["cli_equals_harness_yuv"] = row["cli"]["yuv_sha256"] == row["harness"]["yuv_sha256"]
            rows.append(row)
            with (output / "decodes.jsonl").open("a") as jsonl:
                jsonl.write(json.dumps(row) + "\n")
    print(f"[decode] {len(rows)} decodes recorded in {output / 'decodes.jsonl'}")
    return 0


# ---------------------------------------------------------------- traces

def read_trace(path: Path):
    """Parses a nvcr_dcvcuf_entropy_trace file into a list of units."""
    data = Path(path).read_bytes()
    if data[:4] != b"UFTR" or struct.unpack_from("<I", data, 4)[0] != 1:
        raise ValueError(f"{path}: not a version-1 UF trace")
    position = 8
    units, current, end = [], None, None

    def take(fmt):
        nonlocal position
        values = struct.unpack_from(fmt, data, position)
        position += struct.calcsize(fmt)
        return values

    while position < len(data):
        (tag,) = take("<B")
        if tag == 1:
            unit, kind, size = take("<IBI")
            current = {"unit": unit, "type": "I" if kind == 0 else "P", "bytes": size,
                       "calls": [], "end_valid": None, "ok": None}
            units.append(current)
        elif tag == 2:
            call, complete, count = take("<BBQ")
            indexes = None
            if call == 1:
                indexes = np.frombuffer(data, np.uint8, count, position)
                position += count
            symbols = np.frombuffer(data, np.int8, count, position)
            position += count
            current["calls"].append({"call": "z" if call == 0 else "y", "complete": bool(complete),
                                     "indexes": indexes, "symbols": symbols})
        elif tag == 3:
            (valid,) = take("<B")
            current["end_valid"] = bool(valid)
        elif tag == 4:
            ok, frames, code, length = take("<BIII")
            message = data[position:position + length].decode("utf-8", "replace")
            position += length
            current.update(ok=bool(ok), frames=frames, error_code=code, error=message or None)
        elif tag == 5:
            units_attempted, frames = take("<IQ")
            end = {"units": units_attempted, "frames": frames}
        else:
            raise ValueError(f"{path}: unknown record tag {tag} at {position - 1}")
    return units, end


def call_name(unit, index):
    """Names the entropy call: intra z, y step 0-3; group z, y."""
    calls = unit["calls"]
    if calls[index]["call"] == "z":
        return "z"
    step = sum(1 for call in calls[:index] if call["call"] == "y")
    return f"y{step}" if unit["type"] == "I" else "y"


def compare_traces(reference, candidate):
    """First divergent entropy index and symbol between two decodes of one stream."""
    result = {"first_index_divergence": None, "first_symbol_divergence": None,
              "units_identical": 0, "index_differences_in_first_divergent_call": None}
    for ref_unit, cand_unit in zip(reference, candidate):
        identical = True
        for index, (ref_call, cand_call) in enumerate(zip(ref_unit["calls"], cand_unit["calls"])):
            where = {"unit": ref_unit["unit"], "unit_type": ref_unit["type"],
                     "call": call_name(ref_unit, index)}
            if ref_call["call"] == "y" and result["first_index_divergence"] is None:
                ref_idx, cand_idx = ref_call["indexes"], cand_call["indexes"]
                if len(ref_idx) != len(cand_idx):
                    result["first_index_divergence"] = {
                        **where, "position": None, "reason": "coded symbol count differs",
                        "reference_count": int(len(ref_idx)), "candidate_count": int(len(cand_idx))}
                else:
                    differing = np.nonzero(ref_idx != cand_idx)[0]
                    if len(differing):
                        first = int(differing[0])
                        result["first_index_divergence"] = {
                            **where, "position": first, "count": int(len(ref_idx)),
                            "reference_index": int(ref_idx[first]),
                            "candidate_index": int(cand_idx[first])}
                        result["index_differences_in_first_divergent_call"] = {
                            "differing": int(len(differing)),
                            "max_abs_difference": int(np.max(np.abs(
                                ref_idx[differing].astype(int) - cand_idx[differing].astype(int))))}
            if result["first_symbol_divergence"] is None:
                count = min(len(ref_call["symbols"]), len(cand_call["symbols"]))
                differing = np.nonzero(ref_call["symbols"][:count] != cand_call["symbols"][:count])[0]
                if len(differing) or len(ref_call["symbols"]) != len(cand_call["symbols"]) \
                        or ref_call["complete"] != cand_call["complete"]:
                    result["first_symbol_divergence"] = {
                        **where, "position": int(differing[0]) if len(differing) else None,
                        "candidate_call_complete": cand_call["complete"]}
            if not (np.array_equal(ref_call["symbols"], cand_call["symbols"]) and
                    (ref_call["indexes"] is None or
                     np.array_equal(ref_call["indexes"], cand_call["indexes"]))):
                identical = False
        if len(ref_unit["calls"]) != len(cand_unit["calls"]):
            identical = False
        if identical and result["first_index_divergence"] is None and \
                result["first_symbol_divergence"] is None:
            result["units_identical"] += 1
    return result


# ---------------------------------------------------------------- compare

def frame_psnr_y(reference: Path, candidate: Path, width, height, frames):
    """Per-frame PSNR-Y (dB; inf when equal) of candidate against reference."""
    frame_bytes = width * height * 3 // 2
    values = []
    with Path(reference).open("rb") as ref, Path(candidate).open("rb") as cand:
        for _ in range(frames):
            a = ref.read(frame_bytes)
            b = cand.read(frame_bytes)
            if len(a) < frame_bytes or len(b) < frame_bytes:
                break
            ya = np.frombuffer(a, np.uint8, width * height).astype(np.int32)
            yb = np.frombuffer(b, np.uint8, width * height).astype(np.int32)
            mse = float(np.mean((ya - yb) ** 2))
            values.append(math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse))
    return values


def source_psnr_y(source: Path, decoded: Path, width, height, frames):
    return frame_psnr_y(source, decoded, width, height, frames)


def compare(args):
    rows = []
    for path in args.decodes:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    roots = {}
    for path in args.decodes:
        for line in Path(path).read_text().splitlines():
            if line:
                row = json.loads(line)
                roots[(row["decoder"], row["decoder_gpu"], row["stream_sha256"])] = Path(path).parent
    by_stream = {}
    for row in rows:
        by_stream.setdefault(row["stream_sha256"], []).append(row)
    results = []
    for digest, decodes in by_stream.items():
        reference = next((row for row in decodes
                          if row["decoder"] == row["encoder_path"] and
                          row["decoder_gpu"] == row["encoder_gpu"]), None)
        for row in decodes:
            root = roots[(row["decoder"], row["decoder_gpu"], digest)]
            entry = {key: row[key] for key in ("stream", "stream_sha256", "sequence", "qp",
                                               "encoder_path", "encoder_gpu", "decoder",
                                               "decoder_gpu", "frames")}
            entry["cli_exit"] = row["cli"]["exit"]
            entry["cli_error"] = row["cli"]["stderr"] or None
            entry["frames_written"] = row["cli"]["frames_written"]
            entry["partial_frame_bytes"] = row["cli"]["partial_frame_bytes"]
            entry["cli_equals_harness_yuv"] = row["cli_equals_harness_yuv"]
            entry["yuv_sha256"] = row["cli"]["yuv_sha256"]
            units, end = read_trace(root / row["harness"]["trace"])
            failed = next((unit for unit in units if unit["ok"] is False), None)
            entry["completed"] = row["cli"]["exit"] == 0 and failed is None
            entry["failure"] = None if failed is None else {
                "unit": failed["unit"], "unit_type": failed["type"],
                "error_code": failed["error_code"], "error": failed["error"],
                "entropy_end_valid": failed["end_valid"],
                "failed_call": next((call_name(failed, index)
                                     for index, call in enumerate(failed["calls"])
                                     if not call["complete"]), None)}
            # The CLI reports an NVCR error as one "nvcr: ..." line (a provider
            # log line may precede it) and exits 1; a crash or abort does not.
            entry["defined_error"] = (row["cli"]["exit"] == 1 and any(
                line.startswith("nvcr: ") for line in row["cli"]["stderr"].splitlines()))
            if args.dataset_root is not None and row["cli"]["yuv"]:
                sequence = next(s for s in SEQUENCES if s["name"] == row["sequence"])
                values = source_psnr_y(args.dataset_root / sequence["path"], root / row["cli"]["yuv"],
                                       row["width"], row["height"], row["cli"]["frames_written"])
                entry["psnr_y_vs_source_mean"] = (sum(values) / len(values)) if values else None
            if reference is None:
                entry["reference"] = None
                results.append(entry)
                continue
            ref_root = roots[(reference["decoder"], reference["decoder_gpu"], digest)]
            entry["reference"] = f"{reference['decoder']}@{reference['decoder_gpu']}"
            ref_units, _ = read_trace(ref_root / reference["harness"]["trace"])
            entry["entropy"] = compare_traces(ref_units, units)
            psnr = []
            if row["cli"]["yuv"] and reference["cli"]["yuv"]:
                psnr = frame_psnr_y(ref_root / reference["cli"]["yuv"], root / row["cli"]["yuv"],
                                    row["width"], row["height"], row["cli"]["frames_written"])
            finite = [value for value in psnr if math.isfinite(value)]
            entry["bit_exact"] = (entry["completed"] and
                                  row["cli"]["yuv_sha256"] == reference["cli"]["yuv_sha256"])
            entry["frames_identical_to_reference"] = sum(1 for v in psnr if not math.isfinite(v))
            entry["psnr_y_vs_reference"] = {
                "min": min(finite) if finite else None,
                "mean_finite": sum(finite) / len(finite) if finite else None,
                "first_nonidentical_frame": next((i for i, v in enumerate(psnr)
                                                  if math.isfinite(v)), None),
                "by_frame": [None if not math.isfinite(v) else round(v, 3) for v in psnr]}
            garbage = [i for i, v in enumerate(psnr) if v < GARBAGE_PSNR_DB]
            entry["garbage_frames"] = garbage
            if entry["bit_exact"]:
                verdict = "bit-exact"
            elif entry["completed"]:
                verdict = "FAIL silent garbage" if garbage else "completed, not bit-exact"
            elif entry["defined_error"] and not garbage and entry["partial_frame_bytes"] == 0:
                verdict = "clean failure"
            else:
                verdict = "FAIL unclean failure"
            entry["verdict"] = verdict
            results.append(entry)
    args.output.write_text("".join(json.dumps(entry) + "\n" for entry in results))
    for entry in results:
        divergence = (entry.get("entropy") or {}).get("first_index_divergence")
        print(f"{entry['encoder_path']}@{entry['encoder_gpu']} -> {entry['decoder']}@"
              f"{entry['decoder_gpu']} {entry['sequence']} qp{entry['qp']}: "
              f"{entry.get('verdict')} written={entry['frames_written']} "
              f"first_index_divergence={divergence}")
    return 0 if all(not str(e.get("verdict", "")).startswith("FAIL") for e in results) else 1


def compare_upstream(args):
    """G4: pinned upstream decodes (decode_with_upstream.py) against NVCR's reference."""
    references = {}
    for path in args.decodes:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line) if line else None
            if row and row["decoder"] == row["encoder_path"] and \
                    row["decoder_gpu"] == row["encoder_gpu"]:
                references[row["stream_sha256"]] = (row, Path(path).parent)
    upstream_root = args.upstream.parent
    results = []
    for line in args.upstream.read_text().splitlines():
        row = json.loads(line)
        entry = {key: row[key] for key in ("stream", "stream_sha256", "sequence", "qp",
                                           "encoder_path", "encoder_gpu", "exit", "signal",
                                           "timed_out", "frames_written", "yuv_sha256")}
        entry["decoder"] = "upstream"
        sequence = next(s for s in SEQUENCES if s["name"] == row["sequence"])
        size = (row["width"], row["height"], row["frames_written"])
        decoded = upstream_root / row["yuv"] if row["yuv"] else None
        reference = references.get(row["stream_sha256"])
        entry["nvcr_reference"] = None
        if reference and decoded:
            ref_row, ref_root = reference
            entry["nvcr_reference"] = f"{ref_row['decoder']}@{ref_row['decoder_gpu']}"
            entry["nvcr_reference_completed"] = ref_row["cli"]["exit"] == 0
            entry["bit_exact_with_nvcr"] = (ref_row["cli"]["exit"] == 0 and
                                            row["yuv_sha256"] == ref_row["cli"]["yuv_sha256"])
            if ref_row["cli"]["yuv"] and ref_row["cli"]["frames_written"]:
                psnr = frame_psnr_y(ref_root / ref_row["cli"]["yuv"], decoded, *size)
                finite = [v for v in psnr if math.isfinite(v)]
                entry["psnr_y_vs_nvcr"] = {
                    "frames_compared": len(psnr),
                    "identical_frames": len(psnr) - len(finite),
                    "min": min(finite) if finite else None,
                    "by_frame": [None if not math.isfinite(v) else round(v, 3) for v in psnr]}
        if args.dataset_root is not None and decoded:
            values = source_psnr_y(args.dataset_root / sequence["path"], decoded, *size)
            entry["psnr_y_vs_source"] = {
                "mean": sum(values) / len(values) if values else None,
                "first_8": [round(v, 2) for v in values[:8]],
                "last": round(values[-1], 2) if values else None}
        results.append(entry)
        print(f"upstream <- {row['stream']}: exit={row['exit']} frames={row['frames_written']} "
              f"bit_exact_with_nvcr={entry.get('bit_exact_with_nvcr')} "
              f"vs_source_mean={(entry.get('psnr_y_vs_source') or {}).get('mean')}")
    args.output.write_text("".join(json.dumps(entry) + "\n" for entry in results))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--cli-native", type=Path, required=True)
    e.add_argument("--cli-tensorrt", type=Path, required=True)
    e.add_argument("--native-bundle", type=Path, required=True)
    e.add_argument("--tensorrt-bundle", type=Path, required=True)
    e.add_argument("--extra-tensorrt", action="append",
                   help="LABEL=BUNDLE: an additional TensorRT bundle (supplementary rows)")
    e.add_argument("--only-qcif-for-extra", action="store_true")
    e.add_argument("--dataset-root", type=Path, required=True)
    e.add_argument("--frames", type=int, default=100)
    e.add_argument("--qp", type=int, nargs="+", default=[21, 42])
    e.add_argument("--gpu-label", required=True, help="for example rtx4070")
    e.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    e.add_argument("--output-dir", type=Path, required=True)
    d = sub.add_parser("decode")
    d.add_argument("--streams", type=Path, required=True, help="directory with streams.json")
    d.add_argument("--label", required=True, help="decoder path label, for example tensorrt")
    d.add_argument("--cli", type=Path, required=True)
    d.add_argument("--harness", type=Path, required=True)
    d.add_argument("--bundle", type=Path, required=True)
    d.add_argument("--gpu-label", required=True)
    d.add_argument("--encoder-path", action="append", help="restrict to these encoder paths")
    d.add_argument("--output-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--decodes", type=Path, nargs="+", required=True)
    c.add_argument("--dataset-root", type=Path)
    c.add_argument("--output", type=Path, required=True)
    u = sub.add_parser("compare-upstream")
    u.add_argument("--upstream", type=Path, required=True, help="upstream.jsonl")
    u.add_argument("--decodes", type=Path, nargs="+", required=True)
    u.add_argument("--dataset-root", type=Path)
    u.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return {"encode": encode, "decode": decode, "compare": compare,
            "compare-upstream": compare_upstream}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
