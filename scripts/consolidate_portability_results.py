#!/usr/bin/env python3
"""Consolidate NVCR-only Docker portability diagnostics; never relabel common metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from measurement_metrics import condition_statistics, digest, write_json

PROFILES = {"qcif": "176x144", "cif": "352x288", "360p": "640x360",
            "540p": "960x540", "720p": "1280x720", "1080p": "1920x1080"}
QPS = (0, 21, 42, 63)
GOPS = (1, 30, 100)


def consolidate(rows, image):
    labels = image["Config"]["Labels"]
    revision = labels["org.opencontainers.image.revision"]
    image_ref, = image["RepoDigests"]
    groups = defaultdict(list)
    for row in rows:
        if row["schema"] != "nvcr.benchmark.resolution-matrix.v1":
            raise ValueError("unsupported diagnostic schema")
        if row["container_digest"] != image_ref or row["execution_mode"] != "docker":
            raise ValueError("mixed image or execution identities")
        if row["run_index"] == "average":
            continue
        if row["size"] != PROFILES.get(row["resolution"]):
            raise ValueError("unexpected input dimensions")
        if row["frames"] != 100 or row["warmup_frames"] != 10 or row["runs_planned"] != 10:
            raise ValueError("unexpected frame or repetition contract")
        if row["memory_sampler"] != "none" or any(row[k] is not None for k in
                ("peak_memory_mb", "peak_gpu_memory_mb", "peak_system_memory_mb")):
            raise ValueError("memory-instrumented data cannot enter this cohort")
        for field in ("codec_time_seconds", "throughput_fps", "payload_bpp"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {field}")
        if abs(row["payload_bpp"] - row["payload_bytes"] * 8 /
               (math.prod(map(int, row["size"].split("x"))) * 100)) > 5.1e-10:
            raise ValueError("payload BPP mismatch")
        # CLI FPS and duration are separately rounded; verify within that precision.
        if abs(row["codec_time_seconds"] - 100 / row["throughput_fps"]) > 0.000501 + 100 * 0.0005 / (row["throughput_fps"] * (row["throughput_fps"] - 0.0005)):
            raise ValueError("codec FPS/duration mismatch")
        groups[(row["resolution"], row["qp"], row["gop_size"], row["operation"])].append(row)
    expected = {(p, q, g, op) for p in PROFILES for q in QPS for g in GOPS
                for op in ("encode", "decode")}
    if set(groups) != expected:
        raise ValueError("incomplete or unexpected condition matrix")
    result = []
    for key in sorted(groups, key=lambda k: (list(PROFILES).index(k[0]), *k[1:])):
        values = sorted(groups[key], key=lambda r: int(r["run_index"]))
        if [int(r["run_index"]) for r in values] != list(range(1, 11)):
            raise ValueError(f"missing or duplicate repetition: {key}")
        fps = [r["throughput_fps"] for r in values]
        psnr = [r["psnr_yuv"] for r in values]
        if key[3] == "decode" and any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in psnr):
            raise ValueError("missing finite decode diagnostic quality")
        if key[3] == "encode" and any(v is not None for v in psnr):
            raise ValueError("unexpected encode quality")
        result.append({
            "schema": "nvcr.portability.condition.v1",
            "hardware": values[0]["hardware"], "resolution": key[0], "size": values[0]["size"],
            "qp": key[1], "gop_size": key[2], "operation": key[3],
            "frames": 100, "warmup_frames": 10, "execution_mode": "docker",
            "container_digest": image_ref, "nvcr_version": labels["org.opencontainers.image.version"],
            "nvcr_commit": revision,
            "codec_fps": fps, "codec_time_seconds": [r["codec_time_seconds"] for r in values],
            "payload_bytes": [r["payload_bytes"] for r in values],
            "payload_bpp": [r["payload_bpp"] for r in values],
            "diagnostic_psnr_yuv_db": psnr,
            "codec_fps_statistics": condition_statistics(fps),
        })
    return result


def smoke_check(rows, image_ref):
    raw = [r for r in rows if r["run_index"] != "average"]
    expected = {(p, g, op) for p in PROFILES for g in (1, 3) for op in ("encode", "decode")}
    keys = [(r["resolution"], r["gop_size"], r["operation"]) for r in raw]
    if len(keys) != 24 or set(keys) != expected:
        raise ValueError("incomplete smoke matrix")
    for row in raw:
        if row["frames"] != 3 or row["qp"] != 0 or row["container_digest"] != image_ref:
            raise ValueError("smoke contract or image mismatch")
        if row["throughput_fps"] <= 0 or row["payload_bytes"] <= 0:
            raise ValueError("invalid smoke result")
        if row["operation"] == "decode" and not math.isfinite(row["psnr_yuv"]):
            raise ValueError("invalid smoke quality")
    return {"status": "complete", "measured_operations": 24, "frames": 3,
            "qps": [0], "gops": [1, 3], "profiles": list(PROFILES),
            "warmup_frames": 2, "warmup_method": "separate encode/decode processes",
            "same_session_reset_equivalence": "not tested"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def export(source, output):
    image, = json.loads((source / "image-inspect.json").read_text(encoding="utf-8-sig"))
    paths = [source / "throughput" / f"qp-{q}" / "results.jsonl" for q in QPS]
    rows = [r for p in paths for r in read_rows(p)]
    conditions = consolidate(rows, image)
    smoke = smoke_check(read_rows(source / "smoke/results.jsonl"), image["RepoDigests"][0])
    artifact_checks = {p: json.loads((source / f"artifact-{p}.json").read_text(encoding="utf-8-sig")) for p in PROFILES}
    if any(v.get("valid") is not True for v in artifact_checks.values()):
        raise ValueError("recorded artifact validation failed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in conditions), encoding="utf-8")
    fields = ["resolution", "size", "qp", "gop_size", "operation", "n", "mean", "median",
              "sample_std", "minimum", "maximum", "cv_percent", "ci95_low", "ci95_high"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in conditions:
            writer.writerow({k: r[k] for k in fields[:5]} | r["codec_fps_statistics"])
    write_json(output / "smoke.json", smoke)
    source_files = paths + [source / "smoke/results.jsonl", source / "image-inspect.json",
        source / "docker-version.txt", source / "wsl-status.txt"] + [
        source / f"artifact-{p}.json" for p in PROFILES]
    write_json(output / "provenance.json", {
        "schema": "nvcr.portability.provenance.v1", "status": "complete",
        "measured_operations": 1440, "conditions": 144, "repetitions": 10,
        "binary": {"version": image["Config"]["Labels"]["org.opencontainers.image.version"],
                   "source_revision": image["Config"]["Labels"]["org.opencontainers.image.revision"],
                   "container_digest": image["RepoDigests"][0]},
        "harness": {"recorded_commits": sorted({r["nvcr_commit"] for r in rows}),
                    "recorded_dirty_values": sorted({r["nvcr_dirty"] for r in rows})},
        "provenance_correction": "Original nvcr_commit/nvcr_dirty were supplied by the mounted harness checkout. "
            "Binary revision now comes from the recorded OCI label; binary dirty state is not inferred. "
            "Original files are unchanged; hashes bind this correction to the source records.",
        "source_sha256": {p.relative_to(source).as_posix(): digest(p) for p in source_files},
        "artifact_validation": artifact_checks,
        "artifact_digest_at_execution": "not captured in the supplied records",
        "input_digest_at_execution": "not captured in the supplied records",
        "metrics": {"throughput": "CLI-reported codec-loop FPS; diagnostic interval",
                    "quality": "CLI diagnostic PSNR-YUV; not the common offline quality contract",
                    "rate": "serialized NVAU payload BPP; not entropy-only or complete-file BPP",
                    "memory": "not measured", "process_throughput": "not retained in source JSONL",
                    "warmup": "separate-process warm-up; not same-session reset"},
    })
    lines = ["# RTX 5060 Laptop portability — NVCR 1.0.7", "",
        "Published Linux/amd64 image executed through Windows Docker Desktop/WSL2. "
        "These are NVCR-only deployment diagnostics, not matched Python comparisons or native Windows inference.", "",
        "- Six resolutions; QPs 0/21/42/63; GOPs 1/30/100; 100 frames.",
        "- Ten measured repetitions per encode/decode condition: 144 conditions, 1,440 measured operations.",
        "- All 24 measured smoke operations are present. Separate-process warm-up does not test reset equivalence.",
        "- Memory profiling was disabled. Process throughput was not retained in the diagnostic JSONL.",
        "- Image: `" + image["RepoDigests"][0] + "`.",
        "- Binary release/revision are from OCI metadata; the original dirty flag describes the local harness.",
        "", "## Codec throughput", "",
        "Mean ± sample SD over ten independent process executions. Individual rounded codec FPS values "
        "are retained in results.jsonl. CI95 fields in results.csv are descriptive Student-t intervals; "
        "they do not establish independence/stationarity or compare devices.", "",
        "| Resolution | QP | GOP | Encode FPS | Decode FPS |",
        "|---|---:|---:|---:|---:|"]
    lookup = {(r["resolution"], r["qp"], r["gop_size"], r["operation"]): r for r in conditions}
    for p in PROFILES:
        for q in QPS:
            for g in GOPS:
                stats = [lookup[p, q, g, op]["codec_fps_statistics"] for op in ("encode", "decode")]
                lines.append(f"| {p} | {q} | {g} | " + " | ".join(
                    f"{s['mean']:.3f} ± {s['sample_std']:.3f}" for s in stats) + " |")
    lines += ["", "## Reproduction and limits", "",
        "Run `python3 scripts/consolidate_portability_results.py --source <original-campaign-directory> "
        "--output results/rtx5060`. The original source files remain local; provenance.json records their hashes. "
        "The compact condition observations retain all repetitions needed to regenerate the statistics.", "",
        "The prior QP32/two-repetition result is superseded. These four QPs contain no QP32 observation; "
        "do not reuse the earlier Figure 5 values as new measurements. The paper awaits RTX 3050 integration.", "",
        "The recorded bundle checks passed, but at-execution engine/input digests and full operation logs "
        "were not supplied. No warning-free, common-quality, memory-efficiency or cross-device speed claim is made."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(conditions), "measured_operations": 1440, "smoke_operations": 24}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
