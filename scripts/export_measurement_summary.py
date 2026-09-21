#!/usr/bin/env python3
"""Export compact, shareable CSVs from a complete measurement campaign."""

import argparse
import csv
import json
from pathlib import Path


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = json.loads((args.campaign / "run.json").read_text())
    analysis = json.loads((args.campaign / "analysis.json").read_text())
    rd = json.loads((args.campaign / "rd-points.json").read_text())
    if analysis["status"] != "complete" or rd["schema"] != "nvcr.measurement.rd.v1":
        raise ValueError("complete campaign analysis and RD points required")

    manifest = run["manifest"]
    cases = {
        f'{s["sequence_id"]}-q{qp}-g{gop}': (s, qp, gop)
        for s in manifest["sequences"] for qp in manifest["qps"]
        for gop in manifest["gops"]
    }
    stats = []
    metrics = {
        "throughput": ("codec_seconds", "throughput_fps", "process_seconds"),
        "memory": ("process_peak_rss_mib",),
    }
    for item in analysis["aggregates"]:
        if item["mode"] == "quality":
            continue
        sequence, qp, gop = cases[item["case_id"]]
        for metric in metrics[item["mode"]]:
            value = item[metric]
            stats.append({
                "sequence": sequence["sequence_id"], "width": sequence["width"],
                "height": sequence["height"], "qp": qp, "gop": gop,
                "implementation": item["implementation"],
                "operation": item["operation"], "mode": item["mode"],
                "metric": metric, "n": value["n"], "mean": value["mean"],
                "sample_std": value["sample_std"],
                "cv_percent": 100 * value["sample_std"] / value["mean"] if value["mean"] else "",
                "minimum": value["minimum"], "maximum": value["maximum"],
            })
    stats.sort(key=lambda r: (r["sequence"], r["qp"], r["gop"],
                              r["implementation"], r["operation"], r["mode"], r["metric"]))
    expected_stats = len(cases) * len(manifest["implementations"]) * 2 * 4
    expected_rd = len(cases) * len(manifest["implementations"])
    if len(stats) != expected_stats or len(rd["points"]) != expected_rd:
        raise ValueError("aggregate or RD coverage mismatch")

    observations = []
    execution_ids = set()
    with (args.campaign / "observations.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            execution_id = item["execution_id"]
            if (item["status"] != "passed" or
                    item["fingerprint"] != run["fingerprint"] or
                    execution_id in execution_ids):
                raise ValueError("invalid or duplicate campaign observation")
            execution_ids.add(execution_id)
            timing = item["metrics"]
            size = item.get("bytes", {})
            quality = item.get("quality", {})
            observations.append({
                "execution_id": execution_id, "case_id": item["case_id"],
                "sequence": item["sequence"], "width": item["width"],
                "height": item["height"], "qp": item["qp"], "gop": item["gop"],
                "implementation": item["implementation"],
                "operation": item["operation"], "mode": item["mode"],
                "repeat": item["repeat"], "frames": item["frames"],
                "timed_frames": timing["timed_frames"],
                "warmup_frames": timing["warmup_frames"],
                "codec_seconds": timing["codec_seconds"],
                "throughput_fps": timing["throughput_fps"],
                "initialization_seconds": timing["initialization_seconds"],
                "process_seconds": item["process_seconds"],
                "process_peak_rss_mib": item.get("process_peak_rss_mib", ""),
                "file_bytes": size.get("file_bytes", ""),
                "entropy_bytes": size.get("entropy_bytes", ""),
                "file_bpp": size.get("file_bpp", ""),
                "entropy_bpp": size.get("entropy_bpp", ""),
                "pooled_psnr_yuv_db": quality.get("pooled_psnr_yuv_db", ""),
            })
    if len(observations) != run["expected_operations"]:
        raise ValueError("observation coverage mismatch")
    observations.sort(key=lambda r: (r["sequence"], r["qp"], r["gop"],
                                     r["implementation"], r["mode"],
                                     r["repeat"], r["operation"]))
    rd_rows = sorted(rd["points"], key=lambda r: (
        r["sequence"], r["gop"], r["implementation"], r["qp"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "operation-measurements.csv",
              list(observations[0]), observations)
    write_csv(args.output / "condition-statistics.csv",
              ["sequence", "width", "height", "qp", "gop", "implementation",
               "operation", "mode", "metric", "n", "mean", "sample_std",
               "cv_percent", "minimum", "maximum"], stats)
    write_csv(args.output / "rd-points.csv",
              ["sequence", "width", "height", "gop", "qp", "implementation",
               "content_id", "input_sha256", "frames", "entropy_bpp",
               "file_bpp", "psnr_yuv_db", "quality_contract", "fingerprint"],
              rd_rows)
    print(f"Exported {len(observations)} operations, {len(stats)} statistics "
          f"and {len(rd_rows)} RD points")


if __name__ == "__main__":
    main()
