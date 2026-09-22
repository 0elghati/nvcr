#!/usr/bin/env python3
"""Export compact, shareable measurements from a complete campaign."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from measurement_metrics import condition_statistics


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_repeat_campaign(base, repeat):
    if repeat["manifest"]["modes"] != ["throughput"]:
        raise ValueError("repeat campaign must contain throughput mode only")
    for key in ("schema", "implementations", "execution", "target_profile",
                "model_profile", "nvcr", "build_dir", "engine_root",
                "reference_root", "reference_python", "repetitions",
                "warmup_frames", "reference_reset_interval"):
        if repeat["manifest"][key] != base["manifest"][key]:
            raise ValueError(f"repeat campaign differs in {key}")
    for key in ("qps", "gops"):
        if not set(repeat["manifest"][key]) <= set(base["manifest"][key]):
            raise ValueError(f"repeat campaign has unexpected {key}")
    original = {s["sequence_id"]: s for s in base["manifest"]["sequences"]}
    for sequence in repeat["manifest"]["sequences"]:
        if original.get(sequence["sequence_id"]) != sequence:
            raise ValueError("repeat campaign has an unmatched sequence")
    for key in ("binary", "build", "device", "host_identity", "platform",
                "python_dependencies", "python_device", "reference_source",
                "reference_source_policy"):
        if repeat["preflight_identity"][key] != base["preflight_identity"][key]:
            raise ValueError(f"repeat campaign differs in {key}")
    for key in ("cli/main.cpp", "scripts/measure_python_reference.py",
                "scripts/measurement_environment.py", "scripts/measurement_metrics.py"):
        if (repeat["preflight_identity"]["measurement_sources"][key] !=
                base["preflight_identity"]["measurement_sources"][key]):
            raise ValueError(f"repeat campaign differs in {key}")
    for kind, assets in repeat["preflight_identity"]["assets"].items():
        original_assets = base["preflight_identity"]["assets"][kind]
        if any(original_assets.get(name) != value for name, value in assets.items()):
            raise ValueError(f"repeat campaign differs in {kind}")


def append_analysis_stats(stats, analysis, manifest, campaign):
    cases = {
        f'{sequence["sequence_id"]}-q{qp}-g{gop}': (sequence, qp, gop)
        for sequence in manifest["sequences"] for qp in manifest["qps"]
        for gop in manifest["gops"]
    }
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
                "campaign": campaign,
                "sequence": sequence["sequence_id"], "width": sequence["width"],
                "height": sequence["height"], "qp": qp, "gop": gop,
                "implementation": item["implementation"],
                "operation": item["operation"], "mode": item["mode"],
                "metric": metric, "n": value["n"], "mean": value["mean"],
                "sample_std": value["sample_std"],
                "minimum": value["minimum"], "maximum": value["maximum"],
            })


def append_process_fps_stats(stats, observations, campaigns):
    runs = {label: run for label, _, run in campaigns}
    groups = defaultdict(list)
    for row in observations:
        if row["mode"] == "throughput":
            key = (row["campaign"], row["case_id"], row["implementation"],
                   row["operation"])
            groups[key].append(row)
    expected_groups = {
        (label, f'{sequence["sequence_id"]}-q{qp}-g{gop}', implementation,
         operation)
        for label, _, run in campaigns
        for sequence in run["manifest"]["sequences"]
        for qp in run["manifest"]["qps"]
        for gop in run["manifest"]["gops"]
        for implementation in run["manifest"]["implementations"]
        for operation in ("encode", "decode")
    }
    if groups.keys() != expected_groups:
        raise ValueError("process FPS condition coverage mismatch")
    for (campaign, _, implementation, operation), rows in groups.items():
        expected = runs[campaign]["manifest"]["repetitions"]
        if expected != 10 or len(rows) != expected or (
                {row["repeat"] for row in rows} != set(range(expected))):
            raise ValueError("process FPS requires all ten distinct repetitions")
        first = rows[0]
        stats.append({
            "campaign": campaign,
            "sequence": first["sequence"], "width": first["width"],
            "height": first["height"], "qp": first["qp"], "gop": first["gop"],
            "implementation": implementation, "operation": operation,
            "mode": "throughput", "metric": "process_fps",
            **condition_statistics([row["process_fps"] for row in rows]),
        })


def enrich_analysis_stats(stats, observations, campaigns):
    metrics = {
        "throughput": ("codec_seconds", "throughput_fps", "process_seconds"),
        "memory": ("process_peak_rss_mib",),
    }
    fields = ("campaign", "sequence", "qp", "gop", "implementation",
              "operation", "mode", "metric")
    existing = {tuple(row[field] for field in fields): row
                for row in stats if row["metric"] != "process_fps"}
    if len(existing) != len(stats) - sum(row["metric"] == "process_fps" for row in stats):
        raise ValueError("duplicate aggregate statistic")
    groups = defaultdict(list)
    for row in observations:
        for metric in metrics.get(row["mode"], ()):
            key = tuple(row[field] for field in fields[:-1]) + (metric,)
            groups[key].append(row)
    if groups.keys() != existing.keys():
        raise ValueError("aggregate observation coverage mismatch")
    runs = {label: run for label, _, run in campaigns}
    for key, rows in groups.items():
        expected = runs[key[0]]["manifest"]["repetitions"]
        if expected != 10 or len(rows) != expected or (
                {row["repeat"] for row in rows} != set(range(expected))):
            raise ValueError("statistics require all ten distinct repetitions")
        metric = key[-1]
        calculated = condition_statistics([row[metric] for row in rows])
        original = existing[key]
        for field in ("n", "mean", "sample_std", "minimum", "maximum"):
            if not math.isclose(original[field], calculated[field], rel_tol=1e-12):
                raise ValueError("analysis aggregate differs from observations")
        original.update(calculated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat-campaign", nargs=2, action="append",
                        metavar=("LABEL", "PATH"), default=[],
                        help="append a compatible targeted throughput campaign")
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
    append_analysis_stats(stats, analysis, manifest, "full")
    expected_stats = len(cases) * len(manifest["implementations"]) * 2 * 4
    expected_rd = len(cases) * len(manifest["implementations"])
    if len(stats) != expected_stats or len(rd["points"]) != expected_rd:
        raise ValueError("aggregate or RD coverage mismatch")

    campaigns = [("full", args.campaign, run)]
    for label, path in args.repeat_campaign:
        path = Path(path)
        extra = json.loads((path / "run.json").read_text())
        extra_analysis = json.loads((path / "analysis.json").read_text())
        if extra_analysis["status"] != "complete":
            raise ValueError("repeat campaign analysis must be complete")
        validate_repeat_campaign(run, extra)
        before = len(stats)
        append_analysis_stats(stats, extra_analysis, extra["manifest"], label)
        repeat_cases = (len(extra["manifest"]["sequences"]) *
                        len(extra["manifest"]["qps"]) * len(extra["manifest"]["gops"]))
        if len(stats) - before != repeat_cases * len(extra["manifest"]["implementations"]) * 2 * 3:
            raise ValueError("repeat aggregate coverage mismatch")
        campaigns.append((label, path, extra))
    if len({label for label, _, _ in campaigns}) != len(campaigns):
        raise ValueError("campaign labels must be unique")

    observations = []
    execution_ids = set()
    for label, path, source_run in campaigns:
        count = 0
        with (path / "observations.jsonl").open() as stream:
            for line in stream:
                item = json.loads(line)
                execution_id = item["execution_id"]
                if (item["status"] != "passed" or
                        item["fingerprint"] != source_run["fingerprint"] or
                        execution_id in execution_ids or
                        item["timing_contract"] != "host-yuv420p8-completed-frame-v1"):
                    raise ValueError("invalid or duplicate campaign observation")
                execution_ids.add(execution_id)
                count += 1
                timing = item["metrics"]
                if (timing["timed_frames"] != item["frames"] or
                        timing["warmup_frames"] != source_run["manifest"]["warmup_frames"]):
                    raise ValueError("inconsistent frame counts")
                size = item.get("bytes", {})
                quality = item.get("quality", {})
                process_seconds = item["process_seconds"]
                if not math.isfinite(process_seconds) or process_seconds <= 0:
                    raise ValueError("nonpositive process duration")
                observations.append({
                    "campaign": label,
                    "campaign_fingerprint": source_run["fingerprint"],
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
                    "process_seconds": process_seconds,
                    "process_fps": (item["frames"] / process_seconds
                                    if item["mode"] == "throughput" else ""),
                    "process_peak_rss_mib": item.get("process_peak_rss_mib", ""),
                    "file_bytes": size.get("file_bytes", ""),
                    "entropy_bytes": size.get("entropy_bytes", ""),
                    "file_bpp": size.get("file_bpp", ""),
                    "entropy_bpp": size.get("entropy_bpp", ""),
                    "pooled_psnr_yuv_db": quality.get("pooled_psnr_yuv_db", ""),
                })
        if count != source_run["expected_operations"]:
            raise ValueError(f"{label} observation coverage mismatch")
    campaign_order = {label: index for index, (label, _, _) in enumerate(campaigns)}
    observations.sort(key=lambda r: (campaign_order[r["campaign"]],
                                     r["sequence"], r["qp"], r["gop"],
                                     r["implementation"], r["mode"],
                                     r["repeat"], r["operation"]))
    append_process_fps_stats(stats, observations, campaigns)
    enrich_analysis_stats(stats, observations, campaigns)
    stats.sort(key=lambda r: (campaign_order[r["campaign"]],
                              r["sequence"], r["qp"], r["gop"],
                              r["implementation"], r["operation"], r["mode"],
                              r["metric"]))
    rd_rows = sorted(rd["points"], key=lambda r: (
        r["sequence"], r["gop"], r["implementation"], r["qp"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "operation-measurements.csv",
              list(observations[0]), observations)
    with (args.output / "operation-measurements.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_csv(args.output / "condition-statistics.csv",
              ["campaign", "sequence", "width", "height", "qp", "gop",
               "implementation", "operation", "mode", "metric", "n", "mean",
               "median", "sample_std", "cv_percent", "minimum", "maximum",
               "ci95_low", "ci95_high"], stats)
    write_csv(args.output / "rd-points.csv",
              ["sequence", "width", "height", "gop", "qp", "implementation",
               "content_id", "input_sha256", "frames", "entropy_bpp",
               "file_bpp", "psnr_yuv_db", "quality_contract", "fingerprint"],
              rd_rows)
    print(f"Exported {len(observations)} operations, {len(stats)} statistics "
          f"and {len(rd_rows)} RD points")


if __name__ == "__main__":
    main()
