#!/usr/bin/env python3
"""Export ten-run condition statistics from retained compact observations."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from measurement_metrics import TIMING_CONTRACT, condition_statistics


FIELDS = ("campaign", "sequence", "width", "height", "qp", "gop",
          "implementation", "operation", "mode", "metric", "n", "mean",
          "median", "sample_std", "cv_percent", "minimum", "maximum",
          "ci95_low", "ci95_high")
METRICS = {
    "throughput": ("codec_seconds", "throughput_fps", "process_seconds", "process_fps"),
    "memory": ("process_peak_rss_mib",),
}


def export(rows, campaign, expected_conditions):
    groups = defaultdict(list)
    conditions = set()
    executions = set()
    fingerprints = set()
    for row in rows:
        if (row["status"] != "passed" or row["timing_contract"] != TIMING_CONTRACT or
                row.get("campaign", campaign) != campaign or
                row["execution_id"] in executions or
                row["metrics"]["timed_frames"] != row["frames"]):
            raise ValueError("invalid or duplicate compact observation")
        executions.add(row["execution_id"])
        fingerprints.add(row["fingerprint"])
        condition = tuple(row[key] for key in
                          ("sequence", "width", "height", "qp", "gop",
                           "implementation", "operation"))
        conditions.add(condition)
        mode = row["mode"]
        seconds = row["process_seconds"]
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid process duration")
        if mode == "quality":
            continue
        if mode not in METRICS:
            raise ValueError(f"unexpected observation mode: {mode}")
        values = ({"codec_seconds": row["metrics"]["codec_seconds"],
                   "throughput_fps": row["metrics"]["throughput_fps"],
                   "process_seconds": row["process_seconds"],
                   "process_fps": row["frames"] / row["process_seconds"]}
                  if mode == "throughput" else
                  {"process_peak_rss_mib": row["process_peak_rss_mib"]})
        if mode == "throughput" and not math.isclose(
                row["process_fps"], values["process_fps"], rel_tol=1e-12):
            raise ValueError("stored process FPS disagrees with frames / process_seconds")
        for metric, value in values.items():
            groups[(condition, mode, metric)].append((row["repeat"], value))
    if len(fingerprints) != 1 or len(conditions) != expected_conditions:
        raise ValueError("campaign fingerprint or condition coverage mismatch")
    expected = {(condition, mode, metric) for condition in conditions
                for mode, metrics in METRICS.items() for metric in metrics}
    if groups.keys() != expected:
        raise ValueError("incomplete condition or metric coverage")
    output = []
    for (condition, mode, metric), repetitions in groups.items():
        if (len(repetitions) != 10 or
                {repeat for repeat, _ in repetitions} != set(range(10))):
            raise ValueError("statistics require all ten distinct repetitions")
        sequence, width, height, qp, gop, implementation, operation = condition
        output.append(dict(zip(FIELDS[:8], (campaign, sequence, width, height,
                                            qp, gop, implementation, operation))) | {
            "mode": mode, "metric": metric,
            **condition_statistics([value for _, value in repetitions]),
        })
    return sorted(output, key=lambda row: (row["campaign"], row["sequence"],
                                            row["qp"], row["gop"],
                                            row["implementation"], row["operation"],
                                            row["mode"], row["metric"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--expected-conditions", type=int, required=True)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.observations.open() as stream:
        rows = export((json.loads(line) for line in stream), args.campaign,
                      args.expected_conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} condition statistics")


if __name__ == "__main__":
    main()
