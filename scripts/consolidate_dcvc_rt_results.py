#!/usr/bin/env python3
"""Consolidate DCVC-RT GOP JSON results into commit-friendly datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE = Path("/home/oelghati/DCVC-RT/runner/dcvc_rt/gop")
DEFAULT_OUTPUT = Path("results/jetson-orin-python")
SCHEMA = "nvcr.python-dcvc-rt.gop.v1"

RESULT_NAME = re.compile(
    r",q=(?P<qp>\d+),gop=(?P<gop>\d+),time=(?P<timestamp>\d+)_encode\.json$"
)

# These are the comparable codec and runtime measurements. Energy and power
# fields are deliberately absent instead of being removed after collection.
SCALAR_FIELDS = (
    "mode",
    "frame_num",
    "i_frame_num",
    "p_frame_num",
    "total_bytes",
    "bitrate_kbps",
    "elapsed_time_s",
    "total_encoding_time_s",
    "encoding_fps",
    "decoding_fps",
    "avg_frame_encoding_time_s",
    "p95_frame_encoding_time_s",
    "peak_memory_mb",
    "peak_gpu_memory_mb",
    "reset_interval",
)


def _records(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested result dictionaries identified by the mode field."""
    if isinstance(value, dict):
        if "mode" in value and "frame_num" in value:
            yield value
        else:
            for child in value.values():
                yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def _timing_fields(record: dict[str, Any]) -> dict[str, Any]:
    timing = record.get("timing", {}).get("encoding", {})
    return {
        "timing_frame_count": timing.get("count"),
        "timing_total_s": timing.get("total_s"),
        "timing_avg_s": timing.get("avg_s"),
        "timing_p50_s": timing.get("p50_s"),
        "timing_p95_s": timing.get("p95_s"),
        "timing_min_s": timing.get("min_s"),
        "timing_max_s": timing.get("max_s"),
    }


def _row(path: Path, source_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    match = RESULT_NAME.search(path.name)
    if match is None:
        raise ValueError(f"result filename does not match expected format: {path}")

    resolution = path.relative_to(source_root).parts[0]
    sequence = path.parent.name if match is not None else path.parent.name
    width, height = resolution.split("x", 1)
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "hardware": "jetson-orin",
        "codec": "DCVC-RT",
        "operation": "encode",
        "resolution": resolution,
        "width": int(width),
        "height": int(height),
        "sequence": sequence,
        "source_kind": "summary",
        "qp": int(match.group("qp")),
        "gop_size": int(match.group("gop")),
        "source_timestamp": int(match.group("timestamp")),
        "source_json": path.relative_to(source_root).as_posix(),
    }
    for field in SCALAR_FIELDS:
        if field in record:
            row[field] = record[field]
    row.update(_timing_fields(record))
    total_encoding_time = row.get("total_encoding_time_s")
    if total_encoding_time and row.get("frame_num") is not None:
        row["encoding_fps"] = row["frame_num"] / total_encoding_time
    return row


def collect(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path for path in source_root.rglob("*_encode.json")
        if "streams" not in path.relative_to(source_root).parts
    )
    if not paths:
        raise ValueError(f"no *_encode.json files found below {source_root}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = list(_records(payload))
        if len(records) != 1:
            raise ValueError(f"expected one result record in {path}, found {len(records)}")
        rows.append(_row(path, source_root, records[0]))
    return sorted(rows, key=lambda row: (
        row["resolution"], row["sequence"], row["qp"], row["gop_size"], row["source_timestamp"]
    ))


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    resolutions = sorted({row["resolution"] for row in rows})
    sequences = sorted({row["sequence"] for row in rows})
    gops = sorted({row["gop_size"] for row in rows})
    lines = [
        "# DCVC-RT Python GOP Results",
        "",
        f"- Rows: {len(rows)}",
        f"- Resolutions: {', '.join(resolutions)}",
        f"- Sequences: {', '.join(sequences)}",
        f"- GOP sizes: {', '.join(str(value) for value in gops)}",
        "- Metrics: frame counts, payload size, bitrate, encode timing/FPS, memory, and timing percentiles",
        "- Energy and power results: excluded",
        "",
        "The JSONL file is the canonical dataset; CSV is provided for spreadsheet and plotting workflows.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = collect(args.source.expanduser().resolve())
    write_dataset(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())