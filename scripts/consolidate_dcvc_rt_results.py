#!/usr/bin/env python3
"""Consolidate DCVC-RT GOP JSON results into commit-friendly datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SOURCE = Path("DCVC-RT/runner/dcvc_rt/gop")
DEFAULT_OUTPUT = Path("results/jetson-orin/python/data")
SCHEMA = "nvcr.python-dcvc-rt.gop.v1"

RESULT_NAME = re.compile(
    r",q=(?P<qp>\d+),gop=(?P<gop>\d+),time=(?P<timestamp>\d+)_(?P<operation>encode|decode)\.json$"
)

# These are the comparable process-level measurements. Energy and power fields
# are deliberately absent instead of being removed after collection.
SCALAR_FIELDS = (
    "mode",
    "frame_num",
    "i_frame_num",
    "p_frame_num",
    "total_bytes",
    "bitrate_kbps",
    "process_time_s",
    "process_frame_count",
    "process_avg_frame_time_s",
    "model_initialization_time_s",
    "peak_memory_mb",
    "peak_gpu_memory_mb",
    "reset_interval",
)

QUALITY_FIELDS = (
    "ave_i_frame_bpp",
    "ave_i_frame_psnr",
    "ave_i_frame_msssim",
    "ave_i_frame_psnr_y",
    "ave_i_frame_psnr_u",
    "ave_i_frame_psnr_v",
    "ave_i_frame_msssim_y",
    "ave_i_frame_msssim_u",
    "ave_i_frame_msssim_v",
    "ave_p_frame_bpp",
    "ave_p_frame_psnr",
    "ave_p_frame_msssim",
    "ave_p_frame_psnr_y",
    "ave_p_frame_psnr_u",
    "ave_p_frame_psnr_v",
    "ave_p_frame_msssim_y",
    "ave_p_frame_msssim_u",
    "ave_p_frame_msssim_v",
    "ave_all_frame_bpp",
    "ave_all_frame_psnr",
    "ave_all_frame_msssim",
    "ave_all_frame_psnr_y",
    "ave_all_frame_psnr_u",
    "ave_all_frame_psnr_v",
    "ave_all_frame_msssim_y",
    "ave_all_frame_msssim_u",
    "ave_all_frame_msssim_v",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _frame_count(record: dict[str, Any]) -> int | None:
    if isinstance(record.get("frame_num"), int):
        return record["frame_num"]
    i_frames = record.get("i_frame_num")
    p_frames = record.get("p_frame_num")
    if isinstance(i_frames, int) and isinstance(p_frames, int):
        return i_frames + p_frames
    return None


def _process_time(record: dict[str, Any], operation: str) -> float | None:
    if operation == "encode":
        return _number(record.get("process_encode_time_s"))
    return (
        _number(record.get("process_decode_time_s"))
        or _number(record.get("process_decoding_time_s"))
        or _number(record.get("elapsed_time_s"))
        or _number(record.get("test_time"))
    )


def _process_frame_count(record: dict[str, Any], operation: str) -> int | None:
    if operation == "encode" and isinstance(
        record.get("process_encode_frame_count"), int
    ):
        return record["process_encode_frame_count"]
    frame_count = _frame_count(record)
    return frame_count


def _throughput_fps(
    record: dict[str, Any],
    operation: str,
    frame_count: int | None,
    process_time: float | None,
) -> float | None:
    direct_throughput = _number(record.get("throughput_fps"))
    if direct_throughput is not None:
        return direct_throughput
    if operation == "encode":
        source_fps = _number(record.get("process_encode_fps"))
    else:
        source_fps = (
            _number(record.get("process_decode_fps"))
            or _number(record.get("process_decoding_fps"))
        )
    if source_fps is not None:
        return source_fps
    if (
        frame_count is not None
        and process_time is not None
        and process_time > 0.0
    ):
        return round(frame_count / process_time, 6)
    return _number(record.get("fps"))


def _records(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested result dictionaries identified by the mode field."""
    if isinstance(value, dict):
        if "mode" in value and (
            "frame_num" in value or "i_frame_num" in value or "p_frame_num" in value
        ):
            yield value
        else:
            for child in value.values():
                yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def _row(
    path: Path,
    source_root: Path,
    record: dict[str, Any],
    hardware: str,
) -> dict[str, Any]:
    match = RESULT_NAME.search(path.name)
    if match is None:
        raise ValueError(f"result filename does not match expected format: {path}")

    operation = match.group("operation")
    resolution = path.relative_to(source_root).parts[0]
    sequence = path.parent.name
    width, height = resolution.split("x", 1)
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "hardware": hardware,
        "codec": "DCVC-RT",
        "operation": operation,
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
    frame_count = _frame_count(record)
    if frame_count is not None:
        row["frame_num"] = frame_count
    process_time = _process_time(record, operation)
    process_frame_count = _process_frame_count(record, operation)
    throughput_fps = _throughput_fps(record, operation, process_frame_count, process_time)
    if throughput_fps is not None:
        row["throughput_fps"] = throughput_fps
    if process_time is not None:
        row["process_time_s"] = process_time
    if process_frame_count is not None:
        row["process_frame_count"] = process_frame_count
    if (
        process_frame_count is not None
        and process_time is not None
        and process_frame_count > 0
    ):
        row["process_avg_frame_time_s"] = round(process_time / process_frame_count, 6)
    for field in QUALITY_FIELDS:
        if field in record:
            row[field] = record[field]
    return row


def collect(source_root: Path, hardware: str) -> list[dict[str, Any]]:
    paths = sorted(
        path for path in source_root.rglob("*.json")
        if path.name.endswith(("_encode.json", "_decode.json"))
        if "streams" not in path.relative_to(source_root).parts
    )
    if not paths:
        raise ValueError(f"no encode/decode JSON results found below {source_root}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        records = list(_records(payload))
        if len(records) != 1:
            raise ValueError(f"expected one result record in {path}, found {len(records)}")
        rows.append(_row(path, source_root, records[0], hardware))
    return sorted(
        rows,
        key=lambda row: (
            row["resolution"],
            row["sequence"],
            row["qp"],
            row["gop_size"],
            row["source_timestamp"],
        ),
    )


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
        "- Metrics: frame counts, payload size, bitrate, process timing, throughput FPS, memory, and decode quality",
        "- Energy and power results: excluded",
        "",
        "The JSONL file is the canonical dataset; CSV is provided for spreadsheet and plotting workflows.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_summary(output, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hardware", default="jetson-orin")
    args = parser.parse_args()
    rows = collect(args.source.expanduser().resolve(), args.hardware)
    write_dataset(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
