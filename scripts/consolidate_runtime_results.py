#!/usr/bin/env python3
"""Generate a Python DCVC-RT versus NVCR comparison report from two JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RESOLUTION_ORDER = ("QCIF", "CIF", "360p", "540p", "720p", "1080p")
DIMENSION_TO_LABEL = {
    "176x144": "QCIF",
    "352x288": "CIF",
    "640x360": "360p",
    "960x540": "540p",
    "1280x720": "720p",
    "1920x1080": "1080p",
}
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no JSON objects found in {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardware(rows: list[dict[str, Any]], source: str) -> str:
    values = {row.get("hardware") for row in rows}
    if len(values) != 1 or None in values:
        raise ValueError(f"{source} must contain exactly one hardware value, got {values}")
    return str(values.pop())


def python_key(row: dict[str, Any]) -> tuple[str, int, str]:
    resolution = str(row.get("resolution", ""))
    label = DIMENSION_TO_LABEL.get(resolution, resolution)
    return (label, int(row["gop_size"]), str(row["operation"]))


def nvcr_key(row: dict[str, Any]) -> tuple[str, int, str]:
    dimension = str(row.get("size", ""))
    if dimension not in DIMENSION_TO_LABEL:
        raise ValueError(f"unsupported NVCR resolution: {dimension}")
    return (DIMENSION_TO_LABEL[dimension], int(row["gop_size"]), str(row["operation"]))


def choose_python(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    chosen: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = python_key(row)
        previous = chosen.get(key)
        if previous is None or int(row.get("source_timestamp", 0)) >= int(previous.get("source_timestamp", 0)):
            chosen[key] = row
    return chosen


def choose_nvcr_averages(rows: list[dict[str, Any]]) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]], dict[tuple[str, int, str], list[dict[str, Any]]]
]:
    averages: dict[tuple[str, int, str], dict[str, Any]] = {}
    repetitions: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = nvcr_key(row)
        if row.get("run_index") == "average":
            if key in averages:
                raise ValueError(f"duplicate NVCR average row for {key}")
            averages[key] = row
        else:
            repetitions[key].append(row)
    missing_average = sorted(set(repetitions) - set(averages))
    missing_repetitions = sorted(set(averages) - set(repetitions))
    if missing_average or missing_repetitions:
        raise ValueError(
            f"NVCR average/repetition mismatch: missing averages={missing_average}, "
            f"missing repetitions={missing_repetitions}"
        )
    return averages, repetitions


def validate_matrix(
    python_rows: dict[tuple[str, int, str], dict[str, Any]],
    nvcr_rows: dict[tuple[str, int, str], dict[str, Any]],
) -> None:
    if set(python_rows) != set(nvcr_rows):
        missing_python = sorted(set(nvcr_rows) - set(python_rows))
        missing_nvcr = sorted(set(python_rows) - set(nvcr_rows))
        raise ValueError(f"matrix mismatch: missing Python={missing_python}, missing NVCR={missing_nvcr}")
    if not python_rows:
        raise ValueError("comparison matrix is empty")


def number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def resolution_key(label: str) -> tuple[int, str]:
    try:
        return (RESOLUTION_ORDER.index(label), label)
    except ValueError:
        return (len(RESOLUTION_ORDER), label)


def matrix_keys(keys: set[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    return sorted(keys, key=lambda key: (resolution_key(key[0]), key[1], key[2]))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def nvcr_inner_bpp(row: dict[str, Any], frame_overhead_bytes: int) -> float:
    dimension = str(row.get("size", ""))
    try:
        width_text, height_text = dimension.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid NVCR size for inner-BPP calculation: {dimension}") from error

    frames = number(row, "frames", "frame_num")
    payload_bytes = number(row, "payload_bytes")
    if width <= 0 or height <= 0 or frames is None or frames <= 0:
        raise ValueError(f"invalid NVCR dimensions or frame count for {dimension}")
    if payload_bytes is None:
        raise ValueError(f"NVCR row for {dimension} has no payload_bytes")

    entropy_bytes = payload_bytes - frame_overhead_bytes * frames
    if entropy_bytes < 0:
        raise ValueError(
            f"NVCR payload_bytes is smaller than the selected fixed overhead for {dimension}"
        )
    return entropy_bytes * 8.0 / (width * height * frames)


def hardware_title(value: str) -> str:
    words = value.replace("-", " ").split()
    if words and words[0].lower().startswith("rtx"):
        suffix = words[0][3:]
        return "RTX " + " ".join(([suffix] if suffix else []) + words[1:])
    if words and words[0].lower() == "jetson":
        return "Jetson " + " ".join(words[1:]).title()
    return value.title()


def relative_link(output: Path, source: Path) -> str:
    return Path(os.path.relpath(source, output.parent)).as_posix()


def consistent_value(
    rows: list[dict[str, Any]], fields: tuple[str, ...], source: str
) -> Any | None:
    values = {row[field] for row in rows for field in fields if field in row}
    if len(values) > 1:
        raise ValueError(f"{source} has inconsistent {fields}: {sorted(values, key=str)}")
    return next(iter(values), None)


def build_report(
    python_path: Path,
    nvcr_path: Path,
    output_path: Path,
    python_rows: list[dict[str, Any]],
    nvcr_rows: list[dict[str, Any]],
    python_label: str,
    nvcr_label: str,
    *,
    comparable_gops: tuple[int, ...] = (),
    nvcr_frame_overhead_bytes: int | None = None,
    python_reference_reset: int | None = None,
    nvcr_reference_reset: int | None = None,
    nvcr_checkout_state: str = "unresolved",
) -> str:
    python = choose_python(python_rows)
    nvcr, _repetitions = choose_nvcr_averages(nvcr_rows)
    validate_matrix(python, nvcr)

    python_hardware = hardware(python_rows, "Python source")
    nvcr_hardware = hardware(nvcr_rows, "NVCR source")
    if python_hardware != nvcr_hardware:
        raise ValueError(f"hardware mismatch: Python={python_hardware}, NVCR={nvcr_hardware}")

    python_qp = consistent_value(python_rows, ("qp",), "Python source")
    nvcr_qp = consistent_value(nvcr_rows, ("qp",), "NVCR source")
    python_frames = consistent_value(python_rows, ("frame_num", "frames"), "Python source")
    nvcr_frames = consistent_value(nvcr_rows, ("frame_num", "frames"), "NVCR source")
    if python_qp is not None and nvcr_qp is not None and python_qp != nvcr_qp:
        raise ValueError(f"QP mismatch: Python={python_qp}, NVCR={nvcr_qp}")
    if python_frames is not None and nvcr_frames is not None and python_frames != nvcr_frames:
        raise ValueError(f"frame-count mismatch: Python={python_frames}, NVCR={nvcr_frames}")

    keys = set(python)
    gops = sorted({key[1] for key in keys})
    resolutions = sorted({key[0] for key in keys}, key=resolution_key)
    qp_values = {row.get("qp") for row in python_rows + nvcr_rows if row.get("qp") is not None}
    frame_values = {
        row.get("frame_num", row.get("frames"))
        for row in python_rows + nvcr_rows
        if row.get("frame_num", row.get("frames")) is not None
    }
    commit_values = {row.get("nvcr_commit") for row in nvcr_rows if row.get("nvcr_commit")}
    commit_text = ", ".join(f"`{value}`" for value in sorted(commit_values)) or "not recorded"

    if nvcr_checkout_state not in {"clean", "dirty", "unresolved"}:
        raise ValueError(f"unsupported NVCR checkout state: {nvcr_checkout_state}")
    recorded_dirty = consistent_value(nvcr_rows, ("nvcr_dirty",), "NVCR source")
    if nvcr_checkout_state == "clean" and recorded_dirty is True:
        raise ValueError("NVCR checkout cannot be classified clean when JSONL records dirty")
    if nvcr_checkout_state == "dirty":
        dirty_text = (
            "Retained run-level provenance classifies the NVCR checkout as dirty. "
            f"The canonical JSONL records `nvcr_dirty={str(recorded_dirty).lower()}`; "
            "treat the run as dirty where those sources disagree."
        )
    elif nvcr_checkout_state == "clean":
        dirty_text = (
            "The caller classifies the NVCR checkout as clean and the canonical JSONL "
            f"records `nvcr_dirty={str(recorded_dirty).lower()}`."
        )
    else:
        dirty_text = (
            f"The canonical JSONL records `nvcr_dirty={str(recorded_dirty).lower()}`, but "
            "independent checkout state was not supplied; treat it as unresolved."
        )

    recorded_python_reset = consistent_value(
        python_rows, ("reset_interval",), "Python source"
    )
    if recorded_python_reset is not None:
        recorded_python_reset = int(recorded_python_reset)
    if (
        python_reference_reset is not None
        and recorded_python_reset is not None
        and python_reference_reset != recorded_python_reset
    ):
        raise ValueError(
            "explicit Python reference reset does not match the value recorded in JSONL"
        )
    effective_python_reset = python_reference_reset or recorded_python_reset

    selected_gops = tuple(sorted(set(comparable_gops)))
    missing_gops = sorted(set(selected_gops) - set(gops))
    if missing_gops:
        raise ValueError(f"selected comparable GOPs are absent from the matrix: {missing_gops}")
    if selected_gops and nvcr_frame_overhead_bytes is None:
        raise ValueError("comparable GOPs require --nvcr-frame-overhead-bytes")
    if nvcr_frame_overhead_bytes is not None and not selected_gops:
        raise ValueError("--nvcr-frame-overhead-bytes requires at least one --comparable-gop")
    if nvcr_frame_overhead_bytes is not None and nvcr_frame_overhead_bytes < 0:
        raise ValueError("NVCR frame overhead must be non-negative")

    inter_gops = [gop for gop in selected_gops if gop != 1]
    if inter_gops and (
        effective_python_reset is None or nvcr_reference_reset is None
    ):
        raise ValueError(
            "inter-coded GOP comparison requires explicit Python and NVCR reference-reset policy"
        )
    if inter_gops and effective_python_reset != nvcr_reference_reset:
        raise ValueError(
            "inter-coded GOP comparison requires matching Python and NVCR reference-reset policy"
        )

    rate_lines: list[str] = []
    rate_deltas: list[float] = []
    if selected_gops:
        assert nvcr_frame_overhead_bytes is not None
        selected_keys = {
            key
            for key in keys
            if key[2] == "decode" and key[1] in selected_gops
        }
        for key in matrix_keys(selected_keys):
            label, gop, _ = key
            py_bpp = number(python[key], "ave_all_frame_bpp")
            if py_bpp is None or py_bpp <= 0:
                raise ValueError(f"Python decode row has no positive inner BPP for {key}")
            nv_bpp = nvcr_inner_bpp(nvcr[key], nvcr_frame_overhead_bytes)
            delta = nv_bpp / py_bpp - 1.0
            rate_deltas.append(delta)
            rate_lines.append(
                f"| {label} | {gop} | {fmt(py_bpp, 6)} | {fmt(nv_bpp, 6)} | {pct(delta)} |"
            )

    excluded_inter_gops = [gop for gop in gops if gop != 1 and gop not in selected_gops]
    if effective_python_reset is not None and nvcr_reference_reset is not None:
        if effective_python_reset != nvcr_reference_reset:
            reset_text = (
                f"Python records a {effective_python_reset}-frame feature-reference reset "
                f"interval; the caller identifies NVCR as {nvcr_reference_reset} frames. "
                "Because the inter-reference policies differ, inter-coded GOPs "
                f"{', '.join(str(gop) for gop in excluded_inter_gops)} are excluded. "
                "GOP 1 is all-intra and does not exercise that policy."
            )
        else:
            reset_text = (
                "Python and NVCR are identified with the same "
                f"{effective_python_reset}-frame feature-reference reset interval. Only "
                "the explicitly selected GOPs are reported."
            )
    else:
        reset_text = (
            "The feature-reference reset policy is not fully identified. Inter-coded GOPs "
            f"{', '.join(str(gop) for gop in excluded_inter_gops) or 'in this matrix'} are "
            "not treated as directly comparable."
        )

    lines = [
        f"# {hardware_title(python_hardware)} {python_label} vs {nvcr_label}",
        "",
        "This report is generated from the supplied canonical Python and NVCR JSONL datasets. It is historical diagnostic evidence, not a release, support, or current-performance claim.",
        "",
        "## Inputs and coverage",
        "",
        "| Runtime | Canonical data | SHA-256 | Rows | Coverage |",
        "|---|---|---|---:|---|",
        f"| {python_label} | [{python_path.name}]({relative_link(output_path, python_path)}) | `{sha256(python_path)}` | {len(python_rows)} | {len(python)} comparison cases |",
        f"| {nvcr_label} | [{nvcr_path.name}]({relative_link(output_path, nvcr_path)}) | `{sha256(nvcr_path)}` | {len(nvcr_rows)} | {len(nvcr)} average cases plus retained repetition rows |",
        "",
        f"Both datasets use the hardware label `{python_hardware}`, QP values {', '.join(str(value) for value in sorted(qp_values))}, and frame counts {', '.join(str(value) for value in sorted(frame_values))}. Coverage is {', '.join(resolutions)} with GOPs {', '.join(str(value) for value in gops)} and encode/decode operations.",
        "Matching labels, dimensions, QP, GOP, and frame counts do not establish byte-identical run-time inputs because input hashes were not retained.",
        f"The NVCR recorded commit is {commit_text}. Python duplicate keys are resolved using the latest `source_timestamp`; NVCR rows are selected from `run_index=average`.",
        dirty_text,
        "Exact GPU and target identity, CUDA/TensorRT and engine digests, Python source revision, and checkpoint digest are not recorded, so the matrix is not a controlled baseline.",
        "",
        "## Comparable inner-entropy rate",
        "",
    ]
    if rate_lines:
        assert nvcr_frame_overhead_bytes is not None
        lines.extend(
            [
                f"Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × {nvcr_frame_overhead_bytes}) × 8 / (width × height × frames)`.",
                f"The derivation assumes a fixed {nvcr_frame_overhead_bytes}-byte non-entropy overhead in every retained NVCR access unit. The supplied JSONL does not retain streams or inner-byte counts, so this assumption cannot be verified from the report inputs.",
                "`Relative difference` is `(NVCR inner-entropy BPP / Python inner-bitstream BPP) - 1`.",
                "",
                "| Resolution | GOP | Python inner-bitstream BPP | NVCR derived inner-entropy BPP | Relative difference |",
                "|---|---:|---:|---:|---:|",
                *rate_lines,
                "",
                f"Across these {len(rate_deltas)} explicitly selected cases, the relative-difference range is {pct(min(rate_deltas))} to {pct(max(rate_deltas))}; the median is {pct(statistics.median(rate_deltas))}.",
                "",
                reset_text,
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No GOP was explicitly selected as comparable, so this report contains no cross-runtime rate table.",
                "",
                reset_text,
                "",
            ]
        )

    lines.extend(
        [
            "## Metrics intentionally not compared",
            "",
            f"- Throughput is omitted because Python records process-level timing while {nvcr_label} records codec-loop timing.",
            "- PSNR is omitted because the retained runtimes use different temporal aggregation boundaries.",
            "- Memory is omitted because the samplers and measurement scopes are not identical.",
            "- Wrapper-inclusive NVCR `payload_bpp` is not compared with Python inner-bitstream BPP.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path, help="Python DCVC-RT results.jsonl")
    parser.add_argument("--nvcr", required=True, type=Path, help="NVCR resolution-matrix results.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output summary.md")
    parser.add_argument("--python-label", default="Python DCVC-RT")
    parser.add_argument("--nvcr-label", default="NVCR")
    parser.add_argument(
        "--comparable-gop",
        action="append",
        default=[],
        type=positive_int,
        help="GOP explicitly approved for same-boundary rate comparison; repeat as needed",
    )
    parser.add_argument(
        "--nvcr-frame-overhead-bytes",
        type=nonnegative_int,
        help="fixed per-frame bytes to subtract from NVCR payload_bytes",
    )
    parser.add_argument(
        "--python-reference-reset",
        type=positive_int,
        help="Python feature-reference reset interval when absent from JSONL",
    )
    parser.add_argument(
        "--nvcr-reference-reset",
        type=positive_int,
        help="NVCR feature-reference reset interval for comparability validation",
    )
    parser.add_argument(
        "--nvcr-checkout-state",
        choices=("clean", "dirty", "unresolved"),
        default="unresolved",
        help="run-level checkout classification independent of the JSONL field",
    )
    args = parser.parse_args()

    python_path = args.python.expanduser().resolve()
    nvcr_path = args.nvcr.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report = build_report(
        python_path,
        nvcr_path,
        output_path,
        load_jsonl(python_path),
        load_jsonl(nvcr_path),
        args.python_label,
        args.nvcr_label,
        comparable_gops=tuple(args.comparable_gop),
        nvcr_frame_overhead_bytes=args.nvcr_frame_overhead_bytes,
        python_reference_reset=args.python_reference_reset,
        nvcr_reference_reset=args.nvcr_reference_reset,
        nvcr_checkout_state=args.nvcr_checkout_state,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
