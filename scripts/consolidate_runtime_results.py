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


def cv(values: list[float]) -> float | None:
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return statistics.stdev(values) / statistics.mean(values)


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
) -> str:
    python = choose_python(python_rows)
    nvcr, repetitions = choose_nvcr_averages(nvcr_rows)
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

    decode_lines = []
    decode_deltas = []
    for key in matrix_keys({key for key in keys if key[2] == "decode"}):
        label, gop, _ = key
        py = python[key]
        nr = nvcr[key]
        py_fps = number(py, "fps")
        nv_fps = number(nr, "throughput_fps")
        py_bpp = number(py, "ave_all_frame_bpp")
        nv_bpp = number(nr, "payload_bpp")
        py_psnr = number(py, "ave_all_frame_psnr", "ave_all_frame_psnr_y")
        nv_psnr = number(nr, "psnr_yuv")
        bpp_delta = nv_bpp / py_bpp - 1 if py_bpp and nv_bpp is not None else None
        psnr_delta = nv_psnr - py_psnr if nv_psnr is not None and py_psnr is not None else None
        if psnr_delta is not None:
            decode_deltas.append((key, psnr_delta))
        decode_lines.append(
            f"| {label} | {gop} | {fmt(py_fps)} | {fmt(nv_fps)} | {fmt(py_bpp, 6)} | {fmt(nv_bpp, 6)} | {pct(bpp_delta)} | {fmt(py_psnr)} | {fmt(nv_psnr)} | {fmt(psnr_delta)} |"
        )

    encode_by_resolution = []
    for label in resolutions:
        py_values = [number(python[(label, gop, "encode")], "fps") for gop in gops]
        nv_values = [number(nvcr[(label, gop, "encode")], "throughput_fps") for gop in gops]
        encode_by_resolution.append(
            f"| {label} | {' / '.join(fmt(value) for value in py_values)} | {' / '.join(fmt(value) for value in nv_values)} |"
        )

    mean_psnr = statistics.mean(value for _, value in decode_deltas) if decode_deltas else None
    median_psnr = statistics.median(value for _, value in decode_deltas) if decode_deltas else None
    within = sum(abs(value) <= 0.2 for _, value in decode_deltas)
    largest = sorted(decode_deltas, key=lambda item: abs(item[1]), reverse=True)[:2]
    largest_text = ", ".join(f"{key[0]} GOP {key[1]} (`{value:+.3f} dB`)" for key, value in largest)

    encode_cvs = [cv([number(row, "throughput_fps") for row in repetitions[key] if number(row, "throughput_fps") is not None]) for key in keys if key[2] == "encode"]
    decode_cvs = [cv([number(row, "throughput_fps") for row in repetitions[key] if number(row, "throughput_fps") is not None]) for key in keys if key[2] == "decode"]
    encode_cvs = [value for value in encode_cvs if value is not None]
    decode_cvs = [value for value in decode_cvs if value is not None]

    lines = [
        f"# {hardware_title(python_hardware)} {python_label} vs {nvcr_label}",
        "",
        "This report is generated from the supplied canonical Python and NVCR JSONL datasets. It is diagnostic evidence, not a release or target-support claim.",
        "",
        "## Inputs and coverage",
        "",
        "| Runtime | Canonical data | SHA-256 | Rows | Coverage |",
        "|---|---|---|---:|---|",
        f"| {python_label} | [{python_path.name}]({relative_link(output_path, python_path)}) | `{sha256(python_path)}` | {len(python_rows)} | {len(python)} comparison cases |",
        f"| {nvcr_label} | [{nvcr_path.name}]({relative_link(output_path, nvcr_path)}) | `{sha256(nvcr_path)}` | {len(nvcr_rows)} | {len(nvcr)} average cases; repetition rows used for stability |",
        "",
        f"Both datasets use hardware `{python_hardware}`, QP values {', '.join(str(value) for value in sorted(qp_values))}, and frame counts {', '.join(str(value) for value in sorted(frame_values))}. Coverage is {', '.join(resolutions)} with GOPs {', '.join(str(value) for value in gops)} and encode/decode operations.",
        f"The NVCR recorded commit is {commit_text}. Python duplicate keys are resolved using the latest `source_timestamp`; NVCR comparison rows are selected from `run_index=average`.",
        "",
        "## Decode comparison",
        "",
        f"Python reports process FPS and average all-frame quality. {nvcr_label} reports codec-loop FPS and YUV PSNR. These timing boundaries differ, so FPS values are descriptive and are not a controlled speedup measurement.",
        "",
        "`BPP delta` is `(NVCR BPP / Python BPP) - 1`. `PSNR delta` is NVCR PSNR minus Python PSNR.",
        "",
        "| Resolution | GOP | Python FPS | NVCR FPS | Python BPP | NVCR BPP | BPP delta | Python PSNR | NVCR PSNR | PSNR delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *decode_lines,
        "",
        f"Across the {len(decode_deltas)} decode cases, the mean PSNR delta is `{fmt(mean_psnr)} dB`, the median delta is `{fmt(median_psnr)} dB`, and {within} cases are within `0.2 dB`. The largest quality gaps are {largest_text}.",
        "",
        "## Encode comparison",
        "",
        f"Values are ordered GOP {' / '.join(str(gop) for gop in gops)}.",
        "",
        "| Resolution | Python FPS | NVCR FPS |",
        "|---|---:|---:|",
        *encode_by_resolution,
        "",
        f"The median coefficient of variation across NVCR repetitions is {fmt(statistics.median(encode_cvs) * 100 if encode_cvs else None, 2)}% for encode throughput and {fmt(statistics.median(decode_cvs) * 100 if decode_cvs else None, 2)}% for decode throughput.",
        "",
        "## Assessment",
        "",
        "This matrix is suitable for diagnostic comparison only. Differences in timing boundaries, samplers, and runtime measurement paths mean the FPS and memory values must not be presented as controlled acceleration or efficiency claims. BPP and PSNR differences should be investigated before claiming cross-runtime compression equivalence.",
        "",
        "Memory measurements are available in the NVCR rows and are omitted from the numeric comparison because the Python and NVCR samplers and timing boundaries are not identical.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path, help="Python DCVC-RT results.jsonl")
    parser.add_argument("--nvcr", required=True, type=Path, help="NVCR resolution-matrix results.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output summary.md")
    parser.add_argument("--python-label", default="Python DCVC-RT")
    parser.add_argument("--nvcr-label", default="NVCR")
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
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
