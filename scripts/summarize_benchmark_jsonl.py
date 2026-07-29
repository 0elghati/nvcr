#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize NVCR benchmark JSONL rows as Markdown.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--baseline", default="dcvcrt")
    parser.add_argument("--candidate", default="mlvc-fast")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("benchmark JSONL has no rows")

    latest = {}
    for row in rows:
        key = (str(row["backend"]), str(row["resolution"]), int(row["gop_size"]))
        latest[key] = row

    backends = sorted({str(row["backend"]) for row in rows})
    resolutions = sorted({str(row["resolution"]) for row in rows})
    gops = sorted({int(row["gop_size"]) for row in rows})

    lines = [
        "| Resolution | GOP | Backend | FPS | Payload bytes | Codec time |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for resolution in resolutions:
        for gop in gops:
            for backend in backends:
                row = latest.get((backend, resolution, gop))
                if row is None:
                    continue
                lines.append(
                    f"| {resolution} | {gop} | `{backend}` | "
                    f"{float(row['throughput_fps']):.3f} | "
                    f"{int(row['payload_bytes']):,} | "
                    f"{float(row['codec_time_seconds']):.3f} s |")

    if args.baseline in backends and args.candidate in backends:
        lines.extend([
            "",
            f"Ratios use `{args.baseline}` as the baseline and `{args.candidate}` as the candidate.",
            "",
            "| Resolution | GOP | FPS ratio | Payload ratio |",
            "|---|---:|---:|---:|",
        ])
        for resolution in resolutions:
            for gop in gops:
                base = latest.get((args.baseline, resolution, gop))
                candidate = latest.get((args.candidate, resolution, gop))
                if base is None or candidate is None:
                    continue
                lines.append(
                    f"| {resolution} | {gop} | "
                    f"{float(candidate['throughput_fps']) / float(base['throughput_fps']):.2f}x | "
                    f"{int(candidate['payload_bytes']) / int(base['payload_bytes']):.2f}x |")

    text = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
