#!/usr/bin/env python3
"""Run the NVCR Orin release benchmark and emit one AI-ready JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "scripts/benchmark_resolution_matrix.sh"
ARTIFACTS = ROOT / "scripts/nvcr_artifacts.py"
ENERGY = ROOT / "scripts/profile_energy.py"
RESOLUTIONS = {
    "qcif": ("176x144", 30, "qcif", "qcif/akiyo_qcif.yuv"),
    "cif": ("352x288", 30, "cif", "cif/paris_cif.yuv"),
    "720p": ("1280x720", 30, "720p", "720p/FourPeople_1280x720_60.yuv"),
    "1080p": ("1920x1080", 60, "1080p", "hd/BasketballDrive_1920x1080_50.yuv"),
}
LFB_RE = re.compile(r"\(lfb\s+([0-9]+)x([0-9]+)([KMG]B)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Final consolidated JSON report")
    parser.add_argument("--data-root", type=Path, default=Path("/home/oelghati/datasets"))
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=ROOT / "build/engines/dcvcrt-cvpr2025/orin-nano-l4t3647",
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-release")
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--qp", type=int, default=32)
    parser.add_argument("--gops", nargs="+", type=int, default=[1, 97])
    parser.add_argument("--resolutions", nargs="+", choices=RESOLUTIONS, default=list(RESOLUTIONS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--execution-mode",
        choices=("automatic", "low-memory", "performance"),
        default="automatic",
    )
    parser.add_argument("--prepare-system", action="store_true", help="Run sudo nvpmodel -m 2 and jetson_clocks")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()
    if args.frames <= 0 or args.repetitions <= 0 or args.jobs <= 0:
        parser.error("frames, repetitions, and jobs must be positive")
    if args.warmup_frames < 0 or args.idle_seconds < 0 or args.interval_ms <= 0:
        parser.error("warm-up/idle must be non-negative and interval must be positive")
    return args


def command_result(command: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None) -> dict[str, Any]:
    def decoded(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "return_code": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "return_code": None,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": decoded(getattr(error, "stdout", None)),
            "stderr": decoded(getattr(error, "stderr", None)) or str(error),
        }


def short_command(command: list[str]) -> dict[str, Any]:
    return command_result(command, timeout=10)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tegrastats_sample() -> dict[str, Any]:
    result = command_result(["tegrastats", "--interval", "1000"], timeout=1.5)
    text = result["stdout"].splitlines()[0] if result["stdout"].splitlines() else ""
    match = LFB_RE.search(text)
    result["sample"] = text
    if match:
        unit_scale = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}[match.group(3)]
        result["largest_free_block_count"] = int(match.group(1))
        result["largest_free_block_bytes"] = int(match.group(2)) * unit_scale
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.execution_mode == "automatic":
        environment.pop("NVCR_TENSORRT_LOW_MEMORY_MODE", None)
    else:
        environment["NVCR_TENSORRT_LOW_MEMORY_MODE"] = (
            "1" if args.execution_mode == "low-memory" else "0"
        )
    nvcr = args.build_dir / "cli/nvcr"
    work_dir = Path(tempfile.mkdtemp(prefix="nvcr-orin-benchmark-"))

    report: dict[str, Any] = {
        "schema": "nvcr.benchmark.orin-release.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "protocol": {
            "frames": args.frames,
            "warmup_frames": args.warmup_frames,
            "repetitions": args.repetitions,
            "qp": args.qp,
            "gops": args.gops,
            "resolutions": args.resolutions,
            "tensorrt_execution_mode": args.execution_mode,
            "energy_enabled": not args.skip_energy,
            "energy_idle_seconds": args.idle_seconds,
            "energy_interval_ms": args.interval_ms,
        },
        "platform": {
            "uname": platform.uname()._asdict(),
            "python": sys.version,
            "l4t_release": Path("/etc/nv_tegra_release").read_text(encoding="utf-8", errors="replace")
            if Path("/etc/nv_tegra_release").is_file() else None,
            "nvpmodel": short_command(["nvpmodel", "-q"]),
            "jetson_clocks": short_command(["jetson_clocks", "--show"]),
            "nvidia_smi": short_command(["nvidia-smi"]),
            "compiler": short_command(["c++", "--version"]),
            "cuda": short_command(["/usr/local/cuda/bin/nvcc", "--version"]),
            "tensorrt": short_command(["dpkg-query", "-W", "-f=${Version}", "libnvinfer10"]),
            "memory_before": tegrastats_sample(),
        },
        "repository": {
            "commit": short_command(["git", "rev-parse", "HEAD"])["stdout"].strip(),
            "status": short_command(["git", "status", "--short"])["stdout"].splitlines(),
        },
        "preparation": [],
        "validation": [],
        "matrix_cases": [],
        "energy_cases": [],
    }

    if args.prepare_system:
        report["preparation"].append(command_result(["sudo", "nvpmodel", "-m", "2"]))
        report["preparation"].append(command_result(["sudo", "jetson_clocks"]))

    inputs: dict[str, Path] = {}
    engines: dict[str, Path] = {}
    for label in args.resolutions:
        _, _, profile, relative_input = RESOLUTIONS[label]
        inputs[label] = args.data_root / relative_input
        engines[label] = args.engine_root / profile
        report["validation"].append(command_result([str(ARTIFACTS), "validate", str(engines[label]), "--json"]))

    report["inputs"] = {
        label: {"path": str(path), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256(path)}
        for label, path in inputs.items()
    }
    report["engine_bundles"] = {
        label: {
            "path": str(path),
            "engine_manifest_sha256": sha256(path / "engine_manifest.json"),
            "i_frame_manifest_sha256": sha256(path / "i_frame_manifest.json"),
            "p_frame_manifest_sha256": sha256(path / "p_frame_manifest.json"),
        }
        for label, path in engines.items()
    }

    if not args.skip_build:
        print("[orin-benchmark] building Release", flush=True)
        report["build"] = command_result(["cmake", "--build", str(args.build_dir), "-j", str(args.jobs)])
    if not args.skip_tests:
        print("[orin-benchmark] running registered tests", flush=True)
        report["tests"] = command_result(["ctest", "--test-dir", str(args.build_dir), "--output-on-failure"])

    for label in args.resolutions:
        size, fps, profile, _ = RESOLUTIONS[label]
        for gop in args.gops:
            print(f"[orin-benchmark] matrix resolution={label} gop={gop}", flush=True)
            jsonl = work_dir / f"{label}-gop{gop}.jsonl"
            command = [
                str(MATRIX), "--nvcr", str(nvcr), "--frames", str(args.frames), "--qp", str(args.qp),
                "--gops", str(gop), "--resolutions", label, "--repetitions", str(args.repetitions),
                "--warmup-frames", str(args.warmup_frames), f"--{label}-input", str(inputs[label]),
                f"--{label}-engine-dir", str(engines[label]), "--output-dir", str(work_dir), "--jsonl", str(jsonl),
            ]
            execution = command_result(command, env=environment)
            rows = load_jsonl(jsonl)
            report["matrix_cases"].append({
                "resolution": label, "size": size, "source_fps": fps, "gop": gop,
                "status": "passed" if execution["return_code"] == 0 else "failed",
                "execution": execution, "rows": rows,
                "aggregates": [row for row in rows if row.get("run_index") == "average"],
            })
            print(f"[orin-benchmark] matrix resolution={label} gop={gop} status={report['matrix_cases'][-1]['status']}", flush=True)

            if args.skip_energy or gop != 97 or execution["return_code"] != 0:
                continue
            stream = work_dir / f"energy-{label}-gop{gop}.nvcr"
            for operation in ("encode", "decode"):
                print(f"[orin-benchmark] energy resolution={label} operation={operation}", flush=True)
                energy_json = work_dir / f"energy-{label}-{operation}.json"
                if operation == "encode":
                    codec_command = [str(nvcr), "encode", "-i", str(inputs[label]), "-o", str(stream), "-s", size,
                                     "-r", str(fps), "--frames", str(args.frames), "--gop-size", str(gop),
                                     "--qp", str(args.qp), "--engine-dir", str(engines[label])]
                else:
                    codec_command = [str(nvcr), "decode", "-i", str(stream), "-o", "/dev/null", "--frames",
                                     str(args.frames), "--engine-dir", str(engines[label])]
                energy_command = [str(ENERGY), "--idle-seconds", str(args.idle_seconds), "--interval-ms",
                                  str(args.interval_ms), "--frames", str(args.frames), "--output-json",
                                  str(energy_json), "--", *codec_command]
                energy_execution = command_result(energy_command, env=environment)
                energy_data = json.loads(energy_json.read_text(encoding="utf-8")) if energy_json.is_file() else None
                report["energy_cases"].append({"resolution": label, "gop": gop, "operation": operation,
                                                "execution": energy_execution, "measurement": energy_data})

    report["platform"]["memory_after"] = tegrastats_sample()
    failures = [case for case in report["matrix_cases"] if case["status"] != "passed"]
    validation_failed = any(item["return_code"] != 0 for item in report["validation"])
    build_failed = report.get("build", {}).get("return_code") not in (None, 0)
    tests_failed = report.get("tests", {}).get("return_code") not in (None, 0)
    report["status"] = "complete" if not (failures or validation_failed or build_failed or tests_failed) else "partial"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "matrix_cases_planned": len(args.resolutions) * len(args.gops),
        "matrix_cases_passed": len(report["matrix_cases"]) - len(failures),
        "matrix_cases_failed": len(failures),
        "documentation_instruction": (
            "Update docs/performance.md and ROADMAP.md only from passed aggregate rows. "
            "Preserve failed cases and stderr as labeled history; do not advance release gates when status is partial."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({report['status']})")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    shutil.rmtree(work_dir, ignore_errors=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
