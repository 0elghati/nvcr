#!/usr/bin/env python3
"""Run one command while sampling process and target memory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


RAM_RE = re.compile(r"\bRAM\s+([0-9]+(?:\.[0-9]+)?)/([0-9]+(?:\.[0-9]+)?)MB\b")
LFB_RE = re.compile(r"\(lfb\s+([0-9]+)x([0-9]+)([KMG]B)\)")


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def process_memory_mb(pid: int) -> float | None:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    values: dict[str, float] = {}
    for line in lines:
        if line.startswith(("VmHWM:", "VmRSS:")):
            fields = line.split()
            if len(fields) >= 2:
                values[fields[0].rstrip(":")] = float(fields[1]) / 1024.0
    return values.get("VmHWM", values.get("VmRSS"))


def gpu_process_memory_mb(pid: int) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == pid:
                return float(fields[1])
        except ValueError:
            continue
    return None


def lfb_mb(match: re.Match[str]) -> float:
    scale = {"KB": 1.0 / 1024.0, "MB": 1.0, "GB": 1024.0}[match.group(3)]
    return float(match.group(2)) * scale


class TegrastatsSampler:
    def __init__(self, interval_ms: int) -> None:
        self.interval_ms = max(interval_ms, 20)
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.peak_system_memory_mb: float | None = None
        self.min_largest_free_block_mb: float | None = None
        self.available = False

    def start(self) -> None:
        if shutil.which("tegrastats") is None:
            return
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self.process = None
            return
        self.available = True
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            ram_match = RAM_RE.search(line)
            if ram_match:
                used = float(ram_match.group(1))
                if self.peak_system_memory_mb is None or used > self.peak_system_memory_mb:
                    self.peak_system_memory_mb = used
            lfb_match = LFB_RE.search(line)
            if lfb_match:
                value = lfb_mb(lfb_match)
                if self.min_largest_free_block_mb is None or value < self.min_largest_free_block_mb:
                    self.min_largest_free_block_mb = value

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.sample_ms < 0:
        parser.error("--sample-ms must be non-negative")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def main() -> int:
    args = parse_args()
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    sampler = TegrastatsSampler(args.sample_ms)
    sampler.start()
    samplers = ["procfs"]
    if shutil.which("nvidia-smi") is not None:
        samplers.append("nvidia-smi")
    if sampler.available:
        samplers.append("tegrastats")

    peak_host: float | None = None
    peak_gpu: float | None = None
    return_code: int | None
    try:
        process = subprocess.Popen(
            args.command,
            text=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        args.stdout.write_bytes(b"")
        args.stderr.write_text(str(error) + "\n", encoding="utf-8")
        sampler.stop()
        write_json(
            args.metrics_json,
            {
                "return_code": None,
                "elapsed_seconds": finite(time.monotonic() - started),
                "peak_memory_mb": None,
                "peak_gpu_memory_mb": None,
                "peak_system_memory_mb": finite(sampler.peak_system_memory_mb),
                "min_largest_free_block_mb": finite(sampler.min_largest_free_block_mb),
                "memory_sample_ms": args.sample_ms,
                "memory_sampler": "+".join(samplers),
            },
        )
        return 127

    interval = max(args.sample_ms, 10) / 1000.0 if args.sample_ms > 0 else 0.1
    next_gpu_sample = 0.0
    host = process_memory_mb(process.pid)
    if host is not None:
        peak_host = host
    while process.poll() is None:
        host = process_memory_mb(process.pid)
        if host is not None:
            peak_host = host if peak_host is None else max(peak_host, host)
        now = time.monotonic()
        if args.sample_ms > 0 and now >= next_gpu_sample:
            gpu = gpu_process_memory_mb(process.pid)
            if gpu is not None:
                peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
            next_gpu_sample = now + interval
        time.sleep(min(interval, 0.05))

    stdout, stderr = process.communicate()
    sampler.stop()
    args.stdout.write_bytes(stdout or b"")
    args.stderr.write_bytes(stderr or b"")
    return_code = process.returncode
    write_json(
        args.metrics_json,
        {
            "return_code": return_code,
            "elapsed_seconds": finite(time.monotonic() - started),
            "peak_memory_mb": finite(peak_host),
            "peak_gpu_memory_mb": finite(peak_gpu),
            "peak_system_memory_mb": finite(sampler.peak_system_memory_mb),
            "min_largest_free_block_mb": finite(sampler.min_largest_free_block_mb),
            "memory_sample_ms": args.sample_ms,
            "memory_sampler": "+".join(samplers),
        },
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
