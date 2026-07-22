#!/usr/bin/env python3
"""Measure Jetson tegrastats power while running an NVCR command."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO


RAIL_RE = re.compile(r"\b([A-Za-z0-9_]+)\s+([0-9.]+)(m?W)(?:/[0-9.]+m?W)?")
FRAMES_RE = re.compile(r"\b(?:Encoded|Decoded)\s+([0-9]+)\s+frame")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a command while sampling Jetson tegrastats rails, then report "
            "energy and joules/frame. Pass the profiled command after --."
        )
    )
    parser.add_argument("--tegrastats", default="tegrastats")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--rail", action="append", default=None)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--tegrastats-log", type=Path, default=None)
    parser.add_argument("--stdout-log", type=Path, default=None)
    parser.add_argument("--stderr-log", type=Path, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")
    if args.idle_seconds < 0.0:
        parser.error("--idle-seconds must be non-negative")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def parse_rails(line: str) -> dict[str, float]:
    rails: dict[str, float] = {}
    for match in RAIL_RE.finditer(line):
        value = float(match.group(2))
        if match.group(3) == "mW":
            value /= 1000.0
        rails[match.group(1)] = value
    return rails


class TegrastatsSampler:
    def __init__(self, executable: str, interval_ms: int, log_path: Path | None) -> None:
        self._proc = subprocess.Popen(
            [executable, "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._samples: list[tuple[float, dict[str, float], str]] = []
        self._lock = threading.Lock()
        self._start = time.monotonic()
        self._phase = "idle"
        self._log_file = log_path.open("w", encoding="utf-8") if log_path else None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            timestamp = time.monotonic() - self._start
            text = line.rstrip("\n")
            rails = parse_rails(text)
            with self._lock:
                phase = self._phase
                if rails:
                    self._samples.append((timestamp, rails, phase))
                if self._log_file is not None:
                    self._log_file.write(f"{timestamp:.6f} {phase} {text}\n")
                    self._log_file.flush()

    def snapshot(self) -> list[tuple[float, dict[str, float], str]]:
        with self._lock:
            return list(self._samples)

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._thread.join(timeout=2)
        if self._log_file is not None:
            self._log_file.close()


def select_rails(samples: list[tuple[float, dict[str, float], str]], requested: list[str] | None) -> list[str]:
    seen = {rail for _, rails, _ in samples for rail in rails}
    if requested:
        missing = [rail for rail in requested if rail not in seen]
        if missing:
            raise RuntimeError(f"missing requested rail(s): {', '.join(missing)}")
        return requested
    if "VDD_IN" in seen:
        return ["VDD_IN"]
    if not seen:
        raise RuntimeError("tegrastats did not produce parseable power rails")
    return sorted(seen)


def average_power(samples: list[tuple[float, dict[str, float], str]], rails: list[str], phase: str) -> float:
    values = [sum(values.get(rail, 0.0) for rail in rails) for _, values, sample_phase in samples if sample_phase == phase]
    return sum(values) / len(values) if values else 0.0


def integrate_energy(samples: list[tuple[float, dict[str, float], str]], rails: list[str], phase: str) -> float:
    phase_samples = [(t, values) for t, values, sample_phase in samples if sample_phase == phase]
    if len(phase_samples) < 2:
        return 0.0
    joules = 0.0
    previous_t, previous_values = phase_samples[0]
    previous_power = sum(previous_values.get(rail, 0.0) for rail in rails)
    for current_t, current_values in phase_samples[1:]:
        current_power = sum(current_values.get(rail, 0.0) for rail in rails)
        joules += (previous_power + current_power) * 0.5 * (current_t - previous_t)
        previous_t = current_t
        previous_power = current_power
    return joules


def tee_stream(source: TextIO, sink: TextIO, log_path: Path | None, chunks: list[str]) -> None:
    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        for line in source:
            sink.write(line)
            sink.flush()
            chunks.append(line)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()


def parse_frame_count(stdout: str, stderr: str) -> int | None:
    for text in (stdout, stderr):
        matches = FRAMES_RE.findall(text)
        if matches:
            return int(matches[-1])
    return None


def main() -> int:
    args = parse_args()
    sampler = TegrastatsSampler(args.tegrastats, args.interval_ms, args.tegrastats_log)
    try:
        if args.idle_seconds > 0.0:
            time.sleep(args.idle_seconds)

        sampler.set_phase("active")
        active_started = time.monotonic()
        command = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert command.stdout is not None
        assert command.stderr is not None
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(target=tee_stream, args=(command.stdout, sys.stdout, args.stdout_log, stdout_chunks))
        stderr_thread = threading.Thread(target=tee_stream, args=(command.stderr, sys.stderr, args.stderr_log, stderr_chunks))
        stdout_thread.start()
        stderr_thread.start()
        try:
            return_code = command.wait()
        except KeyboardInterrupt:
            command.send_signal(signal.SIGINT)
            return_code = command.wait()
        active_elapsed = time.monotonic() - active_started
        sampler.set_phase("done")
        stdout_thread.join()
        stderr_thread.join()
        time.sleep(max(0.2, args.interval_ms / 1000.0))

        samples = sampler.snapshot()
        rails = select_rails(samples, args.rail)
        idle_power_w = average_power(samples, rails, "idle")
        active_power_w = average_power(samples, rails, "active")
        active_energy_j = integrate_energy(samples, rails, "active")
        idle_adjusted_j = max(0.0, active_energy_j - idle_power_w * active_elapsed)
        frames = args.frames if args.frames is not None else parse_frame_count("".join(stdout_chunks), "".join(stderr_chunks))
        fps = frames / active_elapsed if frames and active_elapsed > 0.0 else None

        summary = {
            "command": args.command,
            "return_code": return_code,
            "rails": rails,
            "active_sample_count": sum(1 for _, _, phase in samples if phase == "active"),
            "idle_seconds": args.idle_seconds,
            "active_seconds": active_elapsed,
            "idle_power_w": idle_power_w,
            "active_average_power_w": active_power_w,
            "active_energy_j": active_energy_j,
            "idle_adjusted_energy_j": idle_adjusted_j,
            "frames": frames,
            "fps_wall": fps,
            "active_j_per_frame": active_energy_j / frames if frames else None,
            "idle_adjusted_j_per_frame": idle_adjusted_j / frames if frames else None,
        }
        if args.output_json is not None:
            args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print("\nEnergy summary")
        print(f"  rails: {', '.join(rails)}")
        print(f"  active time: {active_elapsed:.3f} s")
        print(f"  idle power: {idle_power_w:.3f} W")
        print(f"  active average power: {active_power_w:.3f} W")
        print(f"  active energy: {active_energy_j:.3f} J")
        print(f"  idle-adjusted energy: {idle_adjusted_j:.3f} J")
        if frames:
            print(f"  frames: {frames}")
            if fps is not None:
                print(f"  wall fps: {fps:.3f}")
            print(f"  active energy/frame: {active_energy_j / frames:.3f} J")
            print(f"  idle-adjusted energy/frame: {idle_adjusted_j / frames:.3f} J")
        else:
            print("  frames: unknown (pass --frames N if needed)")
        return return_code
    finally:
        sampler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
