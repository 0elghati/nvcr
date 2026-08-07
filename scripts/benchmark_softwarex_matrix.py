#!/usr/bin/env python3
"""Run the NVCR SoftwareX matrix and write a self-describing evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import nvcr_artifacts as artifacts  # noqa: E402
import nvcr_device as device  # noqa: E402


RESULT_SCHEMA = "nvcr.softwarex.result.v1"
INPUT_SCHEMA = "nvcr.softwarex.inputs.v1"
PYTHON_REFERENCE_SCHEMA = "nvcr.softwarex.python-reference.v1"
RESULT_FILES = {
    "exact": "exact-results.jsonl",
    "same_compute_capability": "same-compute-results.jsonl",
    "ampere_plus": "ampere-plus-results.jsonl",
}
CORE_TESTS = (
    "nvcr_artifact_profiles",
    "nvcr_engine_catalog",
    "nvcr_softwarex_driver",
    "nvcr_parser_fuzz_boundaries",
    "nvcr_contract_tests",
    "nvcr_format_contract_tests",
    "nvcr_artifact_resolver_tests",
    "nvcr_artifact_catalog_tests",
    "nvcr_dcvcrt_payloads",
    "nvcr_rans_conformance",
    "nvcr_cli_accepts_inter_gop",
)
REQUIRED_EXACT_PROFILES = frozenset(("qcif", "cif", "360p", "720p", "1080p"))
REQUIRED_PERFORMANCE_METRICS = (
    "encode_fps_mean",
    "encode_fps_stddev",
    "decode_fps_mean",
    "decode_fps_stddev",
    "total_wall_time_ms",
    "payload_bytes",
    "bits_per_pixel",
)
REQUIRED_PROFILE_METRICS = (
    "encode_latency_ms_median",
    "encode_latency_ms_p95",
    "decode_latency_ms_median",
    "decode_latency_ms_p95",
    "first_frame_latency_ms",
    "psnr_y",
    "psnr_u",
    "psnr_v",
    "psnr_yuv",
    "peak_gpu_memory_mb",
    "peak_host_memory_mb",
)
NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|inf)"
ENCODE_SUMMARY = re.compile(
    rf"^Encoded ([0-9]+) frame\(s\), ([0-9]+) payload bytes, "
    rf"codec time ({NUMBER}) s \(({NUMBER}) fps\)$",
    re.MULTILINE,
)
DECODE_SUMMARY = re.compile(
    rf"^Decoded ([0-9]+) frame\(s\), codec time ({NUMBER}) s "
    rf"\(({NUMBER}) fps\)$",
    re.MULTILINE,
)
QUALITY_SUMMARY = re.compile(
    rf"^Quality ([0-9]+) frame\(s\): PSNR-Y ({NUMBER}) dB, "
    rf"PSNR-U ({NUMBER}) dB, PSNR-V ({NUMBER}) dB, "
    rf"PSNR-YUV ({NUMBER}) dB$",
    re.MULTILINE,
)
ENCODE_LATENCY = re.compile(
    rf"^frame [0-9]+: encoded [0-9]+ payload bytes in ({NUMBER}) ms$",
    re.MULTILINE,
)
DECODE_LATENCY = re.compile(
    rf"^frame [0-9]+: decoded in ({NUMBER}) ms$",
    re.MULTILINE,
)


class SoftwareXError(RuntimeError):
    """The requested run cannot produce valid SoftwareX evidence."""


@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str
    profile: str
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    pixel_format: str
    redistribution: str

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3 // 2

    @property
    def measured_bytes(self) -> int:
        return self.frame_bytes * self.frames


@dataclass
class CommandResult:
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    elapsed_ms: float
    peak_host_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "return_code": self.return_code,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "peak_host_memory_mb": self.peak_host_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "stdout_tail": self.stdout[-8000:],
            "stderr_tail": self.stderr[-8000:],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SoftwareXError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SoftwareXError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path, byte_count: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    try:
        with path.open("rb") as stream:
            while remaining is None or remaining > 0:
                block_size = 4 * 1024 * 1024
                if remaining is not None:
                    block_size = min(block_size, remaining)
                block = stream.read(block_size)
                if not block:
                    break
                digest.update(block)
                if remaining is not None:
                    remaining -= len(block)
    except OSError as error:
        raise SoftwareXError(f"cannot hash {path}: {error}") from error
    if remaining not in (None, 0):
        raise SoftwareXError(f"{path} is shorter than the measured input prefix")
    return digest.hexdigest()


def engine_bundle_digest(root: Path, manifest: dict[str, Any]) -> str:
    checksum_name = manifest.get("checksum_manifest")
    if not isinstance(checksum_name, str) or not checksum_name:
        raise SoftwareXError(f"engine manifest has no checksum manifest: {root}")
    digest = hashlib.sha256()
    digest.update(b"nvcr.engine-directory.v1\0")
    for name in ("engine_manifest.json", checksum_name):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(root / name)))
    return digest.hexdigest()


def parse_float(value: str) -> float:
    return float("inf") if value == "inf" else float(value)


def finite_json_number(value: float | None) -> float | None:
    if value is None or value == float("inf") or value == float("-inf"):
        return None
    return round(value, 6)


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def parse_encode_output(
    output: str,
    expected_frames: int,
    *,
    require_latencies: bool = True,
) -> dict[str, Any]:
    summary = ENCODE_SUMMARY.search(output)
    if summary is None:
        raise SoftwareXError("NVCR encode summary is missing or malformed")
    frames = int(summary.group(1))
    if frames != expected_frames:
        raise SoftwareXError(
            f"NVCR encoded {frames} frames, expected {expected_frames}"
        )
    latencies = [parse_float(value) for value in ENCODE_LATENCY.findall(output)]
    if require_latencies and len(latencies) != expected_frames:
        raise SoftwareXError(
            f"NVCR emitted {len(latencies)} encode latency rows, expected {expected_frames}"
        )
    return {
        "frames": frames,
        "payload_bytes": int(summary.group(2)),
        "codec_time_seconds": parse_float(summary.group(3)),
        "fps": parse_float(summary.group(4)),
        "latencies_ms": latencies,
    }


def parse_decode_output(
    output: str,
    expected_frames: int,
    *,
    require_latencies: bool = True,
    require_quality: bool = True,
) -> dict[str, Any]:
    summary = DECODE_SUMMARY.search(output)
    quality = QUALITY_SUMMARY.search(output)
    if summary is None:
        raise SoftwareXError("NVCR decode summary is missing or malformed")
    if require_quality and quality is None:
        raise SoftwareXError("NVCR quality summary is missing or malformed")
    frames = int(summary.group(1))
    if frames != expected_frames:
        raise SoftwareXError(
            "NVCR decode frame count does not match the requested frame count"
        )
    if quality is not None and int(quality.group(1)) != expected_frames:
        raise SoftwareXError(
            "NVCR quality frame count does not match the requested frame count"
        )
    latencies = [parse_float(value) for value in DECODE_LATENCY.findall(output)]
    if require_latencies and len(latencies) != expected_frames:
        raise SoftwareXError(
            f"NVCR emitted {len(latencies)} decode latency rows, expected {expected_frames}"
        )
    return {
        "frames": frames,
        "codec_time_seconds": parse_float(summary.group(2)),
        "fps": parse_float(summary.group(3)),
        "latencies_ms": latencies,
        "psnr_y": parse_float(quality.group(2)) if quality is not None else None,
        "psnr_u": parse_float(quality.group(3)) if quality is not None else None,
        "psnr_v": parse_float(quality.group(4)) if quality is not None else None,
        "psnr_yuv": parse_float(quality.group(5)) if quality is not None else None,
    }


def command_text(command: Iterable[str]) -> str:
    return shlex.join(str(item) for item in command)


def run_command(
    command: list[str],
    *,
    cwd: Path = REPOSITORY_ROOT,
    environment: dict[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            (time.monotonic() - started) * 1000.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        stdout = getattr(error, "stdout", "") or ""
        stderr = getattr(error, "stderr", "") or str(error)
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return CommandResult(
            command,
            None,
            stdout,
            stderr,
            (time.monotonic() - started) * 1000.0,
        )


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
    result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=2.0,
    )
    if result.return_code != 0:
        return None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == pid:
                return float(fields[1])
        except ValueError:
            continue
    return None


def run_monitored(
    command: list[str],
    *,
    environment: dict[str, str],
    sample_interval_ms: int,
) -> CommandResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        return CommandResult(command, None, "", str(error), 0.0)
    peak_host: float | None = None
    peak_gpu: float | None = None
    interval = max(sample_interval_ms, 10) / 1000.0
    next_gpu_sample = 0.0
    while process.poll() is None:
        host = process_memory_mb(process.pid)
        if host is not None:
            peak_host = host if peak_host is None else max(peak_host, host)
        now = time.monotonic()
        if sample_interval_ms > 0 and now >= next_gpu_sample:
            gpu = gpu_process_memory_mb(process.pid)
            if gpu is not None:
                peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
            next_gpu_sample = now + interval
        time.sleep(min(interval, 0.05))
    stdout, stderr = process.communicate()
    host = process_memory_mb(process.pid)
    if host is not None:
        peak_host = host if peak_host is None else max(peak_host, host)
    return CommandResult(
        command,
        process.returncode,
        stdout,
        stderr,
        (time.monotonic() - started) * 1000.0,
        finite_json_number(peak_host),
        finite_json_number(peak_gpu),
    )


def load_sequences(path: Path) -> list[SequenceSpec]:
    document = read_json(path)
    if document.get("schema") != INPUT_SCHEMA:
        raise SoftwareXError(f"{path} must use schema {INPUT_SCHEMA}")
    raw_sequences = document.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise SoftwareXError(f"{path} must contain a non-empty sequences list")
    sequences: list[SequenceSpec] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_sequences):
        if not isinstance(raw, dict):
            raise SoftwareXError(f"sequence {index} must be an object")
        try:
            sequence = SequenceSpec(
                sequence_id=str(raw["sequence_id"]),
                profile=artifacts.canonical_profile_name(str(raw["profile"])),
                path=Path(str(raw["path"])).expanduser().resolve(),
                width=int(raw["width"]),
                height=int(raw["height"]),
                fps=float(raw["fps"]),
                frames=int(raw["frames"]),
                pixel_format=str(raw.get("pixel_format", "yuv420p8")),
                redistribution=str(raw.get("redistribution", "unknown")),
            )
        except (KeyError, TypeError, ValueError, artifacts.ValidationError) as error:
            raise SoftwareXError(f"invalid sequence {index}: {error}") from error
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sequence.sequence_id) is None:
            raise SoftwareXError(f"sequence {index} has an invalid sequence_id")
        if sequence.width <= 0 or sequence.height <= 0 or sequence.frames <= 0:
            raise SoftwareXError(f"sequence {sequence.sequence_id} has invalid dimensions/frames")
        if sequence.width % 2 or sequence.height % 2:
            raise SoftwareXError(f"sequence {sequence.sequence_id} must have even dimensions")
        if sequence.fps <= 0 or sequence.pixel_format != "yuv420p8":
            raise SoftwareXError(f"sequence {sequence.sequence_id} must be YUV420P8 with fps > 0")
        identity = (sequence.sequence_id, sequence.profile)
        if identity in identities:
            raise SoftwareXError(f"duplicate sequence/profile identity: {identity}")
        identities.add(identity)
        sequences.append(sequence)
    return sequences


def engine_directory(root: Path, profile: str, overrides: dict[str, Path]) -> Path:
    if profile in overrides:
        return overrides[profile]
    candidates = (
        root,
        root / f"dcvcrt-{profile}",
        root / profile,
        root / "profiles" / "dcvcrt" / profile,
        root / "profiles" / "dcvcrt" / f"{profile}-fp16",
    )
    for candidate in candidates:
        manifest = candidate / "engine_manifest.json"
        if not manifest.is_file():
            continue
        document = read_json(manifest)
        try:
            candidate_profile = artifacts.canonical_profile_name(
                str(document.get("engine_profile_id", ""))
            )
        except artifacts.ValidationError:
            continue
        if candidate_profile == profile:
            return candidate.resolve()
    raise SoftwareXError(f"no engine bundle for profile {profile} under {root}")


def parse_mapping(values: list[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or not raw_path:
            raise SoftwareXError(f"{option} expects PROFILE=PATH: {value}")
        profile = artifacts.canonical_profile_name(key)
        if profile in output:
            raise SoftwareXError(f"duplicate {option} mapping for {profile}")
        output[profile] = Path(raw_path).expanduser().resolve()
    return output


def parse_cuda_version(value: str) -> tuple[int, int]:
    match = re.match(r"^([0-9]+)\.([0-9]+)", value)
    if match is None:
        raise SoftwareXError(f"invalid CUDA version in target profile: {value}")
    return int(match.group(1)), int(match.group(2))


def parse_semver(value: str) -> tuple[int, int, int]:
    match = re.match(r"^([0-9]+)\.([0-9]+)(?:\.([0-9]+))?", value)
    if match is None:
        raise SoftwareXError(f"invalid version in target profile: {value}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def detected_cuda_version(identity: dict[str, Any]) -> tuple[int, int]:
    encoded = int(identity["cuda_runtime_version"])
    return encoded // 1000, (encoded % 1000) // 10


def detected_tensorrt_version(identity: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(identity["tensorrt_version_major"]),
        int(identity["tensorrt_version_minor"]),
        int(identity["tensorrt_version_patch"]),
    )


def validate_test_target(
    target: dict[str, Any], identity: dict[str, Any], compatibility_class: str
) -> None:
    target_id = str(target.get("id", ""))
    host = target.get("host")
    gpu = target.get("gpu")
    if not target_id or not isinstance(host, dict) or not isinstance(gpu, dict):
        raise SoftwareXError("target profile is missing id, host, or gpu identity")
    architecture = str(host.get("architecture", ""))
    if architecture != identity.get("architecture"):
        raise SoftwareXError(
            f"target architecture is {architecture}, detected {identity.get('architecture')}"
        )
    if target.get("precision") != "fp16":
        raise SoftwareXError("SoftwareX v1 requires an FP16 target profile")
    is_jetson = architecture == "aarch64" or "orin" in target_id.lower()
    if is_jetson and compatibility_class != "exact":
        raise SoftwareXError("Jetson/L4T evaluation is exact-only")
    expected_sm = str(gpu.get("compute_capability", ""))
    detected_sm = (
        f"{identity['compute_capability_major']}."
        f"{identity['compute_capability_minor']}"
    )
    if expected_sm != detected_sm:
        raise SoftwareXError(f"target SM is {expected_sm}, detected {detected_sm}")
    if compatibility_class == "exact":
        if str(gpu.get("name", "")) != str(identity.get("device_name", "")):
            raise SoftwareXError(
                f"target GPU is {gpu.get('name')}, detected {identity.get('device_name')}"
            )
        expected_multiprocessors = gpu.get("multiprocessor_count")
        if expected_multiprocessors != identity.get("multiprocessor_count"):
            raise SoftwareXError(
                "target and detected multiprocessor counts do not match"
            )
        if parse_cuda_version(str(target.get("cuda", ""))) != detected_cuda_version(identity):
            raise SoftwareXError("target and detected CUDA runtime versions do not match")
        if parse_semver(str(target.get("tensorrt", ""))) != detected_tensorrt_version(identity):
            raise SoftwareXError("target and detected TensorRT versions do not match")


def validate_engine_identity(
    root: Path,
    manifest: dict[str, Any],
    *,
    profile: str,
    compatibility_class: str,
    model_profile_path: Path,
    test_target_path: Path,
    test_target: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    try:
        artifacts.validate_engine_bundle(root)
    except artifacts.ValidationError as error:
        raise SoftwareXError(f"engine bundle validation failed for {profile}: {error}") from error
    manifest_profile = artifacts.canonical_profile_name(
        str(manifest.get("engine_profile_id", ""))
    )
    if manifest_profile != profile:
        raise SoftwareXError(
            f"engine bundle {root} has profile {manifest_profile}, expected {profile}"
        )
    manifest_class = str(manifest.get("hardware_compatibility", "exact"))
    if manifest_class != compatibility_class:
        raise SoftwareXError(
            f"engine bundle {root} is {manifest_class}, requested {compatibility_class}"
        )
    expected_model_digest = sha256_file(model_profile_path)
    if manifest.get("model_profile_sha256") != expected_model_digest:
        raise SoftwareXError(
            f"engine bundle {root} was not built from the current model profile"
        )
    engine_profile_path = REPOSITORY_ROOT / "configs" / "engine-profiles" / f"{profile}.json"
    if manifest.get("engine_profile_sha256") != sha256_file(engine_profile_path):
        raise SoftwareXError(
            f"engine bundle {root} was not built from the current {profile} profile"
        )
    build_target_id = str(manifest.get("target_profile_id", ""))
    if not build_target_id:
        raise SoftwareXError(f"engine bundle {root} has no target profile id")
    if build_target_id == str(test_target.get("id")):
        build_target_path = test_target_path
    else:
        build_target_path = REPOSITORY_ROOT / "configs" / "targets" / f"{build_target_id}.json"
    if not build_target_path.is_file():
        raise SoftwareXError(
            f"engine build target profile is unavailable: {build_target_path}"
        )
    if manifest.get("target_profile_sha256") != sha256_file(build_target_path):
        raise SoftwareXError(
            f"engine bundle {root} was not built from its current target profile"
        )
    detected_sm = (
        int(identity["compute_capability_major"]),
        int(identity["compute_capability_minor"]),
    )
    manifest_sm = (
        int(manifest.get("compute_capability_major", -1)),
        int(manifest.get("compute_capability_minor", -1)),
    )
    if compatibility_class in ("exact", "same_compute_capability") and manifest_sm != detected_sm:
        raise SoftwareXError(
            f"engine bundle SM {manifest_sm[0]}.{manifest_sm[1]} does not match detected "
            f"SM {detected_sm[0]}.{detected_sm[1]}"
        )
    if compatibility_class == "exact":
        if build_target_id != str(test_target.get("id")):
            raise SoftwareXError("exact engine target does not match the test target")
        if manifest.get("device_name") != identity.get("device_name"):
            raise SoftwareXError("exact engine GPU name does not match the detected GPU")
        if manifest.get("multiprocessor_count") != identity.get("multiprocessor_count"):
            raise SoftwareXError("exact engine SM count does not match the detected GPU")
    manifest_tensorrt = (
        int(manifest.get("tensorrt_version_major", -1)),
        int(manifest.get("tensorrt_version_minor", -1)),
        int(manifest.get("tensorrt_version_patch", -1)),
    )
    if manifest_tensorrt != detected_tensorrt_version(identity):
        raise SoftwareXError("engine and detected TensorRT versions do not match")
    manifest_cuda = int(manifest.get("cuda_runtime_version", -1))
    detected_cuda = int(identity["cuda_runtime_version"])
    if compatibility_class == "exact" and manifest_cuda != detected_cuda:
        raise SoftwareXError("exact engine and detected CUDA runtime versions do not match")
    if compatibility_class != "exact" and (
        manifest_cuda // 1000 != detected_cuda // 1000 or manifest_cuda > detected_cuda
    ):
        raise SoftwareXError("compatible engine CUDA runtime is not usable on this target")
    return {
        "path": str(root),
        "build_target_id": build_target_id,
        "engine_profile_id": profile,
        "hardware_compatibility": manifest_class,
        "engine_manifest_sha256": sha256_file(root / "engine_manifest.json"),
        "engine_bundle_sha256": engine_bundle_digest(root, manifest),
        "engine_bundle_digest_kind": "manifest-and-checksum-sha256",
        "model_profile_sha256": expected_model_digest,
        "target_profile_sha256": str(manifest["target_profile_sha256"]),
        "engine_profile_sha256": str(manifest["engine_profile_sha256"]),
    }


def load_device_identity(path: Path | None, device_id: int) -> dict[str, Any]:
    if path is not None:
        identity = read_json(path)
    else:
        try:
            identity = device.detect_device_identity(device_id)
        except device.DeviceDetectionError as error:
            raise SoftwareXError(str(error)) from error
    required = (
        "operating_system",
        "architecture",
        "device_name",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
        "cuda_runtime_version",
        "tensorrt_version_major",
        "tensorrt_version_minor",
        "tensorrt_version_patch",
    )
    missing = [key for key in required if key not in identity]
    if missing:
        raise SoftwareXError(f"device identity is missing: {', '.join(missing)}")
    return identity


def os_release() -> str:
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    return platform.platform()


def first_line(command: list[str]) -> str:
    result = run_command(command, timeout=5.0)
    text = result.stdout.strip() or result.stderr.strip()
    return text.splitlines()[0] if text else ""


def driver_version() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        timeout=5.0,
    )
    return result.stdout.strip().splitlines()[0] if result.return_code == 0 else ""


def git_identity() -> tuple[str, bool]:
    commit = run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
    status = run_command(["git", "status", "--short"]).stdout.strip()
    if not commit:
        raise SoftwareXError("cannot determine the NVCR Git commit")
    return commit, bool(status)


def load_python_reference(
    path: Path | None,
    model_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    if path is None:
        return [], {}
    rows: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SoftwareXError(f"cannot read Python reference rows: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SoftwareXError(
                f"invalid Python reference JSONL line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict) or row.get("schema") != PYTHON_REFERENCE_SCHEMA:
            raise SoftwareXError(
                f"Python reference line {line_number} must use {PYTHON_REFERENCE_SCHEMA}"
            )
        key = (
            row.get("sequence_id"),
            row.get("width"),
            row.get("height"),
            row.get("frames"),
            row.get("qp"),
            row.get("gop_size"),
        )
        if key in index:
            raise SoftwareXError(f"duplicate Python reference case: {key}")
        for required in (
            "python_psnr_yuv",
            "python_payload_bytes",
            "python_vs_nvcr_psnr_yuv",
        ):
            value = row.get(required)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise SoftwareXError(f"Python reference case {key} is missing {required}")
        for required in (
            "input_sha256",
            "python_command",
            "python_source_commit",
            "image_checkpoint_sha256",
            "video_checkpoint_sha256",
        ):
            if not isinstance(row.get(required), str) or not row[required]:
                raise SoftwareXError(f"Python reference case {key} is missing {required}")
        if re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"]) is None:
            raise SoftwareXError(f"Python reference case {key} has an invalid input_sha256")
        expected_pins = {
            "python_source_commit": model_profile["upstream"]["commit"],
            "image_checkpoint_sha256": model_profile["checkpoints"]["image"]["sha256"],
            "video_checkpoint_sha256": model_profile["checkpoints"]["video"]["sha256"],
        }
        for field, expected in expected_pins.items():
            if row[field] != expected:
                raise SoftwareXError(
                    f"Python reference case {key} does not match pinned {field}"
                )
        rows.append(row)
        index[key] = row
    return rows, index


def reference_key(sequence: SequenceSpec, qp: int, gop_size: int) -> tuple[Any, ...]:
    return (
        sequence.sequence_id,
        sequence.width,
        sequence.height,
        sequence.frames,
        qp,
        gop_size,
    )


def load_exact_baseline(path: Path | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    if path is None:
        return {}
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SoftwareXError(f"cannot read exact baseline rows: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SoftwareXError(
                f"invalid exact baseline JSONL line {line_number}: {error}"
            ) from error
        if (
            not isinstance(row, dict)
            or row.get("schema") != RESULT_SCHEMA
            or row.get("compatibility_class") != "exact"
            or row.get("status") != "pass"
            or row.get("git_dirty") is not False
        ):
            raise SoftwareXError(
                f"exact baseline line {line_number} must be a passing exact {RESULT_SCHEMA} row"
            )
        key = (
            row.get("target_id"),
            row.get("sequence_id"),
            row.get("width"),
            row.get("height"),
            row.get("frames"),
            row.get("qp"),
            row.get("gop_size"),
        )
        if key in index:
            raise SoftwareXError(f"duplicate exact baseline case: {key}")
        for field in ("encode_fps_mean", "decode_fps_mean"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise SoftwareXError(
                    f"exact baseline case {key} has invalid {field}"
                )
        index[key] = row
    return index


def baseline_key(
    target_id: str, sequence: SequenceSpec, qp: int, gop_size: int
) -> tuple[Any, ...]:
    return (
        target_id,
        sequence.sequence_id,
        sequence.width,
        sequence.height,
        sequence.frames,
        qp,
        gop_size,
    )


def comparison_metrics(
    metrics: dict[str, Any], exact: dict[str, Any]
) -> dict[str, Any]:
    exact_encode = float(exact["encode_fps_mean"])
    exact_decode = float(exact["decode_fps_mean"])
    return {
        "compatibility_baseline_available": True,
        "exact_encode_fps_mean": finite_json_number(exact_encode),
        "exact_decode_fps_mean": finite_json_number(exact_decode),
        "encode_fps_ratio_vs_exact": finite_json_number(
            float(metrics["encode_fps_mean"]) / exact_encode
        ),
        "decode_fps_ratio_vs_exact": finite_json_number(
            float(metrics["decode_fps_mean"]) / exact_decode
        ),
    }


def aggregate_performance(
    runs: list[dict[str, Any]],
    *,
    sequence: SequenceSpec,
) -> dict[str, Any]:
    encode_fps = [float(run["encode"]["fps"]) for run in runs]
    decode_fps = [float(run["decode"]["fps"]) for run in runs]
    payloads = [int(run["encode"]["payload_bytes"]) for run in runs]
    payload_bytes = round(statistics.mean(payloads))
    return {
        "encode_fps_mean": finite_json_number(statistics.mean(encode_fps)),
        "encode_fps_stddev": finite_json_number(
            statistics.stdev(encode_fps) if len(runs) > 1 else 0.0
        ),
        "decode_fps_mean": finite_json_number(statistics.mean(decode_fps)),
        "decode_fps_stddev": finite_json_number(
            statistics.stdev(decode_fps) if len(runs) > 1 else 0.0
        ),
        "total_wall_time_ms": finite_json_number(
            statistics.mean(
                run["encode_result"].elapsed_ms + run["decode_result"].elapsed_ms
                for run in runs
            )
        ),
        "payload_bytes": payload_bytes,
        "payload_bytes_runs": payloads,
        "bits_per_pixel": finite_json_number(
            payload_bytes * 8.0 / (sequence.width * sequence.height * sequence.frames)
        ),
    }


def aggregate_profile(runs: list[dict[str, Any]]) -> dict[str, Any]:
    encode_latencies = [
        latency for run in runs for latency in run["encode"]["latencies_ms"]
    ]
    decode_latencies = [
        latency for run in runs for latency in run["decode"]["latencies_ms"]
    ]
    qualities = {
        key: [float(run["decode"][key]) for run in runs]
        for key in ("psnr_y", "psnr_u", "psnr_v", "psnr_yuv")
    }
    return {
        "encode_latency_ms_median": finite_json_number(statistics.median(encode_latencies)),
        "encode_latency_ms_p95": finite_json_number(percentile(encode_latencies, 0.95)),
        "decode_latency_ms_median": finite_json_number(statistics.median(decode_latencies)),
        "decode_latency_ms_p95": finite_json_number(percentile(decode_latencies, 0.95)),
        "first_frame_latency_ms": finite_json_number(
            statistics.mean(
                run["encode"]["latencies_ms"][0] + run["decode"]["latencies_ms"][0]
                for run in runs
            )
        ),
        **{
            key: finite_json_number(statistics.mean(values))
            for key, values in qualities.items()
        },
        "peak_gpu_memory_mb": finite_json_number(
            max(
                (
                    value
                    for run in runs
                    for value in (
                        run["encode_result"].peak_gpu_memory_mb,
                        run["decode_result"].peak_gpu_memory_mb,
                    )
                    if value is not None
                ),
                default=None,
            )
        ),
        "peak_host_memory_mb": finite_json_number(
            max(
                (
                    value
                    for run in runs
                    for value in (
                        run["encode_result"].peak_host_memory_mb,
                        run["decode_result"].peak_host_memory_mb,
                    )
                    if value is not None
                ),
                default=None,
            )
        ),
    }


def aggregate_case(
    performance_runs: list[dict[str, Any]],
    *,
    profile_runs: list[dict[str, Any]] | None,
    sequence: SequenceSpec,
    qp: int,
    gop_size: int,
) -> dict[str, Any]:
    del qp, gop_size
    metrics = empty_metrics()
    metrics.update(aggregate_performance(performance_runs, sequence=sequence))
    if profile_runs:
        metrics.update(aggregate_profile(profile_runs))
    return metrics


def empty_metrics() -> dict[str, Any]:
    return {
        "encode_fps_mean": None,
        "encode_fps_stddev": None,
        "decode_fps_mean": None,
        "decode_fps_stddev": None,
        "encode_latency_ms_median": None,
        "encode_latency_ms_p95": None,
        "decode_latency_ms_median": None,
        "decode_latency_ms_p95": None,
        "first_frame_latency_ms": None,
        "total_wall_time_ms": None,
        "payload_bytes": None,
        "bits_per_pixel": None,
        "psnr_y": None,
        "psnr_u": None,
        "psnr_v": None,
        "psnr_yuv": None,
        "peak_gpu_memory_mb": None,
        "peak_host_memory_mb": None,
    }


def missing_metric_fields(
    metrics: dict[str, Any], fields: Iterable[str]
) -> list[str]:
    missing: list[str] = []
    for field in fields:
        value = metrics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            missing.append(field)
    return missing


def missing_required_metrics(
    metrics: dict[str, Any], *, require_profile: bool = False
) -> list[str]:
    fields = list(REQUIRED_PERFORMANCE_METRICS)
    if require_profile:
        fields.extend(REQUIRED_PROFILE_METRICS)
    return missing_metric_fields(metrics, fields)


def base_row(
    *,
    run_id: str,
    commit: str,
    dirty: bool,
    container_image: str,
    container_digest: str,
    native_build_id: str,
    target: dict[str, Any],
    identity: dict[str, Any],
    compatibility_class: str,
    sequence: SequenceSpec,
    qp: int,
    gop_size: int,
    warmup_runs: int,
    measured_runs: int,
    profiling_enabled: bool,
    profile_runs: int,
    artifact: dict[str, Any] | None,
    input_sha256: str,
) -> dict[str, Any]:
    artifact = artifact or {}
    return {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "nvcr_commit": commit,
        "git_dirty": dirty,
        "container_image": container_image,
        "container_digest": container_digest,
        "native_build_id": native_build_id,
        "os": os_release(),
        "architecture": identity["architecture"],
        "driver_version": driver_version(),
        "cuda_runtime_version": str(identity["cuda_runtime_version"]),
        "tensorrt_version": ".".join(
            str(item) for item in detected_tensorrt_version(identity)
        ),
        "compiler_version": first_line(["c++", "--version"]),
        "target_id": target["id"],
        "gpu_name": identity["device_name"],
        "compute_capability": (
            f"{identity['compute_capability_major']}."
            f"{identity['compute_capability_minor']}"
        ),
        "multiprocessor_count": identity["multiprocessor_count"],
        "compatibility_class": compatibility_class,
        "build_target_id": artifact.get("build_target_id"),
        "codec_id": "dcvc-rt",
        "model_set_id": "dcvcrt-cvpr2025",
        "provider_id": "tensorrt",
        "engine_profile_id": sequence.profile,
        "engine_manifest_sha256": artifact.get("engine_manifest_sha256"),
        "engine_bundle_sha256": artifact.get("engine_bundle_sha256"),
        "engine_bundle_digest_kind": artifact.get("engine_bundle_digest_kind"),
        "model_profile_sha256": artifact.get("model_profile_sha256"),
        "target_profile_sha256": artifact.get("target_profile_sha256"),
        "engine_profile_sha256": artifact.get("engine_profile_sha256"),
        "sequence_id": sequence.sequence_id,
        "input_sha256": input_sha256,
        "input_bytes": sequence.measured_bytes,
        "input_redistribution": sequence.redistribution,
        "pixel_format": sequence.pixel_format,
        "resolution": sequence.profile,
        "width": sequence.width,
        "height": sequence.height,
        "source_fps": sequence.fps,
        "frames": sequence.frames,
        "qp": qp,
        "gop_size": gop_size,
        "mode": "all-intra" if gop_size == 1 else "ip",
        "operation": "encode_decode_roundtrip",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "profiling_enabled": profiling_enabled,
        "profile_runs": profile_runs,
        "performance_instrumentation": "disabled",
        **empty_metrics(),
        "python_reference_available": False,
        "python_psnr_yuv": None,
        "python_payload_bytes": None,
        "python_vs_nvcr_psnr_yuv": None,
        "python_encode_fps": None,
        "python_decode_fps": None,
        "compatibility_baseline_available": False,
        "exact_encode_fps_mean": None,
        "exact_decode_fps_mean": None,
        "encode_fps_ratio_vs_exact": None,
        "decode_fps_ratio_vs_exact": None,
        "status": "planned",
        "error_message": "",
        "command": "",
        "commands": [],
    }


def run_case(
    *,
    args: argparse.Namespace,
    sequence: SequenceSpec,
    engine_dir: Path,
    qp: int,
    gop_size: int,
    work_dir: Path,
    environment: dict[str, str],
    command_log: list[list[str]],
) -> dict[str, Any]:
    case_name = f"{sequence.sequence_id}-{sequence.profile}-qp{qp}-gop{gop_size}"
    performance_runs: list[dict[str, Any]] = []
    profile_runs: list[dict[str, Any]] = []

    def execute_roundtrip(
        *, label: int, kind: str, profiled: bool
    ) -> dict[str, Any]:
        stream = work_dir / f"{case_name}-{kind}{label}.nvcr"
        reconstruction = work_dir / f"{case_name}-{kind}{label}.yuv"
        encode_command = [
            str(args.nvcr),
            "encode",
            "-i",
            str(sequence.path),
            "-o",
            str(stream),
            "-s",
            f"{sequence.width}x{sequence.height}",
            "-r",
            str(sequence.fps),
            "--frames",
            str(sequence.frames),
            "--gop-size",
            str(gop_size),
            "--qp",
            str(qp),
            "--engine-dir",
            str(engine_dir),
        ]
        decode_command = [
            str(args.nvcr),
            "decode",
            "-i",
            str(stream),
            "-o",
            str(reconstruction),
            "--frames",
            str(sequence.frames),
            "--engine-dir",
            str(engine_dir),
        ]
        if profiled:
            encode_command.append("--verbose")
            decode_command.extend(
                ["--quality-metrics", str(sequence.path), "--verbose"]
            )
        for command in (encode_command, decode_command):
            command_log.append(command)
        try:
            if profiled:
                encode_result = run_monitored(
                    encode_command,
                    environment=environment,
                    sample_interval_ms=args.memory_sample_ms,
                )
            else:
                encode_result = run_command(
                    encode_command,
                    environment=environment,
                )
            if encode_result.return_code != 0:
                raise SoftwareXError(
                    f"{kind} encode failed ({encode_result.return_code}): "
                    f"{encode_result.stderr.strip() or encode_result.stdout.strip()}"
                )
            encode = parse_encode_output(
                encode_result.stdout,
                sequence.frames,
                require_latencies=profiled,
            )
            if profiled:
                decode_result = run_monitored(
                    decode_command,
                    environment=environment,
                    sample_interval_ms=args.memory_sample_ms,
                )
            else:
                decode_result = run_command(
                    decode_command,
                    environment=environment,
                )
            if decode_result.return_code != 0:
                raise SoftwareXError(
                    f"{kind} decode failed ({decode_result.return_code}): "
                    f"{decode_result.stderr.strip() or decode_result.stdout.strip()}"
                )
            decode = parse_decode_output(
                decode_result.stdout,
                sequence.frames,
                require_latencies=profiled,
                require_quality=profiled,
            )
            return {
                "run_index": label,
                "encode": encode,
                "decode": decode,
                "encode_result": encode_result,
                "decode_result": decode_result,
            }
        finally:
            for path in (stream, reconstruction):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    for run_index in range(args.warmup_runs + args.measured_runs):
        measured = run_index >= args.warmup_runs
        label = run_index - args.warmup_runs + 1 if measured else run_index + 1
        kind = "run" if measured else "warmup"
        run = execute_roundtrip(label=label, kind=kind, profiled=False)
        if measured:
            performance_runs.append(run)
    if args.profile:
        for profile_index in range(1, args.profile_runs + 1):
            profile_runs.append(
                execute_roundtrip(
                    label=profile_index,
                    kind="profile",
                    profiled=True,
                )
            )
    return aggregate_case(
        performance_runs,
        profile_runs=profile_runs,
        sequence=sequence,
        qp=qp,
        gop_size=gop_size,
    )


def registered_tests(
    build_dir: Path, command_log: list[list[str]]
) -> tuple[set[str], CommandResult]:
    command = ["ctest", "--test-dir", str(build_dir), "-N"]
    command_log.append(command)
    result = run_command(command)
    names = set(re.findall(r"Test\s+#?[0-9]+:\s+([^\s]+)", result.stdout))
    return names, result


def run_test_group(
    build_dir: Path,
    names: list[str],
    command_log: list[list[str]],
) -> CommandResult:
    pattern = "^(?:" + "|".join(re.escape(name) for name in names) + ")$"
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "-R",
        pattern,
    ]
    command_log.append(command)
    return run_command(command)


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SoftwareXError(f"evidence output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for filename in (*RESULT_FILES.values(), "python-reference-results.jsonl", "failures.jsonl"):
        (path / filename).touch(exist_ok=True)


def write_commands(path: Path, commands: list[list[str]]) -> None:
    lines = ["# Commands", ""]
    lines.extend(f"```bash\n{command_text(command)}\n```\n" for command in commands)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# NVCR SoftwareX Run Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Status: **{summary['status']}**",
        f"- Target: `{summary['target_id']}` ({summary['compatibility_class']})",
        f"- Commit: `{summary['nvcr_commit']}`",
        f"- Git dirty: `{str(summary['git_dirty']).lower()}`",
        f"- Cases planned: {summary['cases_planned']}",
        f"- Cases passed: {summary['cases_passed']}",
        f"- Cases failed: {summary['cases_failed']}",
        f"- Cases skipped: {summary['cases_skipped']}",
        f"- Artifact validation: `{summary['artifact_validation']}`",
        f"- Source build: `{summary['source_build']}`",
        f"- Core tests: `{summary['core_tests']}`",
        f"- GPU tests registered: `{summary['gpu_tests_registered']}`",
        f"- GPU tests: `{summary['gpu_tests']}`",
        f"- I/P roundtrip gate: `{summary['ip_roundtrip']}`",
        f"- Required matrix: `{summary['required_matrix']}`",
        f"- Profiling: `{summary['profiling']}`",
        f"- Execution identity: `{summary['execution_identity']}`",
        f"- Python reference: `{summary['python_reference']}`",
        f"- Exact performance baseline: `{summary['compatibility_baseline']}`",
        "",
        "A complete status requires separate profiling and every artifact, build, test,",
        "matrix, and reference gate. Build-only or CPU-only results are not end-to-end evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, default=artifacts.DEFAULT_MODEL_PROFILE)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument(
        "--engine-dir",
        action="append",
        default=[],
        metavar="PROFILE=PATH",
        help="Override one bundle inside --engine-root; repeat as needed.",
    )
    parser.add_argument("--nvcr", type=Path, default=REPOSITORY_ROOT / "build-release/cli/nvcr")
    parser.add_argument("--build-dir", type=Path, default=REPOSITORY_ROOT / "build-release")
    parser.add_argument("--profiles", nargs="+", choices=artifacts.ENGINE_PROFILES)
    parser.add_argument("--qps", nargs="+", type=int, default=[32])
    parser.add_argument(
        "--gops",
        nargs="+",
        default=["1", "normal"],
        help="Positive GOP sizes or 'normal' (the sequence frame count).",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Collect latency, quality, and memory in separate repetitions after "
            "the uninstrumented performance repetitions."
        ),
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Separate profiled repetitions per case when --profile is set.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--device-json", type=Path)
    parser.add_argument(
        "--compatibility-class",
        choices=tuple(RESULT_FILES),
        default="exact",
    )
    parser.add_argument("--artifact-catalog", type=Path)
    parser.add_argument("--python-reference-jsonl", type=Path)
    parser.add_argument(
        "--python-reference-command",
        help=(
            "Quoted, no-shell command that produces --python-reference-jsonl "
            "before rows are loaded."
        ),
    )
    parser.add_argument(
        "--exact-baseline-jsonl",
        type=Path,
        help="Passing exact result rows used for compatibility performance ratios.",
    )
    parser.add_argument("--container-image", default=os.environ.get("NVCR_CONTAINER_IMAGE", ""))
    parser.add_argument("--container-digest", default=os.environ.get("NVCR_CONTAINER_DIGEST", ""))
    parser.add_argument(
        "--native-build-id",
        default=os.environ.get("NVCR_NATIVE_BUILD_ID", ""),
        help="Stable identifier for a native Release build; omit for container runs.",
    )
    parser.add_argument("--memory-sample-ms", type=int, default=100)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-core-tests", action="store_true")
    parser.add_argument("--skip-gpu-tests", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success for a diagnostic partial run; evidence remains marked partial.",
    )
    args = parser.parse_args(argv)
    if args.warmup_runs < 0 or args.measured_runs <= 0 or args.profile_runs <= 0:
        parser.error(
            "warmup runs must be non-negative; measured and profile runs must be positive"
        )
    if args.device_id < 0 or args.memory_sample_ms < 0:
        parser.error("device id and memory sample interval must be non-negative")
    if any(qp < 0 or qp > 63 for qp in args.qps):
        parser.error("DCVC-RT base QP values must be in [0, 63]")
    if len(set(args.qps)) != len(args.qps):
        parser.error("QP values must not be duplicated")
    if args.container_image and args.native_build_id:
        parser.error("use container identity or --native-build-id, not both")
    if bool(args.container_image) != bool(args.container_digest):
        parser.error("--container-image and --container-digest must be supplied together")
    if args.python_reference_command and not args.python_reference_jsonl:
        parser.error("--python-reference-command requires --python-reference-jsonl")
    normalized_gops: list[str] = []
    for raw in args.gops:
        if raw == "normal":
            normalized_gops.append(raw)
            continue
        try:
            value = int(raw)
        except ValueError:
            parser.error(f"invalid GOP value: {raw}")
        if value <= 0 or value > 65535:
            parser.error("GOP values must be in [1, 65535]")
        normalized_gops.append(str(value))
    args.gops = normalized_gops
    if len(set(args.gops)) != len(args.gops):
        parser.error("GOP values must not be duplicated")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.inputs = args.inputs.expanduser().resolve()
    args.target_profile = args.target_profile.expanduser().resolve()
    args.model_profile = args.model_profile.expanduser().resolve()
    args.engine_root = args.engine_root.expanduser().resolve()
    args.nvcr = args.nvcr.expanduser().resolve()
    args.build_dir = args.build_dir.expanduser().resolve()
    prepare_output(args.output_dir)
    command_log: list[list[str]] = []
    execution_identity = bool(args.native_build_id) or bool(
        args.container_image and args.container_digest
    )
    test_summary: dict[str, Any] = {
        "execution_identity": "passed" if execution_identity else "missing"
    }
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    temporary: tempfile.TemporaryDirectory[str] | None = None
    run_started = utc_now()
    commit = "unknown"
    dirty = True
    target_id = "unknown"
    profiles: list[str] = []
    artifact_records: dict[str, dict[str, Any]] = {}
    python_rows: list[dict[str, Any]] = []
    python_reference_execution: dict[str, Any] | None = None
    fatal_error = False
    try:
        commit, dirty = git_identity()
        target = artifacts.load_profile(args.target_profile, "nvcr.target-profile.v1")
        target_id = str(target["id"])
        model_profile = artifacts.load_profile(args.model_profile, "nvcr.model-profile.v1")
        if model_profile.get("id") != "dcvcrt-cvpr2025":
            raise SoftwareXError("SoftwareX v1 requires model profile dcvcrt-cvpr2025")
        identity = load_device_identity(args.device_json, args.device_id)
        write_json(
            args.output_dir / "environment.json",
            {
                "schema": "nvcr.softwarex.environment.v1",
                "timestamp_utc": run_started,
                "nvcr_commit": commit,
                "git_dirty": dirty,
                "container_image": args.container_image,
                "container_digest": args.container_digest,
                "native_build_id": args.native_build_id,
                "os": os_release(),
                "platform": platform.platform(),
                "python": sys.version,
                "compiler": first_line(["c++", "--version"]),
                "driver_version": driver_version(),
                "profiling_enabled": args.profile,
                "profile_runs": args.profile_runs if args.profile else 0,
                "performance_instrumentation": "disabled",
                "memory_sample_ms": args.memory_sample_ms if args.profile else None,
                **identity,
            },
        )
        write_json(
            args.output_dir / "hardware-targets.json",
            {
                "schema": "nvcr.softwarex.hardware-targets.v1",
                "test_target": target,
                "test_target_profile_sha256": sha256_file(args.target_profile),
                "detected_identity": identity,
                "compatibility_class": args.compatibility_class,
            },
        )
        validate_test_target(target, identity, args.compatibility_class)
        sequences = load_sequences(args.inputs)
        if args.profiles:
            selected = set(args.profiles)
            sequences = [sequence for sequence in sequences if sequence.profile in selected]
            missing_profiles = selected - {sequence.profile for sequence in sequences}
            if missing_profiles:
                raise SoftwareXError(
                    "input configuration has no sequence for: "
                    + ", ".join(sorted(missing_profiles))
                )
        profiles = sorted({sequence.profile for sequence in sequences})
        missing_required_profiles = sorted(REQUIRED_EXACT_PROFILES - set(profiles))
        common_matrix = (
            32 in args.qps
            and "1" in args.gops
            and "normal" in args.gops
            and args.warmup_runs == 1
            and args.measured_runs == 3
            and all(sequence.frames > 1 for sequence in sequences)
        )
        required_matrix = common_matrix and (
            args.compatibility_class != "exact" or not missing_required_profiles
        )
        overrides = parse_mapping(args.engine_dir, "--engine-dir")
        engine_dirs: dict[str, Path] = {}
        for profile in profiles:
            root = engine_directory(args.engine_root, profile, overrides)
            manifest = read_json(root / "engine_manifest.json")
            artifact_records[profile] = validate_engine_identity(
                root,
                manifest,
                profile=profile,
                compatibility_class=args.compatibility_class,
                model_profile_path=args.model_profile,
                test_target_path=args.target_profile,
                test_target=target,
                identity=identity,
            )
            engine_dirs[profile] = root
        for sequence in sequences:
            if not sequence.path.is_file():
                raise SoftwareXError(f"input sequence is missing: {sequence.path}")
            if sequence.path.stat().st_size < sequence.measured_bytes:
                raise SoftwareXError(
                    f"input sequence is shorter than {sequence.frames} frames: {sequence.path}"
                )
        if args.python_reference_command:
            try:
                python_command = shlex.split(args.python_reference_command)
            except ValueError as error:
                raise SoftwareXError(
                    f"invalid Python reference command: {error}"
                ) from error
            if not python_command:
                raise SoftwareXError("Python reference command is empty")
            command_log.append(python_command)
            python_result = run_command(python_command)
            python_reference_execution = python_result.summary()
            test_summary["python_reference_command"] = python_reference_execution
            test_summary["python_reference_execution"] = (
                "passed" if python_result.return_code == 0 else "failed"
            )
            if python_result.return_code != 0:
                raise SoftwareXError(
                    "Python reference command failed: "
                    + (python_result.stderr.strip() or python_result.stdout.strip())
                )
        python_rows, python_index = load_python_reference(
            args.python_reference_jsonl, model_profile
        )
        exact_baseline = load_exact_baseline(args.exact_baseline_jsonl)
        for row in python_rows:
            append_jsonl(args.output_dir / "python-reference-results.jsonl", row)

        write_json(
            args.output_dir / "artifact-digests.json",
            {
                "schema": "nvcr.softwarex.artifact-digests.v1",
                "model_profile": str(args.model_profile),
                "model_profile_sha256": sha256_file(args.model_profile),
                "bundles": artifact_records,
            },
        )
        if args.artifact_catalog:
            catalog = read_json(args.artifact_catalog.resolve())
            if catalog.get("schema") != "nvcr.engine-catalog.v1":
                raise SoftwareXError("artifact catalog must use nvcr.engine-catalog.v1")
            write_json(args.output_dir / "artifact-catalog.json", catalog)
        else:
            write_json(
                args.output_dir / "artifact-catalog.json",
                {
                    "schema": "nvcr.softwarex.artifact-catalog-record.v1",
                    "available": False,
                    "reason": (
                        "No catalog was supplied; target-local directories "
                        "were validated directly."
                    ),
                },
            )

        test_summary = {
            "schema": "nvcr.softwarex.test-summary.v1",
            "artifact_validation": "passed",
            "source_build": (
                "planned" if args.plan_only else "skipped" if args.skip_build else "pending"
            ),
            "core_tests": (
                "planned"
                if args.plan_only
                else "skipped"
                if args.skip_core_tests
                else "pending"
            ),
            "gpu_tests_registered": "planned" if args.plan_only else "pending",
            "gpu_tests": (
                "planned"
                if args.plan_only
                else "skipped"
                if args.skip_gpu_tests
                else "pending"
            ),
            "ip_roundtrip": "planned" if args.plan_only else "pending",
            "required_matrix": "passed" if required_matrix else "incomplete",
            "missing_required_profiles": missing_required_profiles,
            "required_qp_32": 32 in args.qps,
            "required_gop_1": "1" in args.gops,
            "required_gop_normal": "normal" in args.gops,
            "required_warmup_runs": args.warmup_runs == 1,
            "required_measured_runs": args.measured_runs == 3,
            "profiling_enabled": args.profile,
            "profile_runs": args.profile_runs if args.profile else 0,
            "profiling": (
                "planned"
                if args.plan_only and args.profile
                else "not-requested"
                if not args.profile
                else "pending"
            ),
            "execution_identity": "passed" if execution_identity else "missing",
            "python_reference_execution": (
                "passed"
                if python_reference_execution is not None
                else "precomputed"
                if args.python_reference_jsonl
                else "not-requested"
            ),
            "commands": [],
        }
        if python_reference_execution is not None:
            test_summary["python_reference_command"] = python_reference_execution
        if not args.plan_only:
            if not args.skip_build:
                command = ["cmake", "--build", str(args.build_dir), "--parallel"]
                command_log.append(command)
                result = run_command(command)
                test_summary["build"] = result.summary()
                test_summary["source_build"] = "passed" if result.return_code == 0 else "failed"
                if result.return_code != 0:
                    raise SoftwareXError("Release build failed")
            if not args.nvcr.is_file() or not os.access(args.nvcr, os.X_OK):
                raise SoftwareXError(f"NVCR executable is missing: {args.nvcr}")
            registered, listing = registered_tests(args.build_dir, command_log)
            test_summary["registered_tests"] = sorted(registered)
            test_summary["ctest_listing"] = listing.summary()
            required_gpu = ["nvcr_cuda_ops"]
            for profile in profiles:
                required_gpu.extend(
                    (
                        f"nvcr_tensorrt_engine_contracts_{profile}",
                        f"nvcr_tensorrt_ip_frame_roundtrip_{profile}",
                    )
                )
            missing_gpu = sorted(set(required_gpu) - registered)
            test_summary["required_gpu_tests"] = required_gpu
            test_summary["missing_gpu_tests"] = missing_gpu
            test_summary["gpu_tests_registered"] = "passed" if not missing_gpu else "failed"
            test_summary["ip_roundtrip"] = (
                "registered"
                if all(
                    f"nvcr_tensorrt_ip_frame_roundtrip_{profile}" in registered
                    for profile in profiles
                )
                else "failed"
            )
            if missing_gpu and not args.skip_gpu_tests:
                raise SoftwareXError(
                    "required GPU tests are not registered: " + ", ".join(missing_gpu)
                )
            if not args.skip_core_tests:
                missing_core = sorted(set(CORE_TESTS) - registered)
                unit_tests = (
                    ["nvcr_smoke_tests"]
                    if "nvcr_smoke_tests" in registered
                    else sorted(name for name in registered if "." in name)
                )
                if not unit_tests:
                    missing_core.append("NVCR smoke or discovered GoogleTest suite")
                if missing_core:
                    test_summary["core_tests"] = "failed"
                    raise SoftwareXError(
                        "required core tests are not registered: " + ", ".join(missing_core)
                    )
                test_summary["unit_tests"] = unit_tests
                result = run_test_group(
                    args.build_dir, [*CORE_TESTS, *unit_tests], command_log
                )
                test_summary["core_test_execution"] = result.summary()
                test_summary["core_tests"] = "passed" if result.return_code == 0 else "failed"
                if result.return_code != 0:
                    raise SoftwareXError("core test gate failed")
            if not args.skip_gpu_tests:
                result = run_test_group(args.build_dir, required_gpu, command_log)
                test_summary["gpu_test_execution"] = result.summary()
                test_summary["gpu_tests"] = "passed" if result.return_code == 0 else "failed"
                test_summary["ip_roundtrip"] = "passed" if result.return_code == 0 else "failed"
                if result.return_code != 0:
                    raise SoftwareXError("GPU test gate failed")

        if args.work_dir:
            work_dir = args.work_dir.expanduser().resolve()
            if work_dir == args.output_dir or args.output_dir in work_dir.parents:
                raise SoftwareXError("--work-dir must remain outside the evidence package")
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="nvcr-softwarex-")
            work_dir = Path(temporary.name)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        for sequence in sequences:
            input_digest = sha256_file(sequence.path, sequence.measured_bytes)
            for qp in args.qps:
                for raw_gop in args.gops:
                    gop_size = sequence.frames if raw_gop == "normal" else int(raw_gop)
                    case_id = (
                        f"{run_started[:10].replace('-', '')}-{target_id}-{sequence.sequence_id}-"
                        f"{sequence.profile}-gop{gop_size}-qp{qp}"
                    )
                    row = base_row(
                        run_id=case_id,
                        commit=commit,
                        dirty=dirty,
                        container_image=args.container_image,
                        container_digest=args.container_digest,
                        native_build_id=args.native_build_id,
                        target=target,
                        identity=identity,
                        compatibility_class=args.compatibility_class,
                        sequence=sequence,
                        qp=qp,
                        gop_size=gop_size,
                        warmup_runs=args.warmup_runs,
                        measured_runs=args.measured_runs,
                        profiling_enabled=args.profile,
                        profile_runs=args.profile_runs if args.profile else 0,
                        artifact=artifact_records[sequence.profile],
                        input_sha256=input_digest,
                    )
                    if args.plan_only:
                        row["status"] = "skipped"
                        row["error_message"] = "plan-only: matrix commands were not executed"
                    else:
                        case_command_start = len(command_log)
                        try:
                            metrics = run_case(
                                args=args,
                                sequence=sequence,
                                engine_dir=engine_dirs[sequence.profile],
                                qp=qp,
                                gop_size=gop_size,
                                work_dir=work_dir,
                                environment=environment,
                                command_log=command_log,
                            )
                            row.update(metrics)
                            missing_metrics = missing_required_metrics(
                                metrics,
                                require_profile=args.profile,
                            )
                            if missing_metrics:
                                raise SoftwareXError(
                                    "required metrics were not recorded: "
                                    + ", ".join(missing_metrics)
                                )
                            row["status"] = "pass"
                        except SoftwareXError as error:
                            row["status"] = "fail"
                            row["error_message"] = str(error)
                    reference = python_index.get(reference_key(sequence, qp, gop_size))
                    if reference is not None:
                        if reference["input_sha256"] != input_digest:
                            row["status"] = "fail"
                            row["error_message"] = (
                                "Python reference input SHA-256 does not match "
                                "the measured input prefix"
                            )
                        else:
                            row.update(
                                python_reference_available=True,
                                python_psnr_yuv=reference["python_psnr_yuv"],
                                python_payload_bytes=reference["python_payload_bytes"],
                                python_vs_nvcr_psnr_yuv=reference["python_vs_nvcr_psnr_yuv"],
                                python_encode_fps=reference.get("python_encode_fps"),
                                python_decode_fps=reference.get("python_decode_fps"),
                            )
                    exact = exact_baseline.get(
                        baseline_key(target_id, sequence, qp, gop_size)
                    )
                    if exact is not None and row["status"] == "pass":
                        row.update(comparison_metrics(row, exact))
                    row_commands = (
                        []
                        if args.plan_only
                        else [
                            command_text(command)
                            for command in command_log[case_command_start:]
                        ]
                    )
                    row["commands"] = row_commands
                    row["command"] = " && ".join(row_commands)
                    rows.append(row)
                    append_jsonl(args.output_dir / RESULT_FILES[args.compatibility_class], row)
                    if row["status"] in ("fail", "skipped"):
                        failures.append(row)
                        append_jsonl(args.output_dir / "failures.jsonl", row)
    except (SoftwareXError, artifacts.ValidationError) as error:
        fatal_error = True
        failure = {
            "schema": "nvcr.softwarex.failure.v1",
            "timestamp_utc": utc_now(),
            "nvcr_commit": commit,
            "target_id": target_id,
            "stage": "preflight-or-gate",
            "status": "fail",
            "error_message": str(error),
        }
        failures.append(failure)
        append_jsonl(args.output_dir / "failures.jsonl", failure)
        test_summary.setdefault("artifact_validation", "failed")
        test_summary["fatal_error"] = str(error)
    finally:
        if temporary is not None:
            temporary.cleanup()

    placeholder_files = {
        "environment.json": "nvcr.softwarex.environment.v1",
        "hardware-targets.json": "nvcr.softwarex.hardware-targets.v1",
        "artifact-catalog.json": "nvcr.softwarex.artifact-catalog-record.v1",
        "artifact-digests.json": "nvcr.softwarex.artifact-digests.v1",
    }
    for filename, schema in placeholder_files.items():
        path = args.output_dir / filename
        if not path.exists():
            write_json(
                path,
                {
                    "schema": schema,
                    "available": False,
                    "reason": test_summary.get(
                        "fatal_error", "preflight did not produce this record"
                    ),
                },
            )

    cases_passed = sum(row.get("status") == "pass" for row in rows)
    cases_failed = sum(row.get("status") == "fail" for row in rows)
    cases_skipped = sum(row.get("status") == "skipped" for row in rows)
    reference_matches = sum(row.get("python_reference_available") is True for row in rows)
    reference_required = args.compatibility_class == "exact" and "rtx4070" in target_id.lower()
    baseline_matches = sum(
        row.get("compatibility_baseline_available") is True for row in rows
    )
    profile_matches = sum(
        row.get("profiling_enabled") is True
        and not missing_metric_fields(row, REQUIRED_PROFILE_METRICS)
        for row in rows
    )
    baseline_required = args.compatibility_class != "exact"
    if reference_required and args.plan_only:
        python_status = "planned"
    elif reference_required and rows and reference_matches == len(rows):
        python_status = "passed"
    elif reference_required and reference_matches:
        python_status = "incomplete-required"
    elif reference_required:
        python_status = "missing-required"
    elif reference_matches:
        python_status = "available"
    else:
        python_status = "not-requested"
    if not baseline_required:
        baseline_status = "not-required"
    elif args.plan_only:
        baseline_status = "planned"
    elif rows and baseline_matches == len(rows):
        baseline_status = "passed"
    elif baseline_matches:
        baseline_status = "incomplete-required"
    else:
        baseline_status = "missing-required"
    if not args.profile:
        profile_status = "not-requested"
    elif args.plan_only:
        profile_status = "planned"
    elif rows and profile_matches == len(rows):
        profile_status = "passed"
    elif profile_matches:
        profile_status = "incomplete-required"
    else:
        profile_status = "missing-required"
    test_summary["python_reference"] = python_status
    test_summary["python_reference_matches"] = reference_matches
    test_summary["compatibility_baseline"] = baseline_status
    test_summary["compatibility_baseline_matches"] = baseline_matches
    test_summary["profiling"] = profile_status
    test_summary["profiling_matches"] = profile_matches
    test_summary["profiling_enabled"] = args.profile
    test_summary["profile_runs"] = args.profile_runs if args.profile else 0
    test_summary["commands"] = command_log
    write_json(args.output_dir / "test-summary.json", test_summary)
    complete = (
        not args.plan_only
        and not failures
        and cases_passed > 0
        and cases_passed == len(rows)
        and test_summary.get("artifact_validation") == "passed"
        and test_summary.get("source_build") == "passed"
        and test_summary.get("core_tests") == "passed"
        and test_summary.get("gpu_tests_registered") == "passed"
        and test_summary.get("gpu_tests") == "passed"
        and test_summary.get("ip_roundtrip") == "passed"
        and test_summary.get("required_matrix") == "passed"
        and profile_status == "passed"
        and test_summary.get("execution_identity") == "passed"
        and (not reference_required or reference_matches == len(rows))
        and (not baseline_required or baseline_matches == len(rows))
        and not dirty
    )
    status = (
        "complete"
        if complete
        else "planned"
        if args.plan_only and not fatal_error
        else "partial"
    )
    summary = {
        "schema": "nvcr.softwarex.summary.v1",
        "run_id": f"softwarex-{run_started[:10].replace('-', '')}-{commit[:8]}",
        "status": status,
        "target_id": target_id,
        "compatibility_class": args.compatibility_class,
        "nvcr_commit": commit,
        "git_dirty": dirty,
        "profiles": profiles,
        "cases_planned": len(rows),
        "cases_passed": cases_passed,
        "cases_failed": cases_failed,
        "cases_skipped": cases_skipped,
        "failure_records": len(failures),
        "artifact_validation": test_summary.get("artifact_validation", "failed"),
        "source_build": test_summary.get("source_build", "not-run"),
        "core_tests": test_summary.get("core_tests", "not-run"),
        "gpu_tests_registered": test_summary.get("gpu_tests_registered", "not-run"),
        "gpu_tests": test_summary.get("gpu_tests", "not-run"),
        "ip_roundtrip": test_summary.get("ip_roundtrip", "not-run"),
        "required_matrix": test_summary.get("required_matrix", "not-run"),
        "profiling": profile_status,
        "profiling_enabled": args.profile,
        "profile_runs": args.profile_runs if args.profile else 0,
        "execution_identity": test_summary.get("execution_identity", "not-run"),
        "python_reference": python_status,
        "compatibility_baseline": baseline_status,
        "started_at": run_started,
        "completed_at": utc_now(),
    }
    write_json(args.output_dir / "run-summary.json", summary)
    write_summary(args.output_dir / "summary.md", summary)
    write_commands(args.output_dir / "commands.md", command_log)
    (args.output_dir / "README.md").write_text(
        "# NVCR SoftwareX evidence package\n\n"
        "This directory was generated by `scripts/benchmark_softwarex_matrix.py`.\n"
        "See `summary.md`, `test-summary.json`, and the compatibility-specific JSONL file.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if complete or args.plan_only and not fatal_error or args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
