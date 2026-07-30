#!/usr/bin/env python3
"""Run the pinned 720p QP-32 Python/native I-frame golden comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile


QUALITY_PATTERN = re.compile(
    r"PSNR-Y ([0-9.]+) dB, PSNR-U ([0-9.]+) dB, "
    r"PSNR-V ([0-9.]+) dB, PSNR-YUV ([0-9.]+) dB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcr", type=Path, required=True)
    parser.add_argument("--dcvcrt-root", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    engine = parser.add_mutually_exclusive_group(required=True)
    engine.add_argument("--engine-dir", type=Path)
    engine.add_argument("--engine-profile")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("i_frame_golden.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    return completed.stdout or ""


def file_sha256(path: Path, byte_count: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        remaining = byte_count
        while remaining is None or remaining > 0:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            chunk = source.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        if remaining not in (None, 0):
            raise RuntimeError(f"{path} is shorter than the required golden frame")
    return digest.hexdigest()


def require_hash(path: Path, expected: str, byte_count: int | None = None) -> None:
    actual = file_sha256(path, byte_count)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}, expected {expected}")


def plane_psnr(reference: bytes, reconstructed: bytes) -> float:
    squared_error = sum((expected - actual) ** 2 for expected, actual in zip(reference, reconstructed))
    if squared_error == 0:
        return math.inf
    mse = squared_error / len(reference)
    return 10.0 * math.log10(65025.0 / mse)


def compare_yuv(reference_path: Path, reconstructed_path: Path, width: int, height: int) -> dict:
    frame_bytes = width * height * 3 // 2
    with reference_path.open("rb") as reference_file:
        reference = reference_file.read(frame_bytes)
    with reconstructed_path.open("rb") as reconstructed_file:
        reconstructed = reconstructed_file.read(frame_bytes)
    if len(reference) != frame_bytes or len(reconstructed) != frame_bytes:
        raise RuntimeError("golden comparison requires one complete YUV420P8 frame")
    y_size = width * height
    uv_size = y_size // 4
    offsets = ((0, y_size), (y_size, y_size + uv_size), (y_size + uv_size, frame_bytes))
    y_psnr, u_psnr, v_psnr = (
        plane_psnr(reference[start:end], reconstructed[start:end])
        for start, end in offsets
    )
    return {
        "psnr_y": y_psnr,
        "psnr_u": u_psnr,
        "psnr_v": v_psnr,
        "psnr_yuv": (6.0 * y_psnr + u_psnr + v_psnr) / 8.0,
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    width = int(manifest["width"])
    height = int(manifest["height"])
    qp = int(manifest["qp"])
    frame_bytes = width * height * 3 // 2
    root = args.dcvcrt_root.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    python = args.python or root / "src/venv/bin/python3"

    if not args.nvcr.is_file():
        raise RuntimeError(f"nvcr CLI not found: {args.nvcr}")
    if not python.exists():
        raise RuntimeError(f"pinned Python interpreter not found: {python}")
    if not args.input.is_file():
        raise RuntimeError(f"720p golden input not found: {args.input}")

    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True).strip()
    native_commit = run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture=True
    ).strip()
    if commit != manifest["reference_commit"]:
        raise RuntimeError(f"DCVC-RT commit is {commit}, expected {manifest['reference_commit']}")
    image_checkpoint = root / "checkpoints/cvpr2025_image.pth.tar"
    video_checkpoint = root / "checkpoints/cvpr2025_video.pth.tar"
    require_hash(image_checkpoint, manifest["image_checkpoint_sha256"])
    require_hash(video_checkpoint, manifest["video_checkpoint_sha256"])
    require_hash(args.input, manifest["source_frame_sha256"], frame_bytes)

    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="nvcr-i-frame-golden-")
        work = Path(temporary.name)

    config = {
        "root_path": str(args.input.resolve().parent),
        "test_classes": {
            "720p": {
                "test": 1,
                "base_path": ".",
                "src_type": "yuv420",
                "sequences": {
                    args.input.name: {
                        "width": width,
                        "height": height,
                        "frames": 1,
                        "intra_period": 1,
                    }
                },
            }
        },
    }
    config_path = work / "dataset.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    streams = work / "python-streams"
    reference_result = work / "python-result.json"
    streams.mkdir(parents=True, exist_ok=True)

    run(
        [
            str(python),
            "test_video.py",
            "--model_path_i", str(image_checkpoint),
            "--model_path_p", str(video_checkpoint),
            "--rate_num", "1",
            "--qp_i", str(qp),
            "--qp_p", str(qp),
            "--force_frame_num", "1",
            "--test_config", str(config_path),
            "--worker", "1",
            "--cuda", "True",
            "--calc_ssim", "False",
            "--write_stream", "True",
            "--check_existing", "False",
            "--stream_path", str(streams),
            "--save_decoded_frame", "True",
            "--output_path", str(reference_result),
            "--verbose_json", "True",
            "--verbose", "0",
        ],
        cwd=root,
    )

    python_dir = streams / "720p"
    python_streams = list(python_dir.glob("*_q32.bin"))
    python_reconstructions = list(python_dir.glob("*kbps.yuv"))
    if len(python_streams) != 1 or len(python_reconstructions) != 1:
        raise RuntimeError("pinned Python run did not produce one stream and reconstruction")
    python_stream = python_streams[0]
    python_reconstruction = python_reconstructions[0]
    require_hash(python_stream, manifest["python_bitstream_sha256"])
    require_hash(python_reconstruction, manifest["python_reconstruction_sha256"])

    native_stream = work / "native.nvcr"
    native_reconstruction = work / "native.yuv"
    engine_args = (
        ["--engine-dir", str(args.engine_dir)]
        if args.engine_dir
        else ["--engine-profile", args.engine_profile]
    )
    run(
        [
            str(args.nvcr), "encode",
            "-i", str(args.input),
            "-o", str(native_stream),
            "-s", f"{width}x{height}",
            "-r", "60",
            "--frames", "1",
            "--gop-size", "1",
            "--qp", str(qp),
            *engine_args,
        ]
    )
    quality_stdout = run(
        [
            str(args.nvcr), "decode",
            "-i", str(native_stream),
            "-o", str(native_reconstruction),
            "--quality-metrics", str(args.input),
            *engine_args,
        ],
        capture=True,
    )
    reported = QUALITY_PATTERN.search(quality_stdout)
    if reported is None:
        raise RuntimeError("native quality summary is missing or malformed")

    source_python = compare_yuv(args.input, python_reconstruction, width, height)
    source_native = compare_yuv(args.input, native_reconstruction, width, height)
    cross_runtime = compare_yuv(python_reconstruction, native_reconstruction, width, height)
    reported_values = tuple(float(value) for value in reported.groups())
    calculated_values = (
        source_native["psnr_y"],
        source_native["psnr_u"],
        source_native["psnr_v"],
        source_native["psnr_yuv"],
    )
    if any(abs(left - right) > 0.00001 for left, right in zip(reported_values, calculated_values)):
        raise RuntimeError("CLI quality summary does not match the golden calculation")

    thresholds = manifest["thresholds"]
    checks = {
        "native_source_psnr_yuv": source_native["psnr_yuv"]
            >= thresholds["native_source_psnr_yuv_min"],
        "source_psnr_yuv_delta": abs(source_python["psnr_yuv"] - source_native["psnr_yuv"])
            <= thresholds["source_psnr_yuv_delta_max"],
        "cross_runtime_psnr_y": cross_runtime["psnr_y"]
            >= thresholds["cross_runtime_psnr_y_min"],
        "cross_runtime_psnr_u": cross_runtime["psnr_u"]
            >= thresholds["cross_runtime_psnr_u_min"],
        "cross_runtime_psnr_v": cross_runtime["psnr_v"]
            >= thresholds["cross_runtime_psnr_v_min"],
        "cross_runtime_psnr_yuv": cross_runtime["psnr_yuv"]
            >= thresholds["cross_runtime_psnr_yuv_min"],
    }
    result = {
        "schema": manifest["schema"],
        "passed": all(checks.values()),
        "checks": checks,
        "pins": {
            "reference_commit": commit,
            "native_commit": native_commit,
            "image_checkpoint_sha256": file_sha256(image_checkpoint),
            "video_checkpoint_sha256": file_sha256(video_checkpoint),
            "source_frame_sha256": file_sha256(args.input, frame_bytes),
            "python_bitstream_sha256": file_sha256(python_stream),
            "python_reconstruction_sha256": file_sha256(python_reconstruction),
            "native_sequence_sha256": file_sha256(native_stream),
            "native_reconstruction_sha256": file_sha256(native_reconstruction),
        },
        "metrics": {
            "source_python": source_python,
            "source_native": source_native,
            "python_native": cross_runtime,
        },
        "thresholds": thresholds,
    }
    if args.engine_dir:
        engine_manifest = args.engine_dir / "engine_manifest.json"
        if engine_manifest.is_file():
            result["pins"]["engine_manifest_sha256"] = file_sha256(engine_manifest)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if temporary is not None:
        temporary.cleanup()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
