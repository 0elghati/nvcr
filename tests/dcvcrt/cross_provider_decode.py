#!/usr/bin/env python3
"""Cross-target DCVC-RT decode must fail cleanly instead of reconstructing garbage,
and an exact-numerics stream must decode bit-identically on the other target.

TensorRT and ONNX Runtime compute DCVC-RT's FP16 entropy parameters with different
kernels, the same way two GPUs with different TensorRT plans do (issue #216). A
stream decoded by the other target either reproduces every entropy index (and
then reconstructs close to the encoder's own decode) or desynchronizes the rANS
decoder, which must be reported as malformed_bitstream. Silently decoding
garbage, crashing, or any other error fails this test.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

DESYNC_DIAGNOSTIC = "DCVC-RT entropy decoding desynchronized"
# Matching entropy decisions leave only FP16 reconstruction drift between
# targets; a desynchronized stream that slipped through measures near 20 dB.
MINIMUM_MATCHED_PSNR_Y = 30.0


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def require_success(result: subprocess.CompletedProcess[str], what: str) -> None:
    if result.returncode != 0:
        raise SystemExit(f"FAIL: {what} failed:\n{result.stdout}{result.stderr}")


def psnr_y(output: str) -> float:
    match = re.search(r"PSNR-Y ([0-9.]+|inf) dB", output)
    if match is None:
        raise SystemExit(f"FAIL: no PSNR-Y in decoder output:\n{output}")
    return math.inf if match.group(1) == "inf" else float(match.group(1))


def check_direction(args: argparse.Namespace, work: Path, encoder: tuple[str, str],
                    decoder: tuple[str, str], name: str) -> str:
    stream = work / f"{name}.nvcr"
    native = work / f"{name}-native.yuv"
    crossed = work / f"{name}-crossed.yuv"
    encoder_provider, encoder_dir = encoder
    decoder_provider, decoder_dir = decoder
    for path in (stream, native, crossed):
        path.unlink(missing_ok=True)

    require_success(run([
        args.nvcr, "encode", "-i", str(work / "input.yuv"), "-o", str(stream),
        "-s", f"{args.width}x{args.height}", "--frames", "3", "--gop-size", "3",
        "--qp", "32", "--provider", encoder_provider, "--engine-dir", encoder_dir,
    ]), f"{name}: {encoder_provider} encode")
    require_success(run([
        args.nvcr, "decode", "-i", str(stream), "-o", str(native),
        "--provider", encoder_provider, "--engine-dir", encoder_dir,
    ]), f"{name}: {encoder_provider} self-decode")

    result = run([
        args.nvcr, "decode", "-i", str(stream), "-o", str(crossed),
        "--provider", decoder_provider, "--engine-dir", decoder_dir,
        "--quality-metrics", str(native),
    ])
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if DESYNC_DIAGNOSTIC not in output:
            raise SystemExit(f"FAIL: {name}: decode failed without the desync diagnostic:\n{output}")
        return f"{name}: rejected as desynchronized"
    quality = psnr_y(output)
    if quality < MINIMUM_MATCHED_PSNR_Y:
        raise SystemExit(
            f"FAIL: {name}: decoded silently to a wrong reconstruction "
            f"(PSNR-Y {quality:.2f} dB against the encoder target's own decode)")
    return f"{name}: entropy decisions matched; PSNR-Y {quality:.2f} dB"


def check_exact(args: argparse.Namespace, work: Path, encoder: tuple[str, str],
                decoder: tuple[str, str], name: str) -> str:
    """Exact numerics: every decoder-side stage runs under nvcr.intconv.v1
    from the bundles' normative programs, so the other provider must
    reproduce the encoder's own decode bit for bit, P-frames included."""
    stream = work / f"{name}-exact.nvcr"
    native = work / f"{name}-exact-native.yuv"
    crossed = work / f"{name}-exact-crossed.yuv"
    encoder_provider, encoder_dir = encoder
    decoder_provider, decoder_dir = decoder
    for path in (stream, native, crossed):
        path.unlink(missing_ok=True)
    require_success(run([
        args.nvcr, "encode", "-i", str(work / "input.yuv"), "-o", str(stream),
        "-s", f"{args.width}x{args.height}", "--frames", "3", "--gop-size", "3",
        "--qp", "32", "--provider", encoder_provider, "--engine-dir", encoder_dir,
        "--numerics", "nvcr.intconv.v1",
    ]), f"{name}: {encoder_provider} exact encode")
    require_success(run([
        args.nvcr, "decode", "-i", str(stream), "-o", str(native),
        "--provider", encoder_provider, "--engine-dir", encoder_dir,
    ]), f"{name}: {encoder_provider} exact self-decode")
    require_success(run([
        args.nvcr, "decode", "-i", str(stream), "-o", str(crossed),
        "--provider", decoder_provider, "--engine-dir", decoder_dir,
    ]), f"{name}: {decoder_provider} exact decode")
    if crossed.read_bytes() != native.read_bytes():
        raise SystemExit(f"FAIL: {name}: exact decode on {decoder_provider} differs from the encoder's own")
    return f"{name}: exact stream (I, P, P) decodes bit-identically on {decoder_provider}"


def check_exact_refusals(args: argparse.Namespace, work: Path) -> list[str]:
    """An exact stream must fail before decoding where the profile or its
    programs are missing, and write no frames."""
    stream = work / "tensorrt-to-onnxruntime-exact.nvcr"
    results = []
    bundle = work / "bundle-without-programs"
    if not bundle.exists():
        bundle.mkdir()
        for item in Path(args.onnxruntime_model_dir).iterdir():
            if item.name != "normative":
                (bundle / item.name).symlink_to(item.resolve())
    cases = (("onnxruntime-cpu", str(bundle), "normative program"),
             ("onnxruntime-cuda", args.onnxruntime_model_dir, "does not implement"))
    for provider, directory, diagnostic in cases:
        output = work / f"refused-{provider}.yuv"
        output.unlink(missing_ok=True)
        result = run([args.nvcr, "decode", "-i", str(stream), "-o", str(output),
                      "--provider", provider, "--engine-dir", directory])
        text = result.stdout + result.stderr
        if "provider is not available" in text or "not registered" in text:
            results.append(f"exact refusal on {provider}: skipped, provider unavailable")
            continue
        if result.returncode == 0 or diagnostic not in text:
            raise SystemExit(f"FAIL: exact stream on {provider} ({directory}) was not refused "
                             f"before decoding:\n{text}")
        if output.exists() and output.stat().st_size > 0:
            raise SystemExit(f"FAIL: refused exact decode on {provider} wrote frames")
        results.append(f"exact stream refused before decoding on {provider} ({diagnostic})")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcr", required=True)
    parser.add_argument("--sample-generator", required=True)
    parser.add_argument("--tensorrt-engine-dir", required=True)
    parser.add_argument("--onnxruntime-model-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    require_success(run([
        sys.executable, args.sample_generator, "--output", str(work / "input.yuv"),
        "--width", str(args.width), "--height", str(args.height), "--frames", "3",
    ]), "sample generation")

    tensorrt = ("tensorrt", args.tensorrt_engine_dir)
    onnxruntime = ("onnxruntime-cpu", args.onnxruntime_model_dir)
    print(check_direction(args, work, tensorrt, onnxruntime, "tensorrt-to-onnxruntime"))
    print(check_direction(args, work, onnxruntime, tensorrt, "onnxruntime-to-tensorrt"))
    if all((Path(directory) / "normative").is_dir()
           for directory in (args.tensorrt_engine_dir, args.onnxruntime_model_dir)):
        print(check_exact(args, work, tensorrt, onnxruntime, "tensorrt-to-onnxruntime"))
        print(check_exact(args, work, onnxruntime, tensorrt, "onnxruntime-to-tensorrt"))
        for line in check_exact_refusals(args, work):
            print(line)
    else:
        print("exact numerics: skipped, the bundles carry no normative/ programs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
