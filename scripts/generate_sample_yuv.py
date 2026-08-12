#!/usr/bin/env python3
"""Generate a small deterministic planar YUV420P8 validation sequence."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_WIDTH = 176
DEFAULT_HEIGHT = 144
DEFAULT_FRAMES = 4


@dataclass(frozen=True)
class SampleYuvResult:
    path: Path
    byte_count: int
    width: int
    height: int
    frames: int
    sha256: str


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def validate_shape(width: int, height: int, frames: int) -> None:
    """Validate dimensions for an 8-bit 4:2:0 planar sequence."""

    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("width, height, and frames must be greater than zero")
    if width % 2 or height % 2:
        raise ValueError("YUV420 dimensions must both be even")


def _frame_planes(width: int, height: int, frame_index: int) -> Iterable[bytes]:
    """Yield one frame in Y, U, V plane order."""

    yield bytes(
        16 + ((3 * x + 5 * y + 17 * frame_index) % 220)
        for y in range(height)
        for x in range(width)
    )

    chroma_width = width // 2
    chroma_height = height // 2
    yield bytes(
        16 + ((7 * x + 11 * y + 13 * frame_index + 41) % 225)
        for y in range(chroma_height)
        for x in range(chroma_width)
    )
    yield bytes(
        16 + ((5 * x + 3 * y + 19 * frame_index + 97) % 225)
        for y in range(chroma_height)
        for x in range(chroma_width)
    )


def write_sample_yuv(
    output: Path | str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    frames: int = DEFAULT_FRAMES,
) -> SampleYuvResult:
    """Write a deterministic YUV420P8 sequence and return its metadata."""

    validate_shape(width, height, frames)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    byte_count = 0
    with output_path.open("wb") as stream:
        for frame_index in range(frames):
            for plane in _frame_planes(width, height, frame_index):
                stream.write(plane)
                digest.update(plane)
                byte_count += len(plane)

    expected_bytes = width * height * 3 // 2 * frames
    if byte_count != expected_bytes:
        raise RuntimeError(
            f"internal size mismatch: generated {byte_count}, expected {expected_bytes}"
        )

    return SampleYuvResult(
        path=output_path,
        byte_count=byte_count,
        width=width,
        height=height,
        frames=frames,
        sha256=digest.hexdigest(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic planar YUV420P8 input for NVCR validation."
    )
    parser.add_argument("--output", type=Path, required=True, help="output .yuv path")
    parser.add_argument("--width", type=_positive_integer, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=_positive_integer, default=DEFAULT_HEIGHT)
    parser.add_argument("--frames", type=_positive_integer, default=DEFAULT_FRAMES)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = write_sample_yuv(args.output, args.width, args.height, args.frames)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"output: {result.path}")
    print(f"bytes: {result.byte_count}")
    print(f"dimensions: {result.width}x{result.height}")
    print(f"frames: {result.frames}")
    print(f"sha256: {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
