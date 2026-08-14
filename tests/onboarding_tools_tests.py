#!/usr/bin/env python3
"""Tests for dependency-free documentation utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_sample_yuv = _load_module(
    "nvcr_generate_sample_yuv",
    REPOSITORY_ROOT / "scripts" / "generate_sample_yuv.py",
)


class GenerateSampleYuvTests(unittest.TestCase):
    def test_default_sequence_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.yuv"
            second_path = Path(directory) / "second.yuv"
            first = generate_sample_yuv.write_sample_yuv(first_path)
            second = generate_sample_yuv.write_sample_yuv(second_path)
            payload = first_path.read_bytes()

            self.assertEqual(first.byte_count, 152064)
            self.assertEqual(first.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                first.sha256,
                "69fa1b70488393267d07be35393882005c720bf82abcb7fd1aa595afb86e34d2",
            )
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(payload, second_path.read_bytes())

            y_bytes = 176 * 144
            chroma_bytes = (176 // 2) * (144 // 2)
            self.assertEqual(payload[0], 16)
            self.assertEqual(payload[y_bytes], 57)
            self.assertEqual(payload[y_bytes + chroma_bytes], 113)

    def test_invalid_shapes_are_rejected(self) -> None:
        invalid_shapes = ((0, 144, 4), (176, 0, 4), (176, 144, 0), (175, 144, 4))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.yuv"
            for width, height, frames in invalid_shapes:
                with self.subTest(width=width, height=height, frames=frames):
                    with self.assertRaises(ValueError):
                        generate_sample_yuv.write_sample_yuv(
                            output, width=width, height=height, frames=frames
                        )

    def test_command_reports_size_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.yuv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "generate_sample_yuv.py"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(f"output: {output.resolve()}", completed.stdout)
            self.assertIn("bytes: 152064", completed.stdout)
            self.assertRegex(completed.stdout, r"sha256: [0-9a-f]{64}")



if __name__ == "__main__":
    unittest.main()
