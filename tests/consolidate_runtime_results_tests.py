#!/usr/bin/env python3
"""Dependency-free tests for the runtime comparison report generator."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "consolidate_runtime_results.py"
SPEC = importlib.util.spec_from_file_location("consolidate_runtime_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)


class RuntimeResultValidationTests(unittest.TestCase):
    def test_requires_nvcr_average_for_each_repetition_key(self) -> None:
        row = {"size": "176x144", "gop_size": 1, "operation": "encode", "run_index": "1"}
        with self.assertRaisesRegex(ValueError, "missing averages"):
            REPORTER.choose_nvcr_averages([row])

    def test_rejects_duplicate_nvcr_average(self) -> None:
        row = {"size": "176x144", "gop_size": 1, "operation": "encode", "run_index": "average"}
        with self.assertRaisesRegex(ValueError, "duplicate NVCR average"):
            REPORTER.choose_nvcr_averages([row, row])

    def test_formats_compact_hardware_name(self) -> None:
        self.assertEqual(REPORTER.hardware_title("rtx4070"), "RTX 4070")
        self.assertEqual(REPORTER.hardware_title("jetson-orin"), "Jetson Orin")

    def test_relative_link_stays_relative_for_external_output(self) -> None:
        link = REPORTER.relative_link(Path("/tmp/report.md"), Path("/data/results.jsonl"))
        self.assertFalse(link.startswith("/"))
        self.assertEqual(link, "../data/results.jsonl")


if __name__ == "__main__":
    unittest.main()
