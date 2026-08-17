#!/usr/bin/env python3
"""Dependency-free tests for the runtime comparison report generator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "consolidate_runtime_results.py"
SPEC = importlib.util.spec_from_file_location("consolidate_runtime_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)


class RuntimeResultValidationTests(unittest.TestCase):
    @staticmethod
    def comparison_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        python_rows: list[dict[str, object]] = []
        nvcr_rows: list[dict[str, object]] = []
        for gop in (1, 30):
            for operation in ("encode", "decode"):
                python_row: dict[str, object] = {
                    "hardware": "rtx4070",
                    "resolution": "176x144",
                    "gop_size": gop,
                    "operation": operation,
                    "qp": 32,
                    "frame_num": 100,
                    "reset_interval": 64,
                    "source_timestamp": 1,
                    "throughput_fps": 400.0,
                }
                if operation == "decode":
                    python_row["ave_all_frame_bpp"] = 0.16
                    python_row["ave_all_frame_psnr"] = 35.0
                python_rows.append(python_row)

                base_nvcr_row: dict[str, object] = {
                    "hardware": "rtx4070",
                    "size": "176x144",
                    "gop_size": gop,
                    "operation": operation,
                    "qp": 32,
                    "frames": 100,
                    "payload_bytes": 55800,
                    "payload_bpp": 0.176136,
                    "throughput_fps": 500.0,
                    "psnr_yuv": 34.8 if operation == "decode" else None,
                    "nvcr_commit": "0123456789abcdef",
                    "nvcr_dirty": False,
                }
                nvcr_rows.append({**base_nvcr_row, "run_index": "1"})
                nvcr_rows.append({**base_nvcr_row, "run_index": "average"})
        return python_rows, nvcr_rows

    def build_synthetic_report(self, **options: object) -> str:
        python_rows, nvcr_rows = self.comparison_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python_path = root / "python.jsonl"
            nvcr_path = root / "nvcr.jsonl"
            output_path = root / "summary.md"
            python_path.write_text("{}\n", encoding="utf-8")
            nvcr_path.write_text("{}\n", encoding="utf-8")
            return REPORTER.build_report(
                python_path,
                nvcr_path,
                output_path,
                python_rows,
                nvcr_rows,
                "Python DCVC-RT",
                "NVCR",
                **options,
            )

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

    def test_inner_bpp_subtracts_fixed_per_frame_overhead(self) -> None:
        row = {"size": "176x144", "frames": 100, "payload_bytes": 55800}
        expected = 50000 * 8 / (176 * 144 * 100)
        self.assertAlmostEqual(REPORTER.nvcr_inner_bpp(row, 58), expected)

    def test_report_only_emits_explicit_same_boundary_rate_cases(self) -> None:
        report = self.build_synthetic_report(
            comparable_gops=(1,),
            nvcr_frame_overhead_bytes=58,
            nvcr_reference_reset=32,
            nvcr_checkout_state="dirty",
        )
        self.assertIn(
            "| QCIF | 1 | 0.160000 | 0.157828 | -1.36% | 34.800 | 35.000 | -0.200 |",
            report,
        )
        self.assertNotIn("| QCIF | 30 | 0.160000 |", report)
        self.assertIn("Python records a 64-frame feature-reference reset", report)
        self.assertIn("NVCR as 32 frames", report)
        self.assertIn("inter-coded GOPs 30 are excluded", report)
        self.assertIn("### Aggregate comparison", report)
        self.assertIn("PSNR delta", report)
        self.assertIn("## Throughput", report)
        self.assertIn("Python encode FPS", report)
        self.assertIn("NVCR encode FPS", report)
        self.assertIn(
            "| QCIF | 1 | 400.000 | 500.000 | 400.000 | 500.000 |",
            report,
        )

    def test_report_rejects_inter_comparison_with_reset_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching Python and NVCR"):
            self.build_synthetic_report(
                comparable_gops=(30,),
                nvcr_frame_overhead_bytes=58,
                nvcr_reference_reset=32,
            )

    def test_report_requires_overhead_for_selected_gop(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --nvcr-frame-overhead-bytes"):
            self.build_synthetic_report(comparable_gops=(1,))
    def test_default_report_keeps_throughput_visible(self) -> None:

        report = self.build_synthetic_report()

        self.assertIn("contains no cross-runtime rate table", report)
        self.assertNotIn("| Resolution | GOP | Python inner-bitstream BPP |", report)
        self.assertIn("## Throughput", report)
        self.assertIn("Python encode FPS", report)
        self.assertIn("NVCR encode FPS", report)
        self.assertIn(
            "| QCIF | 1 | 400.000 | 500.000 | 400.000 | 500.000 |",
            report,
        )
        self.assertNotIn("### Aggregate comparison", report)


if __name__ == "__main__":
    unittest.main()
