import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_measurement_summary as exporter


class MeasurementExportTests(unittest.TestCase):
    def test_complete_export_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            campaign.mkdir()
            output = root / "summary"
            manifest = {
                "sequences": [{"sequence_id": "sample", "width": 176, "height": 144}],
                "qps": [0], "gops": [1], "implementations": ["python"],
            }
            (campaign / "run.json").write_text(json.dumps({
                "manifest": manifest, "fingerprint": "def", "expected_operations": 1}))
            value = {"n": 10, "mean": 2.0, "sample_std": 0.2,
                     "minimum": 1.5, "maximum": 2.5}
            aggregates = []
            for operation in ("encode", "decode"):
                aggregates.extend([
                    {"case_id": "sample-q0-g1", "implementation": "python",
                     "operation": operation, "mode": "throughput",
                     **{field: value for field in
                        ("codec_seconds", "throughput_fps", "process_seconds")}},
                    {"case_id": "sample-q0-g1", "implementation": "python",
                     "operation": operation, "mode": "memory",
                     "process_peak_rss_mib": value},
                ])
            analysis = {"status": "complete", "aggregates": aggregates}
            (campaign / "analysis.json").write_text(json.dumps(analysis))
            point = {"sequence": "sample", "width": 176, "height": 144,
                     "gop": 1, "qp": 0, "implementation": "python",
                     "content_id": "sample", "input_sha256": "abc", "frames": 100,
                     "entropy_bpp": 0.1, "file_bpp": 0.2, "psnr_yuv_db": 30.0,
                     "quality_contract": "test", "fingerprint": "def"}
            (campaign / "rd-points.json").write_text(json.dumps({
                "schema": "nvcr.measurement.rd.v1", "points": [point]}))
            observation = {
                "execution_id": "one", "case_id": "sample-q0-g1",
                "sequence": "sample", "width": 176, "height": 144,
                "qp": 0, "gop": 1, "implementation": "python",
                "operation": "encode", "mode": "throughput", "repeat": 0,
                "frames": 100, "status": "passed", "fingerprint": "def",
                "process_seconds": 3.0,
                "metrics": {"timed_frames": 100, "warmup_frames": 10,
                            "codec_seconds": 2.0, "throughput_fps": 50.0,
                            "initialization_seconds": 0.5},
                "bytes": {"file_bytes": 1000, "entropy_bytes": 900,
                          "file_bpp": 0.2, "entropy_bpp": 0.18},
            }
            (campaign / "observations.jsonl").write_text(json.dumps(observation) + "\n")
            with patch.object(sys, "argv", ["export", str(campaign), str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
                with (output / "condition-statistics.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 8)
                self.assertEqual(float(rows[0]["cv_percent"]), 10.0)
                with (output / "rd-points.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)
                with (output / "operation-measurements.csv").open() as stream:
                    measurements = list(csv.DictReader(stream))
                self.assertEqual(len(measurements), 1)
                self.assertEqual(float(measurements[0]["throughput_fps"]), 50.0)
                self.assertEqual(measurements[0]["entropy_bytes"], "900")
                analysis["status"] = "incomplete"
                (campaign / "analysis.json").write_text(json.dumps(analysis))
                with self.assertRaisesRegex(ValueError, "complete campaign"):
                    exporter.main()


if __name__ == "__main__":
    unittest.main()
