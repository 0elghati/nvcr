import contextlib
import csv
import io
import json
import math
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
                "repetitions": 10, "warmup_frames": 10,
            }
            (campaign / "run.json").write_text(json.dumps({
                "manifest": manifest, "fingerprint": "def", "expected_operations": 20}))
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
            durations = [2.0, 4.0] + [3.0] * 8
            observations = []
            for operation in ("encode", "decode"):
                for repeat, seconds in enumerate(durations):
                    observations.append({
                        "execution_id": f"{operation}-{repeat}",
                        "case_id": "sample-q0-g1", "sequence": "sample",
                        "width": 176, "height": 144, "qp": 0, "gop": 1,
                        "implementation": "python", "operation": operation,
                        "mode": "throughput", "repeat": repeat, "frames": 100,
                        "status": "passed", "fingerprint": "def",
                        "timing_contract": "host-yuv420p8-completed-frame-v1",
                        "process_seconds": seconds,
                        "metrics": {"timed_frames": 100, "warmup_frames": 10,
                                    "codec_seconds": 2.0, "throughput_fps": 50.0,
                                    "initialization_seconds": 0.5},
                        "bytes": {"file_bytes": 1000, "entropy_bytes": 900,
                                  "file_bpp": 0.2, "entropy_bpp": 0.18},
                    })
            (campaign / "observations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in observations))
            with patch.object(sys, "argv", ["export", str(campaign), str(output)]):
                with contextlib.redirect_stdout(io.StringIO()):
                    exporter.main()
                with (output / "condition-statistics.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 10)
                self.assertEqual({row["campaign"] for row in rows}, {"full"})
                process = next(row for row in rows if row["metric"] == "process_fps"
                               and row["operation"] == "encode")
                self.assertEqual(int(process["n"]), 10)
                self.assertAlmostEqual(float(process["mean"]), 34.166666666666664)
                sample_sd = float(process["sample_std"])
                self.assertGreater(sample_sd, 0)
                margin = exporter.T95_DF9 * sample_sd / math.sqrt(10)
                self.assertAlmostEqual(float(process["ci95_low"]),
                                       float(process["mean"]) - margin)
                self.assertAlmostEqual(float(process["ci95_high"]),
                                       float(process["mean"]) + margin)
                self.assertNotAlmostEqual(float(process["mean"]),
                                          1000 / sum(durations))
                with (output / "rd-points.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)
                with (output / "operation-measurements.csv").open() as stream:
                    measurements = list(csv.DictReader(stream))
                self.assertEqual(len(measurements), 20)
                self.assertEqual(float(measurements[0]["throughput_fps"]), 50.0)
                self.assertEqual(measurements[0]["entropy_bytes"], "900")
                analysis["status"] = "incomplete"
                (campaign / "analysis.json").write_text(json.dumps(analysis))
                with self.assertRaisesRegex(ValueError, "complete campaign"):
                    exporter.main()

    def test_process_fps_stats_keep_campaigns_separate(self):
        run = {"manifest": {"sequences": [{"sequence_id": "sample"}],
                            "qps": [0], "gops": [1],
                            "implementations": ["python"], "repetitions": 10}}
        campaigns = [("full", None, run), ("targeted", None, run)]
        observations = []
        for campaign, seconds in (("full", 2.0), ("targeted", 4.0)):
            for operation in ("encode", "decode"):
                for repeat in range(10):
                    observations.append({
                        "campaign": campaign, "case_id": "sample-q0-g1",
                        "implementation": "python", "operation": operation,
                        "mode": "throughput", "repeat": repeat,
                        "process_fps": 100 / seconds, "sequence": "sample",
                        "width": 176, "height": 144, "qp": 0, "gop": 1,
                    })
        stats = []
        exporter.append_process_fps_stats(stats, observations, campaigns)
        self.assertEqual(len(stats), 4)
        self.assertEqual({row["campaign"] for row in stats}, {"full", "targeted"})
        self.assertEqual({row["mean"] for row in stats if row["campaign"] == "full"},
                         {50.0})
        self.assertEqual({row["mean"] for row in stats if row["campaign"] == "targeted"},
                         {25.0})
        self.assertTrue(all(row["n"] == 10 and row["sample_std"] == 0
                            for row in stats))
        with self.assertRaisesRegex(ValueError, "all ten distinct repetitions"):
            exporter.append_process_fps_stats([], observations[:-1], campaigns)


if __name__ == "__main__":
    unittest.main()
