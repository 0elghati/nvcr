import contextlib
import csv
import io
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_measurement_summary as exporter
import export_observation_statistics as compact_exporter


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
                "modes": ["throughput", "memory"],
            }
            (campaign / "run.json").write_text(json.dumps({
                "manifest": manifest, "fingerprint": "def", "expected_operations": 40}))
            durations = [2.0, 4.0] + [3.0] * 8
            rss = [1000.0 + repeat for repeat in range(10)]
            def aggregate(values):
                return {"n": 10, "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values),
                        "minimum": min(values), "maximum": max(values)}
            values = {"codec_seconds": aggregate([2.0] * 10),
                      "throughput_fps": aggregate([50.0] * 10),
                      "process_seconds": aggregate(durations),
                      "process_peak_rss_mib": aggregate(rss)}
            aggregates = []
            for operation in ("encode", "decode"):
                aggregates.extend([
                    {"case_id": "sample-q0-g1", "implementation": "python",
                     "operation": operation, "mode": "throughput",
                     **{field: values[field] for field in
                        ("codec_seconds", "throughput_fps", "process_seconds")}},
                    {"case_id": "sample-q0-g1", "implementation": "python",
                     "operation": operation, "mode": "memory",
                     "process_peak_rss_mib": values["process_peak_rss_mib"]},
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
            observations = []
            for operation in ("encode", "decode"):
                for mode in ("throughput", "memory"):
                    for repeat, seconds in enumerate(durations):
                        observations.append({
                            "execution_id": f"{operation}-{mode}-{repeat}",
                            "case_id": "sample-q0-g1", "sequence": "sample",
                            "width": 176, "height": 144, "qp": 0, "gop": 1,
                            "implementation": "python", "operation": operation,
                            "mode": mode, "repeat": repeat, "frames": 100,
                            "status": "passed", "fingerprint": "def",
                            "timing_contract": "host-yuv420p8-completed-frame-v1",
                            "process_seconds": seconds,
                            "process_peak_rss_mib": rss[repeat] if mode == "memory" else None,
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
                self.assertAlmostEqual(float(process["median"]), 100 / 3)
                rss_stat = next(row for row in rows if row["metric"] == "process_peak_rss_mib"
                                and row["operation"] == "encode")
                self.assertEqual(float(rss_stat["median"]), 1004.5)
                self.assertTrue(rss_stat["ci95_low"] and rss_stat["ci95_high"])
                codec_stat = next(row for row in rows if row["metric"] == "throughput_fps"
                                  and row["operation"] == "encode")
                self.assertEqual(float(codec_stat["median"]), 50.0)
                self.assertEqual(float(codec_stat["ci95_low"]), 50.0)
                self.assertAlmostEqual(float(process["mean"]), 34.166666666666664)
                sample_sd = float(process["sample_std"])
                self.assertGreater(sample_sd, 0)
                margin = 2.2621571628540993 * sample_sd / math.sqrt(10)
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
                self.assertEqual(len(measurements), 40)
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

    def test_compact_observation_statistics(self):
        durations = [2.0, 4.0] + [3.0] * 8
        rows = []
        for mode in ("throughput", "memory"):
            for repeat, seconds in enumerate(durations):
                rows.append({
                    "status": "passed", "fingerprint": "source",
                    "execution_id": f"{mode}-{repeat}", "sequence": "sample",
                    "width": 176, "height": 144, "qp": 0, "gop": 1,
                    "implementation": "nvcr", "operation": "encode", "mode": mode,
                    "repeat": repeat, "frames": 100, "process_seconds": seconds,
                    "process_fps": 100 / seconds if mode == "throughput" else None,
                    "process_peak_rss_mib": 1000 + repeat if mode == "memory" else None,
                    "timing_contract": "host-yuv420p8-completed-frame-v1",
                    "metrics": {"timed_frames": 100, "codec_seconds": 2.0,
                                "throughput_fps": 50.0},
                })
        stats = compact_exporter.export(rows, "full", 1)
        self.assertEqual(len(stats), 5)
        process = next(row for row in stats if row["metric"] == "process_fps")
        self.assertEqual(process["campaign"], "full")
        self.assertEqual(process["n"], 10)
        self.assertAlmostEqual(process["mean"], 34.166666666666664)
        self.assertAlmostEqual(process["median"], 100 / 3)
        self.assertNotAlmostEqual(process["mean"], 1000 / sum(durations))
        rss = next(row for row in stats if row["metric"] == "process_peak_rss_mib")
        self.assertEqual(rss["median"], 1004.5)
        self.assertLess(rss["ci95_low"], rss["mean"])
        with self.assertRaisesRegex(ValueError, "condition coverage"):
            compact_exporter.export(rows, "full", 2)
        with self.assertRaisesRegex(ValueError, "all ten distinct repetitions"):
            compact_exporter.export(rows[:-1], "full", 1)


if __name__ == "__main__":
    unittest.main()
