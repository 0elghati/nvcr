#!/usr/bin/env python3
"""Dependency-free tests for the SoftwareX experiment driver."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_softwarex_matrix as softwarex  # noqa: E402


class OutputParsingTests(unittest.TestCase):
    def test_parses_encode_decode_latency_and_quality(self) -> None:
        encoded = """\
frame 0: encoded 100 payload bytes in 1.25 ms
frame 1: encoded 80 payload bytes in 0.75 ms
Encoded 2 frame(s), 180 payload bytes, codec time 0.002 s (1000.000 fps)
Wrote stream.nvcr
"""
        decoded = """\
frame 0: decoded in 2.50 ms
frame 1: decoded in 1.50 ms
Decoded 2 frame(s), codec time 0.004 s (500.000 fps)
Quality 2 frame(s): PSNR-Y 35.000000 dB, PSNR-U 36.000000 dB, PSNR-V 37.000000 dB, PSNR-YUV 35.375000 dB
Wrote YUV420p8 4x2 to output.yuv
"""
        encode = softwarex.parse_encode_output(encoded, 2)
        decode = softwarex.parse_decode_output(decoded, 2)
        self.assertEqual(encode["payload_bytes"], 180)
        self.assertEqual(encode["latencies_ms"], [1.25, 0.75])
        self.assertEqual(decode["latencies_ms"], [2.5, 1.5])
        self.assertEqual(decode["psnr_yuv"], 35.375)

    def test_rejects_missing_per_frame_latency(self) -> None:
        output = "Encoded 2 frame(s), 180 payload bytes, codec time 0.002 s (1000.000 fps)\n"
        with self.assertRaisesRegex(softwarex.SoftwareXError, "latency rows"):
            softwarex.parse_encode_output(output, 2)

    def test_performance_parser_does_not_require_profile_output(self) -> None:
        encoded = (
            "Encoded 2 frame(s), 180 payload bytes, "
            "codec time 0.002 s (1000.000 fps)\n"
        )
        decoded = "Decoded 2 frame(s), codec time 0.004 s (500.000 fps)\n"
        encode = softwarex.parse_encode_output(
            encoded, 2, require_latencies=False
        )
        decode = softwarex.parse_decode_output(
            decoded,
            2,
            require_latencies=False,
            require_quality=False,
        )
        self.assertEqual(encode["latencies_ms"], [])
        self.assertEqual(decode["latencies_ms"], [])
        self.assertIsNone(decode["psnr_yuv"])


class AggregationTests(unittest.TestCase):
    def test_aggregate_keeps_profile_data_out_of_performance_means(self) -> None:
        sequence = softwarex.SequenceSpec(
            "fixture",
            "qcif",
            Path("fixture.yuv"),
            4,
            2,
            30.0,
            2,
            "yuv420p8",
            "local-only",
        )
        performance_runs = []
        for index, fps in enumerate((100.0, 120.0), 1):
            encode_result = softwarex.CommandResult(
                ["nvcr", "encode"], 0, "", "", 10.0 + index
            )
            decode_result = softwarex.CommandResult(
                ["nvcr", "decode"], 0, "", "", 15.0 + index
            )
            performance_runs.append(
                {
                    "encode": {
                        "fps": fps,
                        "payload_bytes": 100,
                    },
                    "decode": {
                        "fps": fps / 2,
                    },
                    "encode_result": encode_result,
                    "decode_result": decode_result,
                }
            )
        profile_runs = [
            {
                "encode": {
                    "fps": 1.0,
                    "payload_bytes": 100,
                    "latencies_ms": [1.0, 2.0],
                },
                "decode": {
                    "fps": 0.5,
                    "latencies_ms": [2.0, 4.0],
                    "psnr_y": 30.0,
                    "psnr_u": 31.0,
                    "psnr_v": 32.0,
                    "psnr_yuv": 30.375,
                },
                "encode_result": softwarex.CommandResult(
                    ["nvcr", "encode", "--verbose"],
                    0,
                    "",
                    "",
                    1000.0,
                    20.0,
                    30.0,
                ),
                "decode_result": softwarex.CommandResult(
                    ["nvcr", "decode", "--verbose"],
                    0,
                    "",
                    "",
                    2000.0,
                    25.0,
                    35.0,
                ),
            }
        ]
        performance_only = softwarex.aggregate_case(
            performance_runs,
            profile_runs=None,
            sequence=sequence,
            qp=32,
            gop_size=2,
        )
        self.assertIsNone(performance_only["psnr_yuv"])
        self.assertEqual(softwarex.missing_required_metrics(performance_only), [])
        result = softwarex.aggregate_case(
            performance_runs,
            profile_runs=profile_runs,
            sequence=sequence,
            qp=32,
            gop_size=2,
        )
        self.assertEqual(result["encode_fps_mean"], 110.0)
        self.assertGreater(result["encode_fps_stddev"], 0.0)
        self.assertEqual(result["total_wall_time_ms"], 28.0)
        self.assertEqual(result["payload_bytes"], 100)
        self.assertEqual(result["bits_per_pixel"], 50.0)
        self.assertEqual(result["psnr_yuv"], 30.375)
        self.assertEqual(result["peak_gpu_memory_mb"], 35.0)
        self.assertEqual(result["peak_host_memory_mb"], 25.0)

    def test_missing_required_metrics_rejects_unavailable_memory(self) -> None:
        metrics = {
            field: 1.0
            for field in (
                *softwarex.REQUIRED_PERFORMANCE_METRICS,
                *softwarex.REQUIRED_PROFILE_METRICS,
            )
        }
        metrics["peak_gpu_memory_mb"] = None
        self.assertEqual(softwarex.missing_required_metrics(metrics), [])
        self.assertEqual(
            softwarex.missing_required_metrics(metrics, require_profile=True),
            ["peak_gpu_memory_mb"],
        )

    @mock.patch.object(softwarex, "run_monitored")
    @mock.patch.object(softwarex, "run_command")
    def test_profile_pass_uses_only_opt_in_instrumentation(
        self,
        run_command: mock.Mock,
        run_monitored: mock.Mock,
    ) -> None:
        def clean_result(command: list[str], **_: object) -> softwarex.CommandResult:
            if command[1] == "encode":
                output = (
                    "Encoded 2 frame(s), 100 payload bytes, "
                    "codec time 0.020 s (100.000 fps)\n"
                )
                elapsed = 3.0
            else:
                output = "Decoded 2 frame(s), codec time 0.040 s (50.000 fps)\n"
                elapsed = 4.0
            return softwarex.CommandResult(command, 0, output, "", elapsed)

        def profile_result(command: list[str], **_: object) -> softwarex.CommandResult:
            if command[1] == "encode":
                output = (
                    "frame 0: encoded 60 payload bytes in 1.00 ms\n"
                    "frame 1: encoded 40 payload bytes in 2.00 ms\n"
                    "Encoded 2 frame(s), 100 payload bytes, "
                    "codec time 2.000 s (1.000 fps)\n"
                )
                elapsed = 1000.0
                host, gpu = 20.0, 30.0
            else:
                output = (
                    "frame 0: decoded in 2.00 ms\n"
                    "frame 1: decoded in 4.00 ms\n"
                    "Decoded 2 frame(s), codec time 4.000 s (0.500 fps)\n"
                    "Quality 2 frame(s): PSNR-Y 30.000000 dB, "
                    "PSNR-U 31.000000 dB, PSNR-V 32.000000 dB, "
                    "PSNR-YUV 30.375000 dB\n"
                )
                elapsed = 2000.0
                host, gpu = 25.0, 35.0
            return softwarex.CommandResult(
                command, 0, output, "", elapsed, host, gpu
            )

        run_command.side_effect = clean_result
        run_monitored.side_effect = profile_result
        args = SimpleNamespace(
            nvcr=Path("nvcr"),
            warmup_runs=1,
            measured_runs=2,
            profile=True,
            profile_runs=1,
            memory_sample_ms=100,
        )
        sequence = softwarex.SequenceSpec(
            "fixture", "qcif", Path("fixture.yuv"), 4, 2, 30.0, 2,
            "yuv420p8", "local-only"
        )
        with tempfile.TemporaryDirectory(prefix="nvcr-profile-test-") as temporary:
            commands: list[list[str]] = []
            result = softwarex.run_case(
                args=args,
                sequence=sequence,
                engine_dir=Path("engines"),
                qp=32,
                gop_size=2,
                work_dir=Path(temporary),
                environment={},
                command_log=commands,
            )

        self.assertEqual(run_command.call_count, 6)
        self.assertEqual(run_monitored.call_count, 2)
        for call in run_command.call_args_list:
            self.assertNotIn("--verbose", call.args[0])
            self.assertNotIn("--quality-metrics", call.args[0])
        for call in run_monitored.call_args_list:
            self.assertIn("--verbose", call.args[0])
        self.assertEqual(result["encode_fps_mean"], 100.0)
        self.assertEqual(result["decode_fps_mean"], 50.0)
        self.assertEqual(result["total_wall_time_ms"], 7.0)
        self.assertEqual(result["psnr_yuv"], 30.375)
        self.assertEqual(result["peak_gpu_memory_mb"], 35.0)

    def test_computes_compatibility_performance_ratios(self) -> None:
        comparison = softwarex.comparison_metrics(
            {"encode_fps_mean": 75.0, "decode_fps_mean": 120.0},
            {"encode_fps_mean": 100.0, "decode_fps_mean": 100.0},
        )
        self.assertEqual(comparison["encode_fps_ratio_vs_exact"], 0.75)
        self.assertEqual(comparison["decode_fps_ratio_vs_exact"], 1.2)


class ContractTests(unittest.TestCase):
    @staticmethod
    def identity() -> dict[str, object]:
        return {
            "operating_system": "linux",
            "architecture": "x86_64",
            "device_name": "NVIDIA GeForce RTX 4070",
            "compute_capability_major": 8,
            "compute_capability_minor": 9,
            "multiprocessor_count": 46,
            "cuda_runtime_version": 12060,
            "tensorrt_version_major": 10,
            "tensorrt_version_minor": 9,
            "tensorrt_version_patch": 0,
        }

    @staticmethod
    def target() -> dict[str, object]:
        return {
            "id": "rtx4070-ubuntu2404",
            "host": {"architecture": "x86_64"},
            "gpu": {
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": "8.9",
                "multiprocessor_count": 46,
            },
            "cuda": "12.6",
            "tensorrt": "10.9.0",
            "precision": "fp16",
        }

    def test_exact_target_requires_runtime_version_match(self) -> None:
        target = self.target()
        softwarex.validate_test_target(target, self.identity(), "exact")
        target["tensorrt"] = "10.8.0"
        with self.assertRaisesRegex(softwarex.SoftwareXError, "TensorRT"):
            softwarex.validate_test_target(target, self.identity(), "exact")

    def test_base_row_contains_required_schema_fields(self) -> None:
        sequence = softwarex.SequenceSpec(
            "fixture", "qcif", Path("fixture.yuv"), 176, 144, 30.0, 97,
            "yuv420p8", "local-only"
        )
        row = softwarex.base_row(
            run_id="run",
            commit="a" * 40,
            dirty=False,
            container_image="image",
            container_digest="sha256:digest",
            native_build_id="",
            target=self.target(),
            identity=self.identity(),
            compatibility_class="exact",
            sequence=sequence,
            qp=32,
            gop_size=97,
            warmup_runs=1,
            measured_runs=3,
            profiling_enabled=True,
            profile_runs=1,
            artifact={
                "build_target_id": "rtx4070-ubuntu2404",
                "engine_manifest_sha256": "b" * 64,
                "engine_bundle_sha256": "c" * 64,
                "engine_bundle_digest_kind": "fixture",
                "model_profile_sha256": "d" * 64,
                "target_profile_sha256": "e" * 64,
                "engine_profile_sha256": "f" * 64,
            },
            input_sha256="0" * 64,
        )
        required = {
            "schema", "run_id", "timestamp_utc", "nvcr_commit", "git_dirty",
            "container_image", "container_digest", "native_build_id", "os", "architecture",
            "driver_version", "cuda_runtime_version", "tensorrt_version",
            "target_id", "gpu_name", "compute_capability", "compatibility_class",
            "codec_id", "model_set_id", "provider_id", "engine_profile_id",
            "engine_manifest_sha256", "engine_bundle_sha256", "model_profile_sha256",
            "sequence_id", "input_sha256", "pixel_format", "width", "height",
            "frames", "qp", "gop_size", "mode", "warmup_runs", "measured_runs",
            "profiling_enabled", "profile_runs", "performance_instrumentation",
            "encode_fps_mean", "encode_fps_stddev", "decode_fps_mean",
            "decode_fps_stddev", "encode_latency_ms_median", "encode_latency_ms_p95",
            "decode_latency_ms_median", "decode_latency_ms_p95", "payload_bytes",
            "bits_per_pixel", "psnr_y", "psnr_u", "psnr_v", "psnr_yuv",
            "peak_gpu_memory_mb", "peak_host_memory_mb", "python_reference_available",
            "python_psnr_yuv", "python_payload_bytes", "python_vs_nvcr_psnr_yuv",
            "compatibility_baseline_available", "encode_fps_ratio_vs_exact",
            "decode_fps_ratio_vs_exact",
            "status", "error_message", "commands",
        }
        self.assertFalse(required - set(row))

    def test_profile_flag_coexists_with_profile_selection(self) -> None:
        args = softwarex.parse_args(
            [
                "--output-dir", "out",
                "--inputs", "inputs.json",
                "--target-profile", "target.json",
                "--engine-root", "engines",
                "--profile",
                "--profile-runs", "2",
                "--profiles", "720p",
            ]
        )
        self.assertTrue(args.profile)
        self.assertEqual(args.profile_runs, 2)
        self.assertEqual(args.profiles, ["720p"])

    def test_refuses_to_overwrite_nonempty_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nvcr-softwarex-test-") as temporary:
            output = Path(temporary)
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(softwarex.SoftwareXError, "not empty"):
                softwarex.prepare_output(output)
            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "keep")

    def test_input_schema_records_measured_prefix(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nvcr-softwarex-input-") as temporary:
            root = Path(temporary)
            input_path = root / "sequence.yuv"
            input_path.write_bytes(bytes(24))
            config = root / "inputs.json"
            config.write_text(
                json.dumps(
                    {
                        "schema": softwarex.INPUT_SCHEMA,
                        "sequences": [
                            {
                                "sequence_id": "fixture",
                                "profile": "qcif",
                                "path": str(input_path),
                                "width": 4,
                                "height": 2,
                                "fps": 30,
                                "frames": 2,
                                "pixel_format": "yuv420p8",
                                "redistribution": "local-only",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sequence = softwarex.load_sequences(config)[0]
            self.assertEqual(sequence.measured_bytes, 24)

    def test_python_reference_is_bound_to_pinned_source_and_checkpoints(self) -> None:
        model = {
            "upstream": {"commit": "a" * 40},
            "checkpoints": {
                "image": {"sha256": "b" * 64},
                "video": {"sha256": "c" * 64},
            },
        }
        row = {
            "schema": softwarex.PYTHON_REFERENCE_SCHEMA,
            "sequence_id": "fixture",
            "width": 4,
            "height": 2,
            "frames": 2,
            "qp": 32,
            "gop_size": 2,
            "python_psnr_yuv": 30.0,
            "python_payload_bytes": 100,
            "python_vs_nvcr_psnr_yuv": 40.0,
            "input_sha256": "0" * 64,
            "python_command": "python test_video.py",
            "python_source_commit": "a" * 40,
            "image_checkpoint_sha256": "b" * 64,
            "video_checkpoint_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory(prefix="nvcr-python-reference-") as temporary:
            path = Path(temporary) / "reference.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows, index = softwarex.load_python_reference(path, model)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(index), 1)

            row["python_source_commit"] = "d" * 40
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(softwarex.SoftwareXError, "pinned"):
                softwarex.load_python_reference(path, model)


if __name__ == "__main__":
    unittest.main()
