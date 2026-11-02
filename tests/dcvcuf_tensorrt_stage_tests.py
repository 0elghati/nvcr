#!/usr/bin/env python3
"""Dependency-free checks for the C8 DCVC-UF TensorRT stage table and checker mapping."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/backends/dcvcuf"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stages = load("tensorrt_stages")
check = load("check_tensorrt_stages")

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


class StageTableTests(unittest.TestCase):
    def test_tensor_names_are_unique_and_axes_known(self):
        for stage, (inputs, outputs) in stages.STAGES.items():
            names = [name for name, _, _ in inputs + outputs]
            with self.subTest(stage=stage):
                self.assertEqual(len(names), len(set(names)))
                for _name, channels, axis in inputs + outputs:
                    self.assertIn(axis, stages.AXES)
                    self.assertGreater(channels, 0)

    def test_profile_axes_keep_upstream_scale_relations(self):
        for key in ("min", "opt", "max"):
            frame = stages.PROFILE[f"frame_{key}"]
            with self.subTest(point=key):
                latent = [stages.axis_size(size, "latent") for size in frame]
                padded = [stages.axis_size(size, "latent_pad4") for size in frame]
                self.assertEqual([size * 16 for size in latent], list(frame))
                self.assertEqual([stages.axis_size(size, "feature") for size in frame],
                                 [size * 2 for size in latent])
                self.assertTrue(all(size % 4 == 0 for size in padded))
                self.assertEqual([stages.axis_size(size, "hyper") * 4 for size in frame], padded)
        for axis in ("frame", "feature", "latent", "hyper"):
            shapes = stages.profile_shapes(8, axis)
            for position in (2, 3):
                self.assertLessEqual(shapes["min"][position], shapes["opt"][position])
                self.assertLessEqual(shapes["opt"][position], shapes["max"][position])

    def test_scalar_tensors_are_fixed_per_channel_scales(self):
        self.assertEqual(stages.profile_shapes(384, "scalar"),
                         {"min": [1, 384, 1, 1], "opt": [1, 384, 1, 1], "max": [1, 384, 1, 1]})


@unittest.skipIf(numpy is None, "NumPy is not installed")
class CheckerMappingTests(unittest.TestCase):
    @staticmethod
    def tensor(channels, size=4):
        return numpy.zeros((1, channels, size, size), dtype=numpy.float32)

    def recorded_group(self):
        t = self.tensor
        return [
            ("image.enc", [t(3, 64), t(384, 1)], [t(256)]),
            ("image.hyper_enc", [t(256)], [t(128, 1)]),
            ("image.hyper_dec", [t(128, 1)], [t(256)]),
            ("image.y_prior_fusion", [t(256)], [t(512)]),
            ("image.y_spatial_prior_reduction", [t(512)], [t(256)]),
            *[call for k in (1, 2, 3) for call in (
                (f"image.y_spatial_prior_adaptor_{k}", [t(512)], [t(512)]),
                ("image.y_spatial_prior", [t(512)], [t(512)]))],
            ("image.dec", [t(256), t(384, 1)], [t(3, 64)]),
            ("video.feature_adaptor_m", [t(512, 8), t(512, 8)], [t(512, 8)]),
            ("video.feature_extractor", [t(512, 8)], [t(512, 8)]),
            ("video.encoder", [t(24, 64), t(512, 8), t(512, 1)], [t(256)]),
            ("video.hyper_encoder", [t(256)], [t(128, 1)]),
            ("video.temporal_prior_encoder", [t(512, 8), t(512, 1)], [t(512)]),
            ("video.hyper_decoder", [t(128, 1)], [t(256)]),
            ("video.y_prior_fusion", [t(256), t(512)], [t(768)]),
            ("video.y_spatial_prior_reduction", [t(768)], [t(256)]),
            *[call for k in (1, 2, 3) for call in (
                (f"video.y_spatial_prior_adaptor_{k}", [t(256), t(256)], [t(512)]),
                ("video.y_spatial_prior", [t(512)], [t(256)]))],
            ("video.decoder", [t(256), t(512, 8), t(512, 1)], [t(512, 8)]),
            ("video.recon_head", [t(512, 8)], [t(3, 64)] * 8),
            ("video.recon_head", [t(512, 8)], [t(192, 8)]),
            ("video.feature_adaptor_i", [t(192, 8)], [t(512, 8)]),
        ]

    def test_every_stage_is_reached_from_upstream_call_order(self):
        cases = check.stage_cases(self.recorded_group())
        self.assertEqual(sorted({case["stage"] for case in cases}), sorted(stages.STAGES))
        for case in cases:
            inputs, outputs = stages.STAGES[case["stage"]]
            with self.subTest(case=case["label"]):
                self.assertEqual(list(case["inputs"]), [name for name, _, _ in inputs])
                self.assertEqual(list(case["outputs"]), [name for name, _, _ in outputs])
        image_prior = next(case for case in cases if case["stage"] == "i_spatial_prior_1")
        self.assertEqual(image_prior["inputs"]["y_hat"].shape[1], 256)
        self.assertEqual(image_prior["inputs"]["common"].shape[1], 256)
        frames = next(case for case in cases if case["stage"] == "p_reconstruction")
        self.assertEqual(frames["outputs"]["frames"].shape[1], 24)

    def test_unknown_upstream_calls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmapped"):
            check.stage_cases([("video.unknown", [self.tensor(4)], [self.tensor(4)])])

    def test_job_marks_spatial_axes_dynamic_and_scales_fixed(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            contracts = {name: [
                {"name": tensor, "access": access, "channels": channels, "axis": axis,
                 "profile": stages.profile_shapes(channels, axis)}
                for access, tensors in (("read", inputs), ("write", outputs))
                for tensor, channels, axis in tensors]
                for name, (inputs, outputs) in stages.STAGES.items()}
            (bundle / "stages.json").write_text(json.dumps({"stages": contracts}))
            cases = check.stage_cases(self.recorded_group()[:1])
            job = check.write_job(cases, bundle, root, "model").read_text().splitlines()
            self.assertIn("tensor frame read 1 3 -1 -1", job)
            self.assertIn("tensor q_enc read 1 384 1 1", job)
            self.assertIn("run i_encoder.plan 2 1", job)


if __name__ == "__main__":
    unittest.main()
