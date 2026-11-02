#!/usr/bin/env python3
"""Dependency-free checks for the C8 DCVC-UF reference capture tools."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/backends/dcvcuf"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = load("capture_rd_reference")
stages = load("capture_stage_vectors")
MANIFEST = json.loads((SCRIPTS / "c8_reference_sequences.json").read_text(encoding="utf-8"))

try:
    import numpy
except ImportError:  # pragma: no cover - optional for the crop check only
    numpy = None


def tool_args(root, fixture, digests, output):
    return ["--reference-root", str(root), "--dataset-root", str(root),
            "--image-checkpoint", str(fixture), "--image-checkpoint-sha256", digests["image"],
            "--video-checkpoint", str(fixture), "--video-checkpoint-sha256", digests["video"],
            "--output-dir", str(output)]


class ReferenceCaptureTests(unittest.TestCase):
    def test_manifest_identities_are_complete_and_unique(self):
        for group in ("sequences", "stage_vectors"):
            names = [entry["name"] for entry in MANIFEST[group]]
            self.assertEqual(len(names), len(set(names)), group)
            for entry in MANIFEST[group]:
                with self.subTest(group=group, name=entry["name"]):
                    self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                    self.assertFalse(Path(entry["path"]).is_absolute())
                    self.assertTrue(entry["width"] > 0 and entry["height"] > 0)
        for entry in MANIFEST["sequences"]:
            self.assertGreater(entry["frames"], 0)

    def test_stage_vectors_avoid_fused_latent_padding(self):
        for entry in MANIFEST["stage_vectors"]:
            crop = entry.get("crop", {"width": entry["width"], "height": entry["height"]})
            with self.subTest(name=entry["name"]):
                stages.check_latent_alignment(crop["width"], crop["height"])
        with self.assertRaisesRegex(ValueError, "pad_for_y"):
            stages.check_latent_alignment(176, 144)

    def test_rd_capture_rejects_wrong_inputs_before_running_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"fixture")
            good = rd.sha256(fixture)
            for wrong in ("image", "video"):
                digests = {"image": good, "video": good, wrong: "0" * 64}
                output = root / "output"
                with self.subTest(wrong=wrong), \
                        patch.object(rd, "verify_reference", return_value="fixture"), \
                        patch.object(rd.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        rd.main(tool_args(root, fixture, digests, output)
                                + ["--sequence", MANIFEST["sequences"][0]["name"]])
                    run.assert_not_called()
                self.assertFalse(output.exists())

    def test_rd_capture_rejects_unknown_sequences_and_qps(self):
        with self.assertRaisesRegex(ValueError, "unknown manifest sequences"):
            rd.select_sequences(MANIFEST, ["not-a-sequence"])
        base = ["--reference-root", "x", "--dataset-root", "x", "--output-dir", "x",
                "--image-checkpoint", "x", "--image-checkpoint-sha256", "0" * 64,
                "--video-checkpoint", "x", "--video-checkpoint-sha256", "0" * 64]
        for options in (["--qp", "64"], ["--qp", "-1"], ["--frames", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    rd.parse_args(base + options)
        self.assertEqual(rd.parse_args(base).qp, [0, 21, 42, 63])

    def test_capture_tools_require_a_new_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digests = {"image": "0" * 64, "video": "0" * 64}
            for tool in (rd, stages):
                with self.subTest(tool=tool.__name__):
                    with self.assertRaisesRegex(SystemExit, "must be new"):
                        tool.main(tool_args(root, root / "missing", digests, root))

    def test_stage_capture_rejects_wrong_vector_hash_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.write_bytes(b"fixture")
            digest = stages.sha256(fixture)
            vector = MANIFEST["stage_vectors"][0]
            (root / vector["path"]).parent.mkdir(parents=True)
            (root / vector["path"]).write_bytes(b"not the declared sequence")
            output = root / "output"
            with patch.object(stages, "verify_reference", return_value="fixture"):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    stages.main(tool_args(root, fixture, {"image": digest, "video": digest},
                                          output) + ["--vector", vector["name"]])
            self.assertFalse(output.exists())

    @unittest.skipIf(numpy is None, "NumPy is not installed")
    def test_crop_keeps_luma_and_chroma_aligned(self):
        width, height = 8, 4
        luma = numpy.arange(width * height, dtype=numpy.uint8)
        chroma = numpy.arange(100, 100 + width * height // 2, dtype=numpy.uint8)
        frame = numpy.concatenate((luma, chroma))[None]
        cropped = stages.crop_frames(frame, width, height,
                                     {"x": 2, "y": 2, "width": 4, "height": 2})[0]
        self.assertEqual(cropped[:8].tolist(), [18, 19, 20, 21, 26, 27, 28, 29])
        self.assertEqual(cropped[8:].tolist(), [105, 106, 113, 114])
        with self.assertRaisesRegex(ValueError, "even"):
            stages.crop_frames(frame, width, height, {"x": 1, "y": 0, "width": 4, "height": 2})

    def test_tools_pin_the_same_upstream_revision(self):
        self.assertEqual(rd.PINNED_REFERENCE_COMMIT, stages.PINNED_REFERENCE_COMMIT)
        self.assertRegex(rd.PINNED_REFERENCE_COMMIT, re.compile(r"^[0-9a-f]{40}$"))


if __name__ == "__main__":
    unittest.main()
