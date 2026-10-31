#!/usr/bin/env python3
"""Dependency-free checks for UF probe provenance and explicit synthetic mode."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backends/dcvcuf/probe_hts_export.py"
spec = importlib.util.spec_from_file_location("dcvcuf_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProvenanceTests(unittest.TestCase):
    def test_weights_and_feature_identity_must_be_explicit(self):
        base = ["--reference-root", "unused", "--output-dir", "unused"]
        for options in ([], ["--checkpoint", "weights.tar"],
                        ["--synthetic", "--checkpoint-sha256", "abc"],
                        ["--synthetic", "--feature", "feature.npy"],
                        ["--synthetic", "--branch", "8"],
                        ["--synthetic", "--feature-height", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    probe.parse_args(base + options)
        self.assertTrue(probe.parse_args(base + ["--synthetic"]).synthetic)

    def test_wrong_revision_and_modified_or_untracked_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args],
                                               text=True, stderr=subprocess.DEVNULL).strip()
            git("init")
            source = root / "source.py"
            source.write_text("original")
            git("add", "source.py")
            git("-c", "user.name=Probe Test", "-c", "user.email=probe@example.invalid",
                "commit", "-m", "fixture")
            revision = git("rev-parse", "HEAD")
            self.assertEqual(probe.verify_reference(root, revision), revision)
            with self.assertRaisesRegex(ValueError, "expected upstream"):
                probe.verify_reference(root)
            source.write_text("modified")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                probe.verify_reference(root, revision)
            git("add", "source.py")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                probe.verify_reference(root, revision)
            source.write_text("original")
            git("add", "source.py")
            (root / "shadow.py").write_text("untracked")
            with self.assertRaisesRegex(ValueError, "must be clean"):
                probe.verify_reference(root, revision)

    def test_checkpoint_mismatch_fails_before_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.tar"
            checkpoint.write_bytes(b"not the declared checkpoint")
            output = root / "output"
            with patch.object(probe, "verify_reference", return_value=probe.PINNED_REFERENCE_COMMIT):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    probe.main(["--reference-root", directory, "--output-dir", str(output),
                                "--checkpoint", str(checkpoint), "--checkpoint-sha256", "0" * 64])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
