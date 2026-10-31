#!/usr/bin/env python3
"""Dependency-free checks for UF probe provenance and explicit synthetic mode."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backends/dcvcuf/probe_hts_export.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("dcvcuf_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
PROVIDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/backends/dcvcuf/probe_hts_provider.py"
provider_spec = importlib.util.spec_from_file_location("dcvcuf_provider_probe", PROVIDER_SCRIPT)
provider_probe = importlib.util.module_from_spec(provider_spec)
provider_spec.loader.exec_module(provider_probe)
SESSION_SCRIPT = SCRIPT.with_name("probe_nvcr_session.py")
session_spec = importlib.util.spec_from_file_location("dcvcuf_session_probe", SESSION_SCRIPT)
session_probe = importlib.util.module_from_spec(session_spec)
session_spec.loader.exec_module(session_probe)


class ProvenanceTests(unittest.TestCase):
    def test_session_probe_rejects_unverified_native_stream_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "fixture"
            artifact.write_bytes(b"fixture")
            digest = session_probe.sha256(artifact)
            for wrong in ("image-checkpoint", "checkpoint", "extension", "stream"):
                args = ["--reference-root", directory, "--output-dir", str(root / "output")]
                for name in ("image-checkpoint", "checkpoint", "extension", "stream"):
                    args += [f"--{name}", str(artifact), f"--{name}-sha256",
                             "0" * 64 if name == wrong else digest]
                with self.subTest(wrong=wrong), patch.object(
                        session_probe, "verify_reference", return_value="fixture"):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        session_probe.main(args)
                self.assertFalse((root / "output").exists())

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

    def test_provider_probe_requires_real_checkpoint_identity(self):
        base = ["--reference-root", "unused", "--output-dir", "unused",
                "--extension", "unused.so", "--extension-sha256", "0" * 64]
        for options in ([], ["--checkpoint", "weights.tar"],
                        ["--checkpoint", "weights.tar", "--checkpoint-sha256", "0" * 64,
                         "--feature-height", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    provider_probe.parse_args(base + options)
        parsed = provider_probe.parse_args(base + [
            "--checkpoint", "weights.tar", "--checkpoint-sha256", "0" * 64])
        self.assertEqual(parsed.feature_height, 3)

    def test_provider_binary_and_feature_hashes_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.write_bytes(b"fixture")
            digest = provider_probe.sha256(artifact)
            base = ["--reference-root", directory, "--output-dir", str(root / "output")]
            for wrong in ("checkpoint", "extension", "feature"):
                args = list(base)
                for name in ("checkpoint", "extension", "feature"):
                    args += [f"--{name}", str(artifact), f"--{name}-sha256",
                             "0" * 64 if name == wrong else digest]
                with self.subTest(wrong=wrong), patch.object(
                        provider_probe, "verify_reference", return_value="fixture"):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        provider_probe.main(args)
                self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
