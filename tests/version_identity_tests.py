#!/usr/bin/env python3
"""Check that release, build, runtime, and citation identities agree."""

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "version.txt").read_text(encoding="utf-8").strip()


class VersionIdentityTests(unittest.TestCase):
    def test_release_metadata_uses_version_file(self) -> None:
        manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["."], VERSION)

        config = json.loads((ROOT / "release-please-config.json").read_text(encoding="utf-8"))
        package = config["packages"]["."]
        self.assertEqual(package["version-file"], "version.txt")
        extra_paths = {
            entry["path"]
            for entry in package.get("extra-files", [])
        }
        self.assertEqual(extra_paths, set())

        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt"', cmake)
        self.assertNotRegex(cmake, r"VERSION\s+\d+\.\d+\.\d+")

    def test_citation_metadata_matches(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        top_versions = re.findall(r'^version:\s*"?([^"\n]+)"?$', citation, re.MULTILINE)
        self.assertEqual(top_versions, [VERSION])
        preferred_versions = re.findall(
            r'^\s+version:\s*"?([^"\n]+)"?$', citation, re.MULTILINE
        )
        self.assertEqual(preferred_versions, [VERSION])
        self.assertIn(f"v{VERSION} software release using these metadata.", citation)
        self.assertIn(f"/releases/tag/v{VERSION}", citation)

    def test_source_artifact_cli_reports_version(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "nvcr_artifacts.py"), "--version"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, f"nvcr-artifacts {VERSION}\n")
        self.assertEqual(result.stderr, "")
