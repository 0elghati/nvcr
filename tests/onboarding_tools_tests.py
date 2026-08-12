#!/usr/bin/env python3
"""Tests for dependency-free documentation utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_sample_yuv = _load_module(
    "nvcr_generate_sample_yuv",
    REPOSITORY_ROOT / "scripts" / "generate_sample_yuv.py",
)
documentation_check = _load_module(
    "nvcr_check_documentation_consistency",
    REPOSITORY_ROOT / "scripts" / "check_documentation_consistency.py",
)


class GenerateSampleYuvTests(unittest.TestCase):
    def test_default_sequence_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.yuv"
            second_path = Path(directory) / "second.yuv"
            first = generate_sample_yuv.write_sample_yuv(first_path)
            second = generate_sample_yuv.write_sample_yuv(second_path)
            payload = first_path.read_bytes()

            self.assertEqual(first.byte_count, 152064)
            self.assertEqual(first.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                first.sha256,
                "69fa1b70488393267d07be35393882005c720bf82abcb7fd1aa595afb86e34d2",
            )
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(payload, second_path.read_bytes())

            y_bytes = 176 * 144
            chroma_bytes = (176 // 2) * (144 // 2)
            self.assertEqual(payload[0], 16)
            self.assertEqual(payload[y_bytes], 57)
            self.assertEqual(payload[y_bytes + chroma_bytes], 113)

    def test_invalid_shapes_are_rejected(self) -> None:
        invalid_shapes = ((0, 144, 4), (176, 0, 4), (176, 144, 0), (175, 144, 4))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.yuv"
            for width, height, frames in invalid_shapes:
                with self.subTest(width=width, height=height, frames=frames):
                    with self.assertRaises(ValueError):
                        generate_sample_yuv.write_sample_yuv(
                            output, width=width, height=height, frames=frames
                        )

    def test_command_reports_size_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "input.yuv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "generate_sample_yuv.py"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(f"output: {output.resolve()}", completed.stdout)
            self.assertIn("bytes: 152064", completed.stdout)
            self.assertRegex(completed.stdout, r"sha256: [0-9a-f]{64}")


class DocumentationConsistencyTests(unittest.TestCase):
    def _write(self, root: Path, relative: str, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _make_repository(self, root: Path, version: str = "1.4.2") -> None:
        self._write(root, "version.txt", f"{version}\n")
        self._write(
            root,
            "CITATION.cff",
            "cff-version: 1.2.0\n"
            "title: NVCR\n"
            f"version: {version}\n"
            "preferred-citation:\n"
            "  type: software\n"
            f"  version: {version}\n",
        )
        self._write(
            root,
            "README.md",
            "# NVCR\n\n"
            "[Guide](docs/guide.md#install-on-linux)\n"
            "[Duplicate](docs/guide.md#details-1)\n"
            "[Explicit](docs/guide.md#fixed-anchor)\n",
        )
        self._write(
            root,
            "docs/guide.md",
            "# Guide\n\n"
            "## Install on Linux\n\n"
            "## Details\n\n"
            "## Details\n\n"
            '<a id="fixed-anchor"></a>\n\n'
            "`SoftwareX reviewer publication` is code, not prose.\n\n"
            "The benchmark_softwarex_matrix.py name is an immutable identifier.\n\n"
            "Container image publication details are documented separately.\n\n"
            "```text\njournal manuscript submission\n```\n",
        )
        self._write(root, "CHANGELOG.md", "reviewer [missing](missing.md)\n")
        self._write(root, "docs/AGENTS.md", "publication nvrc\n")
        self._write(root, "docs/third_party/README.md", "journal\n")
        self._write(
            root,
            "docs/first-run.md",
            "```bash\n"
            "docker pull omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9\n"
            "docker pull omarelghati/nvcr:latest-jetson-l4t36.4\n"
            "```\n",
        )

    def test_dynamic_version_links_and_legacy_identifiers_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_repository(root)
            version, files, issues = documentation_check.check_repository(root)
            self.assertEqual(version, "1.4.2")
            self.assertEqual(issues, [])
            self.assertEqual(
                {path.relative_to(root).as_posix() for path in files},
                {"README.md", "docs/first-run.md", "docs/guide.md"},
            )

    def test_citation_and_fragment_drift_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_repository(root)
            self._write(
                root,
                "CITATION.cff",
                "cff-version: 1.2.0\ntitle: NVCR\nversion: 1.4.1\n"
                "preferred-citation:\n  version: 1.4.1\n",
            )
            self._write(root, "README.md", "# NVCR\n\n[Bad](docs/guide.md#absent)\n")

            _, _, issues = documentation_check.check_repository(root)
            messages = "\n".join(issue.render() for issue in issues)
            self.assertIn("does not match version.txt '1.4.2'", messages)
            self.assertIn("Markdown fragment does not exist", messages)

    def test_public_prose_paths_and_secrets_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_repository(root)
            self._write(
                root,
                "docs/guide.md",
                "# Guide\n\n"
                "A reviewer prepared this publication-ready table in /home/alice/work.\n"
                "The obsolete name is NVRC.\n"
                "Credential: ghp_abcdefghijklmnopqrstuvwxyz123456.\n",
            )
            self._write(
                root,
                "docs/first-run.md",
                "Use omarelghati/nvcr:9.8.7-amd64-cuda12.8-trt10.9.\n"
                "Do not use omarelghati/nvcr:latest.\n"
                "The latest stable release is v9.8.7.\n",
            )

            _, _, issues = documentation_check.check_repository(root)
            messages = "\n".join(issue.render() for issue in issues)
            self.assertIn("prohibited reviewer wording", messages)
            self.assertIn("prohibited publication-readiness wording", messages)
            self.assertIn("machine-local user path", messages)
            self.assertIn("obsolete repository name", messages)
            self.assertIn("possible GitHub token", messages)
            self.assertIn("pinned semantic-version NVCR image tag", messages)
            self.assertIn("unqualified NVCR latest image tag", messages)
            self.assertIn("hardcoded latest-stable version claim", messages)


if __name__ == "__main__":
    unittest.main()
