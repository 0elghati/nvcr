#!/usr/bin/env python3
"""Contract tests for profile selection in the native installer."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install.sh"
PACKAGE_FAMILY = (
    "linux-aarch64-jetson-l4t36"
    if platform.machine() in ("aarch64", "arm64")
    else "linux-x86_64-nvidia"
)
PACKAGE = f"nvcr-vtest-{PACKAGE_FAMILY}.tar.gz"


class InstallerTests(unittest.TestCase):
    def run_installer(self, *arguments: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            mock_bin = root / "bin"
            fixtures.mkdir()
            mock_bin.mkdir()
            capture = root / "artifact-arguments.txt"
            archive = fixtures / PACKAGE

            artifact_script = (
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$NVCR_TEST_CAPTURE\"\n"
            ).encode()
            with tarfile.open(archive, "w:gz") as output:
                info = tarfile.TarInfo("nvcr-vtest/bin/nvcr-artifacts")
                info.mode = 0o755
                info.size = len(artifact_script)
                output.addfile(info, io.BytesIO(artifact_script))

            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (fixtures / f"{PACKAGE}.sha256").write_text(
                f"{digest}  {PACKAGE}\n", encoding="utf-8"
            )

            mock_curl = mock_bin / "curl"
            mock_curl.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
destination=""
url=""
while (($#)); do
    case "$1" in
        -o) destination="$2"; shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
case "$url" in
    */releases/tags/vtest)
        printf '{"tag_name":"vtest","assets":[' > "$destination"
        printf '{"name":"%s","browser_download_url":"fixture://%s"},' "$NVCR_TEST_PACKAGE" "$NVCR_TEST_PACKAGE" >> "$destination"
        printf '{"name":"%s.sha256","browser_download_url":"fixture://%s.sha256"}]}' "$NVCR_TEST_PACKAGE" "$NVCR_TEST_PACKAGE" >> "$destination"
        ;;
    fixture://*) cp "$NVCR_TEST_FIXTURES/${url#fixture://}" "$destination" ;;
    *) echo "unexpected URL: $url" >&2; exit 1 ;;
esac
""",
                encoding="utf-8",
            )
            mock_curl.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                PATH=f"{mock_bin}:{environment['PATH']}",
                NVCR_PREFIX=str(root / "install"),
                NVCR_ENGINE_ROOT=str(root / "engines"),
                NVCR_TEST_CAPTURE=str(capture),
                NVCR_TEST_FIXTURES=str(fixtures),
                NVCR_TEST_PACKAGE=PACKAGE,
            )
            result = subprocess.run(
                ["bash", str(INSTALLER), "--tag", "vtest", *arguments],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            captured = capture.read_text(encoding="utf-8").splitlines() if capture.exists() else []
            return result, captured

    @staticmethod
    def profiles_from_arguments(arguments: list[str]) -> list[str]:
        return [arguments[index + 1] for index, value in enumerate(arguments) if value == "--profile"]

    def test_default_installs_qcif_only(self) -> None:
        result, arguments = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.profiles_from_arguments(arguments), ["qcif"])

    def test_profile_accepts_multiple_values_and_repetition(self) -> None:
        result, arguments = self.run_installer(
            "--profile", "qcif", "720p", "--profile", "1080p"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.profiles_from_arguments(arguments), ["qcif", "720p", "1080p"])

    def test_all_profiles_forwards_no_filter(self) -> None:
        result, arguments = self.run_installer("--all-profiles")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.profiles_from_arguments(arguments), [])

    def test_all_profiles_rejects_profile_filter(self) -> None:
        result, _ = self.run_installer("--all-profiles", "--profile", "720p")
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot be combined", result.stderr)


if __name__ == "__main__":
    unittest.main()
