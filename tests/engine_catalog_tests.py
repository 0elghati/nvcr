#!/usr/bin/env python3
"""Focused rolling engine catalog, detection, and installation tests."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import nvcr_artifacts as artifacts  # noqa: E402
import nvcr_device as device  # noqa: E402

helper_spec = importlib.util.spec_from_file_location(
    "download_engine_release_assets",
    REPOSITORY_ROOT / "scripts/ci/download_engine_release_assets.py",
)
assert helper_spec and helper_spec.loader
catalog_helper = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = catalog_helper
helper_spec.loader.exec_module(catalog_helper)


def identity() -> dict[str, object]:
    return {
        "operating_system": "linux",
        "architecture": "x86_64",
        "device_id": 0,
        "device_name": "NVIDIA GeForce RTX 4070",
        "compute_capability_major": 8,
        "compute_capability_minor": 9,
        "multiprocessor_count": 46,
        "cuda_runtime_version": 12060,
        "tensorrt_version_major": 10,
        "tensorrt_version_minor": 7,
        "tensorrt_version_patch": 0,
    }


def entry(profile: str, *, target: str = "rtx4070-ubuntu2404") -> dict[str, object]:
    values = identity()
    values.pop("device_id")
    return {
        "backend": "dcvcrt",
        "model_profile_id": "dcvcrt-cvpr2025",
        "target_profile_id": target,
        "profile": profile,
        "precision": "fp16",
        "filename": artifacts.expected_asset_filename(target, "dcvcrt-cvpr2025", profile),
        "size_bytes": 123,
        "sha256": "a" * 64,
        **values,
    }


class DeviceTests(unittest.TestCase):
    def test_architecture_aliases(self) -> None:
        self.assertEqual(device.normalize_architecture("amd64"), "x86_64")
        self.assertEqual(device.normalize_architecture("arm64"), "aarch64")
        with self.assertRaises(device.DeviceDetectionError):
            device.normalize_architecture("riscv64")

    def test_nvidia_smi_device_selection(self) -> None:
        with mock.patch.object(device.shutil, "which", return_value="/usr/bin/nvidia-smi"), mock.patch.object(
            device.subprocess,
            "check_output",
            return_value="NVIDIA GeForce RTX 4070, 8.9\n",
        ) as query:
            detected = device.query_device_with_nvidia_smi(2)
        self.assertEqual(detected["compute_capability_major"], 8)
        self.assertIn("--id=2", query.call_args.args[0])

    def test_cudart_fallback(self) -> None:
        fallback = {
            "device_name": "Orin",
            "compute_capability_major": 8,
            "compute_capability_minor": 7,
            "multiprocessor_count": 8,
        }
        with mock.patch.object(device.platform, "system", return_value="Linux"), mock.patch.object(
            device, "normalize_architecture", return_value="aarch64"
        ), mock.patch.object(device, "query_device_with_nvidia_smi", return_value={}), mock.patch.object(
            device, "query_device_with_cudart", return_value=fallback
        ), mock.patch.object(
            device, "query_device_with_cuda_driver", return_value={}
        ), mock.patch.object(device, "query_cuda_runtime_version", return_value=12060), mock.patch.object(
            device, "query_tensorrt_version", return_value=(10, 3, 0)
        ):
            detected = device.detect_device_identity(0)
        self.assertEqual(detected["device_name"], "Orin")
        self.assertEqual(detected["architecture"], "aarch64")


class CatalogTests(unittest.TestCase):
    def test_profile_arguments_accept_multiple_values_and_repetition(self) -> None:
        self.assertEqual(
            artifacts.flatten_profile_arguments([["360p", "720p"], ["1080p"]]),
            ["360p", "720p", "1080p"],
        )

    def test_install_command_forwards_multiple_profiles(self) -> None:
        release = {
            "assets": [
                {
                    "name": artifacts.CATALOG_FILENAME,
                    "browser_download_url": "https://example.invalid/catalog",
                }
            ]
        }

        def fake_download(_url: str, path: Path, _token: str | None) -> None:
            path.write_text(
                json.dumps({"schema": artifacts.CATALOG_SCHEMA, "assets": []}),
                encoding="utf-8",
            )

        with mock.patch.object(
            artifacts, "github_request_json", return_value=release
        ), mock.patch.object(
            artifacts, "download_file", side_effect=fake_download
        ), mock.patch.object(
            artifacts, "detect_device_identity", return_value=identity()
        ), mock.patch.object(
            artifacts, "install_catalog_assets", return_value=[]
        ) as install:
            result = artifacts.install_command(
                [
                    "--repo",
                    "example/private",
                    "--profile",
                    "360p",
                    "720p",
                    "--profile",
                    "1080p",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(install.call_args.kwargs["requested_profiles"], ["360p", "720p", "1080p"])

    def test_private_release_download_uses_asset_api(self) -> None:
        release = {
            "assets": [
                {
                    "name": artifacts.CATALOG_FILENAME,
                    "url": "https://api.github.com/repos/example/private/releases/assets/1",
                    "browser_download_url": (
                        "https://github.com/example/private/releases/download/"
                        "engine-assets/catalog"
                    ),
                }
            ]
        }
        catalog = {"schema": artifacts.CATALOG_SCHEMA, "assets": []}

        def fake_download(url: str, path: Path, token: str | None) -> None:
            self.assertEqual(
                url,
                "https://api.github.com/repos/example/private/releases/assets/1",
            )
            self.assertEqual(token, "secret")
            path.write_text(json.dumps(catalog), encoding="utf-8")

        with mock.patch.dict(
            artifacts.os.environ, {"GH_TOKEN": "secret"}, clear=False
        ), mock.patch.object(
            artifacts, "github_request_json", return_value=release
        ), mock.patch.object(
            artifacts, "download_file", side_effect=fake_download
        ), mock.patch.object(
            artifacts, "detect_device_identity", return_value=identity()
        ), mock.patch.object(
            artifacts, "install_catalog_assets", return_value=[]
        ):
            result = artifacts.install_command(["--repo", "example/private"])
        self.assertEqual(result, 0)

    def test_catalog_schema_and_exact_match(self) -> None:
        document = {"schema": artifacts.CATALOG_SCHEMA, "assets": [entry("720p")]}
        validated = artifacts.validate_catalog(document)
        self.assertTrue(artifacts.catalog_entry_matches(validated[0], identity()))
        wrong = identity()
        wrong["tensorrt_version_patch"] = 1
        self.assertFalse(artifacts.catalog_entry_matches(validated[0], wrong))

    def test_desktop_catalog_allows_newer_cuda_runtime_within_major(self) -> None:
        candidate = entry("720p")
        newer_runtime = identity()
        newer_runtime["cuda_runtime_version"] = 12080
        self.assertTrue(artifacts.catalog_entry_matches(candidate, newer_runtime))

        older_runtime = identity()
        older_runtime["cuda_runtime_version"] = 12050
        self.assertFalse(artifacts.catalog_entry_matches(candidate, older_runtime))

        next_major = identity()
        next_major["cuda_runtime_version"] = 13000
        self.assertFalse(artifacts.catalog_entry_matches(candidate, next_major))

    def test_jetson_catalog_keeps_exact_cuda_runtime_match(self) -> None:
        candidate = entry("720p", target="orin-nano-l4t3647")
        candidate.update(architecture="aarch64", cuda_runtime_version=12060)
        detected = identity()
        detected.update(architecture="aarch64", cuda_runtime_version=12080)
        self.assertFalse(artifacts.catalog_entry_matches(candidate, detected))

    def test_catalog_ranks_exact_before_hardware_compatible_entries(self) -> None:
        exact = entry("720p")
        same_cc = entry("720p", target="ada-sm89")
        same_cc.update(
            hardware_compatibility="same_compute_capability",
            device_name="Ada compatibility build host",
            multiprocessor_count=128,
            filename=artifacts.expected_asset_filename(
                "ada-sm89",
                "dcvcrt-cvpr2025",
                "720p",
                "same_compute_capability",
                "x86_64",
                8,
                9,
            ),
        )
        broad = entry("720p", target="ampere-plus")
        broad.update(
            hardware_compatibility="ampere_plus",
            device_name="Ampere compatibility build host",
            compute_capability_minor=0,
            multiprocessor_count=108,
            filename=artifacts.expected_asset_filename(
                "ampere-plus", "dcvcrt-cvpr2025", "720p", "ampere_plus"
            ),
        )
        validated = artifacts.validate_catalog(
            {"schema": artifacts.CATALOG_SCHEMA, "assets": [broad, same_cc, exact]}
        )
        self.assertEqual(
            [artifacts.catalog_entry_match_rank(item, identity()) for item in validated],
            [2, 1, 0],
        )

    def test_catalog_rejects_hardware_compatible_jetson_entry(self) -> None:
        candidate = entry("720p", target="orin-family")
        candidate.update(
            architecture="aarch64",
            hardware_compatibility="ampere_plus",
            filename=artifacts.expected_asset_filename(
                "orin-family", "dcvcrt-cvpr2025", "720p", "ampere_plus"
            ),
        )
        with self.assertRaisesRegex(artifacts.ValidationError, "Jetson/AArch64"):
            artifacts.validate_catalog(
                {"schema": artifacts.CATALOG_SCHEMA, "assets": [candidate]}
            )

    def test_catalog_rejects_duplicate_compatibility_filename(self) -> None:
        first = entry("720p", target="ada-sm89-a")
        first.update(
            hardware_compatibility="same_compute_capability",
            filename=artifacts.expected_asset_filename(
                "ada-sm89-a",
                "dcvcrt-cvpr2025",
                "720p",
                "same_compute_capability",
                "x86_64",
                8,
                9,
            ),
        )
        second = dict(first)
        second["target_profile_id"] = "ada-sm89-b"
        with self.assertRaisesRegex(artifacts.ValidationError, "duplicate catalog filename"):
            artifacts.validate_catalog(
                {"schema": artifacts.CATALOG_SCHEMA, "assets": [first, second]}
            )

    def test_catalog_selects_best_rank_per_profile(self) -> None:
        payload = b"catalog asset"
        digest = hashlib.sha256(payload).hexdigest()
        same_cc = entry("720p", target="ada-sm89")
        same_cc.update(
            hardware_compatibility="same_compute_capability",
            device_name="Ada compatibility build host",
            multiprocessor_count=128,
            filename=artifacts.expected_asset_filename(
                "ada-sm89",
                "dcvcrt-cvpr2025",
                "720p",
                "same_compute_capability",
                "x86_64",
                8,
                9,
            ),
            size_bytes=len(payload),
            sha256=digest,
        )
        broad = entry("720p", target="ampere-plus")
        broad.update(
            hardware_compatibility="ampere_plus",
            device_name="Ampere compatibility build host",
            compute_capability_minor=0,
            multiprocessor_count=108,
            filename=artifacts.expected_asset_filename(
                "ampere-plus", "dcvcrt-cvpr2025", "720p", "ampere_plus"
            ),
            size_bytes=len(payload),
            sha256=digest,
        )

        def fake_extract(archive: Path, root: Path) -> Path:
            candidate = same_cc if archive.name == same_cc["filename"] else broad
            bundle = root / "bundle"
            bundle.mkdir(parents=True)
            manifest = {
                "target_profile_id": candidate["target_profile_id"],
                "model_profile_id": candidate["model_profile_id"],
                "engine_profile_id": candidate["profile"],
                "hardware_compatibility": candidate["hardware_compatibility"],
                **{
                    field: candidate[field]
                    for field in (
                        "device_name",
                        "compute_capability_major",
                        "compute_capability_minor",
                        "multiprocessor_count",
                        "cuda_runtime_version",
                        "tensorrt_version_major",
                        "tensorrt_version_minor",
                        "tensorrt_version_patch",
                        "precision",
                    )
                },
            }
            (bundle / "engine_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return bundle

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            artifacts,
            "download_file",
            side_effect=lambda _url, path, _token: path.write_bytes(payload),
        ) as download, mock.patch.object(
            artifacts, "extract_engine_archive", side_effect=fake_extract
        ), mock.patch.object(
            artifacts, "validate_engine_bundle", return_value={}
        ), mock.patch.object(artifacts, "atomic_symlink"):
            installed = artifacts.install_catalog_assets(
                [broad, same_cc],
                {
                    str(same_cc["filename"]): "https://example.invalid/same-cc",
                    str(broad["filename"]): "https://example.invalid/ampere-plus",
                },
                identity(),
                backend="dcvcrt",
                requested_profiles=["720p"],
                engine_root=Path(temporary),
            )
        self.assertEqual(installed, ["720p"])
        self.assertEqual(download.call_count, 1)
        self.assertEqual(Path(download.call_args.args[1]).name, same_cc["filename"])

    def test_catalog_rejects_legacy_public_name_and_duplicates(self) -> None:
        legacy = entry("720p")
        legacy["profile"] = "720p-fp16"
        with self.assertRaises(artifacts.ValidationError):
            artifacts.validate_catalog({"schema": artifacts.CATALOG_SCHEMA, "assets": [legacy]})
        duplicate = entry("720p")
        with self.assertRaises(artifacts.ValidationError):
            artifacts.validate_catalog(
                {"schema": artifacts.CATALOG_SCHEMA, "assets": [duplicate, dict(duplicate)]}
            )

    def test_catalog_merge_requires_complete_new_target(self) -> None:
        with self.assertRaises(SystemExit):
            catalog_helper.merge_catalog([], [entry("720p")])
        complete = [entry(profile) for profile in artifacts.ENGINE_PROFILES]
        merged = catalog_helper.merge_catalog([], complete)
        self.assertEqual(len(merged), len(artifacts.ENGINE_PROFILES))
        updated = entry("720p")
        updated["sha256"] = "b" * 64
        merged = catalog_helper.merge_catalog(merged, [updated])
        selected = [item for item in merged if item["profile"] == "720p"]
        self.assertEqual(selected[0]["sha256"], "b" * 64)

    def test_no_match_and_ambiguous_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                artifacts.ValidationError, r"TensorRT 10\.7\.0 CUDA runtime 12060"
            ):
                artifacts.install_catalog_assets(
                    [entry("720p")],
                    {},
                    {**identity(), "device_name": "Different GPU"},
                    backend="dcvcrt",
                    requested_profiles=[],
                    engine_root=Path(temporary),
                )
            with self.assertRaisesRegex(artifacts.ValidationError, "detected .*available="):
                artifacts.install_catalog_assets(
                    [entry("720p")],
                    {},
                    identity(),
                    backend="dcvcrt",
                    requested_profiles=["unknown"],
                    engine_root=Path(temporary),
                )
            ambiguous = [entry("720p"), entry("720p", target="rtx4070-secondary")]
            with self.assertRaises(artifacts.ValidationError):
                artifacts.install_catalog_assets(
                    ambiguous,
                    {},
                    identity(),
                    backend="dcvcrt",
                    requested_profiles=["720p"],
                    engine_root=Path(temporary),
                )

    def test_no_profile_installs_all_compatible_entries(self) -> None:
        payload = b"catalog archive"
        digest = hashlib.sha256(payload).hexdigest()
        candidates = []
        for profile in artifacts.ENGINE_PROFILES:
            candidate = entry(profile)
            candidate.update(size_bytes=len(payload), sha256=digest)
            candidates.append(candidate)
        by_filename = {str(item["filename"]): item for item in candidates}

        def fake_extract(archive: Path, root: Path) -> Path:
            candidate = by_filename[archive.name]
            bundle = root / "bundle"
            bundle.mkdir(parents=True)
            manifest = {
                "target_profile_id": candidate["target_profile_id"],
                "model_profile_id": candidate["model_profile_id"],
                "engine_profile_id": candidate["profile"],
                **{
                    field: candidate[field]
                    for field in (
                        "device_name",
                        "compute_capability_major",
                        "compute_capability_minor",
                        "multiprocessor_count",
                        "cuda_runtime_version",
                        "tensorrt_version_major",
                        "tensorrt_version_minor",
                        "tensorrt_version_patch",
                        "precision",
                    )
                },
            }
            (bundle / "engine_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return bundle

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            artifacts,
            "download_file",
            side_effect=lambda _url, path, _token: path.write_bytes(payload),
        ) as download, mock.patch.object(
            artifacts, "extract_engine_archive", side_effect=fake_extract
        ), mock.patch.object(
            artifacts, "validate_engine_bundle", return_value={}
        ), mock.patch.object(artifacts, "atomic_symlink"):
            installed = artifacts.install_catalog_assets(
                candidates,
                {str(item["filename"]): "https://example.invalid/archive" for item in candidates},
                identity(),
                backend="dcvcrt",
                requested_profiles=[],
                engine_root=Path(temporary),
            )
        self.assertEqual(installed, list(artifacts.ENGINE_PROFILES))
        self.assertEqual(download.call_count, len(artifacts.ENGINE_PROFILES))

    def test_hash_failure_precedes_extraction(self) -> None:
        candidate = entry("720p")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            artifacts, "download_file", side_effect=lambda _url, path, _token: path.write_bytes(b"bad")
        ):
            with self.assertRaisesRegex(artifacts.ValidationError, "size mismatch"):
                artifacts.install_catalog_assets(
                    [candidate],
                    {str(candidate["filename"]): "https://example.invalid/archive"},
                    identity(),
                    backend="dcvcrt",
                    requested_profiles=["720p-fp16"],
                    engine_root=Path(temporary),
                )

    def test_archive_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("../escape")
                payload = b"bad"
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))
            with self.assertRaises(artifacts.ValidationError):
                artifacts.extract_engine_archive(archive, root / "extract")

    def test_atomic_profile_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            link = root / "profiles" / "dcvcrt" / "720p"
            artifacts.atomic_symlink(first, link)
            self.assertEqual(link.resolve(), first)
            artifacts.atomic_symlink(second, link)
            self.assertEqual(link.resolve(), second)

    def test_build_all_expands_every_registered_profile(self) -> None:
        target = REPOSITORY_ROOT / "configs/targets/rtx4070-ubuntu2404.json"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            artifacts.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            result = artifacts.forward_artifact_command(
                "build",
                [
                    "--all",
                    "--target-profile",
                    str(target),
                    "--engines-root",
                    temporary,
                ],
            )
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, len(artifacts.ENGINE_PROFILES))
        commands = [call.args[0] for call in run.call_args_list]
        profile_paths = [command[command.index("--engine-profile-path") + 1] for command in commands]
        engine_paths = [command[command.index("--engines") + 1] for command in commands]
        self.assertEqual(
            profile_paths,
            [
                str(REPOSITORY_ROOT / f"configs/engine-profiles/{profile}.json")
                for profile in artifacts.ENGINE_PROFILES
            ],
        )
        self.assertEqual(
            engine_paths,
            [str(Path(temporary) / f"dcvcrt-{profile}") for profile in artifacts.ENGINE_PROFILES],
        )

    def test_build_forwards_hardware_compatibility(self) -> None:
        target = REPOSITORY_ROOT / "configs/targets/rtx4070-ubuntu2404.json"
        with mock.patch.object(
            artifacts.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as run:
            artifacts.forward_artifact_command(
                "build",
                [
                    "--profile",
                    "720p",
                    "--target-profile",
                    str(target),
                    "--hardware-compatibility",
                    "same_compute_capability",
                ],
            )
        command = run.call_args.args[0]
        index = command.index("--hardware-compatibility")
        self.assertEqual(command[index + 1], "same_compute_capability")


if __name__ == "__main__":
    unittest.main()
