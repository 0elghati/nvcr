#!/usr/bin/env python3
"""Validate staged engines and merge the rolling NVCR engine catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import nvcr_artifacts as artifacts  # noqa: E402

MAX_GITHUB_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024
SAFE_FILENAME = re.compile(r"^[0-9A-Za-z._+-]+$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StagedAsset:
    filename: str
    sha256: str
    url: str


def fail(message: str) -> None:
    raise SystemExit(f"download-engine-assets: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_assets(text: str, *, allow_file_url: bool) -> list[StagedAsset]:
    assets: list[StagedAsset] = []
    seen_names: set[str] = set()
    seen_urls: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            fail(f"asset row {line_number} must be '<filename> <sha256> <url>'")
        filename, expected_sha256, url = fields
        if (
            not SAFE_FILENAME.fullmatch(filename)
            or "/" in filename
            or "\\" in filename
            or not filename.startswith("nvcr-engines-")
            or not filename.endswith(".tar.gz")
        ):
            fail(f"asset row {line_number} has an invalid stable filename: {filename!r}")
        if not SHA256_HEX.fullmatch(expected_sha256):
            fail(f"asset row {line_number} has an invalid SHA-256")
        parsed = urlparse(url)
        if parsed.scheme != "https" and not (allow_file_url and parsed.scheme == "file"):
            fail(f"asset row {line_number} must use HTTPS")
        if filename in seen_names or url in seen_urls:
            fail(f"duplicate staged asset name or URL: {filename}")
        seen_names.add(filename)
        seen_urls.add(url)
        assets.append(StagedAsset(filename, expected_sha256, url))
    if not assets:
        fail("no engine assets were provided")
    return assets


def download(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        shutil.copyfile(Path(parsed.path), destination)
        return
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--silent",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "20",
            "--max-time",
            "1800",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def validate_archive(
    archive: Path,
    *,
    archive_sha256: str,
    extract_root: Path,
    target_profiles_dir: Path,
) -> dict[str, object]:
    try:
        bundle_root = artifacts.extract_engine_archive(archive, extract_root)
    except artifacts.ValidationError as error:
        fail(str(error))
    manifest_path = bundle_root / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = artifacts.canonical_profile_name(str(manifest["engine_profile_id"]))
    target_profile_path = target_profiles_dir / f"{manifest['target_profile_id']}.json"
    if not target_profile_path.is_file():
        fail(f"missing registered target profile: {target_profile_path}")
    if sha256_file(target_profile_path) != manifest.get("target_profile_sha256"):
        fail(f"target profile digest differs from bundle: {target_profile_path.name}")
    target_profile = json.loads(target_profile_path.read_text(encoding="utf-8"))
    architecture = target_profile.get("host", {}).get("architecture")
    if architecture not in ("x86_64", "aarch64"):
        fail(f"target profile has unsupported architecture: {architecture}")
    expected_name = artifacts.expected_asset_filename(
        str(manifest["target_profile_id"]),
        str(manifest["model_profile_id"]),
        profile,
        str(manifest.get("hardware_compatibility", "exact")),
        architecture,
        int(manifest["compute_capability_major"]),
        int(manifest["compute_capability_minor"]),
    )
    if archive.name != expected_name:
        fail(f"{archive.name} does not match bundle identity; expected {expected_name}")
    return {
        "backend": "dcvcrt",
        "model_profile_id": str(manifest["model_profile_id"]),
        "target_profile_id": str(manifest["target_profile_id"]),
        "profile": profile,
        "precision": "fp16",
        "hardware_compatibility": str(manifest.get("hardware_compatibility", "exact")),
        "operating_system": "linux",
        "architecture": architecture,
        "device_name": str(manifest["device_name"]),
        "compute_capability_major": int(manifest["compute_capability_major"]),
        "compute_capability_minor": int(manifest["compute_capability_minor"]),
        "multiprocessor_count": int(manifest["multiprocessor_count"]),
        "cuda_runtime_version": int(manifest["cuda_runtime_version"]),
        "tensorrt_version_major": int(manifest["tensorrt_version_major"]),
        "tensorrt_version_minor": int(manifest["tensorrt_version_minor"]),
        "tensorrt_version_patch": int(manifest["tensorrt_version_patch"]),
        "filename": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": archive_sha256,
    }


def merge_catalog(
    existing: list[dict[str, object]], updates: list[dict[str, object]]
) -> list[dict[str, object]]:
    def key(entry: dict[str, object]) -> tuple[str, str, str, str, str]:
        return (
            str(entry["backend"]),
            str(entry["model_profile_id"]),
            str(entry["target_profile_id"]),
            str(entry["profile"]),
            str(entry.get("hardware_compatibility", "exact")),
        )

    merged = {key(entry): entry for entry in existing}
    updated_targets: set[tuple[str, str, str, str]] = set()
    for entry in updates:
        merged[key(entry)] = entry
        item_key = key(entry)
        updated_targets.add((item_key[0], item_key[1], item_key[2], item_key[4]))
    for target in updated_targets:
        profiles = {
            item_key[3]
            for item_key in merged
            if (item_key[0], item_key[1], item_key[2], item_key[4]) == target
        }
        if profiles != set(artifacts.ENGINE_PROFILES):
            fail(
                f"rolling target {target} must contain all registered profiles; "
                f"missing={sorted(set(artifacts.ENGINE_PROFILES) - profiles)}"
            )
    return [merged[item_key] for item_key in sorted(merged)]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "## Rolling engine assets",
        "",
        "| Asset | Target | Profile | Device | CUDA | TensorRT |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['filename']}` | `{row['target_profile_id']}` | `{row['profile']}` | "
            f"{row['device_name']} | {row['cuda_runtime_version']} | "
            f"{row['tensorrt_version_major']}.{row['tensorrt_version_minor']}."
            f"{row['tensorrt_version_patch']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-text")
    parser.add_argument("--assets-file", type=Path)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--target-profiles-dir", type=Path, required=True)
    parser.add_argument("--existing-catalog", type=Path)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--allow-file-url", action="store_true")
    args = parser.parse_args()
    if args.assets_file is not None and args.assets_text is not None:
        fail("use either --assets-file or --assets-text")
    assets_text = (
        args.assets_file.read_text(encoding="utf-8")
        if args.assets_file is not None
        else args.assets_text or os.environ.get("NVCR_ENGINE_RELEASE_ASSETS", "")
    )
    staged = parse_assets(assets_text, allow_file_url=args.allow_file_url)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)

    upload_files: list[Path] = []
    updates: list[dict[str, object]] = []
    for asset in staged:
        archive = args.download_dir / asset.filename
        print(f"Downloading {asset.filename}", flush=True)
        download(asset.url, archive)
        if archive.stat().st_size >= MAX_GITHUB_RELEASE_ASSET_BYTES:
            fail(f"{asset.filename} is >= 2 GiB")
        actual_sha256 = sha256_file(archive)
        if actual_sha256 != asset.sha256:
            fail(f"{asset.filename} SHA-256 mismatch")
        checksum = args.download_dir / f"{asset.filename}.sha256"
        checksum.write_text(f"{actual_sha256}  {asset.filename}\n", encoding="utf-8")
        updates.append(
            validate_archive(
                archive,
                archive_sha256=actual_sha256,
                extract_root=args.extract_dir,
                target_profiles_dir=args.target_profiles_dir,
            )
        )
        upload_files.extend((archive, checksum))

    existing: list[dict[str, object]] = []
    if args.existing_catalog is not None and args.existing_catalog.is_file():
        document = json.loads(args.existing_catalog.read_text(encoding="utf-8"))
        existing = artifacts.validate_catalog(document, str(args.existing_catalog))
    merged = merge_catalog(existing, updates)
    catalog = {"schema": artifacts.CATALOG_SCHEMA, "assets": merged}
    artifacts.validate_catalog(catalog)
    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.download_dir / "upload-files.txt").write_text(
        "".join(f"{path}\n" for path in upload_files), encoding="utf-8"
    )
    write_summary(args.summary, updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
