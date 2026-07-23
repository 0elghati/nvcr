#!/usr/bin/env python3
"""Download, verify, and validate staged NVCR engine release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

MAX_GITHUB_RELEASE_ASSET_BYTES = 2 * 1024 * 1024 * 1024
SAFE_FILENAME = re.compile(r"^[0-9A-Za-z._+-]+$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class EngineAsset:
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


def parse_assets(text: str, *, tag: str, allow_file_url: bool) -> list[EngineAsset]:
    if not tag.startswith("v") or not SAFE_FILENAME.fullmatch(tag):
        fail(f"invalid release tag: {tag!r}")
    assets: list[EngineAsset] = []
    seen_names: set[str] = set()
    seen_urls: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            fail(
                f"asset row {line_number} must be '<filename> <sha256> <url>'"
            )
        filename, expected_sha256, url = fields
        if (
            not SAFE_FILENAME.fullmatch(filename)
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            fail(f"asset row {line_number} has an unsafe filename: {filename!r}")
        if not filename.startswith(f"nvcr-{tag}-") or not filename.endswith(
            "-engines.tar.gz"
        ):
            fail(
                f"asset row {line_number} filename must start with nvcr-{tag}- "
                "and end with -engines.tar.gz"
            )
        if not SHA256_HEX.fullmatch(expected_sha256):
            fail(f"asset row {line_number} has an invalid SHA-256 digest")
        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" and not (
            allow_file_url and parsed_url.scheme == "file"
        ):
            fail(f"asset row {line_number} must use an https URL")
        if filename in seen_names:
            fail(f"duplicate asset filename: {filename}")
        if url in seen_urls:
            fail(f"duplicate asset URL for {filename}")
        seen_names.add(filename)
        seen_urls.add(url)
        assets.append(EngineAsset(filename, expected_sha256, url))
    if not assets:
        fail("no engine assets were provided")
    return assets


def download(url: str, destination: Path) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme == "file":
        source = Path(parsed_url.path)
        shutil.copyfile(source, destination)
        return
    command = [
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
    ]
    subprocess.run(command, check=True)


def safe_member_name(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name:
        return False
    return all(part not in {"", ".", ".."} for part in Path(name).parts)


def validate_asset_manifest(asset_root: Path, manifest_path: Path) -> None:
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2 or not SHA256_HEX.fullmatch(fields[0]):
            fail(f"invalid asset manifest line {line_number}: {manifest_path.name}")
        name = fields[1]
        if not safe_member_name(name):
            fail(f"unsafe asset manifest path on line {line_number}: {name!r}")
        if name in expected:
            fail(f"duplicate asset manifest path: {name}")
        expected[name] = fields[0]

    actual: dict[str, Path] = {}
    for path in asset_root.rglob("*"):
        if path == manifest_path or not path.is_file():
            continue
        relative = "./" + path.relative_to(asset_root).as_posix()
        actual[relative] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        fail(f"asset manifest mismatch; missing={missing}, extra={extra}")
    for name, digest in expected.items():
        if sha256_file(actual[name]) != digest:
            fail(f"asset manifest SHA-256 mismatch: {name}")


def validate_archive(
    archive: Path,
    *,
    tag: str,
    extract_root: Path,
    nvcr_artifacts: Path,
) -> dict[str, str]:
    archive_stem = archive.name.removesuffix(".tar.gz")
    target_extract_root = extract_root / archive_stem
    if target_extract_root.exists():
        shutil.rmtree(target_extract_root)
    target_extract_root.mkdir(parents=True)

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if not members:
            fail(f"{archive.name} is empty")
        top_levels: set[str] = set()
        for member in members:
            if not safe_member_name(member.name):
                fail(f"{archive.name} contains an unsafe path: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                fail(f"{archive.name} contains a non-regular archive member")
            parts = Path(member.name).parts
            top_levels.add(parts[0])
            basename = parts[-1]
            if basename.endswith((".onnx", ".pth", ".pth.tar")):
                fail(f"{archive.name} contains forbidden model source asset: {basename}")
        if top_levels != {archive_stem}:
            fail(
                f"{archive.name} must contain exactly one top-level directory "
                f"named {archive_stem}"
            )
        try:
            tar.extractall(target_extract_root, filter="data")
        except TypeError:
            tar.extractall(target_extract_root)

    asset_root = target_extract_root / archive_stem
    bundle_root = asset_root / "dcvcrt"
    manifest_path = bundle_root / "engine_manifest.json"
    checksum_path = bundle_root / "engine.sha256"
    asset_manifest_path = asset_root / "ENGINE-ASSET-MANIFEST.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        fail(f"{archive.name} must contain dcvcrt/engine_manifest.json and dcvcrt/engine.sha256")
    if not asset_manifest_path.is_file():
        fail(f"{archive.name} must contain ENGINE-ASSET-MANIFEST.sha256")
    validate_asset_manifest(asset_root, asset_manifest_path)

    validation = subprocess.run(
        [sys.executable, str(nvcr_artifacts), "validate", str(bundle_root), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if validation.returncode != 0:
        sys.stderr.write(validation.stdout)
        sys.stderr.write(validation.stderr)
        fail(f"{archive.name} failed nvcr-artifacts validation")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_name = (
        f"nvcr-{tag}-{manifest['model_profile_id']}-"
        f"{manifest['target_profile_id']}-{manifest['engine_profile_id']}-"
        "engines.tar.gz"
    )
    if archive.name != expected_name:
        fail(f"{archive.name} does not match engine manifest identity; expected {expected_name}")

    tensorrt = (
        f"{manifest['tensorrt_version_major']}."
        f"{manifest['tensorrt_version_minor']}."
        f"{manifest['tensorrt_version_patch']}"
    )
    compute_capability = (
        f"{manifest['compute_capability_major']}."
        f"{manifest['compute_capability_minor']}"
    )
    return {
        "filename": archive.name,
        "model_profile_id": str(manifest["model_profile_id"]),
        "target_profile_id": str(manifest["target_profile_id"]),
        "engine_profile_id": str(manifest["engine_profile_id"]),
        "device_name": str(manifest["device_name"]),
        "compute_capability": compute_capability,
        "tensorrt": tensorrt,
    }


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "## Engine release assets",
        "",
        "| Asset | Target profile | Engine profile | Device | CC | TensorRT |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['filename']}` | `{row['target_profile_id']}` | "
            f"`{row['engine_profile_id']}` | {row['device_name']} | "
            f"{row['compute_capability']} | {row['tensorrt']} |"
        )
    lines.extend(
        [
            "",
            "These are separate target-specific engine assets. They are not part of "
            "the generic NVCR binary packages.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--assets-text", default=None)
    parser.add_argument("--assets-file", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nvcr-artifacts", type=Path, required=True)
    parser.add_argument("--allow-file-url", action="store_true")
    args = parser.parse_args()

    if args.assets_file is not None and args.assets_text is not None:
        fail("use either --assets-file or --assets-text, not both")
    if args.assets_file is not None:
        assets_text = args.assets_file.read_text(encoding="utf-8")
    else:
        assets_text = args.assets_text
    if assets_text is None:
        assets_text = os.environ.get("NVCR_ENGINE_RELEASE_ASSETS", "")

    if not args.nvcr_artifacts.is_file():
        fail(f"missing nvcr-artifacts validator: {args.nvcr_artifacts}")

    assets = parse_assets(
        assets_text,
        tag=args.tag,
        allow_file_url=args.allow_file_url,
    )
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.extract_dir.mkdir(parents=True, exist_ok=True)

    upload_files: list[Path] = []
    summary_rows: list[dict[str, str]] = []
    for asset in assets:
        archive = args.download_dir / asset.filename
        print(f"Downloading {asset.filename}", flush=True)
        download(asset.url, archive)
        size_bytes = archive.stat().st_size
        if size_bytes >= MAX_GITHUB_RELEASE_ASSET_BYTES:
            fail(f"{asset.filename} is >= 2 GiB and cannot be uploaded as one GitHub Release asset")
        actual_sha256 = sha256_file(archive)
        if actual_sha256 != asset.sha256:
            fail(
                f"{asset.filename} SHA-256 mismatch: expected "
                f"{asset.sha256}, got {actual_sha256}"
            )
        checksum_file = args.download_dir / f"{asset.filename}.sha256"
        checksum_file.write_text(
            f"{asset.sha256}  {asset.filename}\n",
            encoding="utf-8",
        )
        summary_rows.append(
            validate_archive(
                archive,
                tag=args.tag,
                extract_root=args.extract_dir,
                nvcr_artifacts=args.nvcr_artifacts,
            )
        )
        upload_files.extend((archive, checksum_file))

    (args.download_dir / "upload-files.txt").write_text(
        "".join(f"{path}\n" for path in upload_files),
        encoding="utf-8",
    )
    write_summary(args.summary, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
