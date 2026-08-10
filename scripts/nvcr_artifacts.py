#!/usr/bin/env python3
"""Prepare, build, inspect, and validate scoped NVCR DCVC-RT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIRECTORY.parent
if (SOURCE_ROOT / "configs").is_dir() and (SOURCE_ROOT / "scripts").is_dir():
    REPOSITORY_ROOT = SOURCE_ROOT
    HELPER_DIRECTORY = SOURCE_ROOT / "scripts" / "backends" / "dcvcrt"
else:
    install_root = SCRIPT_DIRECTORY.parent
    REPOSITORY_ROOT = install_root / "share" / "nvcr"
    HELPER_DIRECTORY = REPOSITORY_ROOT / "scripts" / "backends" / "dcvcrt"
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
from nvcr_device import (  # noqa: E402
    DeviceDetectionError,
    detect_device_identity,
    identity_summary,
)

DEFAULT_MODEL_PROFILE = REPOSITORY_ROOT / "configs/models/dcvcrt-cvpr2025.json"
ENGINE_PROFILE_DIRECTORY = REPOSITORY_ROOT / "configs/engine-profiles"
TARGET_PROFILE_DIRECTORY = REPOSITORY_ROOT / "configs/targets"
ENGINE_PROFILES = ("qcif", "cif", "360p", "540p", "720p", "1080p")
CATALOG_SCHEMA = "nvcr.engine-catalog.v1"
CATALOG_FILENAME = "nvcr-engine-catalog.json"
DEFAULT_REPOSITORY = "0elghati/nvcr"
DEFAULT_ASSET_RELEASE = "engine-assets"
SAFE_IDENTIFIER = re.compile(r"^[0-9A-Za-z._-]+$")
PINNED_COMMIT = "1feb52a592a9ff2c4e4ba2e5122e2da49a211466"
EXPECTED_CHECKPOINTS = {
    "i_frame_manifest.json": ("555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9", "cvpr2025_image.pth.tar"),
    "p_frame_manifest.json": ("b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6", "cvpr2025_video.pth.tar"),
}
REQUIRED_PLANS = (
    "i_analysis.plan",
    "i_hyper_analysis.plan",
    "i_hyper_synthesis.plan",
    "i_spatial_prior_1.plan",
    "i_spatial_prior_2.plan",
    "i_spatial_prior_3.plan",
    "i_synthesis.plan",
    "p_reference_frame.plan",
    "p_reference_feature.plan",
    "p_analysis.plan",
    "p_hyper_analysis.plan",
    "p_prior.plan",
    "p_spatial_prior.plan",
    "p_synthesis.plan",
)
RUNTIME_ASSETS = (
    "i_entropy.bin",
    "i_quant.bin",
    "i_frame_manifest.json",
    "p_entropy.bin",
    "p_quant.bin",
    "p_frame_manifest.json",
)

# These filenames are part of the pinned DCVC-RT artifact contract. Keeping
# them here prevents a partially exported model from being stamped into an
# engine bundle.
MODEL_GRAPHS = {
    "i_frame_manifest.json": (
        "i_analysis.onnx",
        "i_hyper_analysis.onnx",
        "i_hyper_synthesis.onnx",
        "i_spatial_prior_1.onnx",
        "i_spatial_prior_2.onnx",
        "i_spatial_prior_3.onnx",
        "i_synthesis.onnx",
    ),
    "p_frame_manifest.json": (
        "p_reference_frame.onnx",
        "p_reference_feature.onnx",
        "p_analysis.onnx",
        "p_hyper_analysis.onnx",
        "p_prior.onnx",
        "p_spatial_prior.onnx",
        "p_synthesis.onnx",
    ),
}
MODEL_ASSETS = {
    "i_frame_manifest.json": ("i_entropy.bin", "i_quant.bin"),
    "p_frame_manifest.json": ("p_entropy.bin", "p_quant.bin"),
}


class ValidationError(RuntimeError):
    """A bundle or profile failed the NVCR contract."""


def canonical_profile_name(value: str, *, warn_legacy: bool = False) -> str:
    profile = value
    if profile.endswith("-fp16"):
        profile = profile[:-5]
        if warn_legacy:
            print(
                f"nvcr-artifacts: warning: profile '{value}' is deprecated; use '{profile}'",
                file=sys.stderr,
            )
    if profile not in ENGINE_PROFILES:
        raise ValidationError(
            f"unsupported engine profile: {value}; expected one of {', '.join(ENGINE_PROFILES)}"
        )
    return profile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing JSON file: {path}")
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValidationError(f"JSON file exceeds 4 MiB: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def require_string(document: dict[str, Any], key: str, source: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{source} requires non-empty string '{key}'")
    return value


def validate_visible_dimensions(document: dict[str, Any], source: Path) -> str:
    dimensions = document.get("visible_dimensions")
    if not isinstance(dimensions, dict):
        raise ValidationError(f"{source} requires object 'visible_dimensions'")
    normalized: dict[str, tuple[int, int]] = {}
    for label in ("minimum", "optimum", "maximum"):
        value = dimensions.get(label)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
        ):
            raise ValidationError(
                f"{source} visible_dimensions.{label} must contain two positive integers"
            )
        normalized[label] = (value[0], value[1])
    for axis in range(2):
        if not (
            normalized["minimum"][axis]
            <= normalized["optimum"][axis]
            <= normalized["maximum"][axis]
        ):
            raise ValidationError(f"{source} visible dimensions are not ordered")
    return (
        "fixed"
        if normalized["minimum"] == normalized["optimum"] == normalized["maximum"]
        else "dynamic"
    )


def safe_bundle_name(value: str, source: Path) -> str:
    candidate = Path(value)
    if (candidate.is_absolute() or candidate.name != value or value in (".", "..")
            or "\\" in value):
        raise ValidationError(f"{source} contains non-portable path: {value}")
    return value


def validate_hashed_entries(
    root: Path,
    entries: object,
    source: Path,
    *,
    require_files: bool,
    expected_names: tuple[str, ...] | None = None,
) -> None:
    if not isinstance(entries, list):
        raise ValidationError(f"{source} requires a list of hashed entries")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValidationError(f"{source} has a non-object hashed entry")
        name = safe_bundle_name(require_string(entry, "file", source), source)
        if name in names:
            raise ValidationError(f"{source} contains duplicate entry: {name}")
        names.append(name)
        expected = require_string(entry, "sha256", source)
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValidationError(f"{source} has invalid SHA-256 for {name}")
        path = root / name
        if require_files:
            if not path.is_file():
                raise ValidationError(f"bundle is missing {name}")
            if sha256(path) != expected:
                raise ValidationError(f"SHA-256 mismatch: {name}")
    if expected_names is not None and set(names) != set(expected_names):
        missing = sorted(set(expected_names) - set(names))
        extra = sorted(set(names) - set(expected_names))
        raise ValidationError(
            f"{source} entries do not match the pinned model contract; "
            f"missing={missing}, extra={extra}"
        )


def validate_model_manifests(root: Path, *, require_graphs: bool) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for filename, expected_identity in EXPECTED_CHECKPOINTS.items():
        expected_checkpoint_hash, expected_checkpoint_file = expected_identity
        path = root / filename
        manifest = load_json(path)
        if manifest.get("format") != 2 or manifest.get("schema") != "nvcr.model-manifest.v2":
            raise ValidationError(f"{path} is not an NVCR model manifest v2")
        if manifest.get("model_profile_id") != "dcvcrt-cvpr2025":
            raise ValidationError(f"{path} has the wrong model profile")
        if manifest.get("codec") != "DCVC-RT" or manifest.get("precision") != "fp16":
            raise ValidationError(f"{path} has an unsupported codec or precision")
        expected_frame_type = "I" if filename.startswith("i_") else "P"
        if manifest.get("frame_type") != expected_frame_type:
            raise ValidationError(f"{path} has the wrong frame type")
        if manifest.get("reference_commit") != PINNED_COMMIT:
            raise ValidationError(f"{path} has the wrong upstream commit")
        if manifest.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise ValidationError(f"{path} has the wrong checkpoint hash")
        checkpoint_file = safe_bundle_name(
            require_string(manifest, "checkpoint_file", path), path)
        if checkpoint_file != expected_checkpoint_file:
            raise ValidationError(f"{path} has an unexpected checkpoint name")
        validate_hashed_entries(
            root,
            manifest.get("graphs"),
            path,
            require_files=require_graphs,
            expected_names=MODEL_GRAPHS[filename],
        )
        validate_hashed_entries(
            root,
            manifest.get("assets"),
            path,
            require_files=True,
            expected_names=MODEL_ASSETS[filename],
        )
        manifests.append(manifest)
    return manifests


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    if not path.is_file():
        raise ValidationError(f"missing checksum manifest: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if (
            len(fields) != 2
            or len(fields[0]) != 64
            or any(character not in "0123456789abcdef" for character in fields[0])
        ):
            raise ValidationError(f"invalid checksum line {line_number}: {path}")
        name = safe_bundle_name(fields[1], path)
        if name in checksums:
            raise ValidationError(f"duplicate checksum for {name}")
        checksums[name] = fields[0]
    return checksums


def validate_engine_bundle(root: Path) -> dict[str, Any]:
    validate_model_manifests(root, require_graphs=False)
    manifest_path = root / "engine_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("format") != 2 or manifest.get("schema") != "nvcr.engine-bundle.v2":
        raise ValidationError("engine bundle must use nvcr.engine-bundle.v2")
    if manifest.get("kind") != "nvcr-tensorrt-engine-bundle":
        raise ValidationError("engine manifest kind is invalid")
    if manifest.get("model_profile_id") != "dcvcrt-cvpr2025":
        raise ValidationError("engine manifest model profile is invalid")
    if manifest.get("precision") != "fp16":
        raise ValidationError("only FP16 engine bundles are supported")
    for key in (
        "device_name",
        "target_profile_id",
        "engine_profile_id",
        "model_profile_sha256",
        "engine_profile_sha256",
        "target_profile_sha256",
        "optimization_point",
        "checksum_manifest",
        "checksum_manifest_sha256",
        "i_model_manifest_sha256",
        "p_model_manifest_sha256",
    ):
        require_string(manifest, key, manifest_path)
    for key in (
        "model_profile_sha256",
        "engine_profile_sha256",
        "target_profile_sha256",
    ):
        digest = manifest[key]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValidationError(f"engine manifest has invalid profile SHA-256: {key}")
    for key in (
        "cuda_runtime_version",
        "tensorrt_version_major",
        "tensorrt_version_minor",
        "tensorrt_version_patch",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
        "workspace_mib",
        "builder_optimization_level",
    ):
        if not isinstance(manifest.get(key), int):
            raise ValidationError(f"engine manifest requires integer '{key}'")

    if manifest["workspace_mib"] <= 0 or not 0 <= manifest["builder_optimization_level"] <= 5:
        raise ValidationError("engine workspace or builder level is invalid")
    try:
        manifest_profile = canonical_profile_name(str(manifest["engine_profile_id"]))
    except ValidationError as error:
        raise ValidationError("engine profile and optimization point do not match") from error
    if manifest_profile != manifest["optimization_point"]:
        raise ValidationError("engine profile and optimization point do not match")
    expected_shape_profile = validate_visible_dimensions(manifest, manifest_path)
    shape_profile = manifest.get("shape_profile")
    if shape_profile is None:
        # Existing v2 release bundles predate this field and are all dynamic.
        # A fixed-range bundle must opt in so a hybrid cannot be mislabeled.
        if expected_shape_profile == "fixed":
            raise ValidationError("fixed engine bundle requires shape_profile='fixed'")
        shape_profile = "dynamic"
    if shape_profile not in ("dynamic", "fixed") or shape_profile != expected_shape_profile:
        raise ValidationError("engine shape profile does not match visible dimensions")

    hardware_compatibility = manifest.get("hardware_compatibility", "exact")
    if hardware_compatibility not in (
        "exact",
        "same_compute_capability",
        "ampere_plus",
    ):
        raise ValidationError("engine manifest has invalid hardware_compatibility")

    checksum_name = safe_bundle_name(str(manifest["checksum_manifest"]), manifest_path)
    checksum_path = root / checksum_name
    if sha256(checksum_path) != manifest["checksum_manifest_sha256"]:
        raise ValidationError("engine checksum manifest hash does not match")
    checksums = parse_checksum_manifest(checksum_path)
    required = set(REQUIRED_PLANS + RUNTIME_ASSETS)
    if set(checksums) != required:
        missing = sorted(required - set(checksums))
        extra = sorted(set(checksums) - required)
        raise ValidationError(f"checksum set mismatch; missing={missing}, extra={extra}")
    manifest_hashes = manifest.get("files")
    if not isinstance(manifest_hashes, dict) or manifest_hashes != checksums:
        raise ValidationError("engine manifest file hashes do not match engine.sha256")
    for name, expected in checksums.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValidationError(f"SHA-256 mismatch: {name}")
    if sha256(root / "i_frame_manifest.json") != manifest["i_model_manifest_sha256"]:
        raise ValidationError("I-frame model manifest identity mismatch")
    if sha256(root / "p_frame_manifest.json") != manifest["p_model_manifest_sha256"]:
        raise ValidationError("P-frame model manifest identity mismatch")
    return manifest


def inspect_bundle(root: Path) -> dict[str, Any]:
    if (root / "engine_manifest.json").is_file():
        manifest = load_json(root / "engine_manifest.json")
        return {
            "kind": "engine",
            "path": str(root),
            "schema": manifest.get("schema"),
            "model_profile_id": manifest.get("model_profile_id"),
            "model_profile_sha256": manifest.get("model_profile_sha256"),
            "engine_profile_sha256": manifest.get("engine_profile_sha256"),
            "target_profile_sha256": manifest.get("target_profile_sha256"),
            "engine_profile_id": manifest.get("engine_profile_id"),
            "precision": manifest.get("precision"),
            "device_name": manifest.get("device_name"),
            "compute_capability": (
                f"{manifest.get('compute_capability_major')}."
                f"{manifest.get('compute_capability_minor')}"
            ),
            "tensorrt": (
                f"{manifest.get('tensorrt_version_major')}."
                f"{manifest.get('tensorrt_version_minor')}."
                f"{manifest.get('tensorrt_version_patch')}"
            ),
            "file_count": len(manifest.get("files", {})),
        }
    manifests = validate_model_manifests(root, require_graphs=True)
    return {
        "kind": "model",
        "path": str(root),
        "schema": "nvcr.model-manifest.v2",
        "model_profile_id": manifests[0]["model_profile_id"],
        "precision": manifests[0]["precision"],
        "graph_count": sum(len(item["graphs"]) for item in manifests),
        "asset_count": sum(len(item["assets"]) for item in manifests),
    }


def load_profile(path: Path, expected_schema: str) -> dict[str, Any]:
    profile = load_json(path)
    if profile.get("schema") != expected_schema:
        raise ValidationError(f"{path} must use schema {expected_schema}")
    if profile.get("precision") != "fp16":
        raise ValidationError(f"{path} must select FP16")
    profile_id = require_string(profile, "id", path)
    if expected_schema == "nvcr.engine-profile.v1":
        optimization_point = require_string(profile, "optimization_point", path)
        if canonical_profile_name(profile_id) != optimization_point:
            raise ValidationError(f"{path} engine profile id and optimization point differ")
        validate_visible_dimensions(profile, path)
        workspace = profile.get("workspace_mib")
        builder_level = profile.get("builder_optimization_level")
        if not isinstance(workspace, int) or isinstance(workspace, bool) or workspace <= 0:
            raise ValidationError(f"{path} requires a positive workspace_mib")
        if (
            not isinstance(builder_level, int)
            or isinstance(builder_level, bool)
            or not 0 <= builder_level <= 5
        ):
            raise ValidationError(f"{path} builder_optimization_level must be in [0, 5]")
    return profile


def default_engine_root() -> Path:
    configured = os.environ.get("NVCR_ENGINE_ROOT")
    if configured:
        return Path(configured)
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "nvcr" / "engines"
    return Path.home() / ".local" / "share" / "nvcr" / "engines"


def expected_asset_filename(
    target: str,
    model: str,
    profile: str,
    hardware_compatibility: str = "exact",
    architecture: str = "x86_64",
    compute_capability_major: int = 0,
    compute_capability_minor: int = 0,
) -> str:
    public_architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
        architecture, architecture
    )
    if hardware_compatibility == "same_compute_capability":
        target = (
            f"linux-{public_architecture}-sm{compute_capability_major}{compute_capability_minor}"
        )
    elif hardware_compatibility == "ampere_plus":
        target = f"linux-{public_architecture}-ampere-plus"
    return f"nvcr-engines-{target}-{model}-{profile}.tar.gz"


def validate_catalog(document: object, source: str = CATALOG_FILENAME) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or document.get("schema") != CATALOG_SCHEMA:
        raise ValidationError(f"{source} must use schema {CATALOG_SCHEMA}")
    raw_assets = document.get("assets")
    if not isinstance(raw_assets, list):
        raise ValidationError(f"{source} requires an assets list")
    assets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    seen_filenames: set[str] = set()
    string_fields = (
        "backend",
        "model_profile_id",
        "target_profile_id",
        "profile",
        "precision",
        "operating_system",
        "architecture",
        "device_name",
        "filename",
        "sha256",
    )
    integer_fields = (
        "size_bytes",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
        "cuda_runtime_version",
        "tensorrt_version_major",
        "tensorrt_version_minor",
        "tensorrt_version_patch",
    )
    for index, raw in enumerate(raw_assets):
        if not isinstance(raw, dict):
            raise ValidationError(f"{source} asset {index} must be an object")
        for key in string_fields:
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValidationError(f"{source} asset {index} requires string {key}")
        for key in integer_fields:
            if not isinstance(raw.get(key), int) or isinstance(raw[key], bool):
                raise ValidationError(f"{source} asset {index} requires integer {key}")
        compatibility = raw.get("hardware_compatibility", "exact")
        if compatibility not in ("exact", "same_compute_capability", "ampere_plus"):
            raise ValidationError(
                f"{source} asset {index} has unsupported hardware_compatibility"
            )
        if raw["architecture"] == "aarch64" and compatibility != "exact":
            raise ValidationError(
                f"{source} asset {index} uses hardware compatibility on Jetson/AArch64"
            )
        raw = dict(raw)
        raw["hardware_compatibility"] = compatibility
        for key in ("backend", "model_profile_id", "target_profile_id"):
            if not SAFE_IDENTIFIER.fullmatch(raw[key]):
                raise ValidationError(f"{source} asset {index} has unsafe {key}")
        profile = canonical_profile_name(raw["profile"])
        if raw["profile"] != profile:
            raise ValidationError(f"{source} asset {index} must use canonical profile {profile}")
        if raw["precision"] != "fp16":
            raise ValidationError(f"{source} asset {index} must use internal FP16 precision")
        if raw["operating_system"] != "linux" or raw["architecture"] not in (
            "x86_64",
            "aarch64",
        ):
            raise ValidationError(f"{source} asset {index} has an unsupported platform")
        expected_name = expected_asset_filename(
            raw["target_profile_id"],
            raw["model_profile_id"],
            profile,
            compatibility,
            raw["architecture"],
            raw["compute_capability_major"],
            raw["compute_capability_minor"],
        )
        if raw["filename"] != expected_name:
            raise ValidationError(
                f"{source} asset {index} filename does not match its identity; expected {expected_name}"
            )
        if (
            len(raw["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in raw["sha256"])
            or raw["size_bytes"] <= 0
            or raw["compute_capability_major"] <= 0
            or raw["compute_capability_minor"] < 0
            or raw["multiprocessor_count"] <= 0
            or raw["cuda_runtime_version"] <= 0
            or raw["tensorrt_version_major"] <= 0
            or raw["tensorrt_version_minor"] < 0
            or raw["tensorrt_version_patch"] < 0
        ):
            raise ValidationError(f"{source} asset {index} has invalid hash, size, or runtime data")
        key = (
            raw["backend"],
            raw["model_profile_id"],
            raw["target_profile_id"],
            profile,
            compatibility,
        )
        if key in seen:
            raise ValidationError(f"{source} contains duplicate catalog identity: {key}")
        if raw["filename"] in seen_filenames:
            raise ValidationError(
                f"{source} contains duplicate catalog filename: {raw['filename']}"
            )
        seen.add(key)
        seen_filenames.add(raw["filename"])
        assets.append(dict(raw))
    return assets


def catalog_entry_match_rank(entry: dict[str, Any], identity: dict[str, Any]) -> int | None:
    common_fields = (
        "operating_system",
        "architecture",
        "tensorrt_version_major",
        "tensorrt_version_minor",
        "tensorrt_version_patch",
    )
    if not all(entry[field] == identity[field] for field in common_fields):
        return None
    if not cuda_runtime_compatible(entry, identity):
        return None
    compatibility = entry.get("hardware_compatibility", "exact")
    if compatibility == "exact":
        exact_fields = (
            "device_name",
            "compute_capability_major",
            "compute_capability_minor",
            "multiprocessor_count",
        )
        return 0 if all(entry[field] == identity[field] for field in exact_fields) else None
    if entry["architecture"] != "x86_64":
        return None
    if compatibility == "same_compute_capability":
        return 1 if (
            entry["compute_capability_major"] == identity["compute_capability_major"]
            and entry["compute_capability_minor"] == identity["compute_capability_minor"]
        ) else None
    if compatibility == "ampere_plus":
        return 2 if identity["compute_capability_major"] >= 8 else None
    return None


def catalog_entry_matches(entry: dict[str, Any], identity: dict[str, Any]) -> bool:
    return catalog_entry_match_rank(entry, identity) is not None


def cuda_runtime_major(version: int) -> int:
    return version // 1000


def cuda_runtime_compatible(entry: dict[str, Any], identity: dict[str, Any]) -> bool:
    engine_runtime = int(entry["cuda_runtime_version"])
    active_runtime = int(identity["cuda_runtime_version"])
    if entry["architecture"] != "x86_64":
        return engine_runtime == active_runtime
    return (
        cuda_runtime_major(engine_runtime) == cuda_runtime_major(active_runtime)
        and engine_runtime <= active_runtime
    )


def github_request_json(url: str, token: str | None = None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nvcr-artifacts",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            document = json.load(response)
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot download {url}: {error}") from error
    if not isinstance(document, dict):
        raise ValidationError(f"GitHub returned a non-object response for {url}")
    return document


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def download_file(url: str, destination: Path, token: str | None = None) -> None:
    headers = {"Accept": "application/octet-stream", "User-Agent": "nvcr-artifacts"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        request = Request(url, headers=headers)
        if token and urlparse(url).hostname == "api.github.com":
            try:
                response = build_opener(_NoRedirectHandler).open(request, timeout=1800)
            except HTTPError as error:
                location = error.headers.get("Location")
                if error.code in (301, 302, 303, 307, 308) and location:
                    return download_file(location, destination)
                raise
        else:
            response = urlopen(request, timeout=1800)
        with response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except (HTTPError, URLError, OSError) as error:
        raise ValidationError(f"cannot download {url}: {error}") from error


def safe_archive_member(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name:
        return False
    return all(part not in ("", ".", "..") for part in Path(name).parts)


def validate_asset_manifest(asset_root: Path) -> None:
    manifest_path = asset_root / "ENGINE-ASSET-MANIFEST.sha256"
    if not manifest_path.is_file():
        raise ValidationError("engine archive is missing ENGINE-ASSET-MANIFEST.sha256")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValidationError(f"invalid engine asset manifest line {line_number}")
        name = fields[1]
        if name.startswith("./"):
            name = name[2:]
        if not safe_archive_member(name) or name in expected:
            raise ValidationError(f"invalid engine asset manifest path: {name}")
        expected[name] = fields[0]
    actual = {
        path.relative_to(asset_root).as_posix(): path
        for path in asset_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(expected):
        raise ValidationError(
            "engine asset manifest file set differs; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for name, digest in expected.items():
        if sha256(actual[name]) != digest:
            raise ValidationError(f"engine asset manifest SHA-256 mismatch: {name}")


def extract_engine_archive(archive: Path, extract_root: Path) -> Path:
    archive_stem = archive.name.removesuffix(".tar.gz")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if not members:
            raise ValidationError(f"{archive.name} is empty")
        top_levels: set[str] = set()
        for member in members:
            if not safe_archive_member(member.name):
                raise ValidationError(f"{archive.name} contains unsafe path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValidationError(f"{archive.name} contains a non-regular member")
            if Path(member.name).name.endswith((".onnx", ".pth", ".pth.tar")):
                raise ValidationError(f"{archive.name} contains a model-source asset")
            top_levels.add(Path(member.name).parts[0])
        if top_levels != {archive_stem}:
            raise ValidationError(
                f"{archive.name} must contain one top-level directory named {archive_stem}"
            )
        stream.extractall(extract_root)
    asset_root = extract_root / archive_stem
    validate_asset_manifest(asset_root)
    bundle_root = asset_root / "dcvcrt"
    validate_engine_bundle(bundle_root)
    return bundle_root


def atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


def install_catalog_assets(
    entries: list[dict[str, Any]],
    asset_urls: dict[str, str],
    identity: dict[str, Any],
    *,
    backend: str,
    requested_profiles: list[str],
    engine_root: Path,
    token: str | None = None,
) -> list[str]:
    ranked = [
        (rank, entry)
        for entry in entries
        if entry["backend"] == backend
        if (rank := catalog_entry_match_rank(entry, identity)) is not None
    ]
    best_rank_by_profile: dict[str, int] = {}
    for rank, entry in ranked:
        profile = entry["profile"]
        best_rank_by_profile[profile] = min(rank, best_rank_by_profile.get(profile, rank))
    matching = [
        entry for rank, entry in ranked if rank == best_rank_by_profile[entry["profile"]]
    ]
    if not matching:
        published = sorted(
            {
                f"{entry['architecture']} {entry['device_name']} TensorRT "
                f"{entry['tensorrt_version_major']}.{entry['tensorrt_version_minor']}."
                f"{entry['tensorrt_version_patch']} CUDA runtime "
                f"{entry['cuda_runtime_version']}"
                for entry in entries
                if entry["backend"] == backend
            }
        )
        raise ValidationError(
            f"no compatible published {backend} engine was found for "
            f"{identity_summary(identity)}; published targets: {published or ['none']}; "
            "NVCR does not build engines automatically—follow docs/dcvcrt-artifacts.md "
            "to build a target-local engine manually"
        )
    by_profile: dict[str, dict[str, Any]] = {}
    for entry in matching:
        profile = entry["profile"]
        if profile in by_profile:
            raise ValidationError(
                f"catalog match is ambiguous for {backend}/{profile} and {identity_summary(identity)}"
            )
        by_profile[profile] = entry
    try:
        profiles = (
            [canonical_profile_name(item, warn_legacy=True) for item in requested_profiles]
            if requested_profiles
            else [profile for profile in ENGINE_PROFILES if profile in by_profile]
        )
    except ValidationError as error:
        raise ValidationError(
            f"{error}; detected {identity_summary(identity)}; available={sorted(by_profile)}"
        ) from error
    profiles = list(dict.fromkeys(profiles))
    missing = [profile for profile in profiles if profile not in by_profile]
    if missing:
        raise ValidationError(
            f"profiles {missing} are unavailable for {identity_summary(identity)}; "
            f"available={sorted(by_profile)}"
        )
    installed: list[str] = []
    engine_root = engine_root.expanduser().resolve()
    for profile in profiles:
        entry = by_profile[profile]
        filename = entry["filename"]
        url = asset_urls.get(filename)
        if not url:
            raise ValidationError(f"rolling release is missing catalog asset: {filename}")
        with tempfile.TemporaryDirectory(prefix="nvcr-engine-install-") as temporary:
            temporary_root = Path(temporary)
            archive = temporary_root / filename
            print(f"Downloading {filename}")
            download_file(url, archive, token)
            if archive.stat().st_size != entry["size_bytes"]:
                raise ValidationError(f"downloaded size mismatch: {filename}")
            if sha256(archive) != entry["sha256"]:
                raise ValidationError(f"downloaded SHA-256 mismatch: {filename}")
            bundle_root = extract_engine_archive(archive, temporary_root / "extract")
            manifest = load_json(bundle_root / "engine_manifest.json")
            catalog_manifest_fields = (
                "device_name",
                "compute_capability_major",
                "compute_capability_minor",
                "multiprocessor_count",
                "tensorrt_version_major",
                "tensorrt_version_minor",
                "tensorrt_version_patch",
                "precision",
            )
            manifest_compatibility = manifest.get("hardware_compatibility", "exact")
            if (
                manifest.get("target_profile_id") != entry["target_profile_id"]
                or manifest.get("model_profile_id") != entry["model_profile_id"]
                or canonical_profile_name(str(manifest.get("engine_profile_id", ""))) != profile
                or any(manifest.get(field) != entry[field] for field in catalog_manifest_fields)
                or manifest.get("cuda_runtime_version") != entry["cuda_runtime_version"]
                or manifest_compatibility != entry.get("hardware_compatibility", "exact")
            ):
                raise ValidationError(f"catalog and bundle identity differ: {filename}")
            final = (
                engine_root
                / "bundles"
                / backend
                / entry["target_profile_id"]
                / profile
                / entry["sha256"]
            )
            if not final.exists():
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(bundle_root), str(final))
            validate_engine_bundle(final)
        canonical_link = engine_root / "profiles" / backend / profile
        atomic_symlink(final, canonical_link)
        atomic_symlink(final, engine_root / "profiles" / backend / f"{profile}-fp16")
        installed.append(profile)
    atomic_symlink(engine_root / "profiles" / backend, engine_root / "profiles" / "default")
    return installed


def flatten_profile_arguments(values: list[list[str]]) -> list[str]:
    """Flatten profiles supplied in one or more --profile argument groups."""
    return [profile for group in values for profile in group]


def install_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="nvcr-artifacts install",
        description="Install the best compatible engines from the rolling NVCR asset catalog.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        nargs="+",
        default=[],
        metavar="PROFILE",
        help="resolution profile(s) to install; may be repeated",
    )
    parser.add_argument("--backend", default=os.environ.get("NVCR_BACKEND", "dcvcrt"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repo", default=os.environ.get("NVCR_REPO", DEFAULT_REPOSITORY))
    parser.add_argument(
        "--asset-release",
        default=os.environ.get("NVCR_ENGINE_ASSET_RELEASE", DEFAULT_ASSET_RELEASE),
    )
    parser.add_argument("--engine-root", type=Path, default=default_engine_root())
    args = parser.parse_args(arguments)
    if args.backend in ("default", "dcvc_rt", "dcvc-rt"):
        args.backend = "dcvcrt"
    if args.backend != "dcvcrt":
        raise ValidationError(f"unsupported backend: {args.backend}")
    if (
        args.repo.count("/") != 1
        or args.repo.startswith("/")
        or ".." in args.repo
        or not all(SAFE_IDENTIFIER.fullmatch(part) for part in args.repo.split("/"))
    ):
        raise ValidationError(f"invalid GitHub repository: {args.repo}")
    if not SAFE_IDENTIFIER.fullmatch(args.asset_release):
        raise ValidationError(f"invalid engine asset release: {args.asset_release}")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    api_url = f"https://api.github.com/repos/{args.repo}/releases/tags/{args.asset_release}"
    release = github_request_json(api_url, token)
    release_assets = release.get("assets")
    if not isinstance(release_assets, list):
        raise ValidationError("GitHub engine release has no asset list")
    download_url_field = "url" if token else "browser_download_url"
    asset_urls = {
        item["name"]: item[download_url_field]
        for item in release_assets
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get(download_url_field), str)
    }
    catalog_url = asset_urls.get(CATALOG_FILENAME)
    if not catalog_url:
        raise ValidationError(f"rolling release is missing {CATALOG_FILENAME}")
    with tempfile.TemporaryDirectory(prefix="nvcr-catalog-") as temporary:
        catalog_path = Path(temporary) / CATALOG_FILENAME
        download_file(catalog_url, catalog_path, token)
        catalog = load_json(catalog_path)
    entries = validate_catalog(catalog)
    try:
        identity = detect_device_identity(args.device_id)
    except DeviceDetectionError as error:
        raise ValidationError(str(error)) from error
    print(f"Detected {identity_summary(identity)}")
    installed = install_catalog_assets(
        entries,
        asset_urls,
        identity,
        backend=args.backend,
        requested_profiles=flatten_profile_arguments(args.profile),
        engine_root=args.engine_root,
        token=token,
    )
    print("Installed engine profiles: " + ", ".join(installed))
    return 0


def cuda_runtime_from_profile(value: str) -> int | None:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)) * 1000 + int(match.group(2)) * 10


def target_profile_matches(profile: dict[str, Any], identity: dict[str, Any]) -> bool:
    host = profile.get("host")
    gpu = profile.get("gpu")
    if not isinstance(host, dict) or not isinstance(gpu, dict):
        return False
    compute = str(gpu.get("compute_capability", "")).split(".")
    tensorrt = str(profile.get("tensorrt", "")).split(".")
    if len(compute) != 2 or len(tensorrt) < 3:
        return False
    try:
        expected = {
            "architecture": host["architecture"],
            "compute_capability_major": int(compute[0]),
            "compute_capability_minor": int(compute[1]),
            "multiprocessor_count": int(gpu["multiprocessor_count"]),
            "cuda_runtime_version": cuda_runtime_from_profile(str(profile["cuda"])),
            "tensorrt_version_major": int(tensorrt[0]),
            "tensorrt_version_minor": int(tensorrt[1]),
            "tensorrt_version_patch": int(tensorrt[2]),
        }
    except (KeyError, TypeError, ValueError):
        return False
    return all(identity.get(key) == value for key, value in expected.items())


def detect_target_profile(device_id: int) -> Path:
    try:
        identity = detect_device_identity(device_id)
    except DeviceDetectionError as error:
        raise ValidationError(str(error)) from error
    matches: list[Path] = []
    for path in sorted(TARGET_PROFILE_DIRECTORY.glob("*.json")):
        profile = load_profile(path, "nvcr.target-profile.v1")
        if target_profile_matches(profile, identity):
            matches.append(path)
    if len(matches) != 1:
        raise ValidationError(
            f"could not select exactly one registered target profile for {identity_summary(identity)}; "
            f"matches={[str(path) for path in matches]}; pass --target-profile"
        )
    return matches[0]


def forward_artifact_command(command: str, arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"nvcr-artifacts {command}",
        description="Profile-aware front end; helper-specific options pass through unchanged.",
    )
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL_PROFILE)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--engine-profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--target-profile", type=Path)
    parser.add_argument("--engines-root", type=Path, default=Path("build/engines"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--hardware-compatibility",
        choices=("exact", "same_compute_capability", "ampere_plus"),
        default="exact",
    )
    known, passthrough = parser.parse_known_args(arguments)
    model = load_profile(known.model_profile, "nvcr.model-profile.v1")
    if known.engine_profile is not None:
        profile_paths = [known.engine_profile]
    elif known.all:
        profile_paths = [ENGINE_PROFILE_DIRECTORY / f"{name}.json" for name in ENGINE_PROFILES]
    else:
        profile = canonical_profile_name(str(known.profile), warn_legacy=True)
        profile_paths = [ENGINE_PROFILE_DIRECTORY / f"{profile}.json"]
    if known.all and "--engines" in passthrough:
        raise ValidationError("use --engines-root with --all, not the single-bundle --engines option")
    target_path = known.target_profile or detect_target_profile(known.device_id)
    target = load_profile(target_path, "nvcr.target-profile.v1")
    target_id = str(target["id"])
    helper = "prepare_artifacts.sh" if command == "prepare" else "build_tensorrt.sh"
    for index, profile_path in enumerate(profile_paths):
        engine = load_profile(profile_path, "nvcr.engine-profile.v1")
        profile = canonical_profile_name(str(engine["id"]))
        child = [
            str(HELPER_DIRECTORY / helper),
            "--model-profile-path",
            str(known.model_profile),
            "--engine-profile-path",
            str(profile_path),
            "--target-profile-path",
            str(target_path),
            "--optimization-point",
            str(engine["optimization_point"]),
            "--workspace-mib",
            str(engine["workspace_mib"]),
            "--builder-optimization-level",
            str(engine["builder_optimization_level"]),
            "--model-profile-id",
            str(model["id"]),
            "--target-profile-id",
            target_id,
            "--device-id",
            str(known.device_id),
            "--hardware-compatibility",
            known.hardware_compatibility,
        ]
        if "--engines" not in passthrough:
            child.extend(
                (
                    "--engines",
                    str(known.engines_root / f"dcvcrt-{profile}"),
                )
            )
        if command == "prepare" and index > 0:
            child.extend(("--skip-clone", "--skip-export"))
        child.extend(passthrough)
        result = subprocess.run(child, check=False).returncode
        if result != 0:
            return result
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: nvcr-artifacts {install|prepare|build|inspect|validate} [options]")
        return 0 if len(sys.argv) >= 2 else 2
    command = sys.argv[1]
    try:
        if command == "install":
            return install_command(sys.argv[2:])
        if command in ("prepare", "build"):
            return forward_artifact_command(command, sys.argv[2:])
        parser = argparse.ArgumentParser(prog=f"nvcr-artifacts {command}")
        parser.add_argument("bundle", type=Path)
        parser.add_argument("--json", action="store_true")
        args = parser.parse_args(sys.argv[2:])
        root = args.bundle.resolve()
        if command == "validate":
            if (root / "engine_manifest.json").is_file():
                manifest = validate_engine_bundle(root)
                result = {
                    "valid": True,
                    "kind": "engine",
                    "schema": manifest["schema"],
                    "model_profile_id": manifest["model_profile_id"],
                }
            else:
                manifests = validate_model_manifests(root, require_graphs=True)
                result = {
                    "valid": True,
                    "kind": "model",
                    "schema": manifests[0]["schema"],
                    "model_profile_id": manifests[0]["model_profile_id"],
                }
        elif command == "inspect":
            result = inspect_bundle(root)
        else:
            raise ValidationError(f"unknown command: {command}")
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(" ".join(f"{key}={value}" for key, value in result.items()))
        return 0
    except ValidationError as error:
        print(f"nvcr-artifacts: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
