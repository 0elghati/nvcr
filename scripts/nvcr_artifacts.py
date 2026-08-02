#!/usr/bin/env python3
"""Prepare, build, inspect, and validate scoped NVCR DCVC-RT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SOURCE_ROOT = SCRIPT_DIRECTORY.parent
if (SOURCE_ROOT / "configs").is_dir() and (SOURCE_ROOT / "scripts").is_dir():
    REPOSITORY_ROOT = SOURCE_ROOT
    HELPER_DIRECTORY = SOURCE_ROOT / "scripts" / "backends" / "dcvcrt"
else:
    install_root = SCRIPT_DIRECTORY.parent
    REPOSITORY_ROOT = install_root / "share" / "nvcr"
    HELPER_DIRECTORY = REPOSITORY_ROOT / "scripts" / "backends" / "dcvcrt"
DEFAULT_MODEL_PROFILE = REPOSITORY_ROOT / "configs/models/dcvcrt-cvpr2025.json"
DEFAULT_ENGINE_PROFILE = REPOSITORY_ROOT / "configs/engine-profiles/1080p-fp16.json"
PINNED_COMMIT = "48ab0ac5e5199d78fffb944bfbafafb2b6142f7b"
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
        if len(fields) != 2 or len(fields[0]) != 64:
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
    if manifest["engine_profile_id"] != f"{manifest['optimization_point']}-fp16":
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
        if profile_id != f"{optimization_point}-fp16":
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


def forward_artifact_command(command: str, arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"nvcr-artifacts {command}",
        description="Profile-aware front end; unrecognized options pass to the established helper.",
    )
    parser.add_argument("--model-profile", type=Path, default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--engine-profile", type=Path, default=DEFAULT_ENGINE_PROFILE)
    parser.add_argument("--target-profile", type=Path, required=True)
    known, passthrough = parser.parse_known_args(arguments)
    model = load_profile(known.model_profile, "nvcr.model-profile.v1")
    engine = load_profile(known.engine_profile, "nvcr.engine-profile.v1")
    target_id = "local-auto"
    if known.target_profile is not None:
        target = load_profile(known.target_profile, "nvcr.target-profile.v1")
        target_id = str(target["id"])
    helper = "prepare_artifacts.sh" if command == "prepare" else "build_tensorrt.sh"
    child = [str(HELPER_DIRECTORY / helper)]
    # Keep the profile files themselves bound to the generated engine
    # manifest, not just their human-readable IDs.  This makes changing a
    # profile without changing its ID a detectable bundle invalidation.
    child.extend(("--model-profile-path", str(known.model_profile)))
    child.extend(("--engine-profile-path", str(known.engine_profile)))
    if known.target_profile is not None:
        child.extend(("--target-profile-path", str(known.target_profile)))
    child.extend(("--optimization-point", str(engine["optimization_point"])))
    child.extend(("--workspace-mib", str(engine["workspace_mib"])))
    child.extend(("--builder-optimization-level", str(engine["builder_optimization_level"])))
    if command in ("prepare", "build"):
        child.extend(("--model-profile-id", str(model["id"])))
        child.extend(("--target-profile-id", target_id))
    child.extend(passthrough)
    return subprocess.run(child, check=False).returncode


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: nvcr-artifacts {prepare|build|inspect|validate} [options]")
        return 0 if len(sys.argv) >= 2 else 2
    command = sys.argv[1]
    try:
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
