#!/usr/bin/env python3
"""Dependency-free contract tests for NVCR artifact and profile validation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import nvcr_artifacts as artifacts  # noqa: E402


def write_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return write_bytes(path, payload)


def hashed_entry(root: Path, name: str) -> dict[str, str]:
    return {"file": name, "sha256": artifacts.sha256(root / name)}


def model_manifest(
    root: Path,
    frame: str,
    checkpoint_file: str,
    checkpoint_sha256: str,
    graph: tuple[str, ...],
    assets: tuple[str, str],
) -> dict[str, object]:
    return {
        "format": 2,
        "schema": "nvcr.model-manifest.v2",
        "model_profile_id": "dcvcrt-cvpr2025",
        "exporter": "nvcr-export-v2",
        "codec": "DCVC-RT",
        "frame_type": frame,
        "reference_commit": artifacts.PINNED_COMMIT,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "precision": "fp16",
        "graphs": [hashed_entry(root, name) for name in graph],
        "assets": [hashed_entry(root, name) for name in assets],
    }


def expect_invalid(operation: object, message: str) -> None:
    try:
        operation()  # type: ignore[operator]
    except artifacts.ValidationError:
        return
    raise AssertionError(message)


def main() -> int:
    artifacts.load_profile(
        REPOSITORY_ROOT / "configs/models/dcvcrt-cvpr2025.json",
        "nvcr.model-profile.v1",
    )
    artifacts.load_profile(
        REPOSITORY_ROOT / "configs/engine-profiles/qcif-fp16.json",
        "nvcr.engine-profile.v1",
    )
    artifacts.load_profile(
        REPOSITORY_ROOT / "configs/engine-profiles/1080p-fp16.json",
        "nvcr.engine-profile.v1",
    )
    for target in ("rtx4070-ubuntu2404.json", "orin-nano-l4t3647.json"):
        artifacts.load_profile(
            REPOSITORY_ROOT / "configs/targets" / target,
            "nvcr.target-profile.v1",
        )

    with tempfile.TemporaryDirectory(prefix="nvcr-artifacts-") as temporary:
        root = Path(temporary)
        for name in ("i_entropy.bin", "i_quant.bin", "p_entropy.bin", "p_quant.bin"):
            write_bytes(root / name, f"fixture:{name}".encode())
        for name in artifacts.MODEL_GRAPHS["i_frame_manifest.json"]:
            write_bytes(root / name, f"fixture:{name}".encode())
        for name in artifacts.MODEL_GRAPHS["p_frame_manifest.json"]:
            write_bytes(root / name, f"fixture:{name}".encode())
        image_hash, image_name = artifacts.EXPECTED_CHECKPOINTS["i_frame_manifest.json"]
        video_hash, video_name = artifacts.EXPECTED_CHECKPOINTS["p_frame_manifest.json"]
        i_manifest = model_manifest(
            root,
            "I",
            image_name,
            image_hash,
            artifacts.MODEL_GRAPHS["i_frame_manifest.json"],
            ("i_entropy.bin", "i_quant.bin"),
        )
        p_manifest = model_manifest(
            root,
            "P",
            video_name,
            video_hash,
            artifacts.MODEL_GRAPHS["p_frame_manifest.json"],
            ("p_entropy.bin", "p_quant.bin"),
        )
        write_json(root / "i_frame_manifest.json", i_manifest)
        write_json(root / "p_frame_manifest.json", p_manifest)
        artifacts.validate_model_manifests(root, require_graphs=True)

        # A duplicate entry must not be able to hide a missing required asset.
        malformed = dict(i_manifest)
        malformed["assets"] = [hashed_entry(root, "i_entropy.bin")] * 2
        write_json(root / "i_frame_manifest.json", malformed)
        expect_invalid(
            lambda: artifacts.validate_model_manifests(root, require_graphs=True),
            "duplicate or missing model assets were accepted",
        )
        write_json(root / "i_frame_manifest.json", i_manifest)

        portable = dict(i_manifest)
        portable["checkpoint_file"] = "/host/checkpoints/cvpr2025_image.pth.tar"
        write_json(root / "i_frame_manifest.json", portable)
        expect_invalid(
            lambda: artifacts.validate_model_manifests(root, require_graphs=True),
            "absolute checkpoint path was accepted",
        )
        write_json(root / "i_frame_manifest.json", i_manifest)

        for plan in artifacts.REQUIRED_PLANS:
            write_bytes(root / plan, f"fixture:{plan}".encode())
        checksums = {
            name: artifacts.sha256(root / name)
            for name in artifacts.REQUIRED_PLANS + artifacts.RUNTIME_ASSETS
        }
        checksum_payload = "".join(
            f"{checksums[name]}  {name}\n" for name in sorted(checksums)
        ).encode()
        write_bytes(root / "engine.sha256", checksum_payload)
        engine_manifest = {
            "format": 2,
            "schema": "nvcr.engine-bundle.v2",
            "kind": "nvcr-tensorrt-engine-bundle",
            "model_profile_id": "dcvcrt-cvpr2025",
            "target_profile_id": "test-target",
            "engine_profile_id": "qcif-fp16",
            "model_profile_sha256": checksums["i_frame_manifest.json"],
            "engine_profile_sha256": checksums["p_frame_manifest.json"],
            "target_profile_sha256": "0" * 64,
            "precision": "fp16",
            "optimization_point": "qcif",
            "workspace_mib": 512,
            "builder_optimization_level": 1,
            "cuda_runtime_version": 12060,
            "tensorrt_version_major": 10,
            "tensorrt_version_minor": 7,
            "tensorrt_version_patch": 0,
            "device_name": "test-device",
            "compute_capability_major": 8,
            "compute_capability_minor": 9,
            "multiprocessor_count": 46,
            "i_model_manifest_sha256": checksums["i_frame_manifest.json"],
            "p_model_manifest_sha256": checksums["p_frame_manifest.json"],
            "checksum_manifest": "engine.sha256",
            "checksum_manifest_sha256": artifacts.sha256(root / "engine.sha256"),
            "files": checksums,
        }
        write_json(root / "engine_manifest.json", engine_manifest)
        validated = artifacts.validate_engine_bundle(root)
        assert validated["model_profile_id"] == "dcvcrt-cvpr2025"
        assert artifacts.inspect_bundle(root)["file_count"] == 20

        bad_profile_digest = dict(engine_manifest)
        bad_profile_digest["target_profile_sha256"] = "z" * 64
        write_json(root / "engine_manifest.json", bad_profile_digest)
        expect_invalid(
            lambda: artifacts.validate_engine_bundle(root),
            "invalid profile digest was accepted",
        )
        write_json(root / "engine_manifest.json", engine_manifest)

        write_bytes(root / artifacts.REQUIRED_PLANS[0], b"modified")
        expect_invalid(
            lambda: artifacts.validate_engine_bundle(root),
            "modified engine was accepted",
        )

    print("NVCR artifact/profile validation tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
