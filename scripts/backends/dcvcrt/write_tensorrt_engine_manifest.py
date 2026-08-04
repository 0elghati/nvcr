#!/usr/bin/env python3
"""Write DCVC-RT TensorRT engine bundle metadata for runtime compatibility checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nvcr_device import (  # noqa: E402
    query_cuda_runtime_version,
    query_device_with_cuda_driver,
    query_device_with_cudart,
    query_device_with_nvidia_smi,
    query_tensorrt_version,
)


def query_device_with_trtexec(
    trtexec: Path, engine_dir: Path, device_id: int, reject_device_warning: bool = False
) -> dict[str, object]:
    plan = engine_dir / "i_analysis.plan"
    if not plan.exists():
        return {}
    command = [
        str(trtexec),
        f"--loadEngine={plan}",
        f"--device={device_id}",
        "--skipInference",
        "--duration=0",
        "--verbose",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return {}
    text = completed.stdout
    if reject_device_warning and "across different models of devices" in text:
        raise SystemExit(
            "TensorRT reports this plan was built for a different device model; "
            "rebuild the engine directory instead of stamping it"
        )
    result: dict[str, object] = {}
    patterns = {
        "device_name": r"Selected Device:\s*(.+)",
        "multiprocessor_count": r"SMs:\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        result[key] = match.group(1).strip() if key == "device_name" else int(match.group(1))
    match = re.search(r"Compute Capability:\s*(\d+)\.(\d+)", text)
    if match:
        result["compute_capability_major"] = int(match.group(1))
        result["compute_capability_minor"] = int(match.group(2))
    return result


def require_int(metadata: dict[str, object], key: str) -> None:
    if not isinstance(metadata.get(key), int):
        raise SystemExit(f"could not determine required TensorRT engine metadata: {key}")


def require_string(metadata: dict[str, object], key: str) -> None:
    if not isinstance(metadata.get(key), str) or not metadata[key]:
        raise SystemExit(f"could not determine required TensorRT engine metadata: {key}")


def load_profile(path: Path, schema: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read profile {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise SystemExit(f"{path} must use schema {schema}")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise SystemExit(f"{path} requires a non-empty profile id")
    return value


def validated_visible_dimensions(profile: dict[str, object], path: Path) -> tuple[dict[str, list[int]], str]:
    dimensions = profile.get("visible_dimensions")
    if not isinstance(dimensions, dict):
        raise SystemExit(f"{path} requires visible_dimensions")
    normalized: dict[str, list[int]] = {}
    for label in ("minimum", "optimum", "maximum"):
        value = dimensions.get(label)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
        ):
            raise SystemExit(
                f"{path} visible_dimensions.{label} must contain two positive integers"
            )
        normalized[label] = value
    for axis in range(2):
        if not (
            normalized["minimum"][axis]
            <= normalized["optimum"][axis]
            <= normalized["maximum"][axis]
        ):
            raise SystemExit(f"{path} visible dimensions are not ordered")
    shape_profile = (
        "fixed"
        if normalized["minimum"] == normalized["optimum"] == normalized["maximum"]
        else "dynamic"
    )
    return normalized, shape_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", required=True, type=Path)
    parser.add_argument("--trtexec", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--optimization-point",
        required=True,
        choices=("qcif", "cif", "360p", "540p", "720p", "1080p"),
    )
    parser.add_argument("--workspace-mib", required=True, type=int)
    parser.add_argument("--builder-optimization-level", required=True, type=int)
    parser.add_argument("--model-profile-id", default="dcvcrt-cvpr2025")
    parser.add_argument("--target-profile-id", default="local-auto")
    parser.add_argument("--model-profile-path", required=True, type=Path)
    parser.add_argument("--engine-profile-path", required=True, type=Path)
    parser.add_argument("--target-profile-path", required=True, type=Path)
    parser.add_argument("--enable-int8", action="store_true",
                        help="record experimental int8_fp16 precision")
    parser.add_argument(
        "--reject-device-warning",
        action="store_true",
        help="fail if TensorRT reports a cross-device warning while inspecting an existing plan",
    )
    args = parser.parse_args()

    model_profile = load_profile(args.model_profile_path, "nvcr.model-profile.v1")
    engine_profile = load_profile(args.engine_profile_path, "nvcr.engine-profile.v1")
    target_profile = load_profile(args.target_profile_path, "nvcr.target-profile.v1")
    if model_profile["id"] != args.model_profile_id:
        raise SystemExit("model profile id does not match --model-profile-id")
    expected_engine_id = str(engine_profile.get("id", ""))
    if (
        expected_engine_id not in (args.optimization_point, f"{args.optimization_point}-fp16")
        or engine_profile.get("optimization_point") != args.optimization_point
        or engine_profile.get("precision") != "fp16"
    ):
        raise SystemExit("engine profile identity, optimization point, or precision does not match")
    if (
        engine_profile.get("workspace_mib") != args.workspace_mib
        or engine_profile.get("builder_optimization_level")
        != args.builder_optimization_level
    ):
        raise SystemExit("TensorRT builder settings do not match the selected engine profile")
    target_profile_id = target_profile["id"]
    if args.target_profile_id not in ("local-auto", target_profile_id):
        raise SystemExit("target profile id does not match --target-profile-id")
    visible_dimensions, shape_profile = validated_visible_dimensions(
        engine_profile, args.engine_profile_path
    )

    required_files = (
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
        "i_entropy.bin",
        "i_quant.bin",
        "i_frame_manifest.json",
        "p_entropy.bin",
        "p_quant.bin",
        "p_frame_manifest.json",
    )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    profile_paths = {
        "model_profile_sha256": args.model_profile_path,
        "engine_profile_sha256": args.engine_profile_path,
        "target_profile_sha256": args.target_profile_path,
    }
    missing_profiles = [str(path) for path in profile_paths.values() if not path.is_file()]
    if missing_profiles:
        raise SystemExit("missing profile file(s): " + ", ".join(missing_profiles))
    profile_hashes = {key: sha256(path) for key, path in profile_paths.items()}

    missing = [name for name in required_files if not (args.engines / name).is_file()]
    if missing:
        raise SystemExit("engine bundle is incomplete: " + ", ".join(missing))
    file_hashes = {name: sha256(args.engines / name) for name in required_files}
    checksum_path = args.engines / "engine.sha256"
    checksum_path.write_text(
        "".join(f"{file_hashes[name]}  {name}\n" for name in sorted(file_hashes)),
        encoding="utf-8",
    )

    cuda_runtime_version = query_cuda_runtime_version()
    tensorrt_version = query_tensorrt_version()
    major, minor, patch = tensorrt_version or (None, None, None)
    metadata: dict[str, object] = {
        "format": 2,
        "schema": "nvcr.engine-bundle.v2",
        "kind": "nvcr-tensorrt-engine-bundle",
        "model_profile_id": args.model_profile_id,
        **profile_hashes,
        "target_profile_id": target_profile_id,
        "engine_profile_id": expected_engine_id,
        "precision": "int8_fp16" if args.enable_int8 else "fp16",
        "optimization_point": args.optimization_point,
        "visible_dimensions": visible_dimensions,
        "shape_profile": shape_profile,
        "workspace_mib": args.workspace_mib,
        "builder_optimization_level": args.builder_optimization_level,
        "cuda_runtime_version": cuda_runtime_version,
        "tensorrt_version_major": major,
        "tensorrt_version_minor": minor,
        "tensorrt_version_patch": patch,
        "i_model_manifest_sha256": file_hashes["i_frame_manifest.json"],
        "p_model_manifest_sha256": file_hashes["p_frame_manifest.json"],
        "checksum_manifest": checksum_path.name,
        "checksum_manifest_sha256": sha256(checksum_path),
        "files": file_hashes,
    }
    metadata.update(query_device_with_nvidia_smi(args.device_id))
    if args.reject_device_warning:
        metadata.update(
            query_device_with_trtexec(
                args.trtexec, args.engines, args.device_id, reject_device_warning=True))
    device_keys = (
        "device_name",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
    )
    if not all(key in metadata for key in device_keys):
        metadata.update(query_device_with_cudart(args.device_id))
    metadata.update(query_device_with_cuda_driver(args.device_id))
    if not all(key in metadata for key in device_keys):
        metadata.update(query_device_with_trtexec(args.trtexec, args.engines, args.device_id))

    for key in (
        "format",
        "cuda_runtime_version",
        "tensorrt_version_major",
        "tensorrt_version_minor",
        "tensorrt_version_patch",
        "compute_capability_major",
        "compute_capability_minor",
        "multiprocessor_count",
    ):
        require_int(metadata, key)
    for key in (
        "device_name",
        "model_profile_id",
        "engine_profile_id",
        "checksum_manifest_sha256",
    ):
        require_string(metadata, key)

    path = args.engines / "engine_manifest.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
