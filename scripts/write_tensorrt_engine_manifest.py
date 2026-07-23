#!/usr/bin/env python3
"""Write NVCR TensorRT engine bundle metadata for runtime compatibility checks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import ctypes.util
import json
import re
import shutil
import subprocess
from pathlib import Path


def load_shared_library(name: str, extra_paths: tuple[str, ...] = ()) -> ctypes.CDLL | None:
    candidates = [ctypes.util.find_library(name), *extra_paths]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def infer_tensorrt_version() -> tuple[int | None, int | None, int | None]:
    nvinfer = load_shared_library(
        "nvinfer",
        (
            "/usr/lib/aarch64-linux-gnu/libnvinfer.so",
            "/usr/lib/x86_64-linux-gnu/libnvinfer.so",
        ),
    )
    if nvinfer is None:
        return None, None, None
    try:
        get_version = nvinfer.getInferLibVersion
        get_version.restype = ctypes.c_int32
        version = int(get_version())
    except (AttributeError, OSError):
        return None, None, None
    return version // 10000, (version // 100) % 100, version % 100


def query_device_with_nvidia_smi(device_id: int) -> dict[str, object]:
    if shutil.which("nvidia-smi") is None:
        return {}
    fields = [
        "name",
        "compute_cap",
        "multiprocessor_count",
    ]
    command = [
        "nvidia-smi",
        f"--id={device_id}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    values = [item.strip() for item in output.strip().split(",")]
    if len(values) != len(fields):
        return {}
    result: dict[str, object] = {"device_name": values[0]}
    match = re.fullmatch(r"(\d+)\.(\d+)", values[1])
    if match:
        result["compute_capability_major"] = int(match.group(1))
        result["compute_capability_minor"] = int(match.group(2))
    if values[2].isdigit():
        result["multiprocessor_count"] = int(values[2])
    return result


def query_device_with_cudart(device_id: int) -> dict[str, object]:
    cudart = load_shared_library(
        "cudart",
        (
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so",
            "/usr/local/cuda-12.6/lib64/libcudart.so",
        ),
    )
    if cudart is None:
        return {}
    try:
        get_name = cudart.cudaDeviceGetName
        get_name.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        get_name.restype = ctypes.c_int
        get_attribute = cudart.cudaDeviceGetAttribute
        get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_attribute.restype = ctypes.c_int
    except (AttributeError, OSError):
        return {}

    result: dict[str, object] = {}
    name = ctypes.create_string_buffer(256)
    if get_name(name, len(name), device_id) == 0:
        result["device_name"] = name.value.decode("utf-8", errors="replace")

    attributes = {
        "multiprocessor_count": 16,
        "compute_capability_major": 75,
        "compute_capability_minor": 76,
    }
    for key, attribute in attributes.items():
        value = ctypes.c_int()
        if get_attribute(ctypes.byref(value), attribute, device_id) == 0:
            result[key] = int(value.value)
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", required=True, type=Path)
    parser.add_argument("--trtexec", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--optimization-point", required=True, choices=("qcif", "1080p"))
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

    cuda_runtime_version: int | None = None
    cudart = load_shared_library(
        "cudart",
        (
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so",
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so",
            "/usr/local/cuda-12.6/lib64/libcudart.so",
        ),
    )
    if cudart is not None:
        try:
            get_version = cudart.cudaRuntimeGetVersion
            get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
            get_version.restype = ctypes.c_int
            value = ctypes.c_int()
            if get_version(ctypes.byref(value)) == 0:
                cuda_runtime_version = int(value.value)
        except (AttributeError, OSError):
            pass

    visible_profiles = {
        "qcif": {"minimum": [64, 64], "optimum": [176, 144], "maximum": [176, 144]},
        "1080p": {"minimum": [64, 64], "optimum": [1920, 1080], "maximum": [1920, 1080]},
    }
    major, minor, patch = infer_tensorrt_version()
    metadata: dict[str, object] = {
        "format": 2,
        "schema": "nvcr.engine-bundle.v2",
        "kind": "nvcr-tensorrt-engine-bundle",
        "model_profile_id": args.model_profile_id,
        **profile_hashes,
        "target_profile_id": args.target_profile_id,
        "engine_profile_id": f"{args.optimization_point}-fp16",
        "precision": "int8_fp16" if args.enable_int8 else "fp16",
        "optimization_point": args.optimization_point,
        "visible_dimensions": visible_profiles[args.optimization_point],
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
    if "device_name" not in metadata or "compute_capability_major" not in metadata:
        metadata.update(query_device_with_cudart(args.device_id))
    if "device_name" not in metadata or "compute_capability_major" not in metadata:
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
