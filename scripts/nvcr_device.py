#!/usr/bin/env python3
"""Detect the host and NVIDIA runtime identity used by NVCR artifacts."""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class DeviceDetectionError(RuntimeError):
    """The local host cannot be matched safely to an engine bundle."""


def normalize_architecture(machine: str | None = None) -> str:
    value = (machine or platform.machine()).strip().lower()
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    if value in {"aarch64", "arm64"}:
        return "aarch64"
    raise DeviceDetectionError(f"unsupported host architecture: {value or 'unknown'}")


def load_shared_library(name: str, candidates: tuple[str, ...]) -> ctypes.CDLL | None:
    paths: list[str] = []
    discovered = ctypes.util.find_library(name)
    if discovered:
        paths.append(discovered)
    paths.extend(candidates)
    for path in paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def query_device_with_nvidia_smi(device_id: int) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {}
    # Jetson's nvidia-smi omits multiprocessor_count. CUDA fills that field.
    fields = ("name", "compute_cap")
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
    match = re.fullmatch(r"(\d+)\.(\d+)", values[1])
    if not match or not values[0]:
        return {}
    return {
        "device_name": values[0],
        "compute_capability_major": int(match.group(1)),
        "compute_capability_minor": int(match.group(2)),
    }


def query_device_with_cudart(device_id: int) -> dict[str, Any]:
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
        get_attribute = cudart.cudaDeviceGetAttribute
        get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_attribute.restype = ctypes.c_int
    except AttributeError:
        return {}

    result: dict[str, Any] = {}
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


def query_device_with_cuda_driver(device_id: int) -> dict[str, Any]:
    driver = load_shared_library(
        "cuda",
        (
            "/usr/lib/aarch64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
        ),
    )
    if driver is None:
        return {}
    try:
        initialize = driver.cuInit
        initialize.argtypes = [ctypes.c_uint]
        initialize.restype = ctypes.c_int
        get_device = driver.cuDeviceGet
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        get_device.restype = ctypes.c_int
        get_name = driver.cuDeviceGetName
        get_name.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        get_name.restype = ctypes.c_int
        get_attribute = driver.cuDeviceGetAttribute
        get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        get_attribute.restype = ctypes.c_int
    except AttributeError:
        return {}
    cuda_device = ctypes.c_int()
    if initialize(0) != 0 or get_device(ctypes.byref(cuda_device), device_id) != 0:
        return {}
    result: dict[str, Any] = {}
    name = ctypes.create_string_buffer(256)
    if get_name(name, len(name), cuda_device.value) == 0:
        result["device_name"] = name.value.decode("utf-8", errors="replace")
    for key, attribute in {
        "multiprocessor_count": 16,
        "compute_capability_major": 75,
        "compute_capability_minor": 76,
    }.items():
        value = ctypes.c_int()
        if get_attribute(ctypes.byref(value), attribute, cuda_device.value) == 0:
            result[key] = int(value.value)
    return result


def query_cuda_runtime_version() -> int | None:
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
        return None
    try:
        get_version = cudart.cudaRuntimeGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        value = ctypes.c_int()
        if get_version(ctypes.byref(value)) == 0:
            return int(value.value)
    except (AttributeError, OSError):
        pass
    return None


def query_tensorrt_version() -> tuple[int, int, int] | None:
    nvinfer = load_shared_library(
        "nvinfer",
        (
            "/usr/lib/aarch64-linux-gnu/libnvinfer.so",
            "/usr/lib/x86_64-linux-gnu/libnvinfer.so",
            "/usr/local/TensorRT/lib/libnvinfer.so",
        ),
    )
    if nvinfer is None:
        return None
    try:
        get_version = nvinfer.getInferLibVersion
        get_version.argtypes = []
        get_version.restype = ctypes.c_int
        encoded = int(get_version())
    except (AttributeError, OSError):
        return None
    if encoded <= 0:
        return None
    return encoded // 10000, (encoded // 100) % 100, encoded % 100


def detect_device_identity(device_id: int = 0) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise DeviceDetectionError(f"unsupported host operating system: {platform.system()}")
    if device_id < 0:
        raise DeviceDetectionError("device id must be non-negative")
    device = query_device_with_nvidia_smi(device_id)
    if not all(
        key in device
        for key in (
            "device_name",
            "compute_capability_major",
            "compute_capability_minor",
            "multiprocessor_count",
        )
    ):
        device.update(query_device_with_cudart(device_id))
    # Prefer the CUDA driver name because the runtime compares cudaDeviceProp::name.
    # Jetson nvidia-smi reports "Orin (nvgpu)" while CUDA reports "Orin".
    device.update(query_device_with_cuda_driver(device_id))
    missing = [
        key
        for key in (
            "device_name",
            "compute_capability_major",
            "compute_capability_minor",
            "multiprocessor_count",
        )
        if key not in device
    ]
    cuda_runtime = query_cuda_runtime_version()
    tensorrt = query_tensorrt_version()
    if missing or cuda_runtime is None or tensorrt is None:
        details = []
        if missing:
            details.append("GPU " + ", ".join(missing))
        if cuda_runtime is None:
            details.append("CUDA runtime version")
        if tensorrt is None:
            details.append("TensorRT version")
        raise DeviceDetectionError("could not detect " + "; ".join(details))
    return {
        "operating_system": "linux",
        "architecture": normalize_architecture(),
        "device_id": device_id,
        **device,
        "cuda_runtime_version": cuda_runtime,
        "tensorrt_version_major": tensorrt[0],
        "tensorrt_version_minor": tensorrt[1],
        "tensorrt_version_patch": tensorrt[2],
    }


def identity_summary(identity: dict[str, Any]) -> str:
    return (
        f"{identity['architecture']} {identity['device_name']} "
        f"SM {identity['compute_capability_major']}.{identity['compute_capability_minor']} "
        f"({identity['multiprocessor_count']} multiprocessors), "
        f"CUDA runtime {identity['cuda_runtime_version']}, TensorRT "
        f"{identity['tensorrt_version_major']}.{identity['tensorrt_version_minor']}."
        f"{identity['tensorrt_version_patch']}"
    )


if __name__ == "__main__":
    import json

    print(json.dumps(detect_device_identity(), indent=2, sort_keys=True))
