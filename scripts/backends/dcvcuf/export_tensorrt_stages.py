#!/usr/bin/env python3
"""Export DCVC-UF stages to ONNX and build a TensorRT FP16 engine bundle (C8 phase 2).

Loads pinned upstream image and HT-S models from hash-verified checkpoints in
FP32, exports each stage in tensorrt_stages.py with dynamic spatial axes, checks
every ONNX graph against the PyTorch stage on ONNX Runtime CPU at two sizes,
then builds TensorRT plans with DCVC-RT's build recipe and writes an
nvcr.engine-bundle.v2 manifest that NVCR's TensorRT session validates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorrt_stages  # noqa: E402


PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
MODEL_PROFILE_ID = "dcvcuf-cvpr2026-hts"
ONNX_OPSET = 17
# Export check against PyTorch FP32 on ONNX Runtime CPU: the two FP32 runtimes
# accumulate in different orders, so error is bounded relative to the output's
# peak magnitude rather than per element.
ONNX_ATOL = 1e-4
ONNX_PEAK_RTOL = 1e-5
CHECK_FRAMES = ((256, 256), (128, 192))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_reference(root, expected=PINNED_REFERENCE_COMMIT):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if commit != expected:
        raise ValueError(f"expected upstream {expected}, found {commit}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("reference checkout must be clean, including untracked source files")
    return commit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--image-checkpoint-sha256", required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    parser.add_argument("--video-checkpoint-sha256", required=True)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--builder-optimization-level", type=int, default=2)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                        help="fp32 builds without --fp16 (FP16 I/O, FP32 layers) for diagnosis")
    parser.add_argument("--cuda-runtime-version", type=int, default=12080,
                        help="CUDA runtime the NVCR process uses (major*1000+minor*10)")
    parser.add_argument("--target-profile-id", default="rtx4070-ubuntu2404")
    parser.add_argument("--stage", action="append", help="Restrict to these stages")
    parser.add_argument("--skip-build", action="store_true", help="Export and check ONNX only")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def verify_inputs(args):
    for path, expected, label in (
            (args.image_checkpoint, args.image_checkpoint_sha256, "image checkpoint"),
            (args.video_checkpoint, args.video_checkpoint_sha256, "video checkpoint")):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")


def load_models(args):
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC
    from src.utils.common import ModelStructure, get_state_dict

    image = DMCI().eval()
    image.load_state_dict(get_state_dict(args.image_checkpoint))
    video = DMC(model_structure=ModelStructure.HTS).eval()
    video.load_state_dict(get_state_dict(args.video_checkpoint))
    return image.float(), video.float()


def example_inputs(stage, frame_hw, generator):
    import torch

    inputs, _ = tensorrt_stages.STAGES[stage]
    tensors = []
    for _name, channels, axis in inputs:
        shape = tensorrt_stages.tensor_shape(channels, axis, frame_hw)
        if axis == "scalar":  # per-channel quantization scales are positive
            tensors.append(torch.rand(shape, generator=generator) + 0.5)
        else:
            tensors.append(torch.randn(shape, generator=generator) * 0.5)
    return tensors


def export_stage(stage, module, path):
    import torch

    inputs, outputs = tensorrt_stages.STAGES[stage]
    generator = torch.Generator().manual_seed(0)
    example = example_inputs(stage, CHECK_FRAMES[0], generator)
    dynamic = {}
    for name, _channels, axis in inputs + outputs:
        if axis != "scalar":
            dynamic[name] = {2: f"{name}_height", 3: f"{name}_width"}
    with torch.no_grad():
        torch.onnx.export(
            module, tuple(example), str(path),
            input_names=[name for name, _, _ in inputs],
            output_names=[name for name, _, _ in outputs],
            dynamic_axes=dynamic, opset_version=ONNX_OPSET,
            do_constant_folding=True, dynamo=False)


def check_onnx(stage, module, path):
    import numpy as np
    import onnxruntime
    import torch

    inputs, outputs = tensorrt_stages.STAGES[stage]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    results = []
    for index, frame_hw in enumerate(CHECK_FRAMES):
        generator = torch.Generator().manual_seed(1 + index)
        example = example_inputs(stage, frame_hw, generator)
        with torch.no_grad():
            expected = module(*example)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(None, {name: tensor.numpy()
                                    for (name, _, _), tensor in zip(inputs, example)})
        for (name, _, _), reference, value in zip(outputs, expected, actual):
            reference = reference.numpy()
            same_shape = value.shape == reference.shape
            delta = np.abs(reference.astype(np.float64) - value.astype(np.float64)) \
                if same_shape else np.array([np.inf])
            peak = float(np.abs(reference).max())
            results.append({
                "frame": list(frame_hw), "output": name, "shape": list(value.shape),
                "max_abs_error": float(delta.max()), "reference_peak": peak,
                "error_to_peak": float(delta.max()) / max(peak, 1e-12),
                "passed": bool(same_shape and np.isfinite(value).all() and
                               delta.max() <= ONNX_ATOL + ONNX_PEAK_RTOL * peak),
            })
    return results


def shape_arguments(stage):
    inputs, _ = tensorrt_stages.STAGES[stage]
    shapes = {"min": [], "opt": [], "max": []}
    for name, channels, axis in inputs:
        for key, shape in tensorrt_stages.profile_shapes(channels, axis).items():
            shapes[key].append(f"{name}:{'x'.join(map(str, shape))}")
    return {key: ",".join(values) for key, values in shapes.items()}


def io_formats(tensors):
    return ",".join(f"fp16:{tensorrt_stages.tensor_format(channels)}"
                    for _name, channels, _axis in tensors)


def build_engine(args, stage, onnx_path, bundle, timing_cache):
    shapes = shape_arguments(stage)
    command = [
        str(args.trtexec), f"--onnx={onnx_path}", f"--saveEngine={bundle / (stage + '.plan')}",
        f"--minShapes={shapes['min']}", f"--optShapes={shapes['opt']}",
        f"--maxShapes={shapes['max']}",
        # NVCR's TensorRT session binds FP16 tensors. DCVC-RT gets FP16 I/O by
        # exporting FP16 ONNX; these graphs stay FP32 (as checked above) and
        # request FP16 I/O explicitly, in each tensor's glue layout.
        f"--inputIOFormats={io_formats(tensorrt_stages.STAGES[stage][0])}",
        f"--outputIOFormats={io_formats(tensorrt_stages.STAGES[stage][1])}",
        # DCVC-RT's TensorRT recipe (scripts/backends/dcvcrt/build_tensorrt.sh).
        *(["--fp16"] if args.precision == "fp16" else []), f"--device={args.device}", f"--memPoolSize=workspace:{args.workspace_mib}",
        f"--builderOptimizationLevel={args.builder_optimization_level}",
        f"--timingCacheFile={timing_cache}", "--skipInference",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    (bundle.parent / "logs" / f"{stage}.trtexec.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed for {stage}; see logs/{stage}.trtexec.log")
    return command


def device_identity(device):
    query = subprocess.check_output(
        ["nvidia-smi", f"--id={device}", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader"], text=True).strip()
    name, compute_capability, driver = [part.strip() for part in query.split(",")]
    major, minor = (int(part) for part in compute_capability.split("."))
    return {"name": name, "major": major, "minor": minor, "driver": driver}


def multiprocessor_count(device):
    import torch

    return torch.cuda.get_device_properties(device).multi_processor_count


def write_bundle_manifest(args, bundle, stages, model_profile, device, sm_count):
    files = {path.name: sha256(path) for path in sorted(bundle.glob("*.plan"))}
    files["stages.json"] = sha256(bundle / "stages.json")
    checksums = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    (bundle / "engine.sha256").write_text(checksums, encoding="utf-8")
    engine_profile = json.dumps(
        {"profile": tensorrt_stages.PROFILE,
         "stages": {stage: shape_arguments(stage) for stage in stages}}, sort_keys=True)
    target_profile = json.dumps(
        {"id": args.target_profile_id, "device": device, "multiprocessors": sm_count,
         "cuda_runtime_version": args.cuda_runtime_version, "tensorrt": "10.9.0"},
        sort_keys=True)
    manifest = {
        "format": 2,
        "schema": "nvcr.engine-bundle.v2",
        "kind": "nvcr-tensorrt-engine-bundle",
        "model_profile_id": MODEL_PROFILE_ID,
        "model_profile_sha256": sha256_text(model_profile),
        "target_profile_id": args.target_profile_id,
        "target_profile_sha256": sha256_text(target_profile),
        "engine_profile_id": tensorrt_stages.PROFILE["name"],
        "engine_profile_sha256": sha256_text(engine_profile),
        "optimization_point": tensorrt_stages.PROFILE["name"],
        "precision": "fp16",  # tensor I/O; the build precision is in the report
        "shape_profile": "dynamic",
        "hardware_compatibility": "exact",
        "device_name": device["name"],
        "compute_capability_major": device["major"],
        "compute_capability_minor": device["minor"],
        "multiprocessor_count": sm_count,
        "cuda_runtime_version": args.cuda_runtime_version,
        "tensorrt_version_major": 10,
        "tensorrt_version_minor": 9,
        "tensorrt_version_patch": 0,
        "builder_optimization_level": args.builder_optimization_level,
        "workspace_mib": args.workspace_mib,
        "checksum_manifest": "engine.sha256",
        "checksum_manifest_sha256": sha256(bundle / "engine.sha256"),
        "files": files,
    }
    (bundle / "engine_manifest.json").write_text(
        json.dumps(manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8")


def stage_contracts(stages):
    contracts = {}
    for stage in stages:
        inputs, outputs = tensorrt_stages.STAGES[stage]
        contracts[stage] = [
            {"name": name, "access": access, "channels": channels, "axis": axis,
             "format": tensorrt_stages.tensor_format(channels),
             "profile": tensorrt_stages.profile_shapes(channels, axis)}
            for access, tensors in (("read", inputs), ("write", outputs))
            for name, channels, axis in tensors]
    return contracts


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory must be new: {args.output_dir}")
    stages = args.stage or list(tensorrt_stages.STAGES)
    unknown = sorted(set(stages) - set(tensorrt_stages.STAGES))
    if unknown:
        raise SystemExit(f"unknown stages: {', '.join(unknown)}")
    commit = verify_reference(args.reference_root)
    verify_inputs(args)
    sys.path.insert(0, str(args.reference_root.resolve()))
    import onnx
    import onnxruntime
    import torch

    torch.manual_seed(0)
    onnx_dir = args.output_dir / "onnx"
    bundle = args.output_dir / "bundle"
    for directory in (onnx_dir, bundle, args.output_dir / "logs"):
        directory.mkdir(parents=True)
    image, video = load_models(args)
    modules = tensorrt_stages.modules(image, video)
    report = {
        "schema": "nvcr.dcvcuf.tensorrt_export.v1",
        "upstream": {"commit": commit},
        "checkpoints": {"image": args.image_checkpoint_sha256,
                        "video": args.video_checkpoint_sha256},
        "tools": {"export": sha256(Path(__file__)),
                  "stages": sha256(Path(tensorrt_stages.__file__))},
        "environment": {"torch": torch.__version__, "onnx": onnx.__version__,
                        "onnxruntime": onnxruntime.__version__,
                        "python": platform.python_version()},
        "build": {"precision": args.precision,
                  "builder_optimization_level": args.builder_optimization_level,
                  "workspace_mib": args.workspace_mib},
        "onnx_opset": ONNX_OPSET, "onnx_tolerance": {"atol": ONNX_ATOL, "peak_rtol": ONNX_PEAK_RTOL},
        "profile": tensorrt_stages.PROFILE,
        "stages": {},
    }
    for stage in stages:
        path = onnx_dir / f"{stage}.onnx"
        print(f"[export] {stage}", flush=True)
        export_stage(stage, modules[stage], path)
        onnx.checker.check_model(str(path))
        checks = check_onnx(stage, modules[stage], path)
        report["stages"][stage] = {"onnx_sha256": sha256(path), "onnx_check": checks}
        if not all(check["passed"] for check in checks):
            raise RuntimeError(f"ONNX export of {stage} disagrees with PyTorch: {checks}")

    (bundle / "stages.json").write_text(
        json.dumps({"schema": "nvcr.dcvcuf.tensorrt_stages.v1", "model": MODEL_PROFILE_ID,
                    "stages": stage_contracts(stages)}, indent=2) + "\n", encoding="utf-8")
    if not args.skip_build:
        timing_cache = args.output_dir / "timing.cache"
        for stage in stages:
            print(f"[build] {stage}", flush=True)
            report["stages"][stage]["trtexec"] = build_engine(
                args, stage, onnx_dir / f"{stage}.onnx", bundle, timing_cache)
            report["stages"][stage]["plan_sha256"] = sha256(bundle / f"{stage}.plan")
        model_profile = json.dumps(
            {"upstream": commit, "checkpoints": report["checkpoints"],
             "onnx": {stage: report["stages"][stage]["onnx_sha256"] for stage in stages}},
            sort_keys=True)
        device = device_identity(args.device)
        write_bundle_manifest(args, bundle, stages, model_profile, device,
                              multiprocessor_count(args.device))
        report["bundle_manifest_sha256"] = sha256(bundle / "engine_manifest.json")
    verify_reference(args.reference_root)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[export] wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
