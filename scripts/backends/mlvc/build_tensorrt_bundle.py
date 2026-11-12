#!/usr/bin/env python3
"""Build a TensorRT engine bundle from a published MLVC ONNX bundle (C12).

Copies the ONNX bundle unchanged (MLVC still reads its metadata, PMF tables and
verifies its part digests), builds one FP16 plan per canvas and part named
mlvc_{encoder,decoder}_<canvas>.plan, and writes the nvcr.engine-bundle.v2
manifest that NVCR's TensorRT session and nvcr_artifacts.py validate.
engine.sha256 covers every file, so a plan cannot be paired with a different
ONNX bundle. Package the result with scripts/package_engine_bundle.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

MODEL_PROFILE_ID = "mlvc"
# One engine profile per MLVC model and weights; it covers all of the bundle's
# canvases. Keep in step with BACKEND_PROFILES["mlvc"] in nvcr_artifacts.py.
ENGINE_PROFILES = {
    "dmc61sbr_mini_reglu-perceptual_v1": "s-perceptual",
    "dmc61sbr_reglu-perceptual_v1": "perceptual",
    "dmc61sbr_reglu-psnr_v1": "psnr",
}
PARTS = {"MLVCEncoder": "mlvc_encoder", "MLVCDecoder": "mlvc_decoder"}
TRTEXEC_COMPATIBILITY = {"exact": [], "same_compute_capability":
                         ["--hardwareCompatibilityLevel=sameComputeCapability"]}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Extracted MLVC ONNX bundle")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-profile", type=Path, required=True,
                        help="Registered configs/targets/<id>.json describing this build host")
    parser.add_argument("--hardware-compatibility", choices=sorted(TRTEXEC_COMPATIBILITY),
                        default="exact")
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument("--cuda-runtime-version", type=int, default=12080,
                        help="CUDA runtime the NVCR process uses (major*1000+minor*10)")
    return parser.parse_args(argv)


def build_plan(args, onnx_path, plan_path, timing_cache, log_path):
    command = [
        str(args.trtexec), f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
        # The parts declare FP16 tensors (and an INT32 Q index); --fp16 lets
        # TensorRT keep FP16 kernels as ONNX Runtime does for the same graph.
        "--fp16", f"--device={args.device}", f"--memPoolSize=workspace:{args.workspace_mib}",
        f"--builderOptimizationLevel={args.builder_optimization_level}",
        *TRTEXEC_COMPATIBILITY[args.hardware_compatibility],
        f"--timingCacheFile={timing_cache}", "--skipInference",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed for {onnx_path.name}; see {log_path}")
    return command, completed.stdout


def device_identity(device, trtexec_log):
    query = subprocess.check_output(
        ["nvidia-smi", f"--id={device}", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader"], text=True).strip()
    name, compute_capability, driver = [part.strip() for part in query.split(",")]
    major, minor = (int(part) for part in compute_capability.split("."))
    sms = re.search(r"\bSMs:\s*(\d+)", trtexec_log)
    tensorrt = re.search(r"TensorRT v(\d+)", trtexec_log)
    if not sms or not tensorrt:
        raise RuntimeError("trtexec did not report the multiprocessor count or its version")
    version = int(tensorrt.group(1))
    return {"name": name, "major": major, "minor": minor, "driver": driver,
            "multiprocessors": int(sms.group(1)),
            "tensorrt": [version // 10000, version // 100 % 100, version % 100]}


def check_target(target, device):
    # The target JSON is an assertion about the build host; refuse a mismatch.
    gpu = target["gpu"]
    observed = (device["name"], f"{device['major']}.{device['minor']}",
                device["multiprocessors"], ".".join(map(str, device["tensorrt"])))
    expected = (gpu["name"], gpu["compute_capability"], gpu["multiprocessor_count"],
                target["tensorrt"])
    if observed != expected:
        raise SystemExit(f"build host {observed} does not match target profile {expected}")


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory must be new: {args.output_dir}")
    manifest = json.loads((args.bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    profile = ENGINE_PROFILES.get(manifest["bundle_name"])
    # Other targets carry NPU-specific graph rewrites; TensorRT plans start from generic.
    if profile is None or manifest["target_device"] != "generic":
        raise SystemExit(f"unsupported MLVC bundle {manifest['bundle_name']} "
                         f"({manifest['target_device']}); expected a generic export of "
                         f"{', '.join(ENGINE_PROFILES)}")
    target = json.loads(args.target_profile.read_text(encoding="utf-8"))
    canvases = sorted(manifest["model_manifests"])

    bundle = args.output_dir / "bundle"
    logs = args.output_dir / "logs"
    bundle.mkdir(parents=True)
    logs.mkdir()
    sources = {}
    for path in sorted(args.bundle.iterdir()):
        if path.is_file() and path.suffix in (".json", ".onnx", ".bin"):
            shutil.copy2(path, bundle / path.name)
            sources[path.name] = sha256(path)

    report = {"schema": "nvcr.mlvc.tensorrt_bundle.v1", "bundle_name": manifest["bundle_name"],
              "target_device": manifest["target_device"], "sources": sources,
              "tool_sha256": sha256(Path(__file__)), "plans": {}}
    timing_cache = args.output_dir / "timing.cache"
    device = None
    for canvas in canvases:
        for part, stage in PARTS.items():
            plan = f"{stage}_{canvas}.plan"
            print(f"[build] {plan}", flush=True)
            command, log = build_plan(args, bundle / f"{part}_{canvas}.onnx", bundle / plan,
                                      timing_cache, logs / f"{stage}_{canvas}.trtexec.log")
            if device is None:
                device = device_identity(args.device, log)
                check_target(target, device)
            report["plans"][plan] = {"command": command, "sha256": sha256(bundle / plan)}

    files = {path.name: sha256(path) for path in sorted(bundle.iterdir())}
    (bundle / "engine.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())), encoding="utf-8")
    engine_profile = json.dumps({"id": profile, "canvases": canvases,
                                 "precision": "fp16", "workspace_mib": args.workspace_mib,
                                 "builder_optimization_level": args.builder_optimization_level},
                                sort_keys=True)
    engine_manifest = {
        "format": 2,
        "schema": "nvcr.engine-bundle.v2",
        "kind": "nvcr-tensorrt-engine-bundle",
        "model_profile_id": MODEL_PROFILE_ID,
        # The source ONNX bundle is the model profile.
        "model_profile_sha256": sources["bundle_manifest.json"],
        "target_profile_id": target["id"],
        "target_profile_sha256": sha256(args.target_profile),
        "engine_profile_id": profile,
        "engine_profile_sha256": sha256_text(engine_profile),
        "optimization_point": profile,
        "precision": "fp16",
        "shape_profile": "fixed",
        "hardware_compatibility": args.hardware_compatibility,
        "device_name": device["name"],
        "compute_capability_major": device["major"],
        "compute_capability_minor": device["minor"],
        "multiprocessor_count": device["multiprocessors"],
        "cuda_runtime_version": args.cuda_runtime_version,
        "tensorrt_version_major": device["tensorrt"][0],
        "tensorrt_version_minor": device["tensorrt"][1],
        "tensorrt_version_patch": device["tensorrt"][2],
        "builder_optimization_level": args.builder_optimization_level,
        "workspace_mib": args.workspace_mib,
        "checksum_manifest": "engine.sha256",
        "checksum_manifest_sha256": sha256(bundle / "engine.sha256"),
        "files": files,
    }
    (bundle / "engine_manifest.json").write_text(
        json.dumps(engine_manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    report["device"] = device
    report["engine_manifest_sha256"] = sha256(bundle / "engine_manifest.json")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"[build] wrote {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
