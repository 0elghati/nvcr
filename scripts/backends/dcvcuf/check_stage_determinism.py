#!/usr/bin/env python3
"""Run-to-run determinism of DCVC-UF TensorRT stages on fixed inputs (C13).

Runs each selected stage RUNS times in each of PROCESSES separate
nvcr_tensorrt_stage_runner processes, on the same fixed random FP16 inputs at
each latent size, and counts the distinct output digests. A deterministic
stage gives one digest per (stage, size). Sizes outside a bundle's profile
are skipped.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from check_tensorrt_stages import write_job

DEFAULT_STAGES = ("i_spatial_prior_1", "i_spatial_prior_2", "i_spatial_prior_3",
                  "p_spatial_prior_1")
DEFAULT_SIZES = ((9, 11), (16, 16), (68, 120))  # QCIF, 256x256 and 1080p latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", action="append")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=217)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stages = args.stage or list(DEFAULT_STAGES)
    contracts = json.loads((args.bundle / "stages.json").read_text())["stages"]
    model = json.loads((args.bundle / "engine_manifest.json").read_text())["model_profile_id"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    cases, inputs_digest = [], {}
    for stage in stages:
        tensors = contracts[stage]
        for height, width in DEFAULT_SIZES:
            reads = [t for t in tensors if t["access"] == "read"]
            if any(height > t["profile"]["max"][2] or width > t["profile"]["max"][3] or
                   height < t["profile"]["min"][2] or width < t["profile"]["min"][3]
                   for t in tensors):
                continue
            values = {t["name"]: (rng.standard_normal((1, t["channels"], height, width)) * 0.5)
                      .astype(np.float16) for t in reads}
            inputs_digest[f"{stage}@{height}x{width}"] = hashlib.sha256(
                b"".join(values[name].tobytes() for name in sorted(values))).hexdigest()
            outputs = [t["name"] for t in tensors if t["access"] == "write"]
            for run in range(args.runs):
                cases.append({"stage": stage, "label": f"{stage}.{height}x{width}.r{run}",
                              "inputs": values, "outputs": outputs, "size": (height, width)})
    digests = {}
    for process in range(args.processes):
        directory = args.output_dir / f"process{process}"
        directory.mkdir()
        job = write_job(cases, args.bundle, directory, model)
        completed = subprocess.run([str(args.runner), str(job)], capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(f"stage runner failed: {completed.stderr[-2000:]}")
        for case in cases:
            for name in case["outputs"]:
                path = directory / f"{case['label']}.{name}.out.f16"
                key = f"{case['stage']}@{case['size'][0]}x{case['size'][1]}:{name}"
                digests.setdefault(key, []).append(hashlib.sha256(path.read_bytes()).hexdigest())
                path.unlink()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    manifest = json.loads((args.bundle / "engine_manifest.json").read_text())
    report = {"gpu": gpu, "bundle": str(args.bundle),
              "plans": {stage: manifest["files"].get(f"{stage}.plan") for stage in stages},
              "runs_per_process": args.runs, "processes": args.processes,
              "inputs_sha256": inputs_digest,
              "distinct_outputs": {key: len(set(values)) for key, values in digests.items()},
              "runs": {key: len(values) for key, values in digests.items()}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for key in sorted(report["distinct_outputs"]):
        print(f"{key}: {report['distinct_outputs'][key]} distinct of {report['runs'][key]}")


if __name__ == "__main__":
    main()
