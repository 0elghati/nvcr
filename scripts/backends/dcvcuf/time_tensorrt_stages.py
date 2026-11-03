#!/usr/bin/env python3
"""Time each DCVC-UF TensorRT plan at its profile's optimal shape (C11).

Runs trtexec on every plan of a bundle and adds up the stages each operation
executes, as a model-only estimate comparable to upstream test_video.py's
CUDA-event timer. Glue kernels, entropy coding and copies are not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorrt_stages  # noqa: E402

# Stages each operation runs once; a reset group also runs p_reset.
PIPELINES = {
    "intra_encode": ["i_encoder", "i_hyper_encoder", "i_hyper_decoder", "i_spatial_prior_1",
                     "i_spatial_prior_2", "i_spatial_prior_3", "i_decoder"],
    "intra_decode": ["i_hyper_decoder", "i_spatial_prior_1", "i_spatial_prior_2",
                     "i_spatial_prior_3", "i_decoder"],
    "group_encode": ["p_feature_adaptor_m", "p_feature_extractor", "p_encoder",
                     "p_hyper_encoder", "p_hyper_decoder", "p_prior", "p_spatial_prior_1",
                     "p_spatial_prior_2", "p_spatial_prior_3", "p_decoder"],
    "group_decode": ["p_feature_adaptor_m", "p_feature_extractor", "p_hyper_decoder", "p_prior",
                     "p_spatial_prior_1", "p_spatial_prior_2", "p_spatial_prior_3", "p_decoder",
                     "p_reconstruction"],
}


def time_plan(trtexec, plan, stage, iterations):
    inputs, _ = tensorrt_stages.STAGES[stage]
    shapes = ",".join(
        f"{name}:{'x'.join(map(str, tensorrt_stages.profile_shapes(channels, axis)['opt']))}"
        for name, channels, axis in inputs)
    output = subprocess.run(
        [str(trtexec), f"--loadEngine={plan}", f"--shapes={shapes}",
         f"--iterations={iterations}", "--warmUp=200", "--duration=0", "--useSpinWait"],
        capture_output=True, text=True, check=True).stdout
    match = re.search(r"GPU Compute Time: min = ([0-9.]+) ms, max = ([0-9.]+) ms, "
                      r"mean = ([0-9.]+) ms, median = ([0-9.]+) ms", output)
    if not match:
        raise RuntimeError(f"no timing for {stage}")
    return {"mean_ms": float(match.group(3)), "median_ms": float(match.group(4))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output must be new: {args.output}")
    stages = {stage: time_plan(args.trtexec, args.bundle / f"{stage}.plan", stage, args.iterations)
              for stage in tensorrt_stages.STAGES}
    pipelines = {name: round(sum(stages[stage]["median_ms"] for stage in members), 3)
                 for name, members in PIPELINES.items()}
    report = {"schema": "nvcr.dcvcuf.tensorrt_stage_timing.v1",
              "profile": tensorrt_stages.PROFILE, "frame": tensorrt_stages.PROFILE["frame_opt"],
              "stages": stages, "pipelines_median_ms": pipelines,
              "group_frames": 8,
              "note": "model-only; excludes glue, entropy coding and host copies"}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, total in pipelines.items():
        per_frame = total / 8 if name.startswith("group") else total
        print(f"{name:13s} {total:8.2f} ms ({per_frame:6.2f} ms/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
