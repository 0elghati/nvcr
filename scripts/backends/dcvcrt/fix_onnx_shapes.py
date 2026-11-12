#!/usr/bin/env python3
"""Write a fixed-shape copy of the DCVC-RT ONNX bundle for one frame size.

Accelerators such as the Qualcomm HTP compile static shapes only. The spatial
sizes mirror the host backend: frames pad to multiples of 16, latents are 1/16,
features are 1/8, and the hyperprior is 1/64 after padding latents to 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ASSETS = (
    "i_entropy.bin", "i_quant.bin", "i_frame_manifest.json",
    "p_entropy.bin", "p_quant.bin", "p_frame_manifest.json",
)


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def spatial_sizes(width: int, height: int) -> dict[str, dict[str, tuple[int, int]]]:
    frame = (round_up(height, 16), round_up(width, 16))
    latent = (frame[0] // 16, frame[1] // 16)
    padded = (round_up(latent[0], 4), round_up(latent[1], 4))
    hyper = (padded[0] // 4, padded[1] // 4)
    feature = (frame[0] // 8, frame[1] // 8)
    spatial_prior = {"context": latent}
    return {
        "i_analysis": {"frame": frame},
        "i_hyper_analysis": {"y_padded": padded},
        "i_hyper_synthesis": {"z_hat": hyper},
        "i_spatial_prior_1": spatial_prior,
        "i_spatial_prior_2": spatial_prior,
        "i_spatial_prior_3": spatial_prior,
        "i_synthesis": {"y_hat": latent},
        "p_reference_frame": {"reference_frame": frame},
        "p_reference_feature": {"reference_feature": feature},
        "p_analysis": {"frame": frame, "context": feature},
        "p_hyper_analysis": {"y_padded": padded},
        "p_prior": {"z_hat": hyper, "temporal_context": feature},
        "p_spatial_prior": spatial_prior,
        "p_synthesis": {"y_hat": latent, "context": feature},
        "p_display": {"feature": feature},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width < 64 or args.height < 64 or args.width % 2 or args.height % 2:
        raise SystemExit("--width and --height must be even and at least 64")

    import onnx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage, inputs in spatial_sizes(args.width, args.height).items():
        model = onnx.load(args.model_dir / f"{stage}.onnx")
        for value in model.graph.input:
            dims = value.type.tensor_type.shape.dim
            spatial = inputs.get(value.name)
            if len(dims) != 4 or dims[1].dim_param:
                raise SystemExit(f"{stage}.{value.name} is not [1, C, H, W] with static C")
            fixed = (1, dims[1].dim_value, *spatial) if spatial else None
            if fixed is None and any(dim.dim_param for dim in dims):
                raise SystemExit(f"{stage}.{value.name} has an unmapped symbolic dimension")
            for dim, size in zip(dims, fixed or ()):
                dim.Clear()
                dim.dim_value = size
        # Stale symbolic hints would conflict with the fixed inputs.
        del model.graph.value_info[:]
        for value in model.graph.output:
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param:
                    dim.Clear()
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        onnx.checker.check_model(model)
        onnx.save(model, args.output_dir / f"{stage}.onnx")
        print(f"fixed {stage}: {inputs}")
    for asset in ASSETS:
        shutil.copy2(args.model_dir / asset, args.output_dir / asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
