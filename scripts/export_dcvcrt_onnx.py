#!/usr/bin/env python3
"""Export the pinned DCVC-RT I-frame compute graph into TensorRT-ready ONNX stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dcvcrt-root",
        type=Path,
        default=Path(os.environ.get("NVCR_DCVCRT_ROOT", "/home/oelghati/DCVC/DCVC-RT")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/models/dcvcrt"))
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=176)
    parser.add_argument("--qp", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value: Any) -> list[str]:
    return [dimension.dim_param or str(dimension.dim_value)
            for dimension in value.type.tensor_type.shape.dim]


def export_runtime_assets(model: Any, output_dir: Path) -> list[dict[str, Any]]:
    import struct

    model.update()
    groups = (
        ("image_z", *model.bit_estimator_z.get_cdf_info()),
        ("gaussian_y", *model.gaussian_encoder.get_cdf_info()),
    )
    entropy_path = output_dir / "i_entropy.bin"
    with entropy_path.open("wb") as output:
        output.write(b"NVCRENT1")
        output.write(struct.pack("<I", len(groups)))
        for name, cdf, sizes, offsets in groups:
            encoded_name = name.encode("ascii")
            output.write(struct.pack("<II", len(encoded_name), len(sizes)))
            output.write(encoded_name)
            for row, declared_size, offset in zip(cdf, sizes, offsets, strict=True):
                size = int(declared_size)
                output.write(struct.pack("<ii", size, int(offset)))
                output.write(row[:size].astype("<i4", copy=False).tobytes(order="C"))

    q_enc = model.q_scale_enc.detach().cpu().float().contiguous().numpy()
    q_dec = model.q_scale_dec.detach().cpu().float().contiguous().numpy()
    if q_enc.shape != q_dec.shape or q_enc.ndim != 4 or q_enc.shape[2:] != (1, 1):
        raise RuntimeError("unexpected image quantization tensor shape")
    quant_path = output_dir / "i_quant.bin"
    with quant_path.open("wb") as output:
        output.write(b"NVCRQNT1")
        output.write(struct.pack("<II", q_enc.shape[0], q_enc.shape[1]))
        output.write(q_enc.astype("<f4", copy=False).tobytes(order="C"))
        output.write(q_dec.astype("<f4", copy=False).tobytes(order="C"))

    return [
        {"name": "i_entropy", "file": entropy_path.name, "sha256": sha256(entropy_path),
         "bytes": entropy_path.stat().st_size, "groups": [name for name, *_ in groups]},
        {"name": "i_quant", "file": quant_path.name, "sha256": sha256(quant_path),
         "bytes": quant_path.stat().st_size, "shape": list(q_enc.shape)},
    ]


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must be positive multiples of 16")
    if not 0 <= args.qp < 64:
        raise SystemExit("--qp must be in [0, 63]")

    os.environ["SUPPRESS_CUSTOM_KERNEL_WARNING"] = "1"
    sys.path.insert(0, str(args.dcvcrt_root))

    import onnx
    import torch
    from torch import nn

    from src.models.image_model import DMCI
    from src.utils.common import get_state_dict

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA export requested but torch.cuda.is_available() is false")

    checkpoint = args.dcvcrt_root / "checkpoints/cvpr2025_image.pth.tar"
    if not checkpoint.is_file():
        raise SystemExit(f"image checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = DMCI()
    model.load_state_dict(get_state_dict(checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_assets = export_runtime_assets(model, args.output_dir)
    model = model.to(device=device, dtype=dtype).eval()

    class HyperSynthesis(nn.Module):
        def __init__(self, source: DMCI):
            super().__init__()
            self.hyper_dec = source.hyper_dec
            self.prior_fusion = source.y_prior_fusion
            self.prior_reduction = source.y_spatial_prior_reduction

        def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            params = self.prior_fusion(self.hyper_dec(z))
            return params, self.prior_reduction(params)

    class SpatialPriorStage(nn.Module):
        def __init__(self, adaptor: nn.Module, prior: nn.Module):
            super().__init__()
            self.adaptor = adaptor
            self.prior = prior

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return self.prior(self.adaptor(context))

    frame_h = 16 * torch.export.Dim("frame_h16", min=4, max=135)
    frame_w = 16 * torch.export.Dim("frame_w16", min=4, max=240)
    latent_h = torch.export.Dim("latent_h", min=4, max=135)
    latent_w = torch.export.Dim("latent_w", min=4, max=240)
    padded_y_h = 4 * torch.export.Dim("padded_y_h4", min=1, max=34)
    padded_y_w = 4 * torch.export.Dim("padded_y_w4", min=1, max=60)
    hyper_h = torch.export.Dim("hyper_h", min=1, max=34)
    hyper_w = torch.export.Dim("hyper_w", min=1, max=60)

    frame = torch.zeros((1, 3, args.height, args.width), device=device, dtype=dtype)
    q_enc = model.q_scale_enc[args.qp:args.qp + 1]
    q_dec = model.q_scale_dec[args.qp:args.qp + 1]
    with torch.no_grad():
        y = model.enc(frame, q_enc)
        y_pad = model.pad_for_y(y)
        z = model.hyper_enc(y_pad)
        spatial_context = torch.zeros(
            (1, 512, y.shape[2], y.shape[3]), device=device, dtype=dtype)
        y_hat = torch.zeros_like(y)

    exports: list[dict[str, Any]] = []

    def export(
        name: str,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        input_names: list[str],
        output_names: list[str],
        dynamic_shapes: tuple[Any, ...] | None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        dynamo: bool = True,
    ) -> None:
        output = args.output_dir / f"{name}.onnx"
        module = module.eval()
        with torch.no_grad():
            torch.onnx.export(
                module,
                inputs,
                output,
                input_names=input_names,
                output_names=output_names,
                opset_version=args.opset,
                dynamo=dynamo,
                dynamic_shapes=dynamic_shapes if dynamo else None,
                dynamic_axes=dynamic_axes if not dynamo else None,
                external_data=False,
            )
        graph = onnx.load(output)
        onnx.checker.check_model(graph)
        exports.append({
            "name": name,
            "exporter": "dynamo" if dynamo else "torchscript",
            "file": output.name,
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
            "nodes": len(graph.graph.node),
            "inputs": {value.name: tensor_shape(value) for value in graph.graph.input},
            "outputs": {value.name: tensor_shape(value) for value in graph.graph.output},
        })
        print(f"exported {name}: {output.stat().st_size / (1024 * 1024):.1f} MiB")

    export(
        "i_analysis",
        model.enc,
        (frame, q_enc),
        ["frame", "q_enc"],
        ["y"],
        ({2: frame_h, 3: frame_w}, None),
    )
    export(
        "i_hyper_analysis",
        model.hyper_enc,
        (y_pad,),
        ["y_padded"],
        ["z"],
        ({2: padded_y_h, 3: padded_y_w},),
    )
    export(
        "i_hyper_synthesis",
        HyperSynthesis(model).to(device=device, dtype=dtype),
        (z,),
        ["z_hat"],
        ["params_padded", "common_padded"],
        None,
        {
            "z_hat": {2: "hyper_h", 3: "hyper_w"},
            "params_padded": {2: "padded_y_h", 3: "padded_y_w"},
            "common_padded": {2: "padded_y_h", 3: "padded_y_w"},
        },
        False,
    )
    adaptors = (
        model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3,
    )
    for stage, adaptor in enumerate(adaptors, start=1):
        export(
            f"i_spatial_prior_{stage}",
            SpatialPriorStage(adaptor, model.y_spatial_prior).to(device=device, dtype=dtype),
            (spatial_context,),
            ["context"],
            ["scales_means"],
            ({2: latent_h, 3: latent_w},),
        )
    export(
        "i_synthesis",
        model.dec,
        (y_hat, q_dec),
        ["y_hat", "q_dec"],
        ["frame_hat"],
        None,
        {
            "y_hat": {2: "latent_h", 3: "latent_w"},
            "frame_hat": {2: "frame_h", 3: "frame_w"},
        },
        False,
    )

    manifest = {
        "format": 1,
        "codec": "DCVC-RT",
        "reference_commit": "dae827ffcc812566adbeaf4554f0fe2d9b4b9e0c",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset": args.opset,
        "precision": "fp16" if dtype == torch.float16 else "fp32",
        "sample_resolution": [args.height, args.width],
        "qp": args.qp,
        "graphs": exports,
        "assets": runtime_assets,
    }
    manifest_path = args.output_dir / "i_frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
