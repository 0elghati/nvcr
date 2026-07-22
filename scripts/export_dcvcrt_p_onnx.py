#!/usr/bin/env python3
"""Export the pinned DCVC-RT P-frame compute graph and runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import struct
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dcvcrt_root = Path(__file__).resolve().parent.parent / "assets"
    parser.add_argument(
        "--dcvcrt-root",
        type=Path,
        default=Path(os.environ.get("NVCR_DCVCRT_ROOT", str(default_dcvcrt_root))),
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


def resolve_export_device(requested: str, torch: Any) -> tuple[Any, Any]:
    if requested == "cpu":
        return torch.device("cpu"), torch.float32
    if not torch.cuda.is_available():
        print("CUDA export unavailable; falling back to CPU export.", file=sys.stderr)
        return torch.device("cpu"), torch.float32
    try:
        _ = torch.zeros((1,), device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        message = str(exc).lower()
        if "no kernel image is available" in message or "cuda error" in message:
            print(
                "CUDA export failed on this runtime; falling back to CPU export.",
                file=sys.stderr,
            )
            return torch.device("cpu"), torch.float32
        raise
    return torch.device("cuda"), torch.float16


def derive_dynamic_axes(
    input_names: list[str],
    dynamic_shapes: tuple[Any, ...] | None,
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> dict[str, dict[int, str]] | None:
    if dynamic_axes is not None:
        return dynamic_axes
    if dynamic_shapes is None:
        return None
    axes: dict[str, dict[int, str]] = {}
    for index, spec in enumerate(dynamic_shapes):
        if index >= len(input_names) or not isinstance(spec, dict):
            continue
        names: dict[int, str] = {}
        for axis, dim in spec.items():
            if isinstance(axis, int):
                names[axis] = str(dim)
        if names:
            axes[input_names[index]] = names
    return axes or None


def export_entropy(model: Any, output_dir: Path) -> dict[str, Any]:
    model.update()
    groups = (
        ("video_z", *model.bit_estimator_z.get_cdf_info()),
        ("gaussian_y", *model.gaussian_encoder.get_cdf_info()),
    )
    path = output_dir / "p_entropy.bin"
    with path.open("wb") as output:
        output.write(b"NVCRPEN1")
        output.write(struct.pack("<I", len(groups)))
        for name, cdf, sizes, offsets in groups:
            encoded_name = name.encode("ascii")
            output.write(struct.pack("<II", len(encoded_name), len(sizes)))
            output.write(encoded_name)
            for row, declared_size, offset in zip(cdf, sizes, offsets, strict=True):
                size = int(declared_size)
                output.write(struct.pack("<ii", size, int(offset)))
                output.write(row[:size].astype("<i4", copy=False).tobytes(order="C"))
    return {
        "name": "p_entropy",
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "groups": [name for name, *_ in groups],
    }


def export_quant(model: Any, output_dir: Path) -> dict[str, Any]:
    arrays = (
        ("q_encoder", model.q_encoder),
        ("q_decoder", model.q_decoder),
        ("q_feature", model.q_feature),
        ("q_recon", model.q_recon),
    )
    path = output_dir / "p_quant.bin"
    descriptions: list[dict[str, Any]] = []
    with path.open("wb") as output:
        output.write(b"NVCRPQN1")
        output.write(struct.pack("<I", len(arrays)))
        for name, tensor in arrays:
            values = tensor.detach().cpu().float().contiguous().numpy()
            if values.ndim != 4 or values.shape[0] != 72 or values.shape[2:] != (1, 1):
                raise RuntimeError(f"unexpected {name} shape: {values.shape}")
            encoded_name = name.encode("ascii")
            output.write(struct.pack("<III", len(encoded_name), values.shape[0], values.shape[1]))
            output.write(encoded_name)
            output.write(values.astype("<f4", copy=False).tobytes(order="C"))
            descriptions.append({"name": name, "shape": list(values.shape)})
    return {
        "name": "p_quant",
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "arrays": descriptions,
    }


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise SystemExit("--height and --width must be positive")
    if not 0 <= args.qp < 64:
        raise SystemExit("--qp must be in [0, 63]")

    os.environ["SUPPRESS_CUSTOM_KERNEL_WARNING"] = "1"
    sys.path.insert(0, str(args.dcvcrt_root))

    import onnx
    import torch
    import torch.nn.functional as functional
    from torch import nn

    from src.models.video_model import DMC
    from src.utils.common import get_state_dict

    checkpoint = args.dcvcrt_root / "checkpoints/cvpr2025_video.pth.tar"
    if not checkpoint.is_file():
        raise SystemExit(f"video checkpoint not found: {checkpoint}")

    device, dtype = resolve_export_device(args.device, torch)
    model = DMC()
    model.load_state_dict(get_state_dict(checkpoint))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = [export_entropy(model, args.output_dir), export_quant(model, args.output_dir)]
    model = model.to(device=device, dtype=dtype).eval()

    class ReferenceFromFrame(nn.Module):
        def __init__(self, source: DMC):
            super().__init__()
            self.adaptor = source.feature_adaptor_i
            self.extractor = source.feature_extractor

        def forward(self, frame: torch.Tensor, q_feature: torch.Tensor):
            feature = self.adaptor(functional.pixel_unshuffle(frame, 8))
            return self.extractor(feature, q_feature)

    class ReferenceFromFeature(nn.Module):
        def __init__(self, source: DMC):
            super().__init__()
            self.adaptor = source.feature_adaptor_p
            self.extractor = source.feature_extractor

        def forward(self, feature: torch.Tensor, q_feature: torch.Tensor):
            return self.extractor(self.adaptor(feature), q_feature)

    class Prior(nn.Module):
        def __init__(self, source: DMC):
            super().__init__()
            self.hyper_decoder = source.hyper_decoder
            self.temporal = source.temporal_prior_encoder
            self.fusion = source.y_prior_fusion

        def forward(self, z_hat: torch.Tensor, ctx_t: torch.Tensor):
            hierarchical = self.hyper_decoder(z_hat)
            temporal = self.temporal(ctx_t)
            height, width = temporal.shape[2:]
            hierarchical = hierarchical[:, :, :height, :width].contiguous()
            return self.fusion(torch.cat((hierarchical, temporal), dim=1))

    class Synthesis(nn.Module):
        def __init__(self, source: DMC):
            super().__init__()
            self.decoder = source.decoder
            self.reconstruction = source.recon_generation_net

        def forward(
            self,
            y_hat: torch.Tensor,
            ctx: torch.Tensor,
            q_decoder: torch.Tensor,
            q_recon: torch.Tensor,
        ):
            feature = self.decoder(y_hat, ctx, q_decoder)
            frame_hat = self.reconstruction(feature, q_recon)
            return frame_hat, feature

    padded_height = (args.height + 63) // 64 * 64
    padded_width = (args.width + 63) // 64 * 64
    frame = torch.zeros((1, 3, padded_height, padded_width), device=device, dtype=dtype)
    feature = torch.zeros(
        (1, 256, padded_height // 8, padded_width // 8), device=device, dtype=dtype)
    q_encoder = model.q_encoder[args.qp:args.qp + 1]
    q_decoder = model.q_decoder[args.qp:args.qp + 1]
    q_feature = model.q_feature[args.qp:args.qp + 1]
    q_recon = model.q_recon[args.qp:args.qp + 1]

    with torch.no_grad():
        ctx, ctx_t = ReferenceFromFrame(model)(frame, q_feature)
        y = model.encoder(frame, ctx, q_encoder)
        y_padded = model.pad_for_y(y)
        z = model.hyper_encoder(y_padded)
        params = Prior(model)(z, ctx_t)
        spatial_context = torch.cat((torch.zeros_like(y), params), dim=1)

    block_h = torch.export.Dim("block_h", min=1, max=17)
    block_w = torch.export.Dim("block_w", min=1, max=30)
    frame_h = 64 * block_h
    frame_w = 64 * block_w
    feature_h = 8 * block_h
    feature_w = 8 * block_w
    latent_h = 4 * block_h
    latent_w = 4 * block_w

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
        path = args.output_dir / f"{name}.onnx"
        with torch.no_grad():
            export_signature = inspect.signature(torch.onnx.export)
            export_kwargs: dict[str, Any] = {
                "input_names": input_names,
                "output_names": output_names,
                "opset_version": args.opset,
            }
            if "external_data" in export_signature.parameters:
                export_kwargs["external_data"] = False
            if "dynamo" in export_signature.parameters:
                export_kwargs["dynamo"] = dynamo
            if dynamic_shapes is not None and "dynamic_shapes" in export_signature.parameters:
                export_kwargs["dynamic_shapes"] = dynamic_shapes
            if dynamic_axes is not None and "dynamic_axes" in export_signature.parameters:
                export_kwargs["dynamic_axes"] = dynamic_axes
            try:
                torch.onnx.export(
                    module.eval(),
                    inputs,
                    path,
                    **export_kwargs,
                )
            except Exception:
                if not dynamo:
                    raise
                if "dynamo" in export_signature.parameters:
                    export_kwargs["dynamo"] = False
                export_kwargs.pop("dynamic_shapes", None)
                compat_dynamic_axes = derive_dynamic_axes(input_names, dynamic_shapes, dynamic_axes)
                if compat_dynamic_axes is not None and "dynamic_axes" in export_signature.parameters:
                    export_kwargs["dynamic_axes"] = compat_dynamic_axes
                print(
                    f"dynamo ONNX export failed for {name}; retrying with legacy exporter",
                    file=sys.stderr,
                )
                torch.onnx.export(
                    module.eval(),
                    inputs,
                    path,
                    **export_kwargs,
                )
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        exports.append({
            "name": name,
            "file": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "nodes": len(graph.graph.node),
            "inputs": {value.name: tensor_shape(value) for value in graph.graph.input},
            "outputs": {value.name: tensor_shape(value) for value in graph.graph.output},
        })
        print(f"exported {name}: {path.stat().st_size / (1024 * 1024):.1f} MiB")

    export(
        "p_reference_frame", ReferenceFromFrame(model), (frame, q_feature),
        ["reference_frame", "q_feature"], ["context", "temporal_context"],
        ({2: frame_h, 3: frame_w}, None),
    )
    export(
        "p_reference_feature", ReferenceFromFeature(model), (feature, q_feature),
        ["reference_feature", "q_feature"], ["context", "temporal_context"],
        ({2: feature_h, 3: feature_w}, None),
    )
    export(
        "p_analysis", model.encoder, (frame, ctx, q_encoder),
        ["frame", "context", "q_encoder"], ["y"],
        ({2: frame_h, 3: frame_w}, {2: feature_h, 3: feature_w}, None),
    )
    export(
        "p_hyper_analysis", model.hyper_encoder, (y_padded,),
        ["y_padded"], ["z"], ({2: latent_h, 3: latent_w},),
    )
    export(
        "p_prior", Prior(model), (z, ctx_t),
        ["z_hat", "temporal_context"], ["params"],
        None,
        {
            "z_hat": {2: "block_h", 3: "block_w"},
            "temporal_context": {2: "feature_h", 3: "feature_w"},
            "params": {2: "latent_h", 3: "latent_w"},
        },
        False,
    )
    export(
        "p_spatial_prior", model.y_spatial_prior, (spatial_context,),
        ["context"], ["scales_means"], ({2: latent_h, 3: latent_w},),
    )
    export(
        "p_synthesis", Synthesis(model),
        (torch.zeros_like(y), ctx, q_decoder, q_recon),
        ["y_hat", "context", "q_decoder", "q_recon"],
        ["frame_hat", "feature"],
        ({2: latent_h, 3: latent_w}, {2: feature_h, 3: feature_w}, None, None),
    )

    manifest = {
        "format": 1,
        "codec": "DCVC-RT",
        "frame_type": "P",
        "reference_commit": "dae827ffcc812566adbeaf4554f0fe2d9b4b9e0c",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset": args.opset,
        "precision": "fp16" if dtype == torch.float16 else "fp32",
        "sample_resolution": [args.height, args.width],
        "sample_padded_resolution": [padded_height, padded_width],
        "qp": args.qp,
        "graphs": exports,
        "assets": assets,
    }
    path = args.output_dir / "p_frame_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
