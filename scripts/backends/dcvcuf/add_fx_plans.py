#!/usr/bin/env python3
"""Adds DCVC-UF's nvcr.fx.v1 stages to a DCVC-UF TensorRT engine bundle.

The fx stages (export_fx_programs.py) are compiled over the bundle's frame sizes with nvcr-fx-plans. Next to them
the codec runs five FP16 networks that nothing reads back, exported here from the pinned model and built strongly
typed (the int32 inputs are scaled in FP32): the I-frame analysis and hyper analysis, the chunk's analysis from its
8-bit frames and the exact context (p_encode) and its hyper analysis, and the chunk's eight pictures from the exact
feature (p_display, 8-bit 4:2:0 planes). Every new file is listed in engine.sha256 and engine_manifest.json.

usage: add_fx_plans.py --bundle DIR --programs DIR --dcvcuf-root DIR --checkpoints DIR --nvcr-fx-plans PATH
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fx_bundle import add_fx_plans, sha256  # noqa: E402

IMAGE_SHA256 = "b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302"
VIDEO_SHA256 = "565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47"


def sizes(width: int, height: int) -> dict:
    """Frame, feature, latent and hyper-latent sizes of a padded frame size."""
    h, w = -(-height // 16) * 16, -(-width // 16) * 16
    yh, yw = h // 16, w // 16
    return {"frame": [h, w], "feature": [h // 8, w // 8], "latent": [yh, yw],
            "latent4": [-(-yh // 4) * 4, -(-yw // 4) * 4], "z": [-(-yh // 4), -(-yw // 4)]}


def fx_shapes(width: int, height: int) -> dict:
    """Input shapes [channels, height, width] of every DCVC-UF fx stage."""
    s = sizes(width, height)
    lat, feat = s["latent"], s["feature"]
    qp, parity = [1, 1, 1], [4, *lat]
    istep = {"symbols": [64, *lat], "symbol_means": [64, *lat], "parity": parity}
    pstep = {"symbols": [256, *lat], "symbol_means": [256, *lat], "parity": parity}
    return {
        "i_prior": {"z_fixed": [128, *s["z"]], "latent_ref": [1, *lat], "parity": parity},
        "i_spatial_prior_1": {**istep, "common": [256, *lat]},
        **{f"i_spatial_prior_{k}": {**istep, "y_hat": [256, *lat], "common": [256, *lat]} for k in (2, 3)},
        "i_synthesis": {**istep, "y_hat": [256, *lat], "qp": qp},
        "p_reference_frame": {"frame": [3, *s["frame"]]},
        "p_reset_reference": {"feature": [512, *feat]},
        "p_reference_memory": {"memory": [512, *feat], "feature": [512, *feat]},
        "p_prior": {"z_fixed": [128, *s["z"]], "memory": [512, *feat], "qp": qp, "latent_ref": [1, *lat]},
        "p_spatial_prior_1": {**pstep, "reduced": [256, *lat]},
        **{f"p_spatial_prior_{k}": {**pstep, "y_hat": [256, *lat], "reduced": [256, *lat]} for k in (2, 3)},
        "p_synthesis": {**pstep, "y_hat": [256, *lat], "q_dec": [256, *lat], "context": [512, *feat], "qp": qp},
    }


def picture(luma, chroma):
    """8-bit luma and 4:2:0 chroma planes (one frame per channel group) -> pictures in [-1/2, 1/2], 3 channels each,
    chroma repeated over 2x2 like DCVC-UF's frame upload."""
    luma, chroma = luma.float(), F.interpolate(chroma.float(), scale_factor=2.0, mode="nearest")
    planes = [torch.cat((luma[:, i:i + 1], chroma[:, 2 * i:2 * i + 2]), 1) for i in range(luma.shape[1])]
    return torch.cat(planes, 1) / 255 - 0.5


class IAnalysis(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.enc = net.enc

    def forward(self, luma, chroma, q_enc):
        return self.enc(picture(luma, chroma).to(q_enc.dtype), q_enc)


class PEncode(nn.Module):
    """The chunk's analysis from its eight 8-bit frames and the exact context (int32 at 2^-14)."""

    def __init__(self, net):
        super().__init__()
        self.encoder = net.encoder

    def forward(self, luma, chroma, context, q_encoder):
        frames = picture(luma, chroma).to(q_encoder.dtype)
        return self.encoder(frames, (context.float() * 2.0 ** -14).to(q_encoder.dtype), q_encoder)


class PDisplay(nn.Module):
    """The chunk's eight pictures from the exact feature (int32 at 2^-14), as shown: clamped to [-1/2, 1/2], luma
    floor(255 (v + 1/2) + 1/2), chroma floor(255 (mean of 2x2 + 1/2) + 1/2)."""

    def __init__(self, net):
        super().__init__()
        self.recon_head = net.recon_head

    def forward(self, feature):
        frames = self.recon_head((feature.float() * 2.0 ** -14).half())
        frames = [f.float().clamp(-0.5, 0.5) + 0.5 for f in frames]
        luma = torch.cat([torch.floor(f[:, :1] * 255 + 0.5) for f in frames], 1)
        chroma = torch.cat([torch.floor(F.avg_pool2d(f[:, 1:], 2) * 255 + 0.5) for f in frames], 1)
        return luma.to(torch.uint8), chroma.to(torch.uint8)


def load(path: Path, expected: str, net):
    if sha256(path) != expected:
        raise SystemExit(f"{path}: unexpected SHA-256")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    net.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()})
    return net.eval().cuda().half()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--dcvcuf-root", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, help="default: <dcvcuf-root>/checkpoints")
    ap.add_argument("--nvcr-fx-plans", type=Path, required=True)
    ap.add_argument("--trtexec", default="trtexec")
    ap.add_argument("--builder-optimization-level", type=int, default=5)
    ap.add_argument("--onnx-dir", type=Path, help="also keep the five FP16 graphs here (for ONNX Runtime)")
    args = ap.parse_args()
    sys.path.insert(0, str(args.dcvcuf_root))
    from src.layers.layers import WSiLUChunkAdd
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, ModelStructure

    def chunk_add(self, x):
        """Upstream sums channels 4k..4k+3; with the rows of the conv before it reordered (contiguous()), the same
        sums are of four contiguous channel blocks, which TensorRT fuses instead of copying strided slices."""
        x = self.wsilu(x)
        c = x.shape[1] // 4
        return x[:, :c] + x[:, c:2 * c] + x[:, 2 * c:3 * c] + x[:, 3 * c:]

    WSiLUChunkAdd.forward = chunk_add

    def contiguous(net):
        """Reorder the output rows of each chunk-add's 1x1 conv from 4k + j to j C + k."""
        for m in net.modules():
            if isinstance(m, nn.Sequential) and len(m) == 3 and isinstance(m[1], WSiLUChunkAdd):
                conv = m[0]
                out = conv.out_channels
                order = torch.arange(out).view(out // 4, 4).t().reshape(-1)
                conv.weight.data = conv.weight.data[order]
                if conv.bias is not None:
                    conv.bias.data = conv.bias.data[order]

    ck = args.checkpoints or args.dcvcuf_root / "checkpoints"
    inet = load(ck / "cvpr2026_image.pth.tar", IMAGE_SHA256, DMCI())
    pnet = load(ck / "cvpr2026_video_hts.pth.tar", VIDEO_SHA256, DMC(ModelStructure.HTS))
    for net in (inet, pnet):
        contiguous(net)
    stages = json.loads((args.bundle / "stages.json").read_text())["stages"]
    profile = next(t["profile"] for t in stages["i_encoder"] if t["name"] == "frame")
    frames = {key: (profile[key][3], profile[key][2]) for key in ("min", "opt", "max")}  # (width, height)
    low, opt, high = (sizes(*frames[key]) for key in ("min", "opt", "max"))

    u8, half, i32 = torch.uint8, torch.float16, torch.int32
    dev = "cuda"
    graphs = {
        # name: (module, inputs {name: (tensor, size key, channels or None for static)}, outputs, typed)
        "i_analysis": (IAnalysis(inet), {"luma": ("frame", 1, u8), "chroma": ("chroma", 2, u8),
                                         "q_enc": (None, 384, half)}, ["y"]),
        "i_hyper_analysis": (inet.hyper_enc, {"y_padded": ("latent4", 256, half)}, ["z"]),
        "p_encode": (PEncode(pnet), {"luma": ("frame", 8, u8), "chroma": ("chroma", 16, u8),
                                     "context": ("feature", 512, i32), "q_encoder": (None, 512, half)}, ["y"]),
        "p_hyper_analysis": (pnet.hyper_encoder, {"y_padded": ("latent4", 256, half)}, ["z"]),
        "p_display": (PDisplay(pnet), {"feature": ("feature", 512, i32)}, ["luma", "chroma"]),
    }

    def dims(key, size):
        return [d // 2 for d in size["frame"]] if key == "chroma" else size[key]

    manifest_path = args.bundle / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if args.onnx_dir:
        args.onnx_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for name, (module, inputs, outputs) in graphs.items():
            sample, axes, shapes = [], {}, {"min": [], "opt": [], "max": []}
            for key, (axis, channels, dtype) in inputs.items():
                hw = [1, 1] if axis is None else dims(axis, opt)
                sample.append(torch.zeros((1, channels, *hw), dtype=dtype, device=dev))
                if axis is not None:
                    axes[key] = {2: f"{axis}_h", 3: f"{axis}_w"}
                    for which, size in (("min", low), ("opt", opt), ("max", high)):
                        shapes[which].append(f"{key}:1x{channels}x{'x'.join(map(str, dims(axis, size)))}")
            onnx = (args.onnx_dir or Path(tmp)) / f"{name}.onnx"
            torch.onnx.export(module, tuple(sample), onnx, input_names=list(inputs), output_names=outputs,
                              dynamic_axes=axes, opset_version=18, dynamo=False)
            plan = args.bundle / f"{name}.plan"
            built = subprocess.run([args.trtexec, f"--onnx={onnx}", f"--saveEngine={plan}",
                            *(f"--{which}Shapes={','.join(shapes[which])}" for which in ("min", "opt", "max")),
                            f"--builderOptimizationLevel={args.builder_optimization_level}", "--stronglyTyped",
                            "--skipInference"], capture_output=True, text=True)
            if built.returncode != 0:
                raise SystemExit(f"{name}: trtexec failed\n" + "\n".join((built.stdout + built.stderr).splitlines()[-15:]))
            manifest["files"][plan.name] = sha256(plan)
            print(f"{name}: built")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lo, op, hi = (fx_shapes(*frames[key]) for key in ("min", "opt", "max"))
    ranged = {stage: {name: [lo[stage][name], op[stage][name], hi[stage][name]] for name in op[stage]} for stage in op}
    add_fx_plans(args.bundle, args.programs, args.nvcr_fx_plans, ranged)
    print(f"{args.bundle}: fx stages for {frames['min']} to {frames['max']}")


if __name__ == "__main__":
    main()
