#!/usr/bin/env python3
"""Rate-distortion cost of exact integer arithmetic on DCVC-RT's decoder side, over full GOPs.

Roadmap C12 / issue #216. A stream decodes identically on every target only when
every computation that reaches an entropy decision is exact. For P-frames that
is nearly the whole decoder: the temporal context comes from the previous
decoded feature (or, after an I-frame and at reference resets, the previous
decoded frame). This study emulates the planned nvcr.det-int.v1 profile on
every decoder-side pointwise convolution of the I- and P-frame models and
measures the BD-rate over real GOPs, so drift along the GOP is included.

Decoder-side modules (emulated):
  I: hyper_dec, y_prior_fusion, y_spatial_prior_reduction, y_spatial_prior_adaptor_1..3,
     y_spatial_prior, dec
  P: feature_adaptor_i, feature_adaptor_p, feature_extractor, temporal_prior_encoder,
     hyper_decoder, y_prior_fusion, y_spatial_prior, decoder, recon_generation_net
Encoder-only modules (enc, hyper_enc, encoder, hyper_encoder) stay FP32, as do
depthwise convolutions (the exact profile runs them in det-fp32) and every
nonlinear op.

Integer emulation of one dense conv (see src/deterministic/program.hpp; a
spatial conv first becomes im2col, so its activation scale covers each
output position's receptive field):
  weights      per output channel: q = round(w / s), s = max|w| / (2^(b-1) - 1)
  activations  per position over the input channels: q = round(x * 2^-e), with
               a power-of-two scale chosen so max|x| fits in b bits
  output       (sum q_w q_x) * s * 2^e + bias
The sum is formed in FP32 here; its rounding error is far below the
quantization error being measured, so the rate-distortion result is that of the
exact profile.

Coding follows the pinned test_video.py: one I-frame, then P-frames with the
QP shift map [0, 1, 0, 2, 0, 2, 0, 2] and a reference reset every
--reset-interval frames. Rate is the real bitstream; distortion is PSNR of the
reconstruction the decoder would produce.

Example (the pinned DCVC-RT checkout, its checkpoints and entropy extension):

  python3 integer_gop_study.py --dcvcrt-root ~/DCVC-RT-nvcr-1feb \\
      --sequence ~/datasets/hd/Kimono_1920x1080_24.yuv:1920x1080 \\
      --variants fp32 fp16 w16a16 --frames 96 --output gop.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

I_DECODER_MODULES = (
    "hyper_dec", "y_prior_fusion", "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1", "y_spatial_prior_adaptor_2", "y_spatial_prior_adaptor_3",
    "y_spatial_prior", "dec",
)
P_DECODER_MODULES = (
    "feature_adaptor_i", "feature_adaptor_p", "feature_extractor", "temporal_prior_encoder",
    "hyper_decoder", "y_prior_fusion", "y_spatial_prior", "decoder", "recon_generation_net",
)
DEFAULT_QPS = (0, 21, 42, 63)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcvcrt-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", default=[],
                        help="PATH:WIDTHxHEIGHT; repeat for each sequence")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--qps", type=int, nargs="+", default=list(DEFAULT_QPS))
    parser.add_argument("--variants", nargs="+", default=["fp32", "fp16", "w16a16"],
                        help="fp32, fp16, wNaM (every decoder-side pointwise conv), "
                             "or map=PATH with {\"default\": SPEC, \"layers\": {NAME: SPEC}}; "
                             "layer names are I.<module path> or P.<module path> (--list-layers)")
    parser.add_argument("--anchor", default="fp32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list-layers", action="store_true")
    return parser.parse_args()


def parse_spec(spec: str) -> dict:
    match = re.fullmatch(r"w(\d+)a(\d+)", spec)
    if match is None:
        raise SystemExit(f"unknown bit-width spec {spec}")
    return {"weight_bits": int(match[1]), "activation_bits": int(match[2])}


def parse_variant(name: str) -> dict:
    """fp32, fp16, wNaM, or map=PATH; a "@fp16" suffix on an integer variant runs
    every other op and every stored tensor in FP16 (FP16 intermediate storage).

    Research ablations (no training) of the P-frame temporal prior, the
    entropy model's only dependence on the decoded reference:
      fp32@temporal=zero  the temporal prior parameters are replaced by zeros
      fp32@temporal=mean  they are replaced by their per-channel spatial mean
    Both are distribution shifts for the trained fusion network, so the rate
    cost they show is an upper bound on what the temporal prior is worth."""
    if "@temporal=" in name:
        base, _, mode = name.partition("@temporal=")
        if mode not in ("zero", "mean"):
            raise SystemExit(f"unknown temporal ablation {mode}")
        variant = parse_variant(base)
        variant.update(name=name, temporal=mode)
        return variant
    if name.endswith("@fp16") and name != "fp16":
        variant = parse_variant(name[:-5])
        variant.update(name=name, precision="fp16")
        return variant
    if name in ("fp32", "fp16"):
        return {"name": name, "precision": name}
    if name.startswith("map="):
        document = json.loads(Path(name[4:]).read_text(encoding="utf-8"))
        default = parse_spec(document["default"]) if document.get("default") else None
        layers = {layer: parse_spec(spec) for layer, spec in document["layers"].items()}
        return {"name": name, "precision": "fp32",
                "resolve": lambda layer: layers.get(layer, default)}
    parsed = parse_spec(name)
    return {"name": name, "precision": "fp32", "resolve": lambda layer: parsed}


def main() -> int:
    args = parse_args()
    os.environ["DCVC_DISABLE_CUSTOM_CUDA_INFERENCE"] = "1"
    os.environ["SUPPRESS_CUSTOM_KERNEL_WARNING"] = "1"
    sys.path.insert(0, str(args.dcvcrt_root.resolve()))

    import numpy as np
    import torch
    import torch.nn.functional as F
    # The pinned checkout's fused CUDA kernels read conv weights directly and
    # would bypass the emulated layers. Modules copy this flag at import, so
    # it must be cleared before the models load.
    import src.layers.cuda_inference as cuda_inference
    cuda_inference.CUSTOMIZED_CUDA_INFERENCE = False
    from src.layers.cuda_inference import replicate_pad
    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.utils.common import get_state_dict
    from src.utils.transforms import ycbcr420_to_444_np, yuv_444_to_420

    def calc_psnr(reference, test):
        # As the pinned src/utils/metrics.py (which also imports scipy).
        mse = np.mean(np.square(reference.astype(np.float64) - test.astype(np.float64)))
        return 100.0 if mse <= 1e-10 else 10 * np.log10(255.0 * 255.0 / mse)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    root = args.dcvcrt_root

    def load_models(precision: str):
        i_net = DMCI()
        i_net.load_state_dict(get_state_dict(str(root / "checkpoints/cvpr2025_image.pth.tar")))
        p_net = DMC()
        p_net.load_state_dict(get_state_dict(str(root / "checkpoints/cvpr2025_video.pth.tar")))
        for net in (i_net, p_net):
            net.to(device).eval()
            net.update(None)
            if precision == "fp16":
                net.half()
        return i_net, p_net

    def integer_conv(conv: torch.nn.Conv2d, spec: dict):
        out_channels = conv.weight.shape[0]
        in_channels = conv.weight[0].numel()
        # Artifacts carry FP16 weights; quantize from those.
        weights = conv.weight.detach().half().float().reshape(out_channels, in_channels)
        bias = (conv.bias.detach().half().float().reshape(1, out_channels, 1)
                if conv.bias is not None else None)
        weight_limit = 2 ** (spec["weight_bits"] - 1) - 1
        activation_bits = spec["activation_bits"]
        activation_limit = 2 ** (activation_bits - 1) - 1
        maximum = weights.abs().amax(dim=1, keepdim=True)
        weight_scale = torch.where(maximum > 0, maximum / weight_limit, torch.ones_like(maximum))
        weight_q = torch.round(weights / weight_scale).clamp(-weight_limit, weight_limit)

        spatial = conv.kernel_size != (1, 1) or conv.stride != (1, 1) or conv.padding != (0, 0)

        def forward(x):
            if spatial:
                # A dense spatial conv runs as im2col plus a pointwise conv: one
                # scale per output position over its receptive field.
                padded = F.pad(x, (conv.padding[1], conv.padding[1], conv.padding[0], conv.padding[0]))
                columns = F.unfold(padded, kernel_size=conv.kernel_size, stride=conv.stride)
                height = (padded.shape[2] - conv.kernel_size[0]) // conv.stride[0] + 1
                x = columns.reshape(1, -1, height, columns.shape[-1] // height)
            _, _, height, width = x.shape
            values = x.detach().float().reshape(in_channels, height * width)
            amax = values.abs().amax(dim=0, keepdim=True)
            _, exponent = torch.frexp(amax)
            exponent = torch.where(amax > 0, exponent - (activation_bits - 1),
                                   torch.zeros_like(exponent))
            scale = torch.ldexp(torch.ones_like(amax), exponent)
            activation_q = torch.round(values / scale).clamp(-activation_limit, activation_limit)
            output = (weight_q @ activation_q) * weight_scale * scale
            output = output.reshape(1, out_channels, height * width)
            if bias is not None:
                output = output + bias
            return output.to(x.dtype).reshape(1, out_channels, height, width)

        return forward

    def decoder_layers(i_net, p_net):
        for prefix, net, modules in (("I", i_net, I_DECODER_MODULES), ("P", p_net, P_DECODER_MODULES)):
            for module_name in modules:
                for sub_name, conv in getattr(net, module_name).named_modules():
                    if isinstance(conv, torch.nn.Conv2d) and conv.groups == 1:
                        name = f"{module_name}.{sub_name}" if sub_name else module_name
                        yield f"{prefix}.{name}", conv

    if args.list_layers:
        for layer, conv in decoder_layers(*load_models("fp32")):
            print(layer, conv.in_channels, conv.out_channels)
        return 0

    def read_frame(path: Path, width: int, height: int, index: int):
        frame_bytes = width * height * 3 // 2
        with path.open("rb") as handle:
            handle.seek(frame_bytes * index)
            raw = handle.read(frame_bytes)
        if len(raw) != frame_bytes:
            return None
        data = np.frombuffer(raw, dtype=np.uint8)
        y = data[:width * height].reshape(1, height, width).astype(np.float32)
        uv = data[width * height:].reshape(2, height // 2, width // 2).astype(np.float32)
        return y, uv

    sequences = []
    for entry in args.sequence:
        path, _, size = entry.rpartition(":")
        width, height = (int(item) for item in size.split("x"))
        sequences.append((Path(path), width, height))
    variants = [parse_variant(name) for name in args.variants]
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    records = []
    for variant in variants:
        i_net, p_net = load_models(variant["precision"])
        replaced = 0
        if "resolve" in variant:
            for layer, conv in decoder_layers(i_net, p_net):
                spec = variant["resolve"](layer)
                if spec is not None:
                    conv.forward = integer_conv(conv, spec)
                    replaced += 1
        called = set()
        if "resolve" in variant:
            for layer, conv in decoder_layers(i_net, p_net):
                if variant["resolve"](layer) is not None:
                    inner = conv.forward

                    def tracked(x, inner=inner, layer=layer):
                        called.add(layer)
                        return inner(x)

                    conv.forward = tracked
        if variant.get("temporal"):
            original = p_net.temporal_prior_encoder.forward

            def ablated(x, original=original, mode=variant["temporal"]):
                params = original(x)
                if mode == "zero":
                    return torch.zeros_like(params)
                return params.mean(dim=(2, 3), keepdim=True).expand_as(params).contiguous()

            p_net.temporal_prior_encoder.forward = ablated
        dtype = torch.float16 if variant["precision"] == "fp16" else torch.float32
        started = time.perf_counter()
        for path, width, height in sequences:
            padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
            two_coders = height * width > 1280 * 720
            i_net.set_use_two_entropy_coders(two_coders)
            p_net.set_use_two_entropy_coders(two_coders)
            for qp in args.qps:
                bits = 0
                psnr_y, psnr_yuv, per_frame = [], [], []
                last_qp = 0
                p_net.set_curr_poc(0)
                with torch.no_grad():
                    for frame_index in range(args.frames):
                        planes = read_frame(path, width, height, frame_index)
                        if planes is None:
                            break
                        y, uv = planes
                        x = torch.from_numpy(ycbcr420_to_444_np(y, uv)).to(device) / 255.0
                        x = replicate_pad(x.unsqueeze(0).to(dtype), padding_b, padding_r)
                        if frame_index == 0:
                            encoded = i_net.compress(x, qp)
                            p_net.clear_dpb()
                            p_net.add_ref_frame(None, encoded["x_hat"])
                            x_hat = encoded["x_hat"]
                        else:
                            if args.reset_interval > 0 and frame_index % args.reset_interval == 1:
                                p_net.prepare_feature_adaptor_i(last_qp)
                            curr_qp = p_net.shift_qp(qp, index_map[frame_index % 8])
                            encoded = p_net.compress(x, curr_qp)
                            last_qp = curr_qp
                            q_recon = p_net.q_recon[curr_qp:curr_qp + 1]
                            x_hat = p_net.recon_generation_net(p_net.dpb[0].feature, q_recon).clamp_(0, 1)
                        bits += len(encoded["bit_stream"]) * 8
                        x_hat = x_hat[:, :, :height, :width].float()
                        y_rec, uv_rec = yuv_444_to_420(x_hat)
                        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()[0]
                        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
                        p_y = calc_psnr(y[0], y_rec)
                        p_u = calc_psnr(uv[0], uv_rec[0])
                        p_v = calc_psnr(uv[1], uv_rec[1])
                        psnr_y.append(p_y)
                        psnr_yuv.append((6 * p_y + p_u + p_v) / 8)
                        per_frame.append(round(p_y, 4))
                frames = len(psnr_y)
                records.append({
                    "variant": variant["name"], "sequence": path.name, "width": width,
                    "height": height, "qp": qp, "frames": frames,
                    "bpp": bits / (frames * width * height),
                    "psnr_y": sum(psnr_y) / frames, "psnr_yuv": sum(psnr_yuv) / frames,
                    "psnr_y_per_frame": per_frame,
                })
                print(f"  {variant['name']} {path.name} qp {qp}: {records[-1]['bpp']:.4f} bpp, "
                      f"PSNR-Y {records[-1]['psnr_y']:.3f}", flush=True)
        if "resolve" in variant:
            # Layers only a reference reset or a later frame type reaches may
            # legitimately stay unused in short runs; anything else is bypassed.
            missing = [layer for layer, _ in decoder_layers(i_net, p_net)
                       if variant["resolve"](layer) is not None and layer not in called]
            if missing:
                raise SystemExit(f"{variant['name']}: emulated layers never ran: {missing[:5]} "
                                 f"({len(missing)} total)")
        print(f"{variant['name']}: {replaced} integer convs, "
              f"{time.perf_counter() - started:.1f} s", flush=True)
        del i_net, p_net
        torch.cuda.empty_cache()

    def bd_rate(anchor, test, metric):
        anchor = sorted(anchor, key=lambda item: item[metric])
        test = sorted(test, key=lambda item: item[metric])
        q1 = np.array([item[metric] for item in anchor])
        q2 = np.array([item[metric] for item in test])
        r1 = np.log(np.array([item["bpp"] for item in anchor]))
        r2 = np.log(np.array([item["bpp"] for item in test]))
        p1 = np.polyfit(q1, r1, 3)
        p2 = np.polyfit(q2, r2, 3)
        low, high = max(q1.min(), q2.min()), min(q1.max(), q2.max())
        i1 = np.polyval(np.polyint(p1), high) - np.polyval(np.polyint(p1), low)
        i2 = np.polyval(np.polyint(p2), high) - np.polyval(np.polyint(p2), low)
        return (math.exp((i2 - i1) / (high - low)) - 1) * 100

    summary = {}
    for variant in variants:
        if variant["name"] == args.anchor:
            continue
        per_sequence = {}
        for path, _, _ in sequences:
            anchor = [r for r in records if r["variant"] == args.anchor and r["sequence"] == path.name]
            test = [r for r in records if r["variant"] == variant["name"] and r["sequence"] == path.name]
            if len(anchor) < 4 or len(test) < 4:
                continue
            per_sequence[path.name] = {
                "bd_rate_yuv_percent": bd_rate(anchor, test, "psnr_yuv"),
                "bd_rate_y_percent": bd_rate(anchor, test, "psnr_y"),
            }
        if not per_sequence:
            continue
        summary[variant["name"]] = {
            "mean_bd_rate_yuv_percent": float(np.mean([v["bd_rate_yuv_percent"] for v in per_sequence.values()])),
            "mean_bd_rate_y_percent": float(np.mean([v["bd_rate_y_percent"] for v in per_sequence.values()])),
            "per_sequence": per_sequence,
        }
        print(f"{variant['name']} vs {args.anchor}: BD-rate YUV "
              f"{summary[variant['name']]['mean_bd_rate_yuv_percent']:+.3f}%, Y "
              f"{summary[variant['name']]['mean_bd_rate_y_percent']:+.3f}%", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "anchor": args.anchor, "qps": args.qps, "frames": args.frames,
            "reset_interval": args.reset_interval, "summary": summary, "records": records,
        }, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
