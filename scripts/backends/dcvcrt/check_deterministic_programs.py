#!/usr/bin/env python3
"""Check lowered DCVC-RT deterministic programs against the pinned PyTorch modules.

Each program (<stage>.detprog.json + .detw) is interpreted with PyTorch FP32
operations and compared with the pinned DCVC-RT module it was exported from,
run in FP32 with the same FP16-rounded weights. Arithmetic order differs
between the two, so outputs agree to FP32 rounding, not bit for bit; a lowering
error (wrong channel order, crop, concat, or scale) shows as a large
difference. Bit-exactness across executors is tested separately by
nvcr_deterministic_executor.

  python3 check_deterministic_programs.py --dcvcrt-root DCVC-RT \\
      --program-dir build/models/dcvcrt/normative i_synthesis p_prior ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def interpret(program: dict, weights: "torch.Tensor", inputs: dict) -> list:
    import torch
    import torch.nn.functional as F

    tensors = dict(inputs)
    for node in program["nodes"]:
        op = node["op"]
        if op == "conv":
            groups = node["in_channels"] if node["depthwise"] else 1
            per_out = (1 if node["depthwise"] else node["in_channels"]) * node["kernel"][0] * node["kernel"][1]
            count = node["out_channels"] * per_out
            w = weights[node["weight_offset"]:node["weight_offset"] + count].reshape(
                node["out_channels"], per_out // (node["kernel"][0] * node["kernel"][1]), *node["kernel"])
            b = weights[node["bias_offset"]:node["bias_offset"] + node["out_channels"]]
            top, left, bottom, right = node["pads"]
            x = F.pad(tensors[node["input"]], (left, right, top, bottom))
            out = F.conv2d(x, w, b, stride=node["stride"], groups=groups)
        elif op in ("mul", "add"):
            a, b = (tensors[name] for name in node["inputs"])
            out = a * b if op == "mul" else a + b
        elif op == "scale_channels":
            a, b = (tensors[name] for name in node["inputs"])
            out = a * b
        elif op == "mul_scalar":
            out = tensors[node["input"]] * node["value"]
        elif op == "sigmoid":
            out = torch.sigmoid(tensors[node["input"]])
        elif op == "clamp":
            out = torch.clamp(tensors[node["input"]], node["min"], node["max"])
        elif op == "slice_channels":
            out = tensors[node["input"]][:, node["begin"]:node["end"]]
        elif op == "concat":
            out = torch.cat([tensors[name] for name in node["inputs"]], dim=1)
        elif op == "crop":
            source, reference = (tensors[name] for name in node["inputs"])
            out = (source[:, :, :reference.shape[2]] if node["axis"] == 2
                   else source[:, :, :, :reference.shape[3]])
        elif op == "depth_to_space":
            out = F.pixel_shuffle(tensors[node["input"]], node["block"])
        elif op == "space_to_depth":
            out = F.pixel_unshuffle(tensors[node["input"]], node["block"])
        elif op == "im2col":
            top, left, bottom, right = node["pads"]
            x = F.pad(tensors[node["input"]], (left, right, top, bottom))
            columns = F.unfold(x, kernel_size=node["kernel"], stride=node["stride"])
            height = (x.shape[2] - node["kernel"][0]) // node["stride"][0] + 1
            out = columns.reshape(1, -1, height, columns.shape[-1] // height)
        else:
            raise SystemExit(f"unknown op {op}")
        tensors[node["output"]] = out.contiguous()
    return [tensors[item["name"]] for item in program["outputs"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcvcrt-root", type=Path, required=True)
    parser.add_argument("--program-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs=2, default=(2, 3),
                        help="frame size in 64x64 blocks (height, width)")
    parser.add_argument("--qp", type=int, default=21)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="maximum |difference| relative to the reference output's max |value|")
    parser.add_argument("stages", nargs="+")
    args = parser.parse_args()

    sys.path.insert(0, str(args.dcvcrt_root.resolve()))
    import torch
    import torch.nn.functional as F
    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.utils.common import get_state_dict

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    i_net = DMCI()
    i_net.load_state_dict(get_state_dict(str(args.dcvcrt_root / "checkpoints/cvpr2025_image.pth.tar")))
    p_net = DMC()
    p_net.load_state_dict(get_state_dict(str(args.dcvcrt_root / "checkpoints/cvpr2025_video.pth.tar")))
    for net in (i_net, p_net):
        net.half().float().to(device).eval()

    bh, bw = args.blocks
    qp = args.qp

    def rand(channels, height, width, scale=1.0):
        return (torch.randn(1, channels, height, width, device=device) * scale).half().float()

    def q(table, index):
        return table[index:index + 1].detach().half().float()

    frame = torch.rand(1, 3, 64 * bh, 64 * bw, device=device).half().float()
    feature = rand(256, 8 * bh, 8 * bw)
    y_hat = torch.round(rand(128, 4 * bh, 4 * bw, 3.0))
    q_feature = q(p_net.q_feature, qp)
    q_decoder = q(p_net.q_decoder, qp)
    q_recon = q(p_net.q_recon, qp)
    with torch.no_grad():
        ctx, ctx_t = p_net.feature_extractor(p_net.feature_adaptor_p(feature), q_feature)
    ctx = ctx.half().float()
    ctx_t = ctx_t.half().float()
    z_hat = torch.round(rand(128, bh, bw, 3.0))

    def prior(z, temporal):
        hierarchical = p_net.hyper_decoder(z)
        params = p_net.temporal_prior_encoder(temporal)
        height, width = params.shape[2:]
        return p_net.y_prior_fusion(torch.cat((hierarchical[:, :, :height, :width], params), dim=1))

    def synthesis(y, context, q_dec, q_rec):
        decoded = p_net.decoder(y, context, q_dec)
        return p_net.recon_generation_net(decoded, q_rec), decoded

    def hyper_synthesis(z):
        params = i_net.y_prior_fusion(i_net.hyper_dec(z))
        return [params, i_net.y_spatial_prior_reduction(params)]

    i_context = rand(512, 4 * bh, 4 * bw)
    cases = {
        "i_hyper_synthesis": (
            {"z_hat": torch.round(rand(128, bh, bw, 3.0))},
            lambda t: hyper_synthesis(t["z_hat"])),
        "i_spatial_prior_1": (
            {"context": i_context},
            lambda t: [i_net.y_spatial_prior(i_net.y_spatial_prior_adaptor_1(t["context"]))]),
        "i_spatial_prior_2": (
            {"context": i_context},
            lambda t: [i_net.y_spatial_prior(i_net.y_spatial_prior_adaptor_2(t["context"]))]),
        "i_spatial_prior_3": (
            {"context": i_context},
            lambda t: [i_net.y_spatial_prior(i_net.y_spatial_prior_adaptor_3(t["context"]))]),
        "i_synthesis": (
            {"y_hat": torch.round(rand(256, 4 * bh, 4 * bw, 3.0)), "q_dec": q(i_net.q_scale_dec, qp)},
            lambda t: [i_net.dec(t["y_hat"], t["q_dec"])]),
        "p_reference_frame": (
            {"reference_frame": frame, "q_feature": q_feature},
            lambda t: list(p_net.feature_extractor(
                p_net.feature_adaptor_i(F.pixel_unshuffle(t["reference_frame"], 8)), t["q_feature"]))),
        "p_reference_feature": (
            {"reference_feature": feature, "q_feature": q_feature},
            lambda t: list(p_net.feature_extractor(p_net.feature_adaptor_p(t["reference_feature"]),
                                                   t["q_feature"]))),
        "p_prior": (
            {"z_hat": z_hat, "temporal_context": ctx_t},
            lambda t: [prior(t["z_hat"], t["temporal_context"])]),
        "p_spatial_prior": (
            {"context": rand(512, 4 * bh, 4 * bw)},
            lambda t: [p_net.y_spatial_prior(t["context"])]),
        "p_synthesis": (
            {"y_hat": y_hat, "context": ctx, "q_decoder": q_decoder, "q_recon": q_recon},
            lambda t: list(synthesis(t["y_hat"], t["context"], t["q_decoder"], t["q_recon"]))),
    }

    failed = False
    for stage in args.stages:
        if stage not in cases:
            print(f"{stage}: no reference module")
            failed = True
            continue
        program = json.loads((args.program_dir / f"{stage}.detprog.json").read_text(encoding="utf-8"))
        blob = np.fromfile(args.program_dir / program["weights"]["file"], dtype="<f2")
        weights = torch.from_numpy(blob.astype(np.float32)).to(device)
        inputs, reference = cases[stage]
        ordered = {item["name"]: inputs[item["name"]] for item in program["inputs"]}
        with torch.no_grad():
            got = interpret(program, weights, ordered)
            expected = reference(ordered)
        for item, actual, wanted in zip(program["outputs"], got, expected):
            if actual.shape != wanted.shape:
                print(f"{stage}.{item['name']}: shape {tuple(actual.shape)} != {tuple(wanted.shape)}")
                failed = True
                continue
            scale = float(wanted.abs().max()) or 1.0
            error = float((actual - wanted).abs().max()) / scale
            status = "ok" if error <= args.tolerance else "MISMATCH"
            failed |= status != "ok"
            print(f"{stage}.{item['name']}: {tuple(actual.shape)} relative max error {error:.2e} {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
