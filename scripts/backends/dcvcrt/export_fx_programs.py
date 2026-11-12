#!/usr/bin/env python3
"""Exports DCVC-RT's normative stages as nvcr.fx.v1 programs (docs/spec/numerics-fx-v1.md).

Reads the pinned cvpr2025 checkpoints with the pinned DCVC-RT model code and writes, for every stage,
<stage>.fxprog.json and <stage>.fxw, plus fx_entropy.json (integer index edges and skip threshold).

The programs also carry DCVC-RT's spatial coding steps, so a provider runs them next to the networks: per step the
active element of each channel group is gathered (upstream's two- and four-way masks, from a `parity` input), its
scale gives the entropy index through the index edges, and after entropy coding symbol 2^fa + mean goes back into
y_hat. The coder exchanges only indexes, means and symbols with the provider.

The quantization rules (per-channel two-digit weights, per-layer GEMM input shifts, the coarse WSiLU table and the
sigmoid table) follow the exact DCVC-RT design they come from, so the programs compute the same integers. All
numbers are derived on the CPU in float64 with exact operations, so every machine writes identical files.

usage: export_fx_programs.py --dcvcrt-root DIR --config configs/numerics/dcvcrt-fx.json --output-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcvc_fx import Emitter, Program, Steps, sigmoid_table, step_inputs, wsilu_delta_table  # noqa: E402

IMAGE_SHA256 = "555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9"
VIDEO_SHA256 = "b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6"


# ---- stages ----------------------------------------------------------------------------------------------------------
# I-frame: z_fixed = z_int 2^fa; latent_ref only gives the latent size; params = [q_enc | q_dec | scales | means].
def i_prior(e, net, edges):
    p = e.p
    z, ref = p.input("z_fixed", 128), p.input("latent_ref", 1)
    steps = Steps(e, 256, 4, edges)
    params = e.run(net.y_prior_fusion, e.run(net.hyper_dec, z))
    params = p.node("crop_like", [params, ref], p.channels(params), "crop")
    common = e.conv(net.y_spatial_prior_reduction, params)
    p.output(params, "params")
    p.output(common, "common")
    steps.emit(0, params, 2, 258)
    p.output(e.slice(params, 0, 1, "q"), "q")  # q_enc before its sigmoid, for the encoder


def i_spatial_prior(k):
    def build(e, net, edges):
        p = e.p
        y_hat = p.input("y_hat", 256) if k > 1 else None
        symbols, means = step_inputs(p, 64)
        common = p.input("common", 256)
        steps = Steps(e, 256, 4, edges)
        y_hat = steps.update(k - 1, y_hat, symbols, means)
        p.output(y_hat, "y_hat_next")
        adaptor = getattr(net, f"y_spatial_prior_adaptor_{k}")
        q = e.run(adaptor, e.concat(y_hat, common, "cat"))
        steps.emit(k, e.run(net.y_spatial_prior, q), 0, 256)
    return build


def i_synthesis(e, net, edges):
    p, fa = e.p, e.fa
    y_hat = p.input("y_hat", 256)
    symbols, means = step_inputs(p, 64)
    params, qp = p.input("params", 514), p.input("qp", 1)
    y_hat = Steps(e, 256, 4, edges).update(3, y_hat, symbols, means)
    sig = sigmoid_table(fa, e.cfg["sigmoid_range"])
    s = p.node("sat_table", [e.slice(params, 1, 2, "q_dec_raw")], 1, "sigmoid",
               table=p.blob(sig.numpy(), np.int32), top=2 ** fa)
    q_dec = p.node("affine", [s], 1, "q_dec", alpha=3, beta=2 ** fa, shift=1)  # 1.5 sigmoid + 0.5
    out = e.run(net.dec.dec_1, e.mul(y_hat, q_dec, fa, "mul_q_dec"))
    out = e.mul(out, e.param_row(qp, net.q_scale_dec, "q_scale_dec"), e.fw, "mul_q_scale_dec")
    out = e.run(net.dec.dec_2, out)
    p.output(e.clamp(e.shuffle(out, 8, "shuffle"), 0, 2 ** fa, "clamp"), "frame")


# P-frame.
def _extract(e, net, f, qp):
    x1 = e.run(net.feature_extractor.conv1, f)
    ctx_t = e.mul(x1, e.param_row(qp, net.q_feature, "q_feature"), e.fw, "mul_q_feature")
    e.p.output(ctx_t, "temporal_context")
    e.p.output(e.run(net.feature_extractor.conv2, x1), "context")


def p_reference_frame(e, net, edges):
    frame, qp = e.p.input("frame", 3), e.p.input("qp", 1)
    _extract(e, net, e.run(net.feature_adaptor_i, e.shuffle(frame, 8, "unshuffle", inverse=True)), qp)


def p_reference_feature(e, net, edges):
    feature, qp = e.p.input("feature", 256), e.p.input("qp", 1)
    _extract(e, net, e.conv(net.feature_adaptor_p, feature), qp)


def p_prior(e, net, edges):
    p = e.p
    z, ctx_t, ref = p.input("z_fixed", 128), p.input("temporal_context", 256), p.input("latent_ref", 1)
    steps = Steps(e, 128, 2, edges)
    hier = e.run(net.hyper_decoder.conv, z)
    hier = p.node("crop_like", [hier, ref], p.channels(hier), "crop")
    temporal = e.run(net.temporal_prior_encoder, ctx_t)
    params = e.run(net.y_prior_fusion.conv, e.concat(hier, temporal, "cat"))
    p.output(params, "params")
    steps.emit(0, params, 128, 256)
    p.output(e.slice(params, 0, 128, "q"), "q")  # the quantization step, for the encoder


def p_spatial_prior(e, net, edges):
    p = e.p
    symbols, means = step_inputs(p, 64)
    params = p.input("params", 384)
    steps = Steps(e, 128, 2, edges)
    y_hat = steps.update(0, None, symbols, means)
    p.output(y_hat, "y_hat_next")
    steps.emit(1, e.run(net.y_spatial_prior.conv, e.concat(y_hat, params, "cat")), 0, 128)


def p_synthesis(e, net, edges):
    p, fa = e.p, e.fa
    y_hat = p.input("y_hat", 128)
    symbols, means = step_inputs(p, 64)
    params, ctx, qp = p.input("params", 384), p.input("context", 256), p.input("qp", 1)
    y_hat = Steps(e, 128, 2, edges).update(1, y_hat, symbols, means)
    q_dec = e.clamp(e.slice(params, 0, 128, "q_dec"), 2 ** (fa - 1), 2 ** 31 - 1, "q_dec_min")  # max(q, 0.5)
    d = net.decoder
    feat = e.run(d.up, e.mul(y_hat, q_dec, fa, "mul_q_dec"))
    feat = e.run(d.conv1, e.concat(feat, ctx, "cat"))
    feat = e.mul(e.conv(d.conv2, feat), e.param_row(qp, net.q_decoder, "q_decoder"), e.fw, "mul_q_decoder")
    p.output(feat, "feature")


def p_reset_frame(e, net, edges):
    p, fa = e.p, e.fa
    feature, qp = p.input("feature", 256), p.input("qp", 1)
    r = net.recon_generation_net
    out = e.mul(e.run(r.conv, feature), e.param_row(qp, net.q_recon, "q_recon"), e.fw, "mul_q_recon")
    p.output(e.clamp(e.shuffle(e.conv(r.head, out), 8, "shuffle"), 0, 2 ** fa, "clamp"), "frame")


STAGES = {
    "i_prior": ("i", i_prior),
    "i_spatial_prior_1": ("i", i_spatial_prior(1)),
    "i_spatial_prior_2": ("i", i_spatial_prior(2)),
    "i_spatial_prior_3": ("i", i_spatial_prior(3)),
    "i_synthesis": ("i", i_synthesis),
    "p_reference_frame": ("p", p_reference_frame),
    "p_reference_feature": ("p", p_reference_feature),
    "p_prior": ("p", p_prior),
    "p_spatial_prior": ("p", p_spatial_prior),
    "p_synthesis": ("p", p_synthesis),
    "p_reset_frame": ("p", p_reset_frame),
}


# ---- entry point -----------------------------------------------------------------------------------------------------
def load(root: Path, name: str, expected: str, cls):
    path = root / "checkpoints" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {expected}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    net = cls()
    net.load_state_dict(state)
    return net.eval().float()


def entropy_assets(net, cfg):
    """Index edges at 2^-fa (index k <=> scale >= edge k) and the integer skip threshold."""
    g = net.gaussian_encoder
    steps = np.arange(1, g.scale_level)
    s_k = np.exp(math.log(g.scale_min) + steps * (math.log(g.scale_max) - math.log(g.scale_min)) / (g.scale_level - 1))
    return {"format": "nvcr.dcvcrt-fx-entropy.v1", "fa": cfg["fa"],
            "index_edges": [int(v) for v in np.ceil(s_k * 2 ** cfg["fa"])],
            "skip_threshold": int(math.floor(cfg["skip_threshold"] * 2 ** cfg["fa"])) if cfg["skip_threshold"] > 0 else -1}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dcvcrt-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    sys.path.insert(0, str(args.dcvcrt_root))
    from src.models.image_model import DMCI
    from src.models.video_model import DMC

    nets = {"i": load(args.dcvcrt_root, "cvpr2025_image.pth.tar", IMAGE_SHA256, DMCI),
            "p": load(args.dcvcrt_root, "cvpr2025_video.pth.tar", VIDEO_SHA256, DMC)}
    names = {id(m): f"{tag}.{n}" for tag, net in nets.items() for n, m in net.named_modules()}
    delta = wsilu_delta_table(cfg["fa"], cfg["wsilu_range"], cfg["wsilu_shift"])
    entropy = entropy_assets(nets["p"], cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stage in args.stages:
            tag, build = STAGES[stage]
            program = Program(stage, cfg["fa"])
            build(Emitter(program, cfg, names, cfg["input_shifts"], delta), nets[tag], entropy["index_edges"])
            program.write(args.output_dir)
            print(f"{stage}: {len(program.nodes)} nodes, {len(program.data)} bytes")
    (args.output_dir / "fx_entropy.json").write_text(json.dumps(entropy, indent=1) + "\n")


if __name__ == "__main__":
    main()
