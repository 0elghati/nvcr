#!/usr/bin/env python3
"""Exports DCVC-UF HT-S's normative stages as nvcr.fx.v1 programs (docs/spec/numerics-fx-v1.md).

Reads the pinned cvpr2026 checkpoints with the pinned DCVC-UF model code and writes, for every stage,
<stage>.fxprog.json and <stage>.fxw, plus fx_entropy.json (integer index edges and skip threshold). The program
writer and the layer lowering are shared with DCVC-RT (../dcvc_fx.py); this file adds DCVC-UF's stages and its
entropy rule.

Normative: everything that selects an entropy table or is carried to the next chunk. The I-frame codes its latent in
four spatial steps, like DCVC-RT. An eight-frame HT-S chunk takes all its entropy indexes from one prior; its four
spatial steps only refine the means. The reference is the recurrent memory and the chunk's decoded feature, or after
a feature-memory reset the exact picture of the chunk's last frame. Pictures shown to the user stay FP16.

Entropy indexes are 255 where the element is skipped (scale at or below the skip threshold, as upstream).

usage: export_fx_programs.py --dcvcuf-root DIR --config configs/numerics/dcvcuf-fx.json --output-dir DIR
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
import dcvc_fx as fx  # noqa: E402

IMAGE_SHA256 = "b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302"
VIDEO_SHA256 = "565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47"
SKIPPED = 255


class Emitter(fx.Emitter):
    def masked(self, k, x, masks):
        """x (all latent channels) kept at step k's active elements, zero elsewhere."""
        n = self.p.channels(x) // len(masks[k])
        out = None
        for g, m in enumerate(masks[k]):
            part = self.mul(self.slice(x, g * n, (g + 1) * n, "group"), m, 0, "mask")
            out = part if out is None else self.concat(out, part, "masked")
        return out


def index_table(edges, threshold):
    """sat_table over [-n, n): the entropy index of a scale, or SKIPPED at or below the skip threshold."""
    n = edges[-1]
    s = np.arange(-n, n)
    return np.where(s > threshold, np.searchsorted(np.asarray(edges), s, side="right"), SKIPPED), n


class Steps(fx.Steps):
    def emit(self, k, x, scales, means):
        p = self.e.p
        table, _ = index_table(self.edges, self.threshold)
        p.output(p.node("sat_table", [self.gather(k, x, scales)], self.per_group, "index",
                        table=p.blob(table, np.int32), top=len(self.edges)), "indexes")
        p.output(self.gather(k, x, means), "means")


def steps(e, entropy):
    s = Steps(e, 256, 4, entropy["index_edges"])
    s.threshold = entropy["skip_threshold"]
    return s


# ---- I-frame: as DCVC-RT's, with DCVC-UF's prior and pictures in [-1/2, 1/2] ------------------------------------------
def i_prior(e, net, entropy):
    p = e.p
    z, ref = p.input("z_fixed", net.bit_estimator_z.channel), p.input("latent_ref", 1)
    s = steps(e, entropy)
    params = e.run(net.y_prior_fusion.conv, e.run(net.hyper_dec.conv, z))
    params = p.node("crop_like", [params, ref], p.channels(params), "crop")
    common = e.conv(net.y_spatial_prior_reduction, params)
    p.output(common, "common")
    s.emit(0, params, 0, 256)


def i_spatial_prior(k):
    def build(e, net, entropy):
        p = e.p
        y_hat = p.input("y_hat", 256) if k > 1 else None
        symbols, means = fx.step_inputs(p, 64)
        common = p.input("common", 256)
        s = steps(e, entropy)
        y_hat = s.update(k - 1, y_hat, symbols, means)
        p.output(y_hat, "y_hat_next")
        adaptor = getattr(net, f"y_spatial_prior_adaptor_{k}")
        s.emit(k, e.run(net.y_spatial_prior.conv, e.run(adaptor, e.concat(y_hat, common, "cat"))), 0, 256)
    return build


def i_synthesis(e, net, entropy):
    p, fa, fw = e.p, e.fa, e.fw
    y_hat = p.input("y_hat", 256)
    symbols, means = fx.step_inputs(p, 64)
    qp = p.input("qp", 1)
    y_hat = steps(e, entropy).update(3, y_hat, symbols, means)
    y = e.mul(y_hat, e.param_row(qp, net.q_scale_y_dec, "q_scale_y_dec"), fw, "mul_q_scale_y_dec")
    out = e.mul(e.run(net.dec.dec_1, y), e.param_row(qp, net.q_scale_dec, "q_scale_dec"), fw, "mul_q_scale_dec")
    out = e.shuffle(e.run(net.dec.dec_2, out), 8, "shuffle")
    p.output(e.clamp(out, -2 ** (fa - 1), 2 ** (fa - 1), "clamp"), "frame")


# ---- HT-S chunk -------------------------------------------------------------------------------------------------------
def _adapt_i(e, net, ref):
    memory = e.run(net.feature_adaptor_i.conv, ref)
    e.p.output(memory, "memory")
    e.p.output(e.run(net.feature_extractor.conv, memory), "context")


def p_reference_frame(e, net, entropy):
    """After an I-frame: memory and context from its picture."""
    _adapt_i(e, net, e.shuffle(e.p.input("frame", 3), 8, "unshuffle", inverse=True))


def p_reset_reference(e, net, entropy):
    """After a feature-memory reset: memory and context from the exact picture of the last chunk's last frame."""
    r = net.recon_head
    _adapt_i(e, net, e.run(r.conv2[-1], e.run(r.conv1[-1], e.p.input("feature", 512))))


def p_reference_memory(e, net, entropy):
    p = e.p
    memory = e.run(net.feature_adaptor_m.conv,
                   e.concat(p.input("memory", 512), p.input("feature", 512), "cat"))
    p.output(memory, "memory_next")
    p.output(e.run(net.feature_extractor.conv, memory), "context")


def p_prior(e, net, entropy):
    """The chunk's entropy indexes (all elements), the first step's means, q_dec and the reduced params."""
    p, fa, fw = e.p, e.fa, e.fw
    z, memory = p.input("z_fixed", net.bit_estimator_z.channel), p.input("memory", 512)
    qp, ref = p.input("qp", 1), p.input("latent_ref", 1)
    temporal = e.run(net.temporal_prior_encoder.conv,
                     e.mul(memory, e.param_row(qp, net.q_feature, "q_feature"), fw, "mul_q_feature"))
    hier = p.node("crop_like", [e.run(net.hyper_decoder.conv, z), ref], 256, "crop")
    params = e.run(net.y_prior_fusion.conv, e.concat(hier, temporal, "cat"))
    p.output(e.clamp(e.slice(params, 0, 256, "q_dec"), 2 ** (fa - 1), 2 ** 31 - 1, "q_dec_min"), "q_dec")
    table, _ = index_table(entropy["index_edges"], entropy["skip_threshold"])
    p.output(p.node("sat_table", [e.slice(params, 256, 512, "scales")], 256, "index",
                    table=p.blob(table, np.int32), top=len(entropy["index_edges"])), "indexes")
    p.output(e.slice(params, 512, 768, "means"), "means")
    p.output(e.conv(net.y_spatial_prior_reduction, params), "reduced")


def _masks(e):
    s = fx.Steps(e, 256, 4, [1])
    return s.masks


def _update(e, masks, k, y_hat, symbols, means):
    """y_hat + (symbol 2^fa + mean) at step k's active elements."""
    p = e.p
    v = p.node("add", [p.node("affine", [symbols], 256, "symbol", alpha=2 ** e.fa, beta=0, shift=0), means],
               256, "value")
    part = e.masked(k, v, masks)
    return part if y_hat is None else p.node("add", [y_hat, part], 256, "y_hat")


def p_spatial_prior(k):
    def build(e, net, entropy):
        p = e.p
        y_hat = p.input("y_hat", 256) if k > 1 else None
        symbols, means, reduced = p.input("symbols", 256), p.input("symbol_means", 256), p.input("reduced", 256)
        y_hat = _update(e, _masks(e), k - 1, y_hat, symbols, means)
        p.output(y_hat, "y_hat_next")
        adaptor = getattr(net, f"y_spatial_prior_adaptor_{k}")
        p.output(e.run(net.y_spatial_prior.conv, e.run(adaptor, e.concat(y_hat, reduced, "cat"))), "means")
    return build


def p_synthesis(e, net, entropy):
    """The chunk's decoded feature: the next chunk's reference and the input of its display."""
    p, fa, fw = e.p, e.fa, e.fw
    y_hat, symbols, means = p.input("y_hat", 256), p.input("symbols", 256), p.input("symbol_means", 256)
    q_dec, ctx, qp = p.input("q_dec", 256), p.input("context", 512), p.input("qp", 1)
    y_hat = _update(e, _masks(e), 3, y_hat, symbols, means)
    d = net.decoder
    feat = e.run(d.conv1, e.concat(e.run(d.up, e.mul(y_hat, q_dec, fa, "mul_q_dec")), ctx, "cat"))
    p.output(e.mul(feat, e.param_row(qp, net.q_decoder, "q_decoder"), fw, "mul_q_decoder"), "feature")


STAGES = {
    "i_prior": ("i", i_prior),
    "i_spatial_prior_1": ("i", i_spatial_prior(1)),
    "i_spatial_prior_2": ("i", i_spatial_prior(2)),
    "i_spatial_prior_3": ("i", i_spatial_prior(3)),
    "i_synthesis": ("i", i_synthesis),
    "p_reference_frame": ("p", p_reference_frame),
    "p_reset_reference": ("p", p_reset_reference),
    "p_reference_memory": ("p", p_reference_memory),
    "p_prior": ("p", p_prior),
    "p_spatial_prior_1": ("p", p_spatial_prior(1)),
    "p_spatial_prior_2": ("p", p_spatial_prior(2)),
    "p_spatial_prior_3": ("p", p_spatial_prior(3)),
    "p_synthesis": ("p", p_synthesis),
}


def load(path: Path, expected: str, net):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {expected}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    net.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()})
    return net.eval().float()


def entropy_assets(net, cfg):
    """Index edges at 2^-fa (index k <=> scale >= edge k) and the integer skip threshold (skipped: scale <= it)."""
    g = net.gaussian_encoder
    steps_ = np.arange(1, g.scale_level)
    s_k = np.exp(math.log(g.scale_min) + steps_ * (math.log(g.scale_max) - math.log(g.scale_min)) / (g.scale_level - 1))
    return {"format": "nvcr.dcvcuf-fx-entropy.v1", "fa": cfg["fa"],
            "index_edges": [int(v) for v in np.ceil(s_k * 2 ** cfg["fa"])],
            "skip_threshold": int(math.floor(cfg["skip_threshold"] * 2 ** cfg["fa"])), "skipped_index": SKIPPED}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dcvcuf-root", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, help="default: <dcvcuf-root>/checkpoints")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    sys.path.insert(0, str(args.dcvcuf_root))
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, ModelStructure

    ck = args.checkpoints or args.dcvcuf_root / "checkpoints"
    nets = {"i": load(ck / "cvpr2026_image.pth.tar", IMAGE_SHA256, DMCI()),
            "p": load(ck / "cvpr2026_video_hts.pth.tar", VIDEO_SHA256, DMC(ModelStructure.HTS))}
    names = {id(m): f"{tag}.{n}" for tag, net in nets.items() for n, m in net.named_modules()}
    delta = fx.wsilu_delta_table(cfg["fa"], cfg["wsilu_range"], cfg["wsilu_shift"])
    entropy = entropy_assets(nets["p"], cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stage in args.stages:
            tag, build = STAGES[stage]
            program = fx.Program(stage, cfg["fa"])
            build(Emitter(program, cfg, names, cfg["input_shifts"], delta), nets[tag], entropy)
            program.write(args.output_dir)
            print(f"{stage}: {len(program.nodes)} nodes, {len(program.data)} bytes")
    (args.output_dir / "fx_entropy.json").write_text(json.dumps(entropy, indent=1) + "\n")


if __name__ == "__main__":
    main()
