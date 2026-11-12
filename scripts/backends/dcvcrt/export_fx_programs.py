#!/usr/bin/env python3
"""Exports DCVC-RT's normative stages as nvcr.fx.v1 programs (docs/spec/numerics-fx-v1.md).

Reads the pinned cvpr2025 checkpoints with the pinned DCVC-RT model code and writes, for every stage,
<stage>.fxprog.json and <stage>.fxw, plus fx_entropy.json (integer index edges and skip threshold).

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
from torch import nn

D2 = (256 ** 2 - 1) * 127 // 255  # 32639: two balanced int8 digits
IMAGE_SHA256 = "555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9"
VIDEO_SHA256 = "b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6"


# ---- tables ----------------------------------------------------------------------------------------------------------
def sigmoid_table(fa, lut_range):
    """floor(sigmoid(x) 2^fa + 1/2) on [-lut_range, lut_range) 2^fa."""
    n = lut_range << fa
    x = torch.arange(-n, n, dtype=torch.float64) / 2 ** fa
    return torch.floor(torch.sigmoid(x) * 2 ** fa + 0.5).to(torch.int64)


def wsilu_delta_table(fa, lut_range, c):
    """WSiLU(x) = x sigmoid(4x) as max(x, 0) + dc[(x + h) >> c] on [-h, h]: (dc, h)."""
    n = lut_range << fa
    x = torch.arange(-n, n, dtype=torch.float64) / 2 ** fa
    lut = torch.floor(x * torch.sigmoid(4 * x) * 2 ** fa + 0.5).to(torch.int64)
    xi = torch.arange(-n, n)
    d = lut - xi.clamp_min(0)
    nz = d.nonzero().flatten()
    h = int(max(-xi[nz.min()].item(), xi[nz.max()].item()))
    nb = (2 * h >> c) + 1
    centre = torch.clamp(-h + torch.arange(nb) * 2 ** c + (2 ** c >> 1), max=h)
    dc = d[centre + n]
    assert int(dc.abs().max()) < 2 ** 15
    return dc.to(torch.int16), h


# ---- program writer --------------------------------------------------------------------------------------------------
class Program:
    def __init__(self, name, fa):
        self.name, self.fa = name, fa
        self.tensors, self.nodes, self.inputs, self.outputs, self.tables = [], [], [], [], []
        self.data = bytearray()

    def tensor(self, name, channels):
        self.tensors.append({"name": name, "channels": int(channels)})
        return len(self.tensors) - 1

    def input(self, name, channels):
        t = self.tensor(name, channels)
        self.inputs.append(t)
        return t

    def output(self, t, name):
        self.tensors[t]["name"] = name
        self.outputs.append(t)

    def blob(self, values, dtype):
        a = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1))
        offset = len(self.data)
        self.data += a.astype(np.dtype(dtype).newbyteorder("<")).tobytes()
        return [offset, int(a.size)]

    def node(self, op, inputs, channels, label, **fields):
        out = self.tensor(f"{label}#{len(self.nodes)}", channels)
        self.nodes.append({"op": op, "inputs": list(inputs), "output": out, **fields})
        return out

    def channels(self, t):
        return self.tensors[t]["channels"]

    def write(self, directory: Path):
        data_name = f"{self.name}.fxw"
        (directory / data_name).write_bytes(bytes(self.data))
        doc = {"format": "nvcr.fx-program.v1", "profile": "nvcr.fx.v1", "name": self.name, "fa": self.fa,
               "data": {"file": data_name, "sha256": hashlib.sha256(self.data).hexdigest()},
               "tensors": self.tensors, "inputs": self.inputs, "outputs": self.outputs,
               "delta_tables": self.tables, "nodes": self.nodes}
        (directory / f"{self.name}.fxprog.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")


# ---- DCVC-RT modules as fx nodes -------------------------------------------------------------------------------------
class Emitter:
    """Emits one program's nodes for DCVC-RT modules, with the exact design's integer parameters."""

    def __init__(self, program, cfg, names, input_shifts, delta):
        self.p, self.cfg, self.names, self.k = program, cfg, names, input_shifts
        self.fa, self.fw = cfg["fa"], cfg["fw"]
        self.delta, self.delta_index = delta, None

    def _delta(self):
        if self.delta_index is None:
            dc, h = self.delta
            self.p.tables.append({"h": h, "shift": self.cfg["wsilu_shift"], "dc": self.p.blob(dc.numpy(), np.int16)})
            self.delta_index = len(self.p.tables) - 1
        return self.delta_index

    def _row_shifts(self, w):
        """Per-output-channel weight exponent: the largest k with max|w_c| 2^k <= D2 (capped at 30)."""
        out = []
        for m in w.abs().flatten(1).amax(1).tolist():
            if m == 0.0:
                out.append(self.fw)
                continue
            k = -math.frexp(m)[1] + int(math.log2(D2)) + 2
            while math.ldexp(m, k) > D2:
                k -= 1
            out.append(min(k, 30))
        return torch.tensor(out, dtype=torch.int64)

    def params(self, conv):
        """Integer weights, bias and output shift of a conv; the layer's GEMM input shift lowers the shift."""
        wf = conv.weight.detach().double()
        s = self._row_shifts(wf)
        w = torch.floor(torch.ldexp(wf, s.view((-1,) + (1,) * (wf.dim() - 1)).to(torch.int32)) + 0.5).to(torch.int64)
        k = self.k.get(self.names[id(conv)], 0)
        assert k == 0 or bool((s > k).all()), self.names[id(conv)]
        s = s - k
        bias = torch.zeros(wf.shape[0], dtype=torch.float64) if conv.bias is None else conv.bias.detach().double()
        b = torch.floor(torch.ldexp(bias, (s + self.fa).to(torch.int32)) + 0.5).to(torch.int64)
        return w, b, s, k

    def gemm(self, conv, x, wmat=None, lut=False, group=1, res=(), perm=False):
        w, b, s, k = self.params(conv)
        wmat = w.view(w.shape[0], -1) if wmat is None else wmat
        if perm:  # put the two halves summed by DCVC-RT's chunk-add on adjacent rows
            o = wmat.shape[0]
            idx = torch.stack((torch.arange(o // 2), torch.arange(o // 2, o)), 1).flatten()
            wmat, b, s = wmat[idx], b[idx], s[idx]
        assert int(wmat.abs().max()) <= D2 and int(s.min()) >= 1 and int(s.max()) <= 31, self.names[id(conv)]
        o, kk = wmat.shape
        p = self.p
        return p.node("gemm", [x, *res], o // group, self.names[id(conv)], in_channels=int(kk), out_channels=int(o),
                      group=group, input_shift=int(k), delta_table=self._delta() if lut else -1,
                      weights=p.blob(wmat.numpy(), np.int16), bias=p.blob(b.numpy(), np.int64),
                      shift=p.blob(s.numpy(), np.int32))

    def depthwise(self, conv, x):
        w, b, s, _ = self.params(conv)
        p = self.p
        return p.node("depthwise3x3", [x], p.channels(x), self.names[id(conv)],
                      weights=p.blob(w.reshape(-1).numpy(), np.int32), bias=p.blob(b.numpy(), np.int64),
                      shift=p.blob(s.numpy(), np.int32))

    def conv(self, conv, x, **kw):
        p = self.p
        if conv.kernel_size == (2, 2) and conv.stride == (2, 2) and conv.padding == (0, 0):
            x = p.node("pixel_unshuffle", [x], p.channels(x) * 4, self.names[id(conv)], factor=2)
            return self.gemm(conv, x, **kw)
        assert conv.stride == (1, 1), self.names[id(conv)]
        if conv.kernel_size == (1, 1):
            return self.gemm(conv, x, **kw)
        assert conv.kernel_size == (3, 3) and conv.padding == (1, 1), self.names[id(conv)]
        if conv.groups == conv.in_channels == conv.out_channels:
            assert not kw
            return self.depthwise(conv, x)
        assert conv.groups == 1
        cols = p.node("im2col", [x], p.channels(x) * 9, self.names[id(conv)], kernel=3, stride=1)
        return self.gemm(conv, cols, **kw)

    def wsilu(self, x, label, chunk_add=False):
        p = self.p
        y = p.node("delta", [x], p.channels(x), label, delta_table=self._delta())
        if not chunk_add:
            return y
        half = p.channels(x) // 2
        a = p.node("slice", [y], half, label, begin=0, end=half)
        b = p.node("slice", [y], half, label, begin=half, end=2 * half)
        return p.node("add", [a, b], half, label)

    def dcb(self, m, x):
        """DepthConvBlock: 1x1 + WSiLU, depthwise, 1x1 + residual, 1x1 + WSiLU chunk-add, 1x1 + residual(s)."""
        if m.adaptor is not None:
            x = self.conv(m.adaptor, x)
        t = self.conv(m.dc[0], x, lut=True)
        t = self.depthwise(m.dc[2], t)
        out = self.conv(m.dc[3], t, res=(x,))
        t = self.conv(m.ffn[0], out, lut=True, group=2, perm=True)
        return self.conv(m.ffn[2], t, res=(out, x) if m.shortcut else (out,))

    def run(self, m, x):
        from src.layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2, \
            SubpelConv2x, WSiLU, WSiLUChunkAdd
        p = self.p
        if isinstance(m, nn.Sequential):
            for sub in m:
                x = self.run(sub, x)
            return x
        if isinstance(m, nn.Conv2d):
            return self.conv(m, x)
        if isinstance(m, WSiLUChunkAdd):
            return self.wsilu(x, self.names.get(id(m), "chunk_add"), chunk_add=True)
        if isinstance(m, WSiLU):
            return self.wsilu(x, self.names.get(id(m), "wsilu"))
        if isinstance(m, nn.PixelShuffle):
            r = m.upscale_factor
            return p.node("pixel_shuffle", [x], p.channels(x) // (r * r), "shuffle", factor=r)
        if isinstance(m, SubpelConv2x):
            return self.run(m.conv, x)
        if isinstance(m, ResidualBlockUpsample):
            return self.run(m.conv, self.run(m.up, x))
        if isinstance(m, ResidualBlockWithStride2):
            return self.run(m.conv, self.conv(m.down, x))
        if isinstance(m, DepthConvBlock):
            return self.dcb(m, x)
        raise TypeError(f"no fx lowering for {type(m).__name__}")

    # ---- small ops ----
    def param_row(self, qp, table, label):
        """Per-QP channel scales at 2^-fw, as stored in the checkpoint (one row per QP)."""
        rows = torch.floor(table.detach().double().reshape(table.shape[0], -1) * 2 ** self.fw + 0.5).to(torch.int64)
        return self.p.node("param_row", [qp], rows.shape[1], label, rows=self.p.blob(rows.numpy(), np.int64))

    def mul(self, a, b, shift, label):
        return self.p.node("mul", [a, b], self.p.channels(a), label, shift=shift)

    def clamp(self, x, lo, hi, label):
        return self.p.node("clamp", [x], self.p.channels(x), label, lo=lo, hi=hi)

    def shuffle(self, x, r, label, inverse=False):
        p = self.p
        ch = p.channels(x) * r * r if inverse else p.channels(x) // (r * r)
        return p.node("pixel_unshuffle" if inverse else "pixel_shuffle", [x], ch, label, factor=r)

    def concat(self, a, b, label):
        return self.p.node("concat", [a, b], self.p.channels(a) + self.p.channels(b), label)

    def slice(self, x, begin, end, label):
        return self.p.node("slice", [x], end - begin, label, begin=begin, end=end)


# ---- stages ----------------------------------------------------------------------------------------------------------
# I-frame: z_fixed = z_int 2^fa; latent_ref only gives the latent size; params = [q_enc | q_dec | scales | means].
def i_prior(e, net):
    p = e.p
    z, ref = p.input("z_fixed", 128), p.input("latent_ref", 1)
    params = e.run(net.y_prior_fusion, e.run(net.hyper_dec, z))
    params = p.node("crop_like", [params, ref], p.channels(params), "crop")
    common = e.conv(net.y_spatial_prior_reduction, params)
    p.output(params, "params")
    p.output(common, "common")


def i_spatial_prior(k):
    def build(e, net):
        p = e.p
        y_hat, common = p.input("y_hat", 256), p.input("common", 256)
        adaptor = getattr(net, f"y_spatial_prior_adaptor_{k}")
        q = e.run(adaptor, e.concat(y_hat, common, "cat"))
        p.output(e.run(net.y_spatial_prior, q), "scales_means")
    return build


def i_synthesis(e, net):
    p, fa = e.p, e.fa
    y_hat, params, qp = p.input("y_hat", 256), p.input("params", 514), p.input("qp", 1)
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


def p_reference_frame(e, net):
    frame, qp = e.p.input("frame", 3), e.p.input("qp", 1)
    _extract(e, net, e.run(net.feature_adaptor_i, e.shuffle(frame, 8, "unshuffle", inverse=True)), qp)


def p_reference_feature(e, net):
    feature, qp = e.p.input("feature", 256), e.p.input("qp", 1)
    _extract(e, net, e.conv(net.feature_adaptor_p, feature), qp)


def p_prior(e, net):
    p = e.p
    z, ctx_t, ref = p.input("z_fixed", 128), p.input("temporal_context", 256), p.input("latent_ref", 1)
    hier = e.run(net.hyper_decoder.conv, z)
    hier = p.node("crop_like", [hier, ref], p.channels(hier), "crop")
    temporal = e.run(net.temporal_prior_encoder, ctx_t)
    p.output(e.run(net.y_prior_fusion.conv, e.concat(hier, temporal, "cat")), "params")


def p_spatial_prior(e, net):
    y_hat, params = e.p.input("y_hat", 128), e.p.input("params", 384)
    e.p.output(e.run(net.y_spatial_prior.conv, e.concat(y_hat, params, "cat")), "scales_means")


def p_synthesis(e, net):
    p, fa = e.p, e.fa
    y_hat, params, ctx, qp = p.input("y_hat", 128), p.input("params", 384), p.input("context", 256), p.input("qp", 1)
    q_dec = e.clamp(e.slice(params, 0, 128, "q_dec"), 2 ** (fa - 1), 2 ** 31 - 1, "q_dec_min")  # max(q, 0.5)
    d = net.decoder
    feat = e.run(d.up, e.mul(y_hat, q_dec, fa, "mul_q_dec"))
    feat = e.run(d.conv1, e.concat(feat, ctx, "cat"))
    feat = e.mul(e.conv(d.conv2, feat), e.param_row(qp, net.q_decoder, "q_decoder"), e.fw, "mul_q_decoder")
    p.output(feat, "feature")


def p_reset_frame(e, net):
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for stage in args.stages:
            tag, build = STAGES[stage]
            program = Program(stage, cfg["fa"])
            build(Emitter(program, cfg, names, cfg["input_shifts"], delta), nets[tag])
            program.write(args.output_dir)
            print(f"{stage}: {len(program.nodes)} nodes, {len(program.data)} bytes")
    entropy = entropy_assets(nets["p"], cfg)
    (args.output_dir / "fx_entropy.json").write_text(json.dumps(entropy, indent=1) + "\n")


if __name__ == "__main__":
    main()
