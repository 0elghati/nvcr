"""nvcr.fx.v1 program writer and layer lowering shared by the DCVC-family exporters (docs/spec/numerics-fx-v1.md).

DCVC-RT and DCVC-UF build their networks from the same upstream layers (DepthConvBlock, WSiLU, sub-pixel and stride-2
blocks); this module writes fx programs and lowers those layers with the exact design's integer parameters. Each
codec's exporter (scripts/backends/<codec>/export_fx_programs.py) adds its stages.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

D2 = (256 ** 2 - 1) * 127 // 255  # 32639: two balanced int8 digits


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


# ---- spatial coding steps --------------------------------------------------------------------------------------------
FOUR_WAY = ((0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2))  # [step][group] -> parity 2 (row % 2) + col % 2
TWO_WAY = ((0, 1), (1, 0))  # [step][group] -> 0: row % 2 == col % 2, 1: they differ


class Steps:
    """DCVC-RT's spatial steps over `groups` channel groups of `channels` latent channels. The reduced tensors
    (indexes, means, symbols) hold, for each channel of a group and each position, the group's active element."""

    def __init__(self, e, channels, groups, edges):
        p = e.p
        self.e, self.per_group, self.edges = e, channels // groups, edges
        parity = p.input("parity", 4)
        one = [e.slice(parity, i, i + 1, "parity") for i in range(4)]
        if groups == 4:
            self.masks = [[one[FOUR_WAY[k][g]] for g in range(4)] for k in range(4)]
        else:
            same = p.node("add", [one[0], one[3]], 1, "parity")
            differ = p.node("add", [one[1], one[2]], 1, "parity")
            self.masks = [[(same, differ)[TWO_WAY[k][g]] for g in range(2)] for k in range(2)]

    def gather(self, k, x, begin):
        e, n = self.e, self.per_group
        parts = [e.mul(e.slice(x, begin + g * n, begin + (g + 1) * n, "group"), m, 0, "mask")
                 for g, m in enumerate(self.masks[k])]
        out = parts[0]
        for part in parts[1:]:
            out = self.e.p.node("add", [out, part], n, "gather")
        return out

    def emit(self, k, x, scales, means):
        """Outputs step k's indexes (scales of x from channel `scales`) and means (from channel `means`)."""
        p, edges = self.e.p, self.edges
        n = edges[-1]
        table = np.searchsorted(np.asarray(edges), np.arange(-n, n), side="right")
        p.output(p.node("sat_table", [self.gather(k, x, scales)], self.per_group, "index",
                        table=p.blob(table, np.int32), top=len(edges)), "indexes")
        p.output(self.gather(k, x, means), "means")

    def update(self, k, y_hat, symbols, means):
        """y_hat + step k's symbol 2^fa + mean at the active elements (y_hat None: zeros)."""
        e, p = self.e, self.e.p
        v = p.node("add", [p.node("affine", [symbols], self.per_group, "symbol", alpha=2 ** e.fa, beta=0, shift=0),
                           means], self.per_group, "value")
        out = None
        for m in self.masks[k]:
            part = e.mul(v, m, 0, "scatter")
            out = part if out is None else e.concat(out, part, "scatter")
        return out if y_hat is None else p.node("add", [y_hat, out], p.channels(y_hat), "y_hat")


def step_inputs(p, per_group):
    return p.input("symbols", per_group), p.input("symbol_means", per_group)
