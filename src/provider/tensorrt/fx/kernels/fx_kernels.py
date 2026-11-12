"""nvcr.fx.v1 GEMM and depthwise kernels (Triton), compiled ahead of time by aot_compile.py.

The integer arithmetic implements docs/spec/numerics-fx-v1.md sections 4.1 and 4.2: products of two balanced int8
digits with int32 accumulation, the exact round-half-up requantization, the per-layer input shift with saturation
(XS, XCAP), the delta-table activation (WSILU, WSH), row grouping (G) and residuals (NRES). amax[1] and amax[2]
report the stored maximum and the clamped pre-activation maximum for the profile's failure rules.
"""
import triton
import triton.language as tl


@triton.jit
def _exact_gemm2_kernel(w_ptr, x_ptr, clo_ptr, chi_ptr, s_ptr, lut_ptr, r1_ptr, r2_ptr, out_ptr, amax_ptr,
               O, K, N, stride_wd, NLUT,
               LW: tl.constexpr, LA: tl.constexpr, OFF: tl.constexpr,
               WSILU: tl.constexpr, G: tl.constexpr, NRES: tl.constexpr, RC: tl.constexpr,
               XS: tl.constexpr, XCAP: tl.constexpr, CNT: tl.constexpr, WSH: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < O
    n_mask = offs_n < N
    a0 = tl.zeros((BM, BN), dtype=tl.int32)
    a1 = tl.zeros((BM, BN), dtype=tl.int32)
    a2 = tl.zeros((BM, BN), dtype=tl.int32)
    a3 = tl.zeros((BM, BN), dtype=tl.int32)
    a4 = tl.zeros((BM, BN), dtype=tl.int32)
    xm = tl.zeros((BK, BN), dtype=tl.int32)
    nclip = tl.zeros((BK, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        k_mask = kk < K
        x = tl.load(x_ptr + kk[:, None] * N + offs_n[None, :], mask=k_mask[:, None] & n_mask[None, :], other=0)
        if XCAP > 0:
            # hybrid per-layer input scale: x' = clamp(floor((x + h) / 2^XS), -XCAP, XCAP), h = 2^XS >> 1.
            # The clamp is done on x with the equivalent bounds [lo, hi]; the rounding shift is folded into the digit
            # extraction (u = x + OFF 2^XS + h, digits of u >> XS), so only the two clamp operations are added.
            lo = -XCAP * (1 << XS) - ((1 << XS) >> 1)
            hi = XCAP * (1 << XS) + (1 << XS) - 1 - ((1 << XS) >> 1)
            if CNT:
                nclip += ((x < lo) | (x > hi)).to(tl.int32)
            x = tl.minimum(tl.maximum(x, lo), hi)
            u = x + (OFF * (1 << XS) + ((1 << XS) >> 1))
        else:
            xm = tl.maximum(xm, tl.abs(x))
            u = x + OFF
        d0 = (((u >> XS) & 255) - 128).to(tl.int8)
        if LA > 1:
            d1 = (((u >> (XS + 8)) & 255) - 128).to(tl.int8)
        if LA > 2:
            d2 = (((u >> (XS + 16)) & 255) - 128).to(tl.int8)
        for i in tl.static_range(LW):
            w = tl.load(w_ptr + i * stride_wd + offs_m[:, None] * K + kk[None, :],
                        mask=m_mask[:, None] & k_mask[None, :], other=0)
            for j in tl.static_range(LA):
                if j == 0:
                    d = d0
                elif j == 1:
                    d = d1
                else:
                    d = d2
                if i + j == 0:
                    a0 = tl.dot(w, d, a0, out_dtype=tl.int32)
                elif i + j == 1:
                    a1 = tl.dot(w, d, a1, out_dtype=tl.int32)
                elif i + j == 2:
                    a2 = tl.dot(w, d, a2, out_dtype=tl.int32)
                elif i + j == 3:
                    a3 = tl.dot(w, d, a3, out_dtype=tl.int32)
                else:
                    a4 = tl.dot(w, d, a4, out_dtype=tl.int32)
    if pid_m == 0:
        if XCAP > 0:  # the clamp bounds every input by XCAP; each input tile is counted once (pid_m == 0 programs)
            tl.atomic_max(amax_ptr, XCAP)
            if CNT:
                tl.atomic_add(amax_ptr + 3, tl.sum(tl.sum(nclip, axis=1), axis=0))
        else:
            tl.atomic_max(amax_ptr, tl.max(tl.max(xm, axis=1), axis=0))
    # exact floor((sum_k a_k 256^k + c) / 2^s) in int32 (LW + LA <= 5):
    #   T = H * 2^24 + Lp, Lp in [0, 2^24); |L| < 2^27 and |H| < 2^27 for K < 2^11 (asserted on the host)
    clo = tl.load(clo_ptr + offs_m, mask=m_mask, other=0)
    chi = tl.load(chi_ptr + offs_m, mask=m_mask, other=0)
    s = tl.load(s_ptr + offs_m, mask=m_mask, other=24)
    L = a0 + clo[:, None]
    H = chi[:, None] + tl.zeros((BM, BN), dtype=tl.int32)
    if LW + LA > 2:
        L += (a1 & 0xFFFF) << 8
        H += a1 >> 16
    if LW + LA > 3:
        L += (a2 & 0xFF) << 16
        H += a2 >> 8
    if LW + LA > 4:
        H += a3
    H += L >> 24
    Lp = L & 0xFFFFFF
    d = tl.maximum(24 - s, 0)
    e = tl.maximum(s - 24, 0)
    lim = (1 << (30 - d))  # |H << d| <= 2^30; a clamped H gives |r| >= 2^30 - 2^d > RC (flagged)
    Hc = tl.minimum(tl.maximum(H, -lim[:, None]), lim[:, None])
    r = ((Hc << d[:, None]) >> e[:, None]) + (Lp >> tl.minimum(s, 31)[:, None])
    r = tl.minimum(tl.maximum(r, -RC), RC)
    rcm = tl.max(tl.max(tl.abs(tl.where(m_mask[:, None] & n_mask[None, :], r, 0)), axis=1), axis=0)
    tl.atomic_max(amax_ptr + 2, rcm)
    if WSILU:
        # WSiLU(r) = max(r, 0) + delta[r] on the table's non-trivial range [-NLUT, NLUT], 0 outside
        inside = (r >= -NLUT) & (r <= NLUT)
        dl = tl.load(lut_ptr + ((r + NLUT) >> WSH), mask=inside, other=0)  # WSH > 0: coarse table
        r = tl.maximum(r, 0) + dl.to(tl.int32)
    if G > 1:
        r = tl.sum(tl.reshape(r, (BM // G, G, BN)), axis=1)
    offs_o = pid_m * (BM // G) + tl.arange(0, BM // G)
    o_mask = offs_o < O // G
    msk = o_mask[:, None] & n_mask[None, :]
    ptr = offs_o[:, None] * N + offs_n[None, :]
    if NRES > 0:
        q = r.to(tl.int64) + tl.load(r1_ptr + ptr, mask=msk, other=0)
        if NRES > 1:
            q += tl.load(r2_ptr + ptr, mask=msk, other=0)
        qmax = tl.max(tl.max(tl.abs(tl.where(msk, q, 0)), axis=1), axis=0)
        tl.atomic_max(amax_ptr + 1, tl.minimum(qmax, 2147483647).to(tl.int32))
        tl.store(out_ptr + ptr, q.to(tl.int32), mask=msk)
    else:
        rmax = tl.max(tl.max(tl.abs(tl.where(msk, r, 0)), axis=1), axis=0)
        tl.atomic_max(amax_ptr + 1, rmax)
        tl.store(out_ptr + ptr, r, mask=msk)


@triton.jit
def _dw3x3_kernel(x_ptr, w_ptr, b_ptr, s_ptr, out_ptr, amax_ptr, H, W,
                  TH: tl.constexpr, BW: tl.constexpr):
    c = tl.program_id(0)
    hs = tl.program_id(1) * TH + tl.arange(0, TH)
    ws = tl.program_id(2) * BW + tl.arange(0, BW)
    base = x_ptr + c * H * W
    om = (hs[:, None] < H) & (ws[None, :] < W)
    v = tl.load(base + hs[:, None] * W + ws[None, :], mask=om, other=0)
    xmax = tl.max(tl.max(tl.abs(v), axis=1), axis=0)
    acc = tl.load(b_ptr + c) + tl.load(w_ptr + c * 9 + 4) * v.to(tl.int64)
    for t in tl.static_range(9):
        if t != 4:
            hh = hs[:, None] + (t // 3 - 1)
            ww = ws[None, :] + (t % 3 - 1)
            m = (hh >= 0) & (hh < H) & (ww >= 0) & (ww < W)
            v = tl.load(base + hh * W + ww, mask=m, other=0)
            acc += tl.load(w_ptr + c * 9 + t) * v.to(tl.int64)
    s = tl.load(s_ptr + c)
    r = (acc + (tl.full((), 1, tl.int64) << (s - 1))) >> s
    tl.store(out_ptr + c * H * W + hs[:, None] * W + ws[None, :], r.to(tl.int32), mask=om)
    rmax = tl.max(tl.max(tl.abs(tl.where(om, r, 0)), axis=1), axis=0)
    tl.atomic_max(amax_ptr, xmax)
    tl.atomic_max(amax_ptr + 1, tl.minimum(rmax, 2147483647).to(tl.int32))
