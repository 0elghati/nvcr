# Numerics profile `nvcr.fx.v1`

Status: v1. Frozen once any stream uses it. Where an implementation (including NVCR's CPU reference)
disagrees with this document, the implementation is wrong.

It was derived from the dcvc-robust research codec (`exactrt/fx.py`, `exactrt/tri.py`, `exactrt/loop.py`) at its
hybrid + coarse-WSiLU operating point. The profile itself names no codec and no provider.

---

## 1. Identity

| Field | Value |
|---|---|
| Id | `nvcr.fx.v1` |
| Owner | NVCR (`include/nvcr/numerics/profiles.hpp`: `nvcr::numerics::fx_v1`) |
| Defines | the integer result of every op in Section 4, and when a stage must fail (Section 5) |
| Does not define | which stages a codec makes normative, entropy coding, weights or tables (program data), kernels, tiling, devices |
| Program format | `nvcr.fx-program.v1` (Section 6), versioned separately |
| Version rule | any change to the integers a valid program produces, or to when a stage fails, is a new profile version. A version is frozen once any stream uses it. |
| Claim rule | a provider may answer `supports_numerics_profile("nvcr.fx.v1")` only after passing the whole conformance suite (Section 7) |

---

## 2. Values

- Every tensor is a batch-1 NCHW tensor of **int32** values. Spatial sizes and channel counts come from the program.
- A value `v` stands for `v · 2^-fa`. `fa` is a property of the program, and the ops use it only where Section 4
  says so. The research setting is `fa = 14` everywhere.
- **Stored bound:** every value an op writes must satisfy `|v| ≤ 2^30`. Otherwise the stage fails (Section 5).
  Inputs a codec passes into a stage must satisfy the same bound.

---

## 3. Notation

All arithmetic in this section is on mathematical integers; there is no overflow and no wrap-around.

- `rsr(a, s)` ("round-shift-right"), for an integer `s ≥ 0`:
  `rsr(a, 0) = a`; for `s ≥ 1`, `rsr(a, s) = floor((a + 2^(s−1)) / 2^s)`.
  This is round-half-up (ties go toward +∞), the same as `(a + (1 << (s−1))) >> s` with an arithmetic shift.
- `clamp(a, lo, hi) = min(max(a, lo), hi)`.
- `D2 = 32639`: the largest magnitude two balanced int8 digits represent (`(256² − 1) · 127 / 255`).

---

## 4. Ops

Each node of a program is one op. The parameters named below are program data (Section 6).

### 4.1 `gemm`: 1×1 convolution

Input `X` with `K` channels; output `Y` with `O` channels (before grouping). Parameters:
- integer weights `w[o][k]`, with `|w| ≤ D2`;
- integer bias `b[o]`;
- shift `s[o]`, with `1 ≤ s[o] ≤ 31`;
- input shift `k_in ≥ 0`;
- table activation (optional);
- group `G ∈ {1, 2, 4}`;
- 0, 1 or 2 residual inputs.

For every output position `p` and output channel `o`:

1. Input rounding and saturation:
   `x'[k][p] = clamp(rsr(X[k][p], k_in), −D2, D2)`.
2. Exact sum and requantization:
   `r[o][p] = rsr( Σ_k w[o][k] · x'[k][p] + b[o],  s[o] )`.
   **Failure:** if `|r[o][p]| ≥ 2^29 − 1` for any `o, p`, the stage fails.
3. Table activation, if present (Section 4.5): `a[o][p] = act(r[o][p])`; otherwise `a = r`.
4. Grouping: `g[o'][p] = Σ_{j=0}^{G−1} a[o'·G + j][p]` for `o' < O / G`. Groups are **adjacent rows**. An
   exporter that needs another pairing permutes rows when it writes the program. DCVC-RT's chunk-add sums channel `c`
   with `c + O/2`, so its exporter interleaves those rows.
5. Residuals: `Y[o'][p] = g[o'][p] + R1[o'][p] (+ R2[o'][p])`.
6. Stored bound (Section 2) on `Y`.

Dense convolutions are a `gemm` after an `im2col` node (Section 4.7). The `im2col` channel order is `(c, i, j)`,
matching the weight layout `[O][C][3][3]` flattened.

### 4.2 `depthwise3x3`

Stride 1 and zero padding 1. Per channel `c`, integer weights `w[c][i][j]`, bias `b[c]`, shift `s[c] ≥ 1`:

`Y[c][y][x] = rsr( b[c] + Σ_{i,j ∈ {0,1,2}} w[c][i][j] · X[c][y+i−1][x+j−1],  s[c] )`, where taps outside the image
contribute 0. Then the stored bound applies.

### 4.3 `mul`: fixed-point product

`Y[c][p] = rsr( A[c][p] · B[c'][p'], t )`, with an integer shift `t` from the program.
- `B` has either `A`'s channel count (`c' = c`) or one channel (`c' = 0`).
- `B` has either `A`'s size (`p' = p`) or is 1×1 (`p' = 0`).

The research uses `t = fw = 18` for per-channel model scales, and `t = fa` for a product of two activations.

### 4.4 `param_row`: per-QP parameter rows

A table `rows[q][c]` of integers with `|rows| ≤ 2^24` is program data. The input is a 1×1 one-channel tensor
holding `q`. The output is the `C`-channel 1×1 tensor `rows[q][·]`. A `q` outside the table fails the stage.

This keeps a codec's per-quality model scales (for example DCVC-RT's `q_scale_dec`, `q_feature`, `q_decoder`,
`q_recon`, quantized at 2^-fw) inside the program, and so inside the digest.

### 4.5 `table` activations

Two kinds, each with its table as program data.

- **Bounded delta table** (`delta`; WSiLU in the research): parameters `h ≥ 0`, `c ≥ 0`, and a table `dc` of
  `(2h >> c) + 1` integers with `|dc| < 2^15`:
  - `act(r) = max(r, 0) + dc[(r + h) >> c]` if `−h ≤ r ≤ h`;
  - `act(r) = max(r, 0)` otherwise.
  
  `c = 0` gives a full-resolution table. The research ships `c = 3`. This kind may be fused into a `gemm` (4.1, step
  3) or stand alone.
- **Saturating table** (`sat_table`; sigmoid in the research): parameters `n ≥ 1`, a table `T` of `2n` integers, and
  a value `top`:
  - `Y = T[X + n]` if `−n ≤ X < n`;
  - `Y = top` if `X ≥ n`;
  - `Y = 0` if `X < −n`.

### 4.6 `affine`

Integers `α`, `β` and shift `t ≥ 0`: `Y = rsr(α · X + β, t)`.

Example: DCVC-RT's decoder quant step `floor(1.5 · sigmoid + 0.5 · 2^fa + 0.5)` is `affine` with `α = 3`,
`β = 2^fa`, `t = 1`.

### 4.7 Exact data movement

None of these changes a value.

- `add`: `Y = A + B`, elementwise, same shape; then the stored bound.
- `crop_like`: `X` cropped from the origin to the height and width of a second input, which must not be larger.
- `clamp`: `Y = clamp(X, lo, hi)` with integer `lo`, `hi` (for example `[0, 2^fa]` for a picture).
- `concat`: channels of the first input, then the second.
- `slice`: channels `[begin, end)`.
- `pixel_shuffle(r)` and `pixel_unshuffle(r)`, as PyTorch defines them: unshuffle channel order `(c, i, j)`.
- `im2col` for 3×3 stride 1 padding 1, 3×3 stride 2 padding 1, and 2×2 stride 2 padding 0. The output channel order
  is `(c, i, j)`; taps outside the image are 0.

---

## 5. Failure

- A stage fails when any condition named above is met: `|r| ≥ 2^29 − 1` in a `gemm`, or a stored value
  `|v| > 2^30`.
- A failed stage produces no usable output. The provider reports a structured error, the codec drops the frame or
  unit, and nothing derived from it is written or returned.
- **Every conforming implementation must fail on exactly the same inputs.** The conditions are therefore stated on
  mathematical values, never on an implementation's internal ranges. An implementation that cannot compute some
  node exactly (for example a kernel whose int32 accumulators are too narrow for a large `K`) must use another path
  for that node or refuse the program at load time. It must never produce a different integer.
- **Load-time validation** (the program is refused, no stage runs):
  - `|w| ≤ D2` for `gemm` weights;
  - `|w| ≤ 2^24` for depthwise weights;
  - `|b| ≤ 2^52`;
  - every shift in range;
  - table sizes consistent;
  - every tensor reference valid.
  
  With these bounds and inputs `|v| ≤ 2^30`, every sum in Section 4 is below 2^63, so 64-bit integer arithmetic is
  exact.

---

## 6. Program format `nvcr.fx-program.v1`

Each normative stage is two files in the bundle's `normative/` directory.

**`<stage>.fxw`:** a little-endian blob of typed arrays, referenced by `[byte offset, element count]`.

**`<stage>.fxprog.json`:**

```json
{
  "format": "nvcr.fx-program.v1", "profile": "nvcr.fx.v1", "fa": 14,
  "data": {"file": "<stage>.fxw", "sha256": "<hex SHA-256 of the .fxw file>"},
  "tensors": [{"name": "y_hat", "channels": 128}, ...],
  "inputs": [0, 1], "outputs": [9],
  "delta_tables": [{"h": 108544, "shift": 3, "dc": [offset, count]}],
  "nodes": [{"op": "gemm", "inputs": [x, residual...], "output": y, ...}, ...]
}
```

- Spatial sizes are not stored. They follow from the inputs a codec passes.
- Nodes run in order. A tensor is written once and read only after it is written.
- The program digest a stream carries is the SHA-256 of `<stage>.fxprog.json`. That file pins the `.fxw` SHA-256,
  so weights, input shifts and tables are all covered.

**Node fields** (arrays in the data file; the element type is fixed per field):

| `op` | inputs | fields |
|---|---|---|
| `gemm` | x, then 0–2 residuals | `in_channels`, `out_channels`, `group` (1, 2, 4), `input_shift`, `delta_table` (index or −1), `weights` int16 `[out][in]`, `bias` int64, `shift` int32 |
| `depthwise3x3` | x | `weights` int32 `[c][3][3]`, `bias` int64, `shift` int32 |
| `mul` | a, b (b: same or one channel, same size or 1×1) | `shift` |
| `param_row` | q (1 channel, 1×1) | `rows` int64 `[q][c]` |
| `crop_like` | x, ref | — |
| `affine` | x | `alpha`, `beta`, `shift` |
| `add` | a, b | — |
| `sat_table` | x | `table` int32 `[2n]`, `top` |
| `delta` | x | `delta_table` |
| `clamp` | x | `lo`, `hi` |
| `concat` | a, b | — |
| `slice` | x | `begin`, `end` (channels) |
| `pixel_shuffle`, `pixel_unshuffle` | x | `factor` |
| `im2col` | x | `kernel` (3 or 2), `stride` (1 or 2; 2×2 needs 2) |

`delta_tables[i].dc` is int16 with `((2h) >> shift) + 1` entries.

**Load-time refusals:**
- any failed check in Section 5;
- a data file whose SHA-256 differs from the one in the JSON (`digest_mismatch`);
- a field out of range, or a reference outside the data file or the tensor list.

**How the research settings map to program fields:**

| Research setting | Program field |
|---|---|
| `EXACTRT_APOLICY` (`rt_hybrid_k1.json`, `uf_hybrid_k1.json`) | `gemm.input_shift` |
| `EXACTRT_WLIMBS=2`, per-channel weight exponents | the integer weights and `shift` |
| `EXACTRT_LIMBS=2` | always: inputs saturate to `D2` |
| `EXACTRT_WSILU_SHIFT=3` | `delta_tables[].shift` and its `dc` |
| sigmoid LUT (`build_sigmoid_lut`, range 16) | `sat_table` with `n = 16 · 2^fa`, `top = 2^fa` |

---

## 7. Conformance suite

1. **Per-op vectors:** rounding ties (positive and negative), saturation edges (`±D2`, `±(D2+1)` after the shift),
   delta-table edges (`±h`, `±(h+1)`, bin boundaries), saturating-table edges (`±n`), and the two failure conditions
   on both sides of their thresholds.
2. **Whole-program vectors:** each normative stage of DCVC-RT and DCVC-UF at the research settings, run on inputs
   recorded from the research codec. The expected outputs come from `exactrt` with `EXACTRT_BACKEND=f64` (float64 on
   integer values, exact by construction), cross-checked against the Triton backend.
3. A provider passes only if every vector matches bit for bit, and every failure vector fails.

---

## 8. Differences from the research code

These change only *when a stage fails*, never a value a successful stage produces.

1. **One failure rule for every `gemm`:** `|r| ≥ 2^29 − 1`. The research applies it only on its fused GEMM path
   (`exact_gemm2`); its unfused path checks the stored bound instead. A single rule is needed so that all providers
   fail on the same inputs.
2. **Load-time weight bounds replace the research's run-time 2^52 partial-sum check.** That check only protects the
   float64 backend. With the bounds in Section 5, 64-bit integers are always exact, so the check can't fire in an
   integer implementation.
3. **GEMM inputs always saturate to `D2`.** That matches the research hybrid setting (`EXACTRT_LIMBS=2` with a
   policy, `k_in = 0` for layers the policy doesn't name). Config B without the hybrid policy (three input digits) is
   **not** representable in v1. Representing it would need a per-node input digit count.
4. **Not in the profile:** the conversion of a stage's int32 output to float for display (non-normative), and the
   entropy-side mapping from scales to table indexes (codec-owned, through its integer edges).
