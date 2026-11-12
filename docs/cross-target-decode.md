# Cross-target decode

Status: C14 (issue #216). A DCVC-RT stream encoded with exact numerics decodes
bit-identically on every provider and target that implements the numerics
profile. Native streams (the default) still decode only on the target that
encoded them, and fail cleanly anywhere else.

## The problem

DCVC-RT's decoder picks an entropy table for every latent symbol from a
predicted scale:

```text
index = floor((ln(clamp(scale, 0.11, 16)) - ln(0.11)) * 25.516)   # 128 tables
```

The scale comes from FP16 TensorRT stages. Different GPUs build different plans
with different kernels, so their scales differ by a few FP16 rounding steps.
When a scale sits near a table boundary, the decoder selects the neighbouring
table and the rANS decoder desynchronizes from that symbol on. A TensorRT
stream decoded by ONNX Runtime (or the reverse) fails the same way on one
machine.

Any deterministic function from a continuous value to a discrete index has
boundaries, so more precision only makes flips rarer. Two routes remain:
transmit the fragile decisions, or compute them identically everywhere. The
first was measured and rejected: at the margin needed to stay synchronized
between TensorRT and ONNX Runtime it cost 37-40 times the payload, and
decodes still drifted. NVCR takes the second route.

## What must be exact

Everything whose output can reach an entropy decision or the decoder's
reference state:

| Frame | Normative stages | Free (native) stages |
|---|---|---|
| I | `i_hyper_synthesis`, `i_spatial_prior_1..3`, `i_synthesis` (its frame is the next P-frame's reference) | `i_analysis`, `i_hyper_analysis` (encoder only) |
| P | `p_reference_frame`, `p_reference_feature`, `p_prior`, `p_spatial_prior`, `p_synthesis` | `p_analysis`, `p_hyper_analysis` (encoder only) |

P-frames depend on nearly the whole decoder, because the temporal context
comes from the previous decoded feature, and after an I-frame or a reference
reset from the previous decoded frame. Exact P-frames therefore also make the
decoded video identical on every conformant target.

The glue between stages (masks, symbol restoration, the scale-to-index map)
must be exact too. The index map uses a fixed threshold table
(`src/dcvcrt/scale_index.hpp`) instead of `logf`; the other glue operations
are single IEEE operations whose only products are with 0/1 masks, so FMA
contraction cannot change them.

## Separation of concerns

Codec and provider stay independent axes (`AGENTS.md`):

- **The codec declares.** DCVC-RT lists its normative stages and the profile
  they require (`src/dcvcrt/numerics.*`), writes the identity into the stream,
  and owns the threshold table and glue. It names no provider or GPU.
- **The runtime defines profiles.** `nvcr.det-int.v1` (`src/deterministic`)
  specifies every rounding step of a small op set, with a CPU reference.
- **Providers implement.** A provider advertises the profiles it executes
  exactly and runs normative programs with them: TensorRT through the CUDA
  executor (SM 8.0 and newer), ONNX Runtime CPU through the CPU reference.
  ONNX Runtime CUDA and QNN do not implement the profile, so they refuse
  exact streams before decoding. Nothing falls back to native arithmetic.
- **Artifacts carry programs.** Normative stages ship as portable programs
  (`<stage>.detprog.json` plus FP16 weights) in the bundle's `normative/`
  directory, identical on every target.
- **The stream names them.** Each random-access unit of an exact stream is
  NVAU v2 with a required `numerics_identity` section: the profile id and one
  SHA-256 over the codec's normative stage ids and program digests (48 bytes
  for `nvcr.det-int.v1`). P-frame units keep NVAU v1 framing and inherit the
  identity of the sequence's last random-access unit, like a parameter set.
  A decoder without the profile, or with different programs, fails before
  decoding; an exact decoder refuses a native random-access unit.

## The nvcr.det-int.v1 profile

The arithmetic contract is in `src/deterministic/program.hpp`:

- Dense convolutions use exact integer arithmetic. Weights are quantized per
  output channel and activations per position with a power-of-two scale, each
  to 8 or 16 bits as the program node says. A dense spatial convolution is
  lowered to `im2col` and a pointwise convolution, so its activation scale
  covers each output position's receptive field. Integer sums are exact in any
  order, so executors may tile, split, or use integer tensor cores freely.
- Every other op (depthwise convolutions, sigmoid, elementwise ops, data
  movement) follows det-fp32: FP32 with a fixed operation order, no
  contraction, and an `exp` built from exactly rounded operations.
- Stage outputs round to FP16 (nearest even) at stage boundaries.

Bit widths come from a per-layer map, `configs/numerics/dcvcrt-det-int-v1.json`,
keyed by pinned DCVC-RT layer names (`I.<module>` and `P.<module>`).

## Using it

```bash
nvcr encode -i in.yuv -o out.nvcr -s 1920x1080 --numerics nvcr.det-int.v1
nvcr decode -i out.nvcr -o out.yuv
```

Decode reads the profile from the stream; there is nothing to pass. Bundles
built by `scripts/backends/dcvcrt/prepare_artifacts.sh` and
`build_tensorrt.sh` carry the programs; `export_normative_programs.sh` exports
them for an existing bundle.

## Cost and status

- Rate: no BD-rate cost. Against FP32 over seven 1080p and 720p sequences and
  96-frame GOPs the shipped width map measures -0.089% (the FP16 TensorRT path
  +0.064%). Each random-access unit carries about 130 bytes of identity and
  NVAU v2 framing; P-frame units are unchanged in size.
- Speed: exact stages are slower than TensorRT FP16. Diagnostic 1080p decode on
  an RTX 4070 runs at 18.4 fps against 48.7 fps native (encode 30.1 against
  64.5 fps). The exact path is
  memory-bound (float32 intermediates, two-pass activation quantization);
  8-bit weights and FP16 intermediate storage both cost rate and are not
  used. Kernel headroom is bounded near 10-20%; see the
  [research report](research/issue-216-cross-target-research.md) for
  codec-level options.
- Open: an RTX 4070 to RTX 5060 conformance result, exact-path speed, and
  `normative/` in published bundle manifests.

## Tooling

| Tool | Purpose |
|---|---|
| `scripts/backends/dcvcrt/export_deterministic_program.py` | Lowers an exported ONNX stage into a program; rejects anything outside the op set |
| `scripts/backends/dcvcrt/check_deterministic_programs.py` | Checks each program against the pinned PyTorch module (FP32 agreement) |
| `scripts/backends/dcvcrt/integer_gop_study.py` | Rate-distortion cost of a width map over full GOPs |
| `nvcr_deterministic_executor` | CPU reference against double precision; CUDA bit-identical to the CPU on every op and every stage program |
| `nvcr_numerics_identity` | Stream identity section round trip and rejection |
| `nvcr_dcvcrt_cross_provider_decode` | Native streams fail cleanly across providers; exact streams decode bit-identically; providers or bundles without the profile refuse before decoding |

Results and measurements: [issue #216 evidence](../evidence/issue-216-cross-hardware-fix-20261104.md).
