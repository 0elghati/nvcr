# DCVC-UF HT-S integration probe

This is the first C7 integration step. It exports a shuffle/clamp stage and one
HT-S frame-specific reconstruction branch. It does not register a codec, produce
a UF bitstream, or establish end-to-end compatibility.

The probe imports the pinned upstream `ReconHead`, selects the shared pair at
`branch // 2` and the frame branch at `branch`, and preserves two outputs:

- `pre_shuffle`: unclamped `[1, 192, H, W]`; branch 7 also uses this as the
  feature-memory reset value.
- `frame`: `[1, 3, 8H, 8W]`, shuffled and clamped to `[-0.5, 0.5]` as in the
  upstream inference implementation. The upstream training head does not clamp.

Branch selection and reset meaning stay inside this codec-specific tool.
The graph uses standard ONNX operations. No TensorRT plugin or generic runtime
change is needed for the stages tested here.

## Reproduce the bounded probe

Use a Python environment with PyTorch, NumPy, SciPy, ONNX, and ONNX Runtime.
SciPy is required by the upstream imports. The CPU check explicitly selects
`CPUExecutionProvider`; it is not evidence for an ONNX Runtime CUDA provider.
The output directory must be new so a failed run cannot leave a stale passing
report. Exported models, raw tensors, extension builds, and logs belong under
ignored `build/` or `assets/` directories.

```sh
rtk git clone https://github.com/microsoft/DCVC.git assets/dcvc-uf-reference
rtk git -C assets/dcvc-uf-reference checkout --detach cbdae87a5445114cdc7f48816da63ea80bdeac40
rtk python scripts/backends/dcvcuf/probe_hts_export.py \
  --reference-root assets/dcvc-uf-reference --synthetic \
  --output-dir build/dcvcuf-probe
```

`--synthetic` explicitly opts into seeded random weights and is not a substitute
for a pretrained vector. `--branch` selects 0 through 7 (default 7); default
feature dimensions 2x3 produce a 16x24 reconstruction. For real weights, replace
`--synthetic` with `--checkpoint PATH --checkpoint-sha256 SHA256`. Full upstream
checkpoints are accepted with the upstream state-dictionary wrapper conventions;
all reconstruction-head keys must load strictly. Supply a captured float32 NCHW
feature via `--feature PATH.npy --feature-sha256 SHA256`. Without it, the feature
is synthetic even when real weights are loaded. Neither mode closes C7 by itself.

Add `--trtexec /usr/src/tensorrt/bin/trtexec` for an FP32 TensorRT comparison.
It disables TF32, uses a 64 MiB workspace limit and optimization level zero, and
runs a bounded invocation. Raw binary outputs are compared; TensorRT's JSON
output rounds floats and cannot establish exact shuffle parity. These runs are
correctness probes, not performance measurements.

To also compare the actual upstream fused CUDA operator:

```sh
rtk git clone --depth 1 --branch v4.4.1 https://github.com/NVIDIA/cutlass.git assets/dcvcuf-cutlass
rtk env CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  python scripts/backends/dcvcuf/probe_hts_export.py \
  --reference-root assets/dcvc-uf-reference --synthetic \
  --cutlass-root assets/dcvcuf-cutlass \
  --trtexec /usr/src/tensorrt/bin/trtexec --output-dir build/dcvcuf-fused-probe
```

Set the CUDA toolkit and architecture to match the local PyTorch installation
and GPU. This compiles the original `elementwise/shuffle.cu` translation unit
against pinned CUTLASS v4.4.1, using a minimal local binding. It tests FP16 input
in both contiguous and channels-last layouts, including values outside and on
both clamp boundaries. It does not build or validate the full UF inference
extension. The GPU option fails if CUDA is unavailable; it does not fall back.

`report.json` records source/model/input hashes, versions, shapes, acceptance
limits, comparisons, and remaining gates. A failed numerical comparison exits
nonzero. Source/checkpoint failures also exit nonzero. Exact equality is required
for shuffle/clamp; the branch must satisfy `atol=1e-5, rtol=1e-4` per element.
The full upstream PyTorch head independently checks frame selection and branch
7's reset feature. `c7_complete` stays false because this probe does not test
entropy or a full encode/decode vector.

The dependency-free provenance checks run through CTest or directly:

```sh
rtk python3 tests/dcvcuf_export_tests.py
```

Current results and the remaining integration gates are in
[the C7 evidence](../../../evidence/phase-c7-dcvcuf-export-20261031.md).
