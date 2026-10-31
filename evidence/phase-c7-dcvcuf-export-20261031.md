# Phase C7 DCVC-UF stage export — 2026-10-31

Status: initial stage feasibility passes; C7 remains open. The user prioritized
starting DCVC-UF integration on 2026-10-31. This experiment builds on the existing
codec-owned sessions and grouped/delayed-output contracts. It does not register
a supported DCVC-UF codec or alter the DCVC-RT production path.

## Inputs and boundary

- Microsoft/DCVC: `cbdae87a5445114cdc7f48816da63ea80bdeac40`, clean checkout.
- CUTLASS: v4.4.1, `4370102f9dacab813282e1d67722fceb0b90a019`, clean checkout.
- GPU: NVIDIA GeForce RTX 4070; an existing workload occupied about 10 GiB.
- PyTorch 2.13.0+cu126; CUDA toolkit 12.6 for the isolated fused kernel build.
- ONNX 1.22.0, ONNX Runtime 1.28.0 with CPUExecutionProvider.
- TensorRT 10.9.0, FP32, TF32 disabled, builder optimization level zero,
  64 MiB workspace limit. No performance conclusion is drawn.
- HT-S branch 7, synthetic seeded weights and feature, seed 0, feature shape
  `[1,512,2,3]`, output `[1,3,16,24]`. No pretrained weights were used.

The source boundaries are `src/models/video_model_ht.py:ReconHead`,
`src/layers/extensions/inference/dmc_hts_proxy.cpp:DMCHTSReconHeadProxy`, and
`src/layers/extensions/inference/elementwise/shuffle.cu:pixel_shuffle_8_cuda`.
The official pinned source is [Microsoft/DCVC](https://github.com/microsoft/DCVC/tree/cbdae87a5445114cdc7f48816da63ea80bdeac40).

HT-S shares one common reconstruction stage between each pair of frames. Its
last branch produces an unclamped, pre-shuffle reset feature as well as the
clamped display frame. The probe preserves both. It imports the real upstream
PyTorch head and compiles the original fused shuffle translation unit, without
patching upstream. UF branch/reset semantics remain codec-owned.

## Acceptance and results

Limits were set before execution: exact equality for shuffle/clamp, and
`abs(candidate-reference) <= 1e-5 + 1e-4 * abs(reference)` for every branch
output element. Neither limit was relaxed after testing. Comparisons reject
nonfinite outputs. See the [compact machine-readable record](phase-c7-dcvcuf-export-20261031.json).

The upstream FP16 fused shuffle/clamp passes exactly in contiguous and
channels-last layouts. The ONNX CPU and TensorRT shuffle stages also pass
exactly. The HT-S branch's raw reset feature and reconstructed frame pass both
ONNX CPU and TensorRT FP32 tolerance checks. The full PyTorch head independently
confirms branch selection and the reset feature. An additional branch-0, 1x1
feature CPU probe passes, covering the first shared-pair boundary.

An initial TensorRT check used `--exportOutput` JSON and observed up to
`5.066394805908203e-7` apparent shuffle error due to decimal formatting.
The current probe uses `--dumpRawBindingsToFile`; binary shuffle output is
identical. This was a measurement correction, not a tolerance change.

CPU Release rebuild and CTest: **24/24 pass**, including three dependency-free
UF provenance tests covering wrong revision, modified/staged/untracked source,
explicit synthetic mode, mandatory hashes, and checkpoint-hash rejection.
The production codec/provider code is unchanged.

## Reproduction

See [the probe instructions](../scripts/backends/dcvcuf/README.md) for source
setup, dependencies, checkpoint/captured-feature inputs, and CPU-only use.
The local reference import needed SciPy, installed into the ignored
`build/dcvcuf-python-deps` directory with no change to the existing environment.
The combined current run is:

```sh
rtk env PYTHONPATH=/home/oelghati/NVCR/build/dcvcuf-python-deps \
  CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  .venv/bin/python scripts/backends/dcvcuf/probe_hts_export.py \
  --reference-root assets/dcvc-uf-reference --synthetic \
  --cutlass-root assets/dcvcuf-cutlass \
  --trtexec /usr/src/tensorrt/bin/trtexec --output-dir build/dcvcuf-c7-current
rtk cmake --build build-cpu -j 2
rtk ctest --test-dir build-cpu --output-on-failure -j 2
```

Raw tensors, ONNX models, generated bindings, build products, and logs remain
in the ignored local build directory. Only the compact current record is kept
with this evidence.

## Remaining gates

Both official checkpoint folder links in the pinned README (primary and backup)
redirected to HTTP 403 on 2026-10-31. No pretrained UF checkpoint was available
in the inspected local asset directories. These checks remain necessary:

1. Obtain and hash-pin the I and HT-S pretrained checkpoints; build the full
   upstream inference/entropy extension and capture a deterministic I/P vector.
2. Compare the provider stages on captured upstream features, including the
   unclamped reset feature, with the selected reconstruction/entropy contract.
   This synthetic FP32 branch result does not establish fused FP16 convolution
   parity or stream interchangeability.
3. Close the C7 pinned encode/decode vector gate before registering an adapter;
   then implement UF-owned I/chunk/tail/reset policy and private AU metadata,
   with real full/short-chunk, drain/reset, and reference tests.

No chunk API, generic runtime UF policy, capability taxonomy, production model
bundle, upstream bitstream compatibility, or full UF performance claim is added.
