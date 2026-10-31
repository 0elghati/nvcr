# Phase C7 DCVC-UF stage export — 2026-10-31

Status: upstream checkpoint/export checks pass; Python/fused arithmetic divergence
localized to the first convolution; C7 remains open. The user prioritized
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

Real checkpoint validation then used `cvpr2026_image.pth.tar` (SHA-256
`b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302`) and
`cvpr2026_video_hts.pth.tar` (SHA-256
`565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47`). The
upstream rANS and CUDA inference extensions built successfully with the pinned
source and CUTLASS v4.4.1.

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

The real upstream `test_video.py` also completed a nine-frame QCIF run with
one I-frame and one eight-frame HT-S P chunk at QP 32. The encode/decode run
passed for both requested rate entries. It reports 1 I frame, 8 P frames,
0.042368 aggregate BPP, 38.545470 aggregate PSNR, and stream JSON SHA-256
`0dcc0c3773f14a2f74eb51fd0be2e0b9acac1aac722a29ffc37cd58412a6a3c6`.
The two rate entries intentionally used the same QP as a smoke check.

The real HT-S diagnostic vector then loaded the same video checkpoint into
the upstream `DMCHTSReconHeadProxy` and compared its eight clamped frame outputs
and unclamped reset feature with the upstream Python reconstruction head. The
proxy preallocation lifecycle passed, but the strict predeclared FP16 parity
gate did not: the largest frame error was `0.009521484375` and the reset error
was `0.0047607421875`, against `atol=3e-3, rtol=3e-3`. The vector used feature
shape `[1,512,3,3]`, seed `20261031`, feature SHA-256
`34b4bda86d6e030ca06c62d5157ca7aad407db83b5911d1cb7a8c1dc942009e9`, and
current probe SHA-256
`99bea7b3a062e0c5270bbb78cb763653ef94efd900223aa0bfef9f0282163097`.

The follow-up removes the temporary upstream binding dependency: the tracked
`provider_probe.cpp` links to the existing unmodified upstream extension, whose
SHA-256 is `cf2be0cc51d3766a9c1c8b19eb513a6fe6337f9b9d864ac527a3bcc9718bf805`.
All outputs are snapshotted before further proxy calls. The original reset check
could read a buffer overwritten by `forward_reset`; correcting that lifetime
issue reproduces the same frame/reset error maxima. Input preservation, repeated
execution, full-head/direct-reset equality, and reset-to-display shuffle/clamp
all pass exact checks. The pinned checkout remains clean throughout this run.

The first differing operation is `recon_head.conv1.0.0.dc.0`, the first 1x1
convolution. Even with bias and activation removed, 3680 of 4608 elements differ,
with maximum absolute error `0.01953125`. With bias, the maximum remains
`0.01953125`; after WSiLU it is `0.015625`. A CPU float64 oracle using identical
FP16-quantized inputs and weights gives mean convolution error `0.000328648939`
for PyTorch and `0.001942670801` for CUTLASS.

The pinned `cutlass/conv1x1_kernel.h:conv1x1_bias_generic_cutlass2` instantiates
the GEMM accumulator as `cutlass::half_t`. The WSiLU epilogue computes in float
after that accumulation. These source facts and the bias-free measurement locate
the first divergence in the convolution's accumulation, before activation,
depthwise-bias folding, shuffle, or reset. They do not attribute every later
head error to that one operation or establish behavior on other GPU targets.

This is a comparison of two upstream paths, **not NVCR provider parity**. The
strict Python/fused gate remains failed without changing its tolerance. The
pinned fused decoder outputs are the reference for the next NVCR stage test;
a passing FP32 ONNX/PyTorch comparison alone cannot establish that equivalence.
The current feature is still synthetic, not captured from a codec decode.
Local `outputs.npz` retains both paths' tensors with SHA-256
`75b4c758137447b961d151d66acc00b799721d97c2ab049d641a4f751895d3b0`.

An initial TensorRT check used `--exportOutput` JSON and observed up to
`5.066394805908203e-7` apparent shuffle error due to decimal formatting.
The current probe uses `--dumpRawBindingsToFile`; binary shuffle output is
identical. This was a measurement correction, not a tolerance change.

CPU Release rebuild and CTest: **24/24 pass**. Five dependency-free
UF provenance tests covering wrong revision, modified/staged/untracked source,
explicit synthetic mode, mandatory hashes, and checkpoint/binary/feature-hash
rejection pass.
The production codec/provider code is unchanged.

## Reproduction

See [the probe instructions](../scripts/backends/dcvcuf/README.md) for source
setup, dependencies, checkpoint/captured-feature inputs, and CPU-only use.
The local reference import needed SciPy, installed into the ignored
`build/dcvcuf-python-deps` directory with no change to the existing environment.
The initial export run is:

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

The current diagnostic run is `build/dcvcuf-c7-diagnosis-current/report.json`.
Use the provider command in the probe instructions, adding
`--feature build/dcvcuf-c7-provider-real-4/feature.f16.npy` and
`--feature-sha256 34b4bda86d6e030ca06c62d5157ca7aad407db83b5911d1cb7a8c1dc942009e9`
to replay the retained failing feature exactly. Exit code 1 records the unchanged
numerical gate failure; the lifecycle checks pass independently.

## Remaining gates

The initial automated checkpoint links redirected to HTTP 403, but the required
files were subsequently provided manually. These checks remain necessary:

1. Implement and compare NVCR stages against the pinned fused outputs, preserving
   the required accumulation behavior. Repeat on captured upstream
   reconstruction/reset features. The upstream Python/fused divergence is
   localized, but no NVCR provider comparison has been executed.
2. Close the C7 pinned encode/decode vector gate inside NVCR before registering an adapter;
   then implement UF-owned I/chunk/tail/reset policy and private AU metadata,
   with real full/short-chunk, drain/reset, and reference tests.

No chunk API, generic runtime UF policy, capability taxonomy, production model
bundle, upstream bitstream compatibility, or full UF performance claim is added.
