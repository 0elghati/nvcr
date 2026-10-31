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

## Fused HT-S provider-stage vector

The separate provider probe loads the real HT-S checkpoint and compares the
upstream fused `DMCHTSReconHeadProxy` with the upstream Python reconstruction
head. The proxy must be preallocated before `forward`, and the comparison covers
all eight clamped frame outputs plus the unclamped reset feature. The tracked
`provider_probe.cpp` binding links to an existing, unmodified upstream inference
shared library. It compiles a small C++ binding, without modifying the reference
checkout or rebuilding CUDA kernels. Build the upstream inference extension and
rANS dependency using the pinned source and CUTLASS revision above, with the same
PyTorch/CUDA environment used by this probe. The library hash below identifies
the local build; supply the hash of your own build when reproducing elsewhere.
The binary digest identifies the executable being tested; a clean source checkout
alone does not prove how an externally supplied binary was built.

```sh
rtk proxy env PYTHONPATH=build/dcvcuf-python-deps CUDA_HOME=/usr/local/cuda-12.6 MAX_JOBS=2 \
  .venv/bin/python scripts/backends/dcvcuf/probe_hts_provider.py \
  --reference-root assets/dcvc-uf-reference \
  --checkpoint /home/oelghati/dcvc-uf/cvpr2026_video_hts.pth.tar \
  --checkpoint-sha256 565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47 \
  --extension build/dcvcuf-python-deps/inference_extensions_cuda.cpython-312-x86_64-linux-gnu.so \
  --extension-sha256 cf2be0cc51d3766a9c1c8b19eb513a6fe6337f9b9d864ac527a3bcc9718bf805 \
  --output-dir build/dcvcuf-provider-probe
```

Supply `--feature FILE.npy --feature-sha256 SHA256` to replay an existing finite
FP16 `[1,512,H,W]` vector. Without those options the input is synthetic, even
though the weights are real. `feature.f16.npy` and `outputs.npz` retain the input
and both paths' outputs locally; their digests are in the report.

The current real-checkpoint vector remains a failed Python/fused comparison:
the fused FP16 path differs from PyTorch by at most `0.009521484375` on clamped
frames and `0.0047607421875` on the reset feature against the predeclared
`atol=3e-3, rtol=3e-3` limit. Snapshotting borrowed outputs before subsequent calls
preserves the same result. Input preservation, repeated execution, direct reset,
and reset-to-display shuffle/clamp agree exactly.

The first-operation diagnostic compares the first 1x1 convolution with and
without bias, then its WSiLU activation. Its float64 CPU oracle uses the same
FP16-quantized weights and input. The pinned SM89 CUTLASS path uses FP16 GEMM
accumulators, so standard PyTorch/ONNX arithmetic is not an established reference
for reproducing the fused decoder. NVCR stage comparisons use the pinned fused
outputs. This diagnostic compares two upstream paths; the separate NVCR
provider-session bridge below tests reconstruction on captured codec features.
Entropy integration and adapter lifecycle remain untested.

## NVCR integration boundary

`probe_nvcr_session.py` exercises the experimental implementation in
`src/dcvcuf/backend/upstream/recon_session.cpp` through the existing C++
`IProviderSession` interface. It loads the pinned HT-S weights, allocates opaque
device buffers, loads a reconstruction stage, submits tensor views, waits for
completion, and resets/reuses the session. The interface uses contiguous NHWC
FP16 tensors; conversion to upstream channels-last NCHW stays inside the bridge.
The stage returns eight clamped frames and one unclamped reset tensor. Only
codec-specific code assigns meaning to these outputs.

For a real vector, the script decodes a hash-verified native UF stream with both
the original upstream binary and a capture-only proxy. The capture proxy is a
renamed copy of the pinned HT-S class with a read-only snapshot method; its
unused encoder entry point is omitted because a Python return helper is hidden
in the upstream binary. Supporting operators and entropy functions link to the
original library. The upstream checkout remains untouched. Captured decode
frames must match the original library exactly before this can count as stage
evidence. The input feature, original frames, reset feature, and NVCR outputs
are retained in `vectors.npz`, with hashes in the report.

Run this after configuring an NVCR build (for its generated version header):

```sh
rtk proxy env PYTHONPATH=build/dcvcuf-python-deps \
  .venv/bin/python assets/dcvc-uf-reference/test_video.py \
  --model_path_i /home/oelghati/dcvc-uf/cvpr2026_image.pth.tar \
  --model_path_p /home/oelghati/dcvc-uf/cvpr2026_video_hts.pth.tar \
  --model_structure hts --test_config build/dcvcuf_qcif.json \
  --rate_num 1 --qp_i 32 --qp_p 32 --force_frame_num 17 \
  --force_intra_period 32 --reset_interval 8 --worker 1 --cuda_idx 0 \
  --stream_path build/dcvcuf-reference-session-20261031 \
  --output_path build/dcvcuf-reference-session-20261031.json

rtk proxy env PYTHONPATH=build/dcvcuf-python-deps CUDA_HOME=/usr/local/cuda-12.6 MAX_JOBS=2 \
  .venv/bin/python scripts/backends/dcvcuf/probe_nvcr_session.py \
  --reference-root assets/dcvc-uf-reference --nvcr-build build-cpu \
  --image-checkpoint /home/oelghati/dcvc-uf/cvpr2026_image.pth.tar \
  --image-checkpoint-sha256 b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302 \
  --checkpoint /home/oelghati/dcvc-uf/cvpr2026_video_hts.pth.tar \
  --checkpoint-sha256 565607d84943ee79072a5760d34a200d5724b4bb72c6bc24bfab41e8ae4f5f47 \
  --extension build/dcvcuf-python-deps/inference_extensions_cuda.cpython-312-x86_64-linux-gnu.so \
  --extension-sha256 cf2be0cc51d3766a9c1c8b19eb513a6fe6337f9b9d864ac527a3bcc9718bf805 \
  --stream build/dcvcuf-reference-session-20261031/QCIF/akiyo_qcif.yuv_q32.bin \
  --stream-sha256 7b2ebb14719082381ea8363ff9511545d5200dfb6a543db84d6782279be3fe7a \
  --output-dir build/dcvcuf-nvcr-session
```

The probe output directory must be new. The recorded stream is a fresh 17-frame
QCIF upstream run with two predicted chunks and reset interval 8. Supply the digest of your own generated native stream when
reproducing elsewhere. The diagnostic requires explicit input digests and uses
the upstream parser. It also uses the reference worker's `set_torch_env()` setup.
The I-frame must decode identically twice before its features are used. The older
nine-frame stream failed this cold replay check with the current binary and is
not a stage reference. Its producing environment is not established by its old
quality report alone; see the evidence for the rejected stream identity.
Reconstruction outputs, repeated execution after provider reset, input
preservation, and independent sessions require exact equality. The GPU check
also rejects foreign buffers/stages/dependencies, invalid shapes, names, types,
access modes, slices, aliasing, and model identities before executing kernels.

This is a synchronous, single-caller correctness bridge with in-memory verified
weights and feature dimensions limited to 256x256. It is built only by the probe,
does not register a production codec/provider, and is not a deployable standalone
backend. Artifact loading, asynchronous execution, UF entropy integration,
native-byte interoperability, adapter lifecycle, and performance remain separate
gates. Provider reset here clears no codec reference state: UF reference-memory
reset belongs to the future codec adapter.

This probe does not add a DCVC-UF option, stage, or lifecycle rule to NVCR's
shared runtime, provider, or configuration code. A future UF adapter owns its
I/chunk/tail/reset and access-unit semantics; a provider session receives only
generic stages, tensor contracts, buffers, submissions, and reset. The generic
runtime has no default codec, provider, model, or stream identity. The current
CLI selects its existing DCVC-RT/TensorRT product pair and identities explicitly
at that CLI entry point.

The dependency-free architecture check guards this separation:

```sh
rtk python3 tests/dcvcuf_architecture_tests.py
```

The dependency-free provenance checks run through CTest or directly:

```sh
rtk python3 tests/dcvcuf_export_tests.py
```

Current results and the remaining integration gates are in
[the C7 evidence](../../../evidence/phase-c7-dcvcuf-export-20261031.md).
