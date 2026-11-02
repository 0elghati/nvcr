# Phase C7 DCVC-UF stage export — 2026-10-31

Updated 2026-11-02. Status: the opt-in native build registers DCVC-UF and its
pinned fused-CUDA provider, and the normal NVCR C++ CLI encodes and decodes
intra plus HT-S grouped access units. The runtime binary has no Python ABI
dependency. The pinned Python/native golden and the 500-frame HD measurement
pass (see the golden section below). C7 stays open for the provider decision,
the upstream README recipe options, and package gates. The earlier Python/fused arithmetic divergence remains
localized to the first convolution. The user prioritized DCVC-UF integration
on 2026-10-31. The integration uses the existing codec-owned sessions and
grouped/delayed-output contracts and does not alter the DCVC-RT production path.

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

The initial upstream nine-frame QCIF smoke run was superseded by the fresh
17-frame reference described below. Its retained stream failed repeat decoding
before the deterministic empty-tensor initialization fix; that historical
compatibility decision is documented below.
The current upstream run reports 1 I-frame and 16 P-frames at QP 32, aggregate
BPP `0.027796345811051695`, and aggregate PSNR `38.68018492283872`.
These are upstream reference metrics, not full NVCR codec results.

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
That diagnostic feature is synthetic; the later NVCR session gate below uses
features captured from native decoding.
Local `outputs.npz` retains both paths' tensors with SHA-256
`75b4c758137447b961d151d66acc00b799721d97c2ab049d641a4f751895d3b0`.

An initial TensorRT check used `--exportOutput` JSON and observed up to
`5.066394805908203e-7` apparent shuffle error due to decimal formatting.
The current probe uses `--dumpRawBindingsToFile`; binary shuffle output is
identical. This was a measurement correction, not a tolerance change.

CPU Release rebuild and CTest: **25/25 pass**. Seven dependency-free
UF provenance tests covering wrong revision, modified/staged/untracked source,
explicit synthetic mode, mandatory hashes, and checkpoint/binary/feature-hash
rejection pass. The dependency-free architecture check also passes: shared
codec, provider, runtime, and configuration code contains no UF-specific terms,
and core configuration has no default codec, provider, model, or stream
identity. The CLI names its current DCVC-RT/TensorRT product selection and
identities explicitly. The production codec/provider implementations are
unchanged; a probe-only UF reconstruction implementation now lives under
`src/dcvcuf/backend/upstream`.

## NVCR provider-session synthesis and reconstruction

The existing C++ `IProviderSession` boundary now runs both the synthesis decoder
and reconstruction head. It allocates opaque device buffers, loads codec-local
stages, submits tensor views, waits for completion, resets, and reuses the
session. The bridge invokes the original `DMCHTSDecoderProxy` and
`DMCHTSReconHeadProxy` from the verified library, preserving UF's fused FP16
arithmetic. No shared runtime/provider interface or codec policy changes.

The vector remains the 17-frame QCIF native UF stream (one I-frame and two
eight-frame predicted chunks, QP 32, reset interval 8), SHA-256
`7b2ebb14719082381ea8363ff9511545d5200dfb6a543db84d6782279be3fe7a`.
A renamed capture copy of the pinned HT-S decoder adds only snapshots of the
latents, context, quantization, feature, and reset tensor. Its frame outputs
must match the unmodified decoder exactly. The unused encoder entry point is
omitted from that copy to avoid a hidden Python helper. The upstream checkout
remains clean; generated capture-source hashes and binary hashes are recorded.

Boundary tensors are contiguous NHWC FP16:

| Tensor | Shape | Owner of meaning |
|---|---|---|
| Latent | `[1,9,11,256]` | UF decoded latent |
| Context | `[1,18,22,512]` | UF temporal context |
| Quantization | `[1,1,1,512]` | UF per-channel decoder scale |
| Feature | `[1,18,22,512]` | UF decoder output/reconstruction input |
| Frames | `[8,144,176,3]` per chunk | UF presentation-ordered outputs |
| Reset feature | `[1,18,22,192]` | UF unclamped reference representation |

The decoder output and reconstruction input share one provider buffer, with
an ordinary execution dependency between submissions. There is no intermediate
host transfer. Both features, all sixteen reconstructed frames, and both reset
features match the native decoder exactly. Input preservation, provider
reset/reuse, and independent sessions pass. Thirty-six contract checks cover
both stages, including foreign handles, malformed bindings, byte bounds,
overlap, model/artifact identity, unknown stages, malformed output contracts,
and accepted same-session dependencies.

Two independent capture-and-stage processes under `CUDA_LAUNCH_BLOCKING=0`
produce identical hashes for all 23 tensors and the vector archive. Separately,
the verified binary replays those frozen vectors with `CUDA_LAUNCH_BLOCKING=0`:
six two-stage runs across two sessions, exact input/output equality, and all
36 contract checks pass. Provider submission itself remains synchronous and
single-caller; concurrent sessions and async completion are not claimed.

### Native UF entropy parity  2026-11-02

NVCR now builds a separate, codec-local `nvcr_dcvcuf_rans` target from the
pinned DCVC-UF rANS source. It is namespaced independently from
`third_party/dcvc_rans`, so DCVC-RT's entropy implementation and bitstream
behavior are unchanged. The UF wrapper enforces upstream CDF identity
(`z=0`, `y=1`), signed bypass mapping, global z-channel indexing,
one-to-eight coder partitioning, paired-stream overlap/header framing, and
z-before-y decode order.

The CPU conformance binary passes exact round trip and post-reset byte
repeatability for all coder counts 1 through 8. The native-stream gate then
uses the verified CVPR 2026 HT-S checkpoint and the retained 17-frame QCIF
stream. For both predicted chunks, decoded z symbols and y symbols match the
upstream fused decoder exactly. The existing decoder feature, all sixteen
predicted frames, both reset features, repeat/reuse checks, independent-session
checks, and all 36 provider contract rejections remain exact.

The passing report is
`build/dcvcuf-nvcr-entropy-parity-20261102-3/report.json`, SHA-256
`a5a9ead4ed8a142f4d805e14ba0e9e70eaccd8f502fff88a52d247c6045cf128`.
Its vector archive SHA-256 is
`97fc56193355ee1d96a40f573ee161fb7bca5060f46065a311f85d616f910af8`.
This proves native UF entropy compatibility for the tested HT-S stream. It does
not register the codec; image/context/prior/encoder artifact loading and the
codec-owned chunk lifecycle remain open.

### Artifact-backed UF stage loading  2026-11-02

The synthesis decoder and reconstruction stages no longer receive in-memory
Python weight maps. The offline exporter writes 385 decoder/reconstruction
tensors to a LibTorch archive. Each C++ provider session receives only the
artifact path and expected SHA-256, verifies the file with NVCR's SHA-256
implementation, deserializes named tensors through LibTorch, enforces CUDA
FP16 stage weights, and caches one artifact identity for all loaded stages.

The same native-stream gate passes after this change with no tensor hash
changes. The stage artifact SHA-256 is
`6f2b418bee783db5daceb08ba8f1338e79cb286f0cb04ec043058ccab53ee181`.
The passing report SHA-256 is
`51f3b180ce093fd76a69ed5f498a29df1c38ad6a9082d8c4a6b8e8313f3e9546`.
The artifact is generated from the already hash-verified CVPR 2026 HT-S
checkpoint and is not committed to the generic NVCR source package.

### Viewable NVCR provider output and speed — 2026-11-02

The verified replay now emits a lossless H.264/YUV420 MP4 containing 17 QCIF
frames at 176x144: one verified native I-frame followed by sixteen predicted
frames produced by the NVCR `dcvcuf.hts.decoder` and
`dcvcuf.hts.reconstruction` provider stages. FFprobe reads all 17 frames at
30 fps. The MP4 SHA-256 is
`ea8d2a6e1c47a7d216caacb6c1af8d2d9bfac18df9714f8ebebef35f02f64efa`.
All provider tensors remain exact, and the 36 contract checks pass.

Three normal-launch timing samples each execute 200 repetitions of two
8-frame chunks after ten warm-up repetitions, including host uploads/downloads.
They report 1471.420, 1424.189, and 1434.829 fps; the median is 1434.829 fps,
or 5.576 ms per eight-frame chunk, on the NVIDIA GeForce RTX 4070. This is
provider synthesis/reconstruction speed from frozen native entropy-decoded
tensors. It is not full UF bitstream decode speed. The I-frame, entropy decode,
remaining context/prior stages, artifact loading, and registered adapter are
still outside this timing boundary.

### Native replay limitation

The initial normal-launch failure came from the pinned upstream setup disabling
deterministic initialization for empty CUDA scratch tensors. The probe now
re-enables `torch.utils.deterministic.fill_uninitialized_memory` after the
upstream environment setup. Three fresh normal-launch processes each decode
the I-frame repeatedly to digest
`04f8e7f2b4edba38f6c130a2879dc3f1eba428967a3ef43a06bf2c53a7b90b3b`, and the
full native-to-provider probe passes without `CUDA_LAUNCH_BLOCKING`. No
reference arithmetic, entropy syntax, checkpoint, or upstream source was
changed.

The older nine-frame stream, SHA-256
`a9bcc803acb31802de58295ca96ddd13149ed592e87ee34e68373436a847ded1`,
was rejected after an earlier repeat failure. Its I payload was 597 bytes,
SHA-256 `75e2ee5efe8d2c3ffc4a18372bc0aa9278f6952776b992b754825ac2806b38e7`;
the current stream's I payload is 601 bytes,
SHA-256 `d1bf03da978629a870a3236750aa8ddd4c503cfb7a583ae19919eeb05004f988`.
The earlier rejection predates the deterministic-initialization fix and is
retained as a compatibility decision; it is not used as current stage evidence.

### Codec-local UF entropy gate

`src/dcvcuf/entropy/uf_rans_codec.cpp` now preserves the UF entropy rules inside
the codec boundary: coder parallelism 1 through 8, packed `y` symbols,
channel-indexed `z` symbols, reversed paired streams with zero-byte overlap,
the offset header and optional tail, decode order, and encoder reset. The CPU
Release conformance executable exercises every partition count, exact decode
round trips, and exact post-reset stream repeatability. This is codec-local
self-conformance only. It does not yet prove exact upstream model-symbol bytes
or connect entropy to the remaining UF model stages and adapter lifecycle.

DCVC-RT regression: the Release engine-contract and I/P round-trip executables
pass on `build/engines-rtx4070-exact-20260805/qcif` (176x144, 202 payload bytes).
Their codec/provider selections are now explicit, repairing the missed test
callers after removal of implicit configuration defaults. No performance
measurement or broad GPU campaign was run.

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

The current NVCR stage run is `build/dcvcuf-nvcr-final-20261101-2/report.json`;
its independent capture replay is run `-1`. The stage-only replay result is
`build/dcvcuf-nvcr-final-20261101-1/replay.json`. Commands are in the probe
README. Raw vectors and build products stay local; compact records are retained
under `nvcr_provider_session`, `provider_tensor_replay`, and
`native_async_replay` in the adjacent JSON evidence.

## Native NVCR CLI gate - 2026-11-02

The opt-in CUDA 12.6/SM89 Release build compiles the pinned DMCI and HT-S fused
sources into libnvcr and registers dcvc-uf with dcvcuf-upstream-cuda. Python is
used only by the offline checkpoint-to-bundle converter. The linked CLI has no
Python or libtorch_python dependency.

A normal NVCR CLI run encoded nine QCIF Akiyo frames as one intra access unit
and one eight-frame predicted access unit, then decoded all nine frames. The
commands used build-dcvcuf-native-cu126/cli/nvcr encode and decode with codec
dcvc-uf, provider dcvcuf-upstream-cuda, QP 32, GOP 32, and the converted
dcvcuf-cvpr2026-hts.pt bundle.

After the golden fixes below, this nine-frame stream is 1,450 bytes with SHA-256
39da424d492bd5dd4b7588b97d4e10e091782446736517f6b2b5d924d047f7bb and the
342,144-byte decoded YUV has SHA-256
aee8172786b4008163c902b3145b46f46d5301cfd86a256924bfefed8fff8352. These
supersede the hashes first recorded here, which came from the earlier FP32
input normalization and matched only NVCR's own proxy, not upstream. A
five-frame run also passes: the encoder pads the short HT-S tail internally and
the decoder emits only the five real timestamps and frames.

## Pinned Python/native golden - 2026-11-02

The first long runs exposed three departures from pinned upstream
`test_video.py`, each now fixed in codec-local DCVC-UF code:

- Frame type was decided per frame, so GOP 32 split an HT-S group at frame 65.
  Upstream decides once per step (one I frame or one eight-frame group); NVCR now
  does the same. With intra period 32 this places I frames only at 0 and 33,
  exactly as upstream does.
- HT-S feature memory was never reset. Upstream resets when
  `(step + 8) % reset_interval == 1` (default 32) and signals the bit in the
  stream. NVCR now does the same and carries the bit in the previously reserved
  predicted-payload byte 6; streams without a reset are unchanged.
- Input normalization ran in FP32 with one rounding. NVCR now runs upstream's
  tensor operations: nearest-neighbour uint8 4:4:4 expansion (equal to
  `scipy.ndimage.zoom(order=0)` for a factor of two), then FP16 conversion,
  `/255` and `-0.5`.

Upstream ran with the pinned checkpoints, QP 32, `--reset_interval 32` and the
default skip threshold 0, which are the values NVCR currently fixes. Every
upstream entropy unit appears byte-identical in the NVCR stream with the same
entropy-parallel and reset fields, and NVCR's decoded YUV equals upstream's
saved reconstruction:

| Sequence | Frames | Intra period | Units identical | Decoded YUV |
|---|---:|---|---:|---|
| Akiyo QCIF | 300 | 32 | 40/40 | identical |
| BasketballDrive 1080p | 500 | 32 | 65/65 | identical |
| BasketballDrive 1080p | 500 | -1 (NVCR GOP 65536) | 64/64 | 32.747 dB mean per-frame PSNR-Y vs upstream float 32.758 dB |

Upstream's intra-period-32 schedule never places another I frame or reset after
frame 33. On BasketballDrive this degrades to 6.79 dB PSNR-Y over frames
400-499 in both runtimes; it is upstream behavior, not an NVCR defect. Upstream's
own test configuration uses intra period -1, which NVCR reproduces with any GOP
larger than the sequence. The CLI's PSNR is computed from pooled MSE; upstream
averages per-frame PSNR, and the two agree when averaged the same way.

The 500-frame 1080p measured run (33 warm-up frames, session reset, rewind)
produces the same stream and decoded YUV as a fresh run.

## Performance - 2026-11-02

Nsight Systems and synchronized stage timers showed NVCR's host-side work costing
almost as much as the fused models: per 1080p eight-frame group, 28.7 ms of CPU
4:4:4 expansion and copies against 38.4 ms of encoder model time, and 18.5 ms of
per-frame output conversion and copies against 49.5 ms of decoder model time.
Every session reset and decoded I frame also rebuilt both fused models, which
discarded processed weights and all captured CUDA graphs.

The session now uploads only 4:2:0 planes in one pinned copy per group and
expands them on the device, converts all decoded frames of a group with one
device-to-host copy, and keeps its models across resets. Upstream also keeps its
models for a whole sequence; an I frame replaces the HT-S reference state. All
golden vectors above were re-run after this change and remain byte-identical.

BasketballDrive 1080p, QP 32, intra only at frame 0, RTX 4070 otherwise idle,
100 measured frames after 10 warm-up frames, ten independent processes per
operation:

| Operation | Process FPS (mean, SD) | Codec FPS (mean) |
|---|---:|---:|
| Encode | 51.6 (0.3) | 183.8 |
| Decode | 43.8 (0.6) | 135.1 |

Process FPS here is a Python `perf_counter` wrapper, not
`scripts/measurement_campaign.py`. About 1.3 s of each process is startup:
0.28 s process and library load, 0.52 s weight load, and first-use CUDA graph
capture for all 64 QPs as upstream does. Upstream's own timer, CUDA events
around `compress`/`decompress` only, gives 252.5 / 183.4 FPS on this GPU. A
100-frame 1080p encode-plus-decode round trip takes 3.93 s in NVCR and 14.81 s
in upstream Python (five runs each; the Python process also computes PSNR).

## Remaining gates

1. Decide the provider. `dcvcuf-upstream-cuda` is a second provider; on
   2026-11-02 the user directed a port to the existing TensorRT provider.
2. Make the upstream README recipe configurable (intra period -1,
   `--reset_interval 128`, `--skip_thres 0.15`). The decoder also depends on the
   skip threshold, so it must be carried in the stream.
3. Close target packaging and clean release-suite gates before calling C7
   complete.
