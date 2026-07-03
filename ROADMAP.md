# NVCR DCVC-RT to FFmpeg roadmap

This is the source of truth for work toward a fast, FFmpeg-integratable DCVC-RT
implementation. Repository agents must follow `AGENTS.md` when updating it.

Last reviewed: 2026-07-02

## Objective

Deliver a native DCVC-RT library that:

- matches pinned Python DCVC-RT behavior;
- meets or beats its warmed encode/decode latency on the same hardware;
- supports complete I/P GOP operation, reset, seeking, drain, and flush;
- exposes a versioned C ABI without C++/CUDA/TensorRT types;
- supports FFmpeg `libdcvcrt` encoder and decoder wrappers;
- defines an elementary stream and a usable FFmpeg container path.

## Milestone status

| Milestone | State | Exit gate |
|---|---|---|
| M0 — Baseline and entropy | Complete | Golden rANS tests and repeatable baseline |
| M1 — GPU-resident I-frame | Active | Warmed native I-frame latency ≤ Python |
| M2 — Elementary stream | Pending | Cross-runtime I/P golden streams |
| M3 — Predicted frames | Active | Two complete conformant GOPs |
| M4 — Deployment artifacts | Pending | Reproducible validated model/engine loading |
| M5 — Stable C ABI | Pending | C-only encode/decode/drain/reset tests |
| M6 — FFmpeg codec wrapper | Pending | `libdcvcrt` transcode tests |
| M7 — Container integration | Pending | Seekable mux/demux round trip |
| M8 — Zero-copy and hardening | Pending | CUDA-frame path and release matrix |

Current milestones: **M1 — GPU-resident execution** and **M3 — Predicted frames**

Project completion rule: an all-intra-only multi-frame path is **incomplete**.
Normal encoding must use the configured I/P GOP and pass M3 before NVCR can be
described as a DCVC-RT video encoder.

Current next action: add CUDA-event stage timing and allocation/transfer counters,
then replace `run_host_engine` with reusable device-bound execution for both I-
and P-frame graphs. Normal I/P encoding works, but remains performance-incomplete
until intermediate tensors and the DPB stay GPU-resident.

## M0 — Baseline and entropy

State: **Complete**

- [x] Pin the authoritative DCVC-RT rANS implementation.
- [x] Add single/two-coder golden vectors and wide-CDF coverage.
- [x] Add an opt-in 1080p-sized entropy benchmark.
- [x] Optimize CDF lookup, coder scheduling, and output copies.
- [x] Record native and Python end-to-end baselines.

Exit criteria:

- [x] Core and TensorRT tests pass.
- [x] Golden streams remain byte-identical.
- [x] Benchmark protocol and results are documented.

## M1 — GPU-resident I-frame

State: **Active**

Measurement:

- [ ] Add per-stage CPU and CUDA-event timing.
- [ ] Count per-frame allocations, transfers, bytes, and synchronizations.
- [ ] Add warmed QCIF, 720p, and 1080p benchmarks with machine-readable output.

Device execution:

- [ ] Add an owned `DeviceTensor` and bounded per-context CUDA arena.
- [ ] Replace `run_host_engine` with reusable device-address binding.
- [ ] Bind TensorRT outputs directly to downstream inputs where possible.
- [ ] Remove steady-state `cudaMalloc`, `cudaFree`, and unconditional syncs.
- [ ] Use pinned staging only at unavoidable CPU/GPU boundaries.

CUDA prior operations:

- [ ] Run padding, quantization, masks, prior processing, quarter reduction,
  restore, scale indexing, and combined-symbol generation on the GPU.
- [ ] Copy only entropy symbols/indexes required by CPU rANS.

FFmpeg-oriented frame boundary:

- [ ] Add planar frame views with per-plane strides.
- [ ] Accept and produce YUV420P8 without RGB intermediates.
- [ ] Separate visible dimensions from padded tensor dimensions.

Exit criteria:

- [ ] I-frame reconstruction is conformant at QCIF, 720p, and 1080p.
- [ ] No device allocation occurs in steady-state frame processing.
- [ ] Intermediate TensorRT tensors do not round-trip through host memory.
- [ ] Warmed 720p and 1080p latency meets or beats Python under
  `docs/performance.md`.

## M2 — Elementary stream

State: **Pending**

- [ ] Decide and document upstream byte compatibility. Preferred: **yes**.
- [ ] Separate codec access units from `NVCR`/`NVCS` envelopes.
- [ ] Define versioned extradata with codec/model identity and required features.
- [ ] Define frame syntax for type, QP, partitions, reset, and payload lengths.
- [ ] Define limits, version behavior, parser/writer, and fuzz targets.
- [ ] Add Python→native and native→Python I/P golden vectors.

Exit criteria:

- [ ] A normative syntax document is reviewed.
- [ ] I/P access units work in both runtime directions.
- [ ] Timestamps/container metadata are not duplicated in access units.

## M3 — Predicted frames

State: **Active**

- [x] Remove the CLI preflight that rejected normal multi-frame GOPs.
- [x] Make default multi-frame encoding produce the configured I/P GOP.

- [x] Export and validate all seven P-frame TensorRT engines.
- [x] Implement temporal context, residual prior, entropy, and reconstruction stages.
- [ ] Keep DPB and latent references GPU-resident.
- [x] Implement QP shifts, feature adaptation, reset intervals, and GOP rules.
- [ ] Implement deterministic drain, flush, reset, and seek recovery.

Exit criteria:

- [ ] Two complete GOPs match pinned Python reconstruction.
- [ ] Cross-runtime I/P stream tests pass.
- [ ] Reset/drain tests and P-frame performance gates pass.
- [ ] The CLI never silently substitutes all-I frames for a requested GOP.

## M4 — Deployment artifacts

State: **Pending**

- [ ] Version ONNX, entropy, quantization, and manifest files as one bundle.
- [ ] Validate hashes and compatibility before frame processing.
- [ ] Key TensorRT caches by GPU, TensorRT, precision, model, and profile.
- [ ] Provide reproducible engine build/cache tooling and clear failures.
- [ ] Resolve checkpoint and generated-engine redistribution licensing.

Exit criteria:

- [ ] A clean install can obtain/build compatible engines reproducibly.
- [ ] Corrupt/incompatible bundles fail during initialization.
- [ ] Stream model identity resolves to an installed decoder bundle.

## M5 — Stable C ABI

State: **Pending**

- [ ] Add opaque encoder/decoder handles and versioned configurations.
- [ ] Add planar frame views and owned packet/frame release functions.
- [ ] Add send/receive, drain, flush, reset, error, log, and allocator APIs.
- [ ] Hide non-ABI symbols and publish pkg-config/CMake metadata.
- [ ] Document one-context-per-stream threading guarantees.

Exit criteria:

- [ ] A C-only example passes encode/decode/drain/reset/reuse tests.
- [ ] No STL, exception, CUDA, or TensorRT type crosses the ABI.
- [ ] ABI compatibility checking runs in CI.

## M6 — FFmpeg codec wrapper

State: **Pending**

FFmpeg support is a compile-time external-library wrapper, not a runtime plugin.
Maintain a patch series or fork pinned to a tested FFmpeg revision.

- [ ] Add `--enable-libnvcr`, codec ID/descriptor, encoder, and decoder.
- [ ] Map AVFrame planes and AVPacket buffers without avoidable copies.
- [ ] Preserve timestamps, duration, color metadata, and keyframe flags.
- [ ] Map drain, flush, reset, seeking, and errors.
- [ ] Add model/device/QP/GOP/reset AVOptions and YUV420P8 advertisement.

Exit criteria:

- [ ] FFmpeg lists `libdcvcrt` as encoder and decoder.
- [ ] Raw YUV → DCVC-RT → raw YUV works through FFmpeg.
- [ ] Transcode, drain, seek/reset, and corrupt-packet tests pass.
- [ ] The wrapper uses only the public C ABI.

## M7 — Container integration

State: **Pending**

- [ ] Add an elementary-stream muxer/demuxer first.
- [ ] Define experimental Matroska mapping and CodecPrivate contents.
- [ ] Preserve time base, dimensions, color metadata, and keyframe index.
- [ ] Validate random access at intra frames.
- [ ] Defer MP4 until a sample entry/configuration record is specified.

Exit criteria:

- [ ] FFmpeg muxes, probes, seeks, demuxes, and decodes a multi-GOP file.
- [ ] Remuxing preserves access units and metadata.
- [ ] Experimental compatibility limits are documented.

## M8 — Zero-copy and hardening

State: **Pending**

- [ ] Add optional CUDA frame descriptors to the C ABI.
- [ ] Add FFmpeg `AV_PIX_FMT_CUDA` via `AVHWFramesContext`.
- [ ] Define CUDA context, stream, event, and frame lifetime ownership.
- [ ] Add bounded-memory, multistream, fuzz, sanitizer, and leak tests.
- [ ] Add supported CUDA/TensorRT/GPU/FFmpeg build and package matrices.

Exit criteria:

- [ ] CPU and CUDA paths pass the release matrix.
- [ ] No unbounded steady-state allocation growth is observed.
- [ ] Performance, conformance, ABI, packaging, and security gates pass.

## Evidence log

Append evidence; never silently replace historical results.

### 2026-07-02 — M0 baseline and entropy optimization

- Hardware: NVIDIA GeForce RTX 4070, driver 580.159.03.
- Pinned Python, FourPeople 1280×720, QP 32, five frames after ten warm-up:
  encode 23.976 ms; decode 21.294 ms.
- Native release with the same protocol: encode 153.674 ms; decode 101.174 ms.
- Entropy before optimization: one coder 91.6 ms; two coders 48.0 ms.
- Entropy after optimization: one coder 46.6 ms; two coders 25.2 ms.
- Core, CUDA, rANS golden/wide-CDF, and TensorRT round-trip tests passed.
- Conclusion: M0 complete; end-to-end performance remains open; M1 active.

### 2026-07-02 — HD timing diagnosis

- Reported native HD encode: 34.224 s for 97 frames, or 352.8 ms/frame.
- The CLI forces `gop_size = 1`; the native backend implements I-frames only.
- A normal Python sequence uses one I-frame followed by P-frames, so its reported
  total is not a like-for-like codec-path comparison.
- Fair Python all-I BQTerrace 1920×1080, QP 32, five frames after ten warm-up:
  encode 55.358 ms; decode 46.239 ms.
- Native all-I measurements are approximately 353 ms encode and 250 ms decode,
  about 6.4× and 5.4× slower respectively.
- Conclusion: M1 host staging is the immediate performance blocker; M3 P-frame
  support is separately required for normal DCVC-RT GOP behavior.

### 2026-07-02 — Reject implicit all-intra encoding

- Removed the CLI override that forced `gop_size = 1`.
- The CLI default is GOP 32 and rejects unsupported multi-frame normal-GOP
  encoding before opening the output.
- `--gop-size 1` remains an explicit development-only I-frame test mode.
- Project completion still requires native P-frames and normal I/P GOP output.

### 2026-07-02 — Native P-frame correctness path and performance audit

- Exported and built seven P-frame TensorRT plans plus pinned P entropy and quant assets.
- Implemented two-pass checkerboard entropy coding, adaptive QP shifts, frame/feature
  reference adaptation, FP16 DPB state, reset interval behavior, and I/P dispatch.
- Native I/P reconstruction equality test passes at 176×144. A 34-frame GOP encoded
  and decoded through frame 33, exercising feature references and the reset branch.
- Debug 1080p BQTerrace: I 2249 ms; steady P 1035–1046 ms.
- Release on the same input: I 383.6 ms; first P 247.8 ms; steady P 210–220 ms.
- The 1–2 s report was a Debug-build artifact; the Release P path is still roughly
  4× slower than the ~55 ms Python GPU target and is not performance-complete.
- Root cause: `run_host_engine` allocates, synchronizes, and round-trips large tensors
  through host memory at every graph boundary; the feature DPB is also host-resident.

### 2026-07-02 — First device-resident P-frame chain

- Nsight baseline over four 1080p frames observed 270 `cudaMalloc`, 299
  `cudaFree`, 482.958 MB H2D, and 280.476 MB D2H operations across initialization
  and frame processing; GPU TensorRT kernels totaled about 38 ms/frame.
- Chained P reference, analysis, hyper-analysis, prior, and synthesis through
  device tensors; only CPU prior/entropy and final DPB data cross the host boundary.
- BQTerrace 1920×1080, QP 32, GOP 97, 97 frames: encode improved from 4.532
  to 5.321 fps (18.230 s), slightly faster than the reported Python 18.70 s run.
- Decode improved from 7.318 to 8.294 fps (11.695 s).
- Native I/P encoder/decoder reconstruction equality and all Release tests pass.
- Remaining gates: CUDA spatial prior/index operations, device-resident DPB,
  allocation arena, I-frame device chain, and same-protocol Python decode evidence.

## Decision log

Append decisions with date, rationale, and consequences.

### 2026-07-02 — Keep the FFmpeg wrapper thin

Decision: finish codec conformance and performance behind a stable C ABI before
building the production wrapper.

Rationale: FFmpeg wrappers are compiled into FFmpeg and use internal `FFCodec`
interfaces. Codec logic in `libnvcr` limits FFmpeg-version coupling.

Consequence: an early experimental patch may validate the boundary but must not
become an alternative codec implementation.

### 2026-07-02 — Keep access units below container framing

Decision: codec access units and decoder configuration will not use the current
`NVCR` packet envelope or `NVCS` sequence framing.

Rationale: FFmpeg owns timestamps, packet metadata, and container framing.

Consequence: current CLI framing remains a development format until M2/M7.
