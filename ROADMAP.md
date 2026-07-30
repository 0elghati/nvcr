# NVCR deployment roadmap

This is the source of truth for the scoped neural video codec runtime
architecture described in `docs/scope-and-support.md`. DCVC-RT is the first and
currently only supported codec backend.

Last reviewed: 2026-07-30

## Objective

Deliver NVCR v1 as a Linux C++20/CUDA/TensorRT runtime architecture and
reproducible local artifact toolchain, validated through the pinned DCVC-RT
CVPR 2025 I/P model pair, that:

- matches pinned Python DCVC-RT behavior on the validated profiles;
- meets or beats its warmed encode/decode latency on the same hardware;
- supports complete I/P GOP operation, reset, drain, and flush;
- validates model and target-local engine bundles before execution;
- defines bounded versioned codec access units; and
- records correctness, performance, memory, bitrate/distortion, and Orin energy
  evidence for RTX 4070 and Jetson Orin Nano.

Additional codec backends, a public C ABI, FFmpeg integration, and container
mapping are post-v1 work. They remain in the roadmap but are not v1 exit
criteria.

## Release status

| Release | State | Exit gate |
|---|---|---|
| 0.2.x | Historical development snapshots | No current readiness claim |
| 0.3.0 — Scope and foundation | Active | Clean RTX checkpoint-to-native I/P workflow |
| 1.0.0 — Scoped product | Pending | RTX 4070 and Orin Nano M1–M4 matrices pass |
| Post-v1 | Pending | M5–M8 gates as separately scheduled |

## Milestone status

| Milestone | State | Exit gate |
|---|---|---|
| M0 — Baseline and entropy | Complete | Golden rANS tests and repeatable baseline |
| M1 — GPU-resident I-frame | Active | Warmed native I-frame latency ≤ Python |
| M2 — Codec access unit and conformance | Active | Cross-runtime I/P golden streams |
| M3 — Predicted frames | Active | Two complete conformant GOPs |
| M4 — Deployment artifacts | Active | Reproducible validated model/engine loading |
| M5 — Stable C ABI | Post-v1 | C-only encode/decode/drain/reset tests |
| M6 — FFmpeg codec wrapper | Post-v1 | `libnvcr` transcode tests |
| M7 — Container integration | Post-v1 | Seekable mux/demux round trip |
| M8 — Zero-copy and hardening | Post-v1 | CUDA-frame path and expanded release matrix |

Current milestones: **M1–M4**, ordered by the 0.3/v1 release gates.

Project completion rule: an all-intra-only multi-frame path is **incomplete**.
Normal encoding must use the configured I/P GOP and pass M3 before NVCR can be
described as a DCVC-RT video encoder.

Current next action: localize the first native/Python entropy index-map
divergence on the pinned 720p QP-32 I frame. Compare z symbols, hyper-synthesis
parameters, and the first y index map before changing any bitstream code; outer
framing, model CDF assets, and the low-level rANS implementation are ruled out.
After restoring a deterministic index contract, rerun bidirectional I/P vectors,
then two complete GOPs and the warmed 720p/1080p release matrix. Remaining
host-staging removal stays open under M1/M3; target support remains pending.

Deployment next action: reproduce the v2 `nvcr-artifacts` workflow and full
registered suite on Orin Nano, then record its clean target, correctness,
performance, memory, rate/distortion, and energy matrix. Engine-cache reuse must
be keyed by model, target, TensorRT/CUDA, precision, and shape profile. The
release installer now downloads every selected-backend engine profile by default
and uses backend/profile aliases plus a backend-neutral default engine slot under
`$XDG_DATA_HOME/nvcr/engines`; exact-tag installer smoke remains pending until
the next published package/engine assets exist. The CLI now warns when
multi-frame `--gop-size 1` all-intra runs are used as performance measurements.
Automatic TensorRT mode now keeps persistent contexts on discrete GPUs and leaves
integrated/Jetson-class devices on the conservative low-memory path. I-frame
encode and decode now keep TensorRT transforms, image-prior processing,
quarter reduction/restore, and synthesis on GPU. Image quantization tensors are
cached once per QP, intermediate TensorRT outputs use the bounded per-session
CUDA arena where lifetimes permit, and obsolete host inference/prior helpers have
been removed. CPU rANS staging and the public decoded-frame download remain the
intentional host boundaries. I-decode now reuses pinned buffers at those entropy
boundaries and uploads packed int8 symbols before converting them to FP16 on the
GPU.

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

- [x] Add per-stage CPU and CUDA-event timing.
- [x] Count per-frame allocations, transfers, bytes, and synchronizations.
- [ ] Add warmed QCIF, 720p, and 1080p benchmarks with machine-readable output.

Device execution:

- [x] Add owned `DeviceTensor` storage.
- [x] Add a bounded per-session CUDA arena.
- [x] Default automatic TensorRT execution mode to persistent contexts on
  discrete GPUs while keeping integrated devices conservative.
- [x] Replace `run_host_engine` with reusable device-address binding for all
  active TensorRT stages and remove the obsolete host-engine implementation.
- [ ] Bind TensorRT outputs directly to downstream inputs where possible.
- [ ] Remove steady-state `cudaMalloc`, `cudaFree`, and unconditional syncs.
  Encode-side manual scratch allocations now use the per-session arena; remaining
  profiled allocations are owned TensorRT outputs, DPB/reference lifetimes,
  entropy downloads, and decode-side staging.
- [ ] Use pinned staging only at unavoidable CPU/GPU boundaries.

CUDA prior operations:

- [x] Run encode-side padding, image-prior quantization, masks, prior processing,
  quarter reduction, scale indexing, and combined-symbol generation on the GPU.
- [x] Copy only encode-side entropy symbols/indexes required by CPU rANS.
- [x] Move decode-side I-frame restore/synthesis to the same device-resident path
  and remove the host-staged reconstruction bridge.

v1 frame boundary:

- [ ] Add planar frame views with per-plane strides.
- [x] Accept and produce YUV420P8 without RGB intermediates.
- [ ] Separate visible dimensions from padded tensor dimensions.

Verification evidence, 2026-07-29 to 2026-07-30:

- [x] Release build after encode-side I-frame GPU residency: `cmake --build build-release -j 8`.
- [x] Release tests after encode-side I-frame GPU residency: `ctest --test-dir build-release --output-on-failure` passed 8/8.
- [x] Release build after encode scratch arena: `cmake --build build-release -j 8` passed.
- [x] Release tests after encode scratch arena: `ctest --test-dir build-release --output-on-failure` passed 8/8.
- [x] Device-resident I-decode recovery and cleanup, 2026-07-30: Release build passed without backend warnings; registered tests passed 6/6; installed v0.4.1 720p/1080p TensorRT bundle validation passed; native I/P round trip passed.
- [x] Repeated 720p GOP-8 decode, FourPeople, 97 frames, QP 32: 42.295, 42.108, and 42.390 fps after a 42.257 fps initial run. Matching encode produced 312,696 payload bytes at 73.891 fps.
- [x] Short 720p profile confirmed no host TensorRT stages. I decode reports one allocation / 5,529,600 bytes, 5 H2D / 1,904,640 bytes, 5 D2H / 6,451,200 bytes, 6 D2D / 11,059,200 bytes, and 5 synchronizations.
- [x] Packed I-decode entropy staging, 2026-07-30: clean temporary Release build passed; registered tests passed 6/6; CUDA coverage includes int8-to-FP16 conversion and decode-side image-prior transforms; native I/P round trip and installed v0.4.1 720p/1080p bundle tests passed.
- [x] The packed path reduced profiled I-frame H2D traffic from 1,904,640 to 952,320 bytes while preserving one allocation / 5,529,600 bytes, 5 H2D, 5 D2H / 6,451,200 bytes, 6 D2D / 11,059,200 bytes, and 5 causal entropy synchronizations.
- [x] Corrected 720p all-I decode, FourPeople, 97 frames, QP 32: 35.434, 35.485, and 35.511 fps; all three output hashes were `3c3da066e973e06352ecc473a9f64e0750ed5075148b43f6105d51bf329fd056`.
- [x] Parent `1407b02` and the corrected device candidate encoded the same 720p GOP-8 access units byte-for-byte: 312,696 payload bytes and SHA-256 `76f6bfc6edc16dea56d073c40eb431256a27600291653fecd7a2dc9c67168c08`. Decoding that stream measured 30.673 fps on the parent host path and 43.098 fps on the corrected device path.
- [x] Development-only source comparison for that 97-frame GOP-8 stream measured 22.872802 dB average PSNR on the parent decoder and 31.569734 dB on the corrected device decoder. This identifies a missing decode-side image-prior sigmoid transform in the recovered path, but does not replace the required pinned-Python golden comparison.
- [x] Superseded experiment: pinned staging and CUDA events without packed symbol uploads averaged 42.109 fps versus a 42.264 fps baseline (-0.37%, within run noise), so it was not accepted as an independent speed claim.
- [x] Encode scratch arena profile: 720p all-I, 3 frames, QP 32, profile `720p-fp16`, `--profile` showed I-frame allocations reduced to 10 allocations / 24,408,512 bytes, with remaining hot TensorRT enqueue stages around `i_analysis` 5.3 ms and `i_synthesis` 10.1 ms.
- [x] Encode scratch arena 720p sweep on FourPeople, 97 frames, QP 32, profile `720p-fp16`: GOP 1 = 30.168 fps, GOP 8 = 73.689 fps, GOP 97 = 94.239 fps.
- [x] CLI logging overhead cleanup: default encode/decode now omit per-frame progress and info startup logs unless `--verbose` is passed; quiet 720p all-I check produced 30.616 fps.
- [x] 720p entropy threshold update: enabling two rANS coders at exactly 1280x720 and removing unused normal I-frame latent serialization improved all-I FourPeople from commit `f88c2b2e24be` 29.154 fps to 30.452 fps in a single 97-frame run. A no-host-reconstruction experiment was rejected after regressing all-I to 24.683 fps.
- [x] Commit `f88c2b2e24be` 720p GOP sweep on FourPeople, 97 frames, QP 32, profile `720p-fp16`: GOP 1 = 29.154 fps, GOP 2 = 44.174 fps, GOP 4 = 60.082 fps, GOP 8 = 72.978 fps, GOP 16 = 81.792 fps, GOP 32 = 87.487 fps, GOP 97 = 93.916 fps.
- [x] 720p all-I profile after making the DPB bridge verification-only: `./build-release/cli/nvcr encode -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv -o /tmp/fourpeople_720p_alli_fast.nvcr -s 1280x720 -r 30 --frames 97 --gop-size 1 --qp 32 --engine-profile 720p-fp16 --profile` produced 97 frames in 3.353 s (28.927 fps); warmed I frames were ~34.2-34.8 ms.
- [x] 720p GOP-8 profile sample before the bridge fix: `./build-release/cli/nvcr encode -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv -o /tmp/fourpeople_720p_opt.nvcr -s 1280x720 -r 30 --frames 97 --gop-size 8 --qp 32 --engine-profile 720p-fp16 --profile` produced 97 frames in 2.405 s (40.331 fps); warmed P frames were ~10.3-10.9 ms.
- [ ] Clean-release warmed 720p/1080p matrix and repeated JSON benchmark output are not recorded yet.
- [x] Paired 720p/1080p diagnostic smoke added with `scripts/benchmark_resolution_pair.sh`; current single-run baseline: 720p GOP 1 = 30.518 fps, 720p GOP 97 = 94.170 fps, 1080p GOP 1 = 13.254 fps, 1080p GOP 97 = 44.103 fps.
- [x] 1080p all-I diagnostic profile after scratch arena: 10 allocations / 54,477,632 bytes per I frame; hot enqueue stages are `i_synthesis` about 22-23 ms, `i_analysis` about 12 ms, and three spatial priors about 9 ms combined.
- [x] RTX 4070 warmed-process diagnostic at commit `4b7e181`, 3 encode and 3 decode runs per point: 720p GOP1 encode/decode = 30.524667/35.340000 fps; 720p GOP97 = 94.773000/44.292000 fps; 1080p GOP1 = 13.160000/14.967000 fps; 1080p GOP97 = 44.182333/19.920667 fps. Per-run JSONL is under `docs/evidence`; the formal gate remains open because warm-up used a separate process.
- [x] Superseded Week-2 baseline: the pre-fix Python/native output diverged at frame 1 and averaged 17.883866 dB over 97 frames; retained as failed history.
- [x] Direct decoded-frame YUV420P8 output removes the full-range BT.709 YCbCr to RGB to limited-range BT.601 YUV conversion chain. At 720p QP 32, source quality is 38.627594 dB for Python and 38.619461 dB for native; Python versus native reconstruction is 43.950954 dB.
- [x] CLI `decode --quality-metrics REFERENCE.yuv` reports aggregate Y, U, V, and 6:1:1 weighted PSNR outside codec timing.
- [x] Pinned 720p QP-32 I-frame golden checks exact upstream commit, checkpoint, source, Python bitstream, and Python reconstruction identities, then gates native source quality and per-plane/weighted Python-native PSNR. Evidence: `docs/evidence/2026-07-30-rtx4070-dcvcrt-i-frame-reference.json`.

Exit criteria:

- [ ] I-frame reconstruction is conformant at QCIF, 720p, and 1080p.
- [ ] No device allocation occurs in steady-state frame processing.
- [x] Intermediate TensorRT tensors do not round-trip through host memory.
  CPU rANS indexes/symbols and the public decoded-frame output remain explicit
  host boundaries.
- [ ] Warmed 720p and 1080p latency meets or beats Python under
  `docs/performance.md`.


## M2 — Codec access unit and conformance

State: **Active**

- [x] Decide and document upstream byte compatibility: v1 does not promise
  upstream byte or payload interchangeability. Pinned two-frame framing-only
  probes decode structurally in both directions but fail quality from the first
  I frame; independent native and Python reconstruction comparison remains the
  reference conformance contract.
- [x] Separate codec access units from `NVCR`/`NVCS` envelopes.
- [x] Define a versioned access unit with model identity, dimensions, and features.
- [x] Define syntax for frame type, effective QP, reset state, and payload lengths.
- [x] Define bounds/version behavior and add parser/writer plus deterministic fuzz tests.
- [x] Separate the generic codec backend/session boundary from the DCVC-RT TensorRT implementation.
- [ ] Add Python→native and native→Python I/P golden vectors. The initial
  720p two-frame probe is preserved as failed evidence: Python→native reaches
  `24.864091 dB`, native→Python reaches `25.152377 dB`, and both diverge at
  the I-frame entropy/index-map boundary despite byte-identical CDF assets.
- [x] 720p frame-1 quality parity restored at QP 32 by returning direct YUV420P8 reconstruction; durable cross-runtime golden vectors remain the next M2 action.
- [x] Add an opt-in deterministic Python/native 720p I-frame reconstruction golden at QP 32 with machine-readable evidence.

Exit criteria:

- [ ] The normative syntax document and compatibility position are reviewed.
- [x] I/P access units work in both native runtime directions.
- [x] Timestamps/container metadata are not duplicated in access units.

## M3 — Predicted frames

State: **Active**

- [x] Remove the CLI preflight that rejected normal multi-frame GOPs.
- [x] Make default multi-frame encoding produce the configured I/P GOP.
- [x] Export and validate all seven P-frame TensorRT engines.
- [x] Implement temporal context, residual prior, entropy, and reconstruction stages.
- [x] Keep P-frame DPB frame/feature references GPU-resident.
- [x] Implement QP shifts, feature adaptation, reset intervals, and GOP rules.
- [x] Cover native two-GOP and reset/reuse behavior on the RTX integration bundle.
- [ ] Complete deterministic drain/flush semantics and their pending-work tests.
- [x] Repeated 720p GOP-97 encode and decode are byte-deterministic. Fresh
  pinned-Python comparison has `45.241436 dB` average and `44.137497 dB`
  minimum Python/native PSNR-YUV; no frame is below `40 dB`. The earlier
  frame-2 divergence is superseded by current evidence.

Exit criteria:

- [ ] Two complete GOPs match pinned Python reconstruction.
- [ ] Cross-runtime I/P stream tests pass.
- [ ] Reset/drain tests and P-frame performance gates pass.
- [x] The CLI never silently substitutes all-I frames for a requested GOP.

## M4 — Deployment artifacts

State: **Active**

- [x] Version ONNX, entropy, quantization, and model manifests as one v2 bundle.
- [x] Validate model/engine hashes and compatibility before plan deserialization.
- [ ] Key reusable TensorRT caches by GPU, TensorRT/CUDA, precision, model, and profile.
- [x] Provide one profile-aware `prepare`/`build`/`inspect`/`validate` command.
- [x] Add a one-command release installer that downloads the latest package plus
  backend-selected engine bundles into profile aliases and a backend-neutral
  default engine slot.
- [x] Adopt a local-build/no-redistribution policy pending an explicit rights review.

Exit criteria:

- [ ] An exact-tag clean install can build compatible engines reproducibly.
- [ ] The release installer succeeds from a published tag and validates every
  downloaded backend engine bundle plus the selected default profile.
- [x] Corrupt/incompatible bundles fail during initialization.
- [x] Access-unit model identity resolves to the configured decoder bundle.

## M5 — Stable C ABI

State: **Post-v1**

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

State: **Post-v1**

FFmpeg support is a compile-time external-library wrapper, not a runtime plugin.
Maintain a patch series or fork pinned to a tested FFmpeg revision.

- [ ] Add `--enable-libnvcr`, codec ID/descriptor, encoder, and decoder.
- [ ] Map AVFrame planes and AVPacket buffers without avoidable copies.
- [ ] Preserve timestamps, duration, color metadata, and keyframe flags.
- [ ] Map drain, flush, reset, seeking, and errors.
- [ ] Add model/device/QP/GOP/reset AVOptions and YUV420P8 advertisement.

Exit criteria:

- [ ] FFmpeg lists `libnvcr` as encoder and decoder.
- [ ] Raw YUV → DCVC-RT → raw YUV works through FFmpeg.
- [ ] Transcode, drain, seek/reset, and corrupt-packet tests pass.
- [ ] The wrapper uses only the public C ABI.

## M7 — Container integration

State: **Post-v1**

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

State: **Post-v1**

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

### 2026-07-29 — RTX 4070 720p TensorRT bundle

- Added the `720p-fp16` engine profile and I/P TensorRT shape routing, built all
  14 plans on RTX 4070 (SM 8.9) with CUDA 12.6 and TensorRT 10.7.0, passed all
  per-plan smoke inferences, and validated the resulting
  `nvcr.engine-bundle.v2` checksums and metadata. Removed the two legacy-format
  local engine directories after validation.



Append evidence; never silently replace historical results.

### 2026-07-29 — Codec runtime architecture boundary alignment

- Refactored the public session boundary so `nvcr::Runtime::create` receives
  generic `nvcr::codec::Components` and owns a generic `nvcr::codec::Runtime`
  session. Generic `CodecBackend`, `SequenceState`, and session orchestration now
  live under `include/nvcr/codec` and `src/codec`; `nvcr::dcvcrt` retains
  compatibility aliases and the only concrete TensorRT backend factory.
- Moved DCVC-RT-specific artifact preparation helpers under
  `scripts/backends/dcvcrt/`, while keeping `scripts/nvcr_artifacts.py` as the
  generic installed `nvcr-artifacts` front end. The backend-local directory now
  contains the DCVC-RT prepare helper, TensorRT builder, I/P exporters, and
  engine-manifest writer.
- Updated README, scope/support, architecture, API reference, compatibility,
  getting-started, CLI, bitstream, and DCVC-RT integration docs to frame NVCR as
  a neural video codec runtime architecture whose current v1 support is limited
  to the DCVC-RT backend.
- Updated CMake install rules, package validation, scripts documentation, and CI
  linting so backend-local helpers are shipped and checked recursively.
- Validation: repository text scan found no remaining old public DCVC-RT session
  namespace usage; `bash -n` passed for
  `scripts/backends/dcvcrt/prepare_artifacts.sh` and
  `scripts/backends/dcvcrt/build_tensorrt.sh`; Python compile passed for
  `scripts/nvcr_artifacts.py`, the moved DCVC-RT Python helpers, and
  `tests/artifact_tool_tests.py`; `python tests/artifact_tool_tests.py` passed;
  `git diff --check` passed.
- Not validated in this Windows workspace: CMake configure/build/CTest. The
  sandboxed PowerShell runner intermittently failed to spawn, and the escalated
  shell reported `cmake` was not installed.

### 2026-07-23 — Local RTX 4070 v0.3 foundation validation

- Hardware: NVIDIA GeForce RTX 4070, driver 580.159.03, 12,282 MiB; CUDA 12.6.85 and TensorRT 10.7.0.
- CPU Release configuration passed all 4 registered tests: artifact/profile validation, smoke/access-unit boundaries, deterministic parser fuzz boundaries, and rANS conformance.
- CUDA/TensorRT Release configuration with a profile-bound v2 engine bundle passed all 7 registered tests, including CUDA operators, model/target/engine manifest validation, high-QP P-frame round trip through effective QP 71, two GOPs, reset/reuse, native I-P reconstruction equality, and recorded profile digest plus engine checksum validation.
- Installed CLI completed a two-frame 176x144 YUV420P8 I-P encode/decode; output framing and reconstructed dimensions were validated.
- Release install and the public x86_64 NVIDIA / Jetson L4T 36 package archives passed required-file, forbidden-asset, internal manifest, and SHA-256 checks; the packaged contents contained the required documentation, licenses, notices, and manifests and excluded checkpoints plus derived model and engine assets.
- Status: exact-tag clean-room execution, pinned Python cross-runtime golden vectors, performance/rate-distortion evidence, and the Jetson Orin Nano target matrix remain open; v0.3 and v1.0 are not complete.

### 2026-07-23 — Generic public package families with explicit validation targets

- Updated release packaging and docs so public archives use generic family names
  (`linux-x86_64-nvidia`, `linux-aarch64-jetson-l4t36`) while release gating
  and evidence remain tied to `rtx4070-ubuntu2404` and `orin-nano-l4t3647`.
- Kept TensorRT engine generation target-local in the public workflow and docs;
  engine bundles remain release test inputs rather than release outputs.
- Hardened PR CI so workflow files are parsed during linting,
  `release-assets.yml` edits trigger hosted packaging coverage, and the hosted
  portable CUDA/TensorRT job now hard-gates the generic
  `linux-x86_64-nvidia` archive smoke package with checksum, manifest-presence,
  and forbidden-asset checks. Archive contents are listed once before grep
  checks so `pipefail` cannot turn a successful `grep -q` match into a tar
  SIGPIPE failure.
- Validation: `bash -n scripts/package_release.sh`,
  `./scripts/package_release.sh --help`, YAML parse of
  `.github/workflows/release-assets.yml` and `.github/workflows/ci.yml`,
  release-please JSON validation, and `git diff --check`.


### 2026-07-23 — Align release automation with standard GitHub-hosted runners

- Reworked release automation so the default repository contract assumes only
  GitHub standard hosted runners: `release-assets.yml` now builds and uploads
  the generic x86_64 NVIDIA package on hosted Ubuntu 24.04 and leaves the
  release in draft.
- Moved exact RTX 4070 / Jetson Orin Nano validation, Jetson package creation,
  and final publication to explicit manual release steps recorded in roadmap
  evidence instead of unreachable self-hosted Actions jobs.
- Converted the old main-push GPU workflow into an optional manual self-hosted
  helper disabled by default, pinned the hosted CUDA/TensorRT CI gate to
  Ubuntu 24.04, minimized its CUDA/TensorRT package install to avoid
  hosted-runner disk pressure, and reduced hosted CPU CI to Release-only
  coverage on the standard x86_64 and arm64 runners.
- Validation: YAML parse of all workflow files, release-please JSON validation,
  `bash -n scripts/ci/install_cuda_tensorrt.sh`, and `git diff --check`.

### 2026-07-23 — Optional engine assets distributed from GitHub Releases

- Added a separate reviewer-convenience engine asset path that keeps the generic
  binary packages engine-free while allowing validated target-bound TensorRT
  bundles to be attached as package-family assets to a draft GitHub Release.
- Added `scripts/package_engine_bundle.sh` so current RTX/Jetson engine
  directories are packaged under manifest-derived package-family names:
  `nvcr-vX.Y.Z-<package-family>-dcvcrt-cvpr2025-<engine-profile>-engines.tar.gz`.
- Added a manual `upload-engine-assets.yml` workflow that downloads staged
  archives, verifies caller-supplied SHA-256 digests, validates safe archive
  structure, runs the tagged `nvcr-artifacts validate` against each extracted
  bundle, checks filename-to-manifest identity, uploads the archive plus checksum
  to the GitHub Release, and only publishes the release when explicitly
  confirmed.
- Decision: staging services such as OneDrive may be used as temporary upload
  inputs, but public/reviewer downloads should come from GitHub Release assets.
  These engine assets remain target-bound evidence artifacts, not generic
  TensorRT portability claims and not package contents.
- Validation: `bash -n scripts/package_release.sh` and
  `scripts/package_engine_bundle.sh`, Python compile of the engine upload helper,
  workflow YAML parse, JSON validation, `git diff --check`, a fake public-package
  manifest smoke test, and an end-to-end fake engine-bundle
  package/download/validate smoke test.

### 2026-07-25 — Engine staging input helper

- Added `scripts/stage_engine_release_asset.sh` to package a validated engine
  bundle, optionally upload it to S3, and generate the `engine_assets.txt` row
  consumed by `upload-engine-assets.yml`.
- Added an AWS CDK app for a private S3 release-assets staging bucket in account
  `<aws-account-id>`; staged engine uploads now use S3 plus presigned HTTPS URLs
  rather than public OneDrive/share links.
- Decision: the helper does not infer direct-download links from browser share
  links because those links are provider-specific and often point to preview or
  login HTML rather than archive bytes. The workflow keeps SHA-256 and tar
  validation as the authority before uploading to GitHub Releases.
- Clarified that `upload-engine-assets.yml` must be available on the selected
  branch before it can be manually dispatched. A local `gh workflow run` command
  only sends the input text to GitHub; the download, validation, and release
  upload execute on the GitHub-hosted runner.


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

### 2026-07-07 — Orin Nano TensorRT build bring-up

- Hardware/software: Jetson Orin Nano, L4T 36.4.7, CUDA toolkit 12.6.68,
  TensorRT 10.3.0.30.
- Installed local CMake 3.29.8 under `.tools/` because Ubuntu Jammy provides
  CMake 3.22.1 and the project requires 3.24 or newer.
- Configured Release TensorRT build with explicit CUDA compiler and Orin
  architecture:
  `-DNVCR_ENABLE_TENSORRT=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCMAKE_CUDA_ARCHITECTURES=87
  -DTensorRT_ROOT=/usr`.
- Built `libnvcr`, the `nvcr` CLI, CUDA ops tests, and TensorRT test targets in
  `build-orin-release`; installed to `install-orin`.
- Verification: sandboxed CTest passed CPU/rANS/CLI tests but could not open
  Jetson GPU memory management. Re-running with direct device access passed all
  four configured tests: `nvcr_smoke_tests`, `nvcr_rans_conformance`,
  `nvcr_cli_accepts_inter_gop`, and `nvcr_cuda_ops`.
- Runtime artifact blocker: no DCVC-RT ONNX files, entropy/quant assets, or
  TensorRT plans were present. Upstream DCVC was cloned to the expected path, but
  the recorded reference commit was not reachable from the current upstream clone,
  and this machine lacks the Python export stack (`torch`, `onnx`, `onnxscript`)
  plus the two pretrained checkpoints needed to generate engines.

### 2026-07-07 — Checkpoint-to-engine artifact pipeline

- Added `scripts/prepare_dcvcrt_artifacts.sh` as the end-user pipeline from the
  public Microsoft DCVC checkout and `cvpr2025_*.pth.tar` checkpoints to
  exported ONNX/runtime assets and target-local TensorRT plans.
- Added `docs/dcvcrt-artifacts.md` with local and remote upstream paths,
  checkpoint URLs, expected checkpoint hashes, Orin-local engine generation, and
  direct `nvcr encode`/`decode` examples.
- Updated README, Getting Started, CLI, scripts, and integration docs to separate
  portable ONNX/assets from non-portable TensorRT `.plan` files and to document
  the platform-tag mismatch recovery path.
- Verification: `bash -n scripts/prepare_dcvcrt_artifacts.sh` passed and
  `./scripts/prepare_dcvcrt_artifacts.sh --help` printed the expected options.
- Remaining gate: actual ONNX export and engine generation still require the two
  Microsoft checkpoint files and a Python environment with `torch`, `onnx`, and
  `onnxscript`.

### 2026-07-08 — Orin Nano local install verification

- Configured a fresh Release TensorRT build in `build-orin-install` with
  `-DNVCR_ENABLE_TENSORRT=ON`,
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`, and
  `-DCMAKE_CUDA_ARCHITECTURES=87`.
- Built `libnvcr`, `nvcr`, examples, CUDA ops tests, and TensorRT test targets.
- Installed successfully to the local prefix `install-orin`; installed artifacts
  include `bin/nvcr`, `lib/libnvcr.a`, public headers, and CMake package files
  under `lib/cmake/NVCR`.
- Verified `install-orin/bin/nvcr --help` runs from the installed prefix.
- In the available environment, `nvcr_smoke_tests`,
  `nvcr_rans_conformance`, and `nvcr_cli_accepts_inter_gop` passed;
  `nvcr_cuda_ops` failed with `NvRmMemInitNvmap failed` because `/dev/nvmap` and
  `/dev/nvhost-*` were not visible. This reflects missing device-node access,
  not an install layout failure.
- Documentation updated with the Jetson Orin Nano configure/install command,
  local-prefix option, and CUDA device-node verification caveat.

### 2026-07-08 — Orin Nano 1080p TensorRT engine generation

- Confirmed copied `build/engines/dcvcrt-1080p` plans were not suitable evidence
  for the current Orin runtime because TensorRT `.plan` files are target-specific.
- Host memory before the successful build: 7.4 GiB total, about 5.6 GiB
  available, swap mostly unused. The largest user processes were VS Code server
  components; no heavyweight Python/export process needed to be stopped.
- Initial 1080p TensorRT build with the default 2048 MiB workspace and builder
  optimization level 3 hit Orin GPU/shared allocation pressure during tactic
  search.
- Added `--workspace-mib` and `--builder-optimization-level` knobs to
  `prepare_dcvcrt_artifacts.sh` and `build_dcvcrt_tensorrt.sh`.
- Generated all fourteen target-local 1080p Orin plans in
  `build/engines/dcvcrt-1080p-orin` using:
  `CUDA_MODULE_LOADING=LAZY ./scripts/prepare_dcvcrt_artifacts.sh --skip-clone
  --skip-export --models build/models/dcvcrt --engines
  build/engines/dcvcrt-1080p-orin --trtexec /usr/src/tensorrt/bin/trtexec
  --optimization-point 1080p --workspace-mib 512
  --builder-optimization-level 1 --skip-smoke`.
- `sha256sum -c build/engines/dcvcrt-1080p-orin/engine.sha256` passed for all
  fourteen plans.
- `./build-orin-install/tests/nvcr_tensorrt_engine_tests
  build/engines/dcvcrt-1080p-orin` passed.
- `./build-orin-install/tests/nvcr_tensorrt_roundtrip_tests
  build/engines/dcvcrt-1080p-orin` passed:
  native DCVC-RT I/P round trip at 176x144, 165-byte payload.
- Documentation updated with the tested low-memory Orin Nano 1080p engine
  generation flags.

### 2026-07-08 — TensorRT engine bundle compatibility guard

- Trigger: observed RTX 4070 P-frame encode latency around 49-50 ms and confirmed
  that reusing TensorRT `.plan` files across GPU models can deserialize with
  warnings or performance loss instead of a clear NVCR-owned failure.
- Added `scripts/write_tensorrt_engine_manifest.py` and updated
  `scripts/build_dcvcrt_tensorrt.sh` to stamp each engine directory with
  `engine_manifest.json` containing TensorRT version, precision, optimization
  point, workspace, builder optimization level, GPU name, compute capability, and
  multiprocessor count.
- Added `--device-id` propagation to `prepare_dcvcrt_artifacts.sh` and passed the
  same device to `trtexec` build and smoke commands.
- Updated `TensorRTBackend::initialize` to require `engine_manifest.json` and to
  reject bundles whose TensorRT version or selected CUDA device metadata differs
  before deserializing any plan.
- Updated README, Getting Started, CLI, scripts, artifact, and integration docs
  to describe the new manifest requirement and rebuild rule.
- Verification: `python3 -m py_compile scripts/write_tensorrt_engine_manifest.py`,
  `bash -n scripts/build_dcvcrt_tensorrt.sh`, and
  `bash -n scripts/prepare_dcvcrt_artifacts.sh` passed.
- Verification: `cmake --build build-release --parallel` passed; existing
  TensorRT backend unused-function warnings remain.
- Verification: direct `build-release/tests/nvcr_tests`,
  `build-release/tests/nvcr_rans_conformance_tests`, and
  `build-release/cli/nvcr --help` passed. Full CTest could not run because no
  `ctest` binary is installed in the environment.
- Remaining gate: repeat manifest validation and encode/round-trip tests on the
  RTX 4070 engine bundle before considering the cross-device guard complete
  across the supported target set.

### 2026-07-08 — Rebuilt Orin engines after manifest-only recovery failed

- Observed command: `nvcr encode -i BasketballDrive_1920x1080_50.yuv
  -o /tmp/basketball.nvcr -s 1920x1080 -r 50 --frames 97 --gop-size 97
  --qp 32 --engine-dir build/engines/dcvcrt-1080p-orin` failed during
  initialization because `engine_manifest.json` was missing.
- A manifest-only stamp allowed initialization but TensorRT still printed the
  cross-device plan warning for all fourteen engines, and encode failed with
  `cudaMallocAsync failed: out of memory`. Conclusion: the directory needed a
  true target-local rebuild, not a metadata stamp.
- Added `--stamp-only` recovery support to `build_dcvcrt_tensorrt.sh`, then
  hardened it so stamp-only validation rejects directories when TensorRT reports
  the cross-device plan warning.
- Rebuilt all fourteen 1080p Orin plans in
  `build/engines/dcvcrt-1080p-orin` with:
  `./scripts/prepare_dcvcrt_artifacts.sh --skip-clone --skip-export --models
  build/models/dcvcrt --engines build/engines/dcvcrt-1080p-orin --trtexec
  /usr/src/tensorrt/bin/trtexec --optimization-point 1080p --workspace-mib 512
  --builder-optimization-level 1 --device-id 0 --python python3 --skip-smoke`.
- Verification: `sha256sum -c build/engines/dcvcrt-1080p-orin/engine.sha256`
  passed for all fourteen plans.
- Verification: hardened stamp-only check passed:
  `./scripts/build_dcvcrt_tensorrt.sh --stamp-only --engines
  build/engines/dcvcrt-1080p-orin --trtexec /usr/src/tensorrt/bin/trtexec
  --optimization-point 1080p --workspace-mib 512 --builder-optimization-level 1
  --device-id 0 --python python3`.
- Verification: `./build-release/cli/nvcr encode ... --frames 97 --gop-size 97
  --engine-dir build/engines/dcvcrt-1080p-orin` completed without TensorRT
  cross-device warnings; output `/tmp/basketball.nvcr`, 97 frames, 339046
  payload bytes, codec time 23.540 s, 4.121 fps.

### 2026-07-08 — P-frame profiling and GPU-only encode DPB reference

- Added CLI `--profile` and TensorRT backend per-frame profiling for CPU stages,
  CUDA-event engine timings, allocation counts/bytes, transfer counts/bytes, and
  stream synchronization counts.
- Profiled `BasketballDrive_1920x1080_50.yuv`, 97-frame GOP, QP 32, target-local
  Orin engine bundle. Before the encode-side P reconstruction removal, steady
  P-frames allocated 32 buffers / 125535616 bytes, copied 12535936 bytes H2D,
  14688000 bytes D2H, 8355840 bytes D2D, and synchronized once per frame.
- Diagnosis: steady P-frame host stages included about 44-56 ms input color
  conversion and about 112-117 ms reconstruction download/RGB conversion; CUDA
  engine time was dominated by `p_reference_feature.plan` around 49-56 ms and
  `p_synthesis.plan` around 78-82 ms.
- Removed the encode-side P-frame reconstruction download/conversion. The device
  DPB remains updated from TensorRT outputs, while sequence state now permits a
  predicted commit with an empty host reference after an intra reference exists.
- After removal, the same profile showed P-frame D2H reduced to 3 copies /
  2154240 bytes and per-frame synchronizations reduced to zero.
- Verification: `cmake --build build-release --parallel` passed.
- Verification: `./build-release/tests/nvcr_tests` passed.
- Verification: `./build-release/tests/nvcr_cuda_ops_tests` passed when run with
  CUDA device access; the sandboxed run failed before kernels with NVIDIA memory
  manager initialization unavailable.
- Verification: `./build-release/cli/nvcr encode -i
  /path/to/BasketballDrive_1920x1080_50.yuv -o
  /tmp/basketball.nvcr -s 1920x1080 -r 50 --frames 97 --gop-size 97 --qp 32
  --engine-dir build/engines/dcvcrt-1080p-orin` completed: 97 frames, 339046
  payload bytes, codec time 18.293 s, 5.303 fps. Steady P-frames were mostly
  176-180 ms; approximate P-frame average after the I-frame was 178.2 ms.

### 2026-07-08 — Orin OOM mitigation for 1080p GOP encode

- Observed Orin run: 1080p GOP encode failed with
  `cudaMallocAsync failed: out of memory` and NVMap allocation errors.
- Added allocator resilience in `tensorrt_backend.cpp`: when stream-ordered
  `cudaMallocAsync` returns OOM, synchronize and retry, then fall back to
  `cudaMalloc` for that allocation if needed.
- Reduced TensorRT context residency by removing persistent
  `IExecutionContext` storage from loaded engines; contexts are now created at
  warm-up/invocation boundaries.
- Added explicit device-engine stream synchronization before returning from
  `run_device_engine` to bound in-flight context/device-memory lifetime.
- Verification: `cmake --build build-orin-release --target nvcr --parallel`
  passed.
- Verification: `./build-orin-release/cli/nvcr encode -i
  /path/to/BasketballDrive_1920x1080_50.yuv -o
  /tmp/basketball.nvcr -s 1920x1080 -r 50 --frames 97 --gop-size 97 --qp 32
  --engine-dir build/engines/dcvcrt-1080p-orin` completed: 97 frames, 339046
  payload bytes, codec time 23.333 s, 4.157 fps.
- Consequence: stability improved on constrained Orin memory, but latency
  regressed due to per-engine synchronization/context churn; this remains a
  temporary bridge until bounded reusable device memory/address binding lands.

### 2026-07-08 — Adaptive TensorRT mode for speed vs memory headroom

- Added adaptive TensorRT execution mode selection in
  `tensorrt_backend.cpp`:
  default `low-memory` on GPUs with <= 12 GiB VRAM, `performance` on larger
  GPUs.
- Added `NVCR_TENSORRT_LOW_MEMORY_MODE` environment override with accepted
  values `0/1`, `true/false`, and `on/off`.
- In `low-memory` mode, execution contexts are short-lived and device-engine
  execution synchronizes per stage to reduce in-flight memory pressure.
- In `performance` mode, per-engine execution contexts are cached and reused,
  and the per-stage synchronization is skipped to recover throughput.
- Verification on Orin 1080p GOP-97, QP 32, 20-frame run:
  low-memory mode 2.256 fps, performance mode 4.354 fps.
- Verification on Orin 1080p GOP-97, QP 32, 97-frame run:
  performance mode completed successfully with 339046 payload bytes,
  codec time 18.416 s, 5.267 fps.

### 2026-07-08 — P-path host-staging trim (quant cache + pinned entropy buffers)

- Added persistent device-side video quant cache for all QP values and all four
  P-path quant tensors (`q_encoder`, `q_decoder`, `q_feature`, `q_recon`) to
  remove per-frame host tensor creation and H2D uploads in predicted encode and
  decode.
- Added reusable pinned host buffers for predicted encode entropy staging
  (`z_symbols`, `indexes0`, `indexes1`) and switched rANS encode calls to
  consume `std::span` views directly from those buffers.
- Verification: `cmake --build build-orin-release --target nvcr_cli --parallel`
  passed.
- Verification on Orin, performance mode, 1080p GOP-97, QP 32:
  20-frame run completed at 4.389 fps (codec time 4.557 s), with steady
  P-frames mostly around 175-179 ms.
- Verification on Orin, performance mode, 1080p GOP-97, QP 32:
  97-frame run completed at 5.322 fps (codec time 18.227 s), with steady
  P-frames mostly around 176-179 ms.
- Follow-up: improvements are incremental; to materially beat the current
  ~176-179 ms steady P-frame band, the next larger step is full GPU input color
  conversion and wider per-frame temporary tensor reuse/arena work under M1.

### 2026-07-08 — GPU-native predicted input color conversion

- Added CUDA kernels in `cuda_ops` to convert RGB24 and YUV420P8 input bytes
  directly into padded FP16 YCbCr tensors:
  `rgb24_to_ycbcr_padded` and `yuv420p8_to_ycbcr_padded`.
- Switched predicted encode input path to GPU-native conversion before
  `p_analysis` by uploading raw frame bytes once and launching the new kernels,
  removing the per-frame CPU `rgb_to_ycbcr` / `yuv420p8_to_ycbcr` loop for
  predicted frames.
- Verification: `cmake --build build-orin-release --target nvcr_cli --parallel`
  passed.
- Verification on Orin, performance mode, 1080p GOP-97, QP 32:
  20-frame run completed at 4.477 fps (codec time 4.467 s), with early P-frames
  around 167-176 ms.
- Verification on Orin, performance mode, 1080p GOP-97, QP 32:
  97-frame run completed at 5.403 fps (codec time 17.954 s), with steady
  P-frames mostly around 174-177 ms and occasional lower/high outliers.

### 2026-07-08 — Jetson energy profiling harness

- Added `scripts/profile_energy.py` to wrap encode/decode commands with
  `tegrastats` rail sampling, idle baseline measurement, raw board energy,
  idle-subtracted energy, joules/frame, wall fps, JSON output, and raw log
  capture.
- Updated `docs/performance.md` and `scripts/README.md` with the Orin energy
  protocol. The preferred first-pass rail is `VDD_IN` when present.
- Current interpretation guidance: a reported 240 ms/frame decode on Orin is
  plausible for the present performance-incomplete native path, especially if
  measured as a full CLI decode with reconstruction download, RGB-to-YUV output
  conversion, file I/O, and any low-memory TensorRT context churn. It is not a
  parity target; same-protocol Python decode and per-frame profile evidence are
  still required.
- Verification: `python3 -m py_compile scripts/profile_energy.py`,
  `./scripts/profile_energy.py --help`, and a short `/bin/sleep` wrapper smoke
  test passed on Orin, parsing `VDD_IN`.
- Encode energy capture attempt, 1080p BasketballDrive GOP-97 QP 32, failed in
  both performance and low-memory mode before frame 0 with NVMap/TensorRT OOM.
  Concurrent `tegrastats` showed `RAM 3190/7620MB (lfb 14x4MB)`, so the blocker
  is contiguous NVMap/GPU allocation pressure, not useful encode-energy data.
- Decode energy capture completed in low-memory mode using `/tmp/basketball.nvcr`
  and `build/engines/dcvcrt-1080p-orin`: 97 frames, codec time 32.509 s
  (2.984 fps), wall time 34.945 s (2.776 fps), `VDD_IN` active energy
  436.992 J, idle-adjusted energy 304.432 J, active 4.505 J/frame,
  idle-adjusted 3.138 J/frame.
- Decode post-warmup timing from frames 10-96 in that run: 87 frames averaged
  327.451 ms/frame, min 277.840 ms, max 346.580 ms. This is slower than the
  earlier 240 ms/frame decode report and reinforces that low-memory context
  churn and host/output staging remain performance blockers.
- Next action: repeat encode energy after restoring contiguous NVMap headroom
  (usually a reboot or stopping the processes that fragmented GPU memory), and
  repeat decode in performance mode when it can initialize without OOM.

### 2026-07-22 — Generalized build automation beyond Orin (device/arch auto-detect, one-command install, auto-tuned artifact pipeline)

- Rationale: the prior Orin bring-up still relied on Orin-specific
  CUDA/TensorRT configure flags (`CMAKE_CUDA_COMPILER`,
  `CMAKE_CUDA_ARCHITECTURES=87`, `TensorRT_ROOT`) and artifact-pipeline tuning
  flags (`--workspace-mib 512`, `--builder-optimization-level 1`,
  `--device-id 0`). NVCR needed a default local-install path that auto-detects
  one machine's CUDA/TensorRT/GPU settings plus an explicit portable
  multi-architecture build mode for redistributable binaries.
- Added `cmake/NVCRAutodetect.cmake`: auto-detects `CMAKE_CUDA_COMPILER` (glob
  `/usr/local/cuda*/bin/nvcc` when not on `PATH`) and, when
  `CMAKE_CUDA_ARCHITECTURES` is unset, either single-GPU `compute_cap` via
  `nvidia-smi` (`NVCR_CUDA_ARCH_SET=auto`, default) or a curated multi-arch
  list `75;80;86;87;89;90` for redistributable builds
  (`NVCR_CUDA_ARCH_SET=portable`, new `cmake/NVCROptions.cmake` cache option).
  Wired into `CMakeLists.txt` before `add_library(nvcr)`/`enable_language(CUDA)`
  (target `CUDA_ARCHITECTURES` is snapshotted at target-creation time, so
  ordering is load-bearing).
- Extended `cmake/FindTensorRT.cmake` with a Python `tensorrt` package fallback
  for `TensorRT_ROOT` and multiarch (`aarch64-linux-gnu`/`x86_64-linux-gnu`)
  search paths, plus `TensorRT_VERSION` parsing from `NvInferVersion.h`.
- Added `scripts/detect_platform.sh`: shared helper reporting Jetson-vs-discrete
  platform, GPU architecture/device index (`nvidia-smi
  --query-gpu=index,compute_cap,memory.free`), CUDA/TensorRT locations, and a
  memory-budget-derived `workspace-mib`/`builder-optimization-level` tier.
- Added `scripts/install.sh`: one-command configure/build/install wrapper with
  a `--arch-set {auto,portable}` flag (`auto` default maps to single-GPU
  detection; `portable` selects `-DNVCR_CUDA_ARCH_SET=portable`); explicit
  `--cuda-arch` always takes precedence over `--arch-set`.
- `scripts/prepare_dcvcrt_artifacts.sh` now auto-tunes `--device-id`,
  `--workspace-mib`, and `--builder-optimization-level` from
  `detect_platform.sh` unless `--no-auto-tune` is passed; hardcoded host-local
  paths were removed from it and from `scripts/export_dcvcrt_onnx.py` and
  `scripts/export_dcvcrt_p_onnx.py` in favor of `$repo_root` and
  `NVCR_DCVCRT_ROOT`.
- Updated `README.md`, `docs/getting-started.md`, `scripts/README.md`, and
  `docs/dcvcrt-artifacts.md` to lead with `scripts/install.sh` and
  `scripts/prepare_dcvcrt_artifacts.sh`, to document `--arch-set portable`, and
  to state explicitly that GPU-architecture portability (the `portable` fat
  binary) is independent of host CPU architecture, and that TensorRT `.plan`
  files are never bundled as portable/prebuilt artifacts (always generated
  per-target, guarded by `engine_manifest.json`, per the 2026-07-08 decision
  below). Removed remaining Orin/user-specific hardcoded example paths from
  `docs/dcvcrt-artifacts.md`'s direct-CLI examples.
- Verification (on the same Jetson Orin Nano used for prior bring-up):
  - `cmake -S . -B <tmp> -DNVCR_ENABLE_TENSORRT=ON` with no manual CUDA/TensorRT
    flags auto-detected `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`,
    `CMAKE_CUDA_ARCHITECTURES=87` (via `nvidia-smi`), and
    `Found TensorRT: /usr/include/aarch64-linux-gnu (found version "10.3.0")`;
    `cmake --build <tmp> --target nvcr` compiled `cuda_ops.cu` for `sm_87` and
    linked `libnvcr.a`.
  - `cmake -S . -B <tmp> -DNVCR_ENABLE_TENSORRT=ON -DNVCR_CUDA_ARCH_SET=portable`
    configured with `CMAKE_CUDA_ARCHITECTURES=75;80;86;87;89;90` and
    `cmake --build <tmp> --target nvcr` succeeded in 29 s, compiling
    `cuda_ops.cu` for all six architectures into one `libnvcr.a`.
  - `./scripts/detect_platform.sh` reported platform=jetson, arch=87,
    nvcc/CUDA-root/TensorRT-root/trtexec all correctly resolved, device id=0,
    workspace=512 MiB, builder level=1 — matching the hand-tuned values from
    the 2026-07-08 entries above.
  - `./scripts/install.sh --no-tensorrt --build-dir <tmp> --prefix <tmp-prefix>`
    ran end-to-end (bootstrap check, configure, build, install) and produced a
    working install layout.
  - `./scripts/prepare_dcvcrt_artifacts.sh --skip-clone` printed the correct
    auto-tune report (workspace=512, level=1, device=0) before failing only on
    missing Python `torch`/`onnx`/`onnxscript` (expected, unrelated to this
    change).
  - `bash -n` passed for `detect_platform.sh`, `install.sh`, and
    `prepare_dcvcrt_artifacts.sh`; `python3 -m py_compile` passed for both
    export scripts.
- Remaining gate: this automation has only been exercised on one Jetson Orin
  Nano; M4's checkpoint/engine redistribution-licensing and hash-validation
  bullets remain open, and no discrete RTX/datacenter GPU run has yet
  re-verified the `portable` arch-set path end-to-end (only configure+build
  evidence exists, gathered on Jetson hardware). Release packaging automation
  remained separate from this change set.

### 2026-07-22 — Binary install and fallback engine-build docs

- Added `docs/install-binary.md` with a binary-first install flow: platform
  detection, release archive download, engine-bundle
  download, and quick CLI verification.
- Updated `docs/getting-started.md` to present a fast user path first and keep
  source-build as a separate developer path.
- Updated `docs/dcvcrt-artifacts.md` with explicit preflight checks and a
  dedicated "no prebuilt engine bundle" fallback section for local
  target-machine engine generation.
- Updated `docs/README.md` reading order and quick links to include the new
  binary install path.
- Scope note: documentation-only restructure; no runtime or performance behavior
  changes were introduced.

### 2026-07-22 — Release-please and GitHub release automation bootstrap

- Added `release-please-config.json`, `.release-please-manifest.json`,
  `version.txt`, and `CHANGELOG.md` to bootstrap manifest-driven
  release-please for the root package.
- Added `.github/workflows/release-please.yml` to open release pull requests,
  bump versions, tag releases, and publish GitHub Releases from conventional
  commits.
- Added `.github/workflows/release-assets.yml` for self-hosted GPU runners to
  build release installs, optionally generate target-local engine bundles, and
  upload release archives to GitHub Releases.
- Added `scripts/package_release.sh` and `docs/releasing.md` to document and
  package binary and engine archives with stable names matching the install
  guide.
- Validation: `bash -n scripts/package_release.sh`,
  `./scripts/package_release.sh --help`, `python3 -m json.tool` for both
  release-please JSON files, and `git diff --check` passed locally.
- Operational note: actual remote release creation still depends on pushing the
  workflow files, enabling repository Actions permissions, and supplying a
  `RELEASE_PLEASE_TOKEN` if release-created workflows should trigger asset
  publication.

### 2026-07-22 — PR validation CI and multi-architecture release CD

- Fulfills the multi-architecture CI/CD packaging pipeline explicitly deferred
  in the 2026-07-22 "Split CUDA architecture selection" decision below.
- Added `.github/workflows/ci.yml`, triggered on every pull request and on
  pushes to `main`, using only GitHub-hosted runners: `lint` (shell syntax,
  shellcheck, Python syntax, release-please JSON validation); `build-cpu`, a
  matrix of `{ubuntu-latest, ubuntu-24.04-arm}` x `{Debug, Release}` with
  `-DNVCR_ENABLE_TENSORRT=OFF` running the full CTest suite, covering both
  host CPU architectures NVCR ships for without needing GPU hardware; and
  `build-cuda-portable`, a best-effort (`continue-on-error: true`)
  compile-only check that installs the CUDA toolkit and TensorRT development
  packages on a hosted runner and configures/builds with
  `-DNVCR_ENABLE_TENSORRT=ON -DNVCR_CUDA_ARCH_SET=portable`, since no GPU is
  present to run CUDA-executing tests there.
- Added `scripts/ci/install_cuda_tensorrt.sh`, a shared installer for CUDA
  toolkit and TensorRT development packages on GitHub-hosted Ubuntu x86_64
  runners from NVIDIA's public apt repositories, used by both the new CI job
  and the new release-asset job below.
- Added a `build-portable` job to `.github/workflows/release-assets.yml` that
  runs on a GitHub-hosted runner (no self-hosted GPU hardware required) and
  packages a `linux-x86_64-portable` archive covering every GPU architecture
  in the portable matrix, alongside the existing self-hosted `build-discrete`
  and `build-jetson` jobs. The portable archive ships no engine bundle, same
  as the other platforms.
- Self-hosted GPU runners (`nvcr-release-discrete`, `nvcr-release-jetson`)
  remain intentionally excluded from the `pull_request` trigger: wiring them
  to PR validation would let a fork-opened PR run arbitrary code on that
  hardware. Real GPU execution stays confined to the release asset workflow
  and manual maintainer runs.
- Updated `docs/releasing.md`, `scripts/README.md`, and
  `docs/install-binary.md` to document the CI gate, the new installer script,
  and the portable platform tag.
- Validation: `bash -n scripts/ci/install_cuda_tensorrt.sh`, YAML parse of
  both workflow files via `python3 -c "import yaml; yaml.safe_load(...)"`.
- Not yet validated: an actual GitHub Actions run, and the hosted-runner
  CUDA/TensorRT apt install steps.
- Follow-up: added a `changes` job (`dorny/paths-filter`) to `ci.yml` that
  gates `build-cpu` and `build-cuda-portable` on whether the push/PR touched
  any path that can affect the compiled output (CMake files, `src/`,
  `include/`, `cli/`, `tests/`, `benchmarks/`, `examples/`, `third_party/`,
  `tools/`, `scripts/`, or the CI workflow itself). Docs-only or
  README/ROADMAP-only changes now skip both compile jobs instead of paying
  for a full rebuild; `lint` still always runs. Skipped jobs report as
  passing (not failing) for branch-protection required-status-check
  purposes.

## Decision log

Append decisions with date, rationale, and consequences.

### 2026-07-29 — Make NVCR a codec-runtime architecture with one supported backend

Decision: the public C++ session boundary is now expressed in generic
`nvcr::codec` terms, while DCVC-RT remains the only implemented and supported
backend for v0.3/v1.

Rationale: NVCR's project identity is a deployment-oriented runtime architecture
for neural video codecs, but the current evidence and release gates are tied to
one DCVC-RT model profile. A generic backend/session boundary lets the
architecture be described honestly without claiming additional codec support.

Consequence: documentation may describe the runtime architecture as codec-backend
oriented, and backend-specific preparation code should live under a backend
directory. Release claims must continue to say that only DCVC-RT is supported
until another backend has model profiles, payload contracts, conformance tests,
artifact validation, and target evidence.

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

### 2026-07-08 — Require device-stamped TensorRT engine bundles

Decision: TensorRT plan directories must carry `engine_manifest.json`, and the
runtime must reject missing or mismatched manifests before plan deserialization.

Rationale: TensorRT plans are optimized executable artifacts for a specific build
runtime and GPU class. A successfully deserialized plan can still carry suboptimal
tactics for a different GPU model, so NVCR should fail early with a rebuild
instruction instead of discovering the problem during encode latency or CUDA
execution.

Consequence: existing engine directories without `engine_manifest.json` must be
rebuilt with the updated scripts before use.

### 2026-07-08 — Encode-side predicted references may be GPU-only

Decision: predicted-frame encode may advance `SequenceState` with an empty host
reference when the backend owns the actual reconstructed reference in a validated
GPU DPB.

Rationale: TensorRT P-frame encode already feeds the next frame from device
resident `frame` and `feature` tensors. Downloading and RGB-converting the
reconstruction only to satisfy a host reference marker cost roughly 112-117 ms
per 1080p P-frame on the Orin run.

Consequence: generic sequence state now tracks reference availability separately
from the optional host `Frame`. Backends that require a host reference must still
return one; TensorRT can keep encode references GPU-resident.

### 2026-07-22 — Split CUDA architecture selection into auto vs. portable, deferred release packaging

Decision: add an explicit `NVCR_CUDA_ARCH_SET` CMake option (`auto` default,
`portable`) and matching `scripts/install.sh --arch-set` flag, instead of
always detecting the local build machine's single GPU or always building a
fat binary.

Rationale: NVCR must support both local single-machine builds and
redistributable binaries for a wider GPU set. Single-GPU auto-detection (via
`nvidia-smi compute_cap`) is correct and fastest for local development or a
single-machine install, but it is the wrong default for a build meant to be
redistributed to machines other than the one it was built on. Conflating the
two would force every local developer build to pay the multi-architecture
compile cost, or force every release build to silently target only the
packaging machine's GPU.

Consequence: `auto` remains the default so existing local/dev workflows and
documentation are unaffected; `portable` is available and verified through
configure/build validation for future packaging use. GPU-architecture portability
(`NVCR_CUDA_ARCH_SET=portable`) is explicitly documented as orthogonal to host
CPU architecture portability — separate builds are still required per CPU
arch — and TensorRT `.plan` engines are still never bundled as portable
artifacts; they must always be generated per-target via
`scripts/prepare_dcvcrt_artifacts.sh`, guarded by `engine_manifest.json`
(2026-07-08 decision above). Release packaging automation remained out of scope
for that change.

### 2026-07-23 — Keep public package families broader than validated reference hardware

Decision: public archive names identify a Linux/NVIDIA package family, while the
validated support claim remains anchored to explicit target profiles in configs,
release automation, and roadmap evidence.

Rationale: package filenames should not imply that NVCR only installs on one
exact GPU SKU when the portable binary may run on a broader NVIDIA/Linux family.
At the same time, TensorRT plans and support evidence remain narrower than the
binary package family, so archive naming must not replace the target matrix.

Consequence: release packages now use generic family labels such as
`linux-x86_64-nvidia` and `linux-aarch64-jetson-l4t36`, while the workflow still
builds and validates against `rtx4070-ubuntu2404` and
`orin-nano-l4t3647`. Public releases remain engine-free and target-local engine
generation remains the default path. If reviewer-convenience compatibility-mode
engines are added later, they stay a non-default discrete-GPU class and not a
Jetson portability path.

### 2026-07-23 — Treat standard GitHub-hosted runners as packaging helpers, not target-evidence gates

Decision: the default Actions contract uses standard GitHub-hosted runners for
lint, CPU tests, hosted CUDA/TensorRT compile/package smoke, and draft x86_64
package upload only. Reference-target validation and Jetson release deliverables
remain explicit external/manual steps until dedicated runner capability exists.

Rationale: standard GitHub-hosted runners provide general-purpose x86_64 and
arm64 Linux VMs, but not the recorded RTX 4070 or Jetson Orin Nano deployment
environments required for target-local TensorRT plan generation, the full GPU
suite, or Jetson/L4T packaging evidence.

Consequence: hosted automation may upload the generic x86_64 package to a draft
release, but it does not publish the release. The Jetson archive, exact target
matrices, and final publication remain manual release responsibilities recorded
in the roadmap.

### 2026-07-29 — Orchestrate engine assets from Release Please tags

Decision: add `scripts/release_engine_assets.sh` as the default operator path for optional TensorRT engine release assets. The helper derives the release tag from the checked-out `version.txt` by default, validates local target-built bundles, stages archives under a caller-provided S3 prefix, generates presigned URLs and `dist/nvcr-engine-assets.txt`, checks the matching GitHub draft release, and dispatches `upload-engine-assets.yml` for that Release Please tag.

Rationale: Release Please should remain the only version and tag authority, S3 should remain temporary staging, and GitHub Releases should remain the public distribution channel. Automating the handoff removes manual manifest copy/paste and reduces the chance of attaching engines to a stale or mismatched release tag.

Consequence: engine assets are still built on validated target machines and are still separate from generic binary packages, but the release operator now has a repeatable one-command path from target-local bundles to draft-release assets. Publishing remains gated on the recorded roadmap evidence for the release track.
