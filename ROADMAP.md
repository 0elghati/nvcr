# NVCR roadmap

Last reviewed: 2026-11-02

## Product direction

NVCR is a native C++ runtime architecture for neural video codecs. The product
direction is a reusable systems layer for stateful codec sessions, execution
providers, target-aware artifacts, bounded access units, and reproducible
validation. It is codec-extensible, but the current supported
end-to-end integration is DCVC-RT through TensorRT FP16 on Linux/NVIDIA targets.
MLVC is an experimental second codec through the ONNX Runtime providers (C10).

## Completed runtime foundations

- C++20 `Frame`, `Packet`, `Runtime`, session, error, statistics, and reset/
  flush contracts.
- Codec descriptors, adapter boundary, provider API, static registry, and
  provider-mediated `RuntimeServices`.
- Artifact descriptors, catalog parsing, resolver ranking, version/digest/
  license checks, and target compatibility classes.
- Bounded `NVAU` v1 and experimental sectioned `NVAU` v2 parsing/serialization.
- Deterministic test codec and CPU provider for conformance fixtures.
- CLI, CMake package, native installer, Docker/Compose surfaces, and evidence
  generation tooling.

## Current supported end-to-end integration

```text
DCVC-RT codec adapter -> TensorRT FP16 provider -> Linux/NVIDIA targets
```

The production boundary assigns GOP/reference semantics, native entropy coding,
codec-private payload syntax, and reference meaning to the registered DCVC-RT
adapter. The registered TensorRT provider session owns target-local plans,
device storage, execution, synchronization, and provider errors. The remaining
`TensorRTBackend` is a private composition facade, not the provider factory. The
completed staged extraction is documented in
[the provider-boundary RFC](docs/provider-boundary-v2.md).

The deterministic test codec and CPU provider are conformance fixtures. They do
not count as additional production codecs/providers or performance baselines.

## Current work

| Area | Status | Remaining gate |
|---|---|---|
| Runtime and stream contracts | Implemented | Maintain parser, reset, flush, delayed-output, and I/P coverage |
| TensorRT provider path | v2 extraction complete through B7 | Maintain the accepted ownership, byte, lifecycle, target, and performance gates; a second provider remains separate |
| ONNX Runtime provider path | Experimental (C8) | CPU self-conformance and TensorRT regression pass. QNN HTP self-conformance passes on a Galaxy S25 (C9). Open: the CUDA execution provider, QNN measurements, and a stream-level provider identity so a mismatched decoder fails instead of reconstructing wrong frames (a stream change requiring separate approval) |
| MLVC codec | Experimental (C10) | Upstream-runtime parity passes on x86 ONNX Runtime CPU. Runs on a Galaxy S25 NPU at about 300 fps (640x360); cross-target drift measured. Open: arm64 parity, DCVC-RT burst mode, ONNX Runtime CUDA, the standard MLVC model, NHWC bundles and encoder rate control; see the [MLVC integration](docs/mlvc-integration.md) |
| Binary and container packaging | Implemented | The engine-bundle packager now carries the NVCR and upstream notices inside the installed bundle; republish existing catalog archives and verify manifests/digests before the release gate. Complete provenance, clean-package, and version-first container-tag checks |
| Public documentation and onboarding | Implemented | Keep the project identity, contributions, reference comparisons, latest-release examples, and platform workflows current |
| CLI and artifact-client build identity | Implemented | Keep `version.txt`, build metadata, both version commands, citation metadata, and package checks consistent |
| Linux container GPU injection | Follow-up | Validate and document configured `nvidia` runtime, Docker `--gpus`, and CDI paths across supported Docker and NVIDIA Container Toolkit versions |
| Exact-target artifacts | In progress | RTX 3050 Laptop TensorRT 10.9.0/CUDA runtime 12080: all six exact bundles published with verified remote digests after 84 independent warning-free plan loads and dimension-matched I/P round trips. Fresh catalog QCIF installation and downloaded-bundle round trip pass with identical stream/reconstruction hashes. See [publication evidence](evidence/rtx3050-trt109-local-20261012/summary.md). Produce current warning-free profile sets and target-local evidence for remaining targets |
| Reproducible evaluation | Matched measurements complete | Process throughput is the primary comparison; codec-interval measurements are reported separately. See the [result inventory](results/README.md) for coverage and retained records. The RTX 4070 matched campaign predates the current stricter preflight gate: it was captured under a forked/unpinned reference source and without a matched smoke. Rerun with a clean pinned reference and a passing matched smoke before further publication claims |
| Compatibility classes | Experimental | Compare against complete exact baselines |
| Public C++ API/ABI | Release policy adopted | Preserve public API compatibility in patch and minor releases; require a major release for breaking changes. Rebuild applications when upgrading; cross-release ABI compatibility is not guaranteed. |

## Active v2 work

The long-term direction is recorded in [the NVCR vision](docs/NVCR_VISION.md).
The table below is the execution record for the active provider-boundary work.

| Sequence | Work item | Status | Exit evidence |
|---|---|---|---|
| A | Baseline coherence | Complete in [PR #145](https://github.com/0elghati/nvcr/pull/145) | Version, status, metadata, and baseline tests agree |
| B0 | Production-path discovery and boundary RFC | Complete in [PR #145](https://github.com/0elghati/nvcr/pull/145) | Live source path traced; ownership, alternatives, migration, and gates documented |
| B1 | Vision and execution governance | Recorded | Vision tracked; roadmap and RFC cross-linked; performance acceptance remains explicit per migration |
| B2 | Provider-session contracts and deterministic fixtures | Complete in [PR #148](https://github.com/0elghati/nvcr/pull/148) | Experimental contracts and deterministic tests cover ownership, bounds, dependencies, completion, reset, and errors without switching production |
| B3 | Pre-refactor target baseline | Complete; recorded comparisons accepted 2026-09-13 | Exact inputs and bundles, per-run values, variance, copies/synchronization, peak memory, context policy, and accepted comparisons recorded |
| B4 | TensorRT execution session behind the facade | Complete; measured regression accepted 2026-09-13 | Engine/context, binding, allocation, stream/event, graph-cache, and enqueue ownership moved behind the session contract while bundle validation and context policy remain intact |
| B5 | I-frame codec orchestration | Complete; measured regression accepted 2026-09-13 | DCVC-RT owns stage order, quantization, entropy, and payload assembly with byte and reconstructed-frame parity |
| B6 | P-frame state and orchestration | Complete; measured regression accepted 2026-09-13 | DCVC-RT owns reference semantics; GOP, reset, flush, repeated-session, and malformed-input gates pass |
| B7 | Production construction cleanup | Complete; gates pass 2026-09-14 | Registered adapter and provider-session factories select both IDs; direct generic-runtime registration, component factory, and unused production stub are removed |

B2 was completed in PR #148. Its clean CPU Release build, test, install,
sanitizer, and TensorRT Release target gates passed. B3 was captured before
TensorRT extraction began in B4. A second provider, a second codec, FFmpeg, and
stream changes remain separate approved work.

B3 recording now retains raw clean samples, per-profile and pooled summaries,
provider copy/synchronization counters, context policy, peak memory, and bounded
CUDA graph-cache behavior. The RTX 4070 baseline and B4 comparison are recorded
in [the B4 performance evidence](evidence/vision-b4-rtx4070-20260911.md).
The recorded B4-B6 regression results were explicitly accepted on 2026-09-13.
This approval is specific to those measured changes and is not a reusable limit.

B4 was explicitly reprioritized before the recorded measurements were accepted.
The private TensorRT execution session now implements the experimental
provider-session contract and owns bundle/stage loading, runtime and contexts,
opaque buffers, stream/events, dependencies/completions, graph caching, enqueue,
and provider profiling. `TensorRTBackend` remains its only production caller and
continues to own codec sequencing, entropy, payload, and reference meaning. The
clean CPU Release and install, sanitizer and bounded fuzz, TensorRT Release,
TensorRT non-GPU, all six exact-profile GPU, pinned Python/native golden, and
B3/B4 byte and reconstruction parity gates pass. The retained 65-frame GOP-8
fixture produced identical full-stream and decoded-YUV hashes across B3 and B4.
See [the B4 closure evidence](evidence/vision-b4-closure-rtx4070-20260913.md).
Relative to B3, pooled B4 encode throughput changed by -0.294% for normal I/P
and -0.640% for all-intra, while decode improved by 1.206% and 0.426%. Peak
device memory and all provider copy/synchronization, context-policy, and
graph-cache records are unchanged. The recorded B4 result was explicitly
accepted on 2026-09-13, completing B4.

B5 was explicitly started before the B3/B4 result was accepted. The DCVC-RT
codec-side `IntraOrchestration` now owns named I-frame stage selection, I-frame
CDF and quantization assets, rANS session state, and NVI1 payload parsing and
assembly. TensorRT continues to own device buffers, CUDA transforms, stage
submission, transfers, synchronization, and profiling. Clean CPU Release/install,
sanitizer/fuzz, TensorRT Release, all six exact-profile GPU
contracts and I/P round trips, and the pinned Python/native golden pass. A
65-frame GOP-8 comparison against B4 produced identical complete `.nvcr` and
decoded-YUV hashes. The matched five-profile result records +0.044% pooled I/P
encode, +0.364% pooled I/P decode, +0.827% pooled all-intra encode, and +1.430%
pooled all-intra decode. Provider counters and peak device memory are unchanged.
See [the B5 evidence](evidence/vision-b5-rtx4070-20260913.md). The recorded B5
result was explicitly accepted on 2026-09-13, completing B5.

B6 was explicitly started on top of B5. The codec-side
`PredictedOrchestration` now owns named P-frame stage selection, effective QP and
reference choice, P-frame CDF and quantization assets, rANS state, and NVP1
parsing and assembly. Codec-side `ReferenceState` owns feature availability
and reference generation/index meaning. TensorRT continues to own provider
buffers, CUDA transforms, stage submission, transfers, synchronization, and
profiling. Clean CPU Release/install, sanitizer/fuzz, TensorRT Release, all six
exact-profile GPU contracts and I/P round trips, the
pinned golden, and the 65-frame GOP-8 byte/reconstruction parity gates pass.
Relative to B5, matched pooled throughput changed by +0.045% I/P encode,
-0.160% I/P decode, -0.019% all-intra encode, and -0.360% all-intra decode.
Payloads, quality, provider counters, engine/input identities, and peak device
memory are unchanged. See
[the B6 evidence](evidence/vision-b6-rtx4070-20260913.md). The recorded B6
result was explicitly accepted on 2026-09-13, completing B6.

B7 was completed on top of B6. `RuntimeConfiguration` now carries explicit
codec and provider IDs, and `Runtime::create(configuration)` resolves both the
registered codec adapter and provider-session factories. The CLI uses this
path. The DCVC-RT entry has a real adapter factory, the TensorRT entry creates
the real `TensorRTExecutionSession`, and generic runtime code no longer
registers DCVC-RT directly. The production-only `component_factory`, unused
TensorRT `IExecutionProvider::load` stub, and test-only component shim were
removed after their last callers.

Clean CPU Release/install, sanitizer/fuzz, clean TensorRT Release, all six
exact-profile GPU contracts and I/P round trips, the pinned golden, and the
65-frame GOP-8 byte/reconstruction parity gates pass. The matched five-profile
normal-I/P result changed pooled encode throughput by -0.038% and decode by
+0.019% relative to B6; the worst per-profile decrease is -0.204%, and peak
device memory remains 2,244 MiB. See
[the B7 evidence](evidence/vision-b7-rtx4070-20260914.md). B7 completes the
staged Phase B execution-boundary migration.

## Phase C — Cross-codec runtime ownership

Phase C executes the existing multi-codec direction in
[`docs/NVCR_VISION.md`](docs/NVCR_VISION.md). The source-grounded requirements
and gap classifications are recorded in
[`docs/cross-codec-requirements.md`](docs/cross-codec-requirements.md).

| Sequence | Work item | Status | Exit evidence |
|---|---|---|---|
| C0 | Cross-codec requirements audit | Complete 2026-09-14 | DCVC-RT, MLVC, and DCVC-UF requirements, current NVCR gaps, exact upstream revisions, and next-step decisions recorded |
| C1 | Codec-runtime ownership cleanup | Complete 2026-09-14 | Registered DCVC-RT sessions own sequence policy, state, output queues, and codec-specific AU handling; the 65-frame GOP-8 stream and reconstruction match the B7 hashes |
| C2 | Buffered/delayed-output lifecycle | Complete 2026-09-14 | Registered eight-input/one-packet and one-packet/eight-frame fixture passes full/short drain, reset, reuse, repeated-session, and independent-direction checks |
| C3 | Configuration ownership split | Complete 2026-09-14 | Runtime, codec, provider, artifact-selection, and stream-policy scopes are distinct while existing CLI/config keys and validation behavior remain intact |
| C4 | Grouped-AU timing prototype | Complete 2026-09-14 | A registered grouped codec preserves distinct non-uniform frame timestamps across the serialized `PacketIO` boundary without a chunk API or NVAU change; see [the C4 evidence](evidence/phase-c4-grouped-au-timing-20260914.md) |
| C5 | MLVC reference/export feasibility | Complete; strict reference-consistency target failed 2026-09-14 | The pinned MLVC-S/default generic FP16 ONNX/ORT-CUDA pair exceeded the predeclared PSNR and BPP limits against the PyTorch CPU reference; see [the C5 evidence](evidence/phase-c5-mlvc-reference-export-20260914.md) |
| C6 | MLVC CUDA equivalence diagnosis | Complete; tested reference-interchange path rejected 2026-09-14 | FP16 encoder raw symbols diverged before entropy coding, so the tested pair is not byte/payload interchangeable with the reference path; see [the C6 evidence](evidence/phase-c6-mlvc-cuda-equivalence-20260914.md) |
| C6.5 | Generic CLI session driving | Complete 2026-09-15 | Encode/decode use session send, full receive drain, directional flush, and final drain through the same driver exercised by the grouped fixture; see [the C6.5 evidence](evidence/phase-c6-5-cli-session-driver-20260915.md) |
| C6.6 | Session send/receive contract clarification | Complete 2026-09-15 | Send success consumes input, receive-side `try_again` reports delayed readiness, and a registered multi-input decoder fixture proves order, timestamps, flush/drain, and reset/reuse without changing production behavior; see [the C6.6 evidence](evidence/phase-c6-6-session-contract-20260915.md) |
| C6.7 | Current-main rebase and minor-release compatibility | Complete 2026-10-26 with documented GPU-contention exception | The twelve-commit `main-advanced` stack is rebased onto v1.0.8 and the public 1.0 source seams are retained. CPU, install-consumer, sanitizer/fuzz, TensorRT non-GPU, available-profile GPU, pinned-golden, and byte-parity gates pass; two 1080p tests were omitted because they could not coexist with the active GPU workload. See [the C6.7 evidence](evidence/main-advanced-rebase-vetting-20261026.md) |
| C7 | DCVC-UF fused-operator feasibility | Working opt-in native CLI; validation open 2026-11-02 | Pinned DMCI and HT-S fused sources now build into NVCR without a Python ABI dependency. The registered dcvc-uf adapter owns I/eight-frame/tail/timestamp/reset lifecycle and the registered dcvcuf-upstream-cuda provider owns weights, CUDA execution, and reference state. Streams and decoded frames are byte-identical to pinned upstream test_video.py on 300 QCIF and 500 BasketballDrive 1080p frames; 1080p RTX 4070 throughput is 51.6/43.8 process FPS and 183.8/135.1 codec FPS (encode/decode). Remaining: provider decision (user directed a TensorRT-provider port on 2026-11-02), upstream README recipe options (which save 2-56% bits at nearly equal PSNR on the C8 reference set), packaging, and clean release gates. See [the C7 evidence](evidence/phase-c7-dcvcuf-export-20261031.md) |
| C8 | Second provider: ONNX Runtime | Exit gates pass 2026-11-01 on the CPU execution provider; not yet merged. Prioritized ahead of C7 on 2026-11-01. Host transforms match the CUDA kernels with zero differences, all six exact-profile TensorRT suites pass, and TensorRT streams are byte-identical to `main`. Cross-provider streams decode without error to wrong frames; see [the C8 evidence](evidence/phase-c8-onnxruntime-provider-20261101.md) | DCVC-RT runs through a provider-neutral host backend whose transforms match the CUDA kernels; an ONNX Runtime session passes I/P self-conformance (encoder and decoder reconstructions identical), determinism, reset/reuse, and lifecycle gates on the exported bundle; TensorRT build and byte parity are unchanged; cross-provider comparison against TensorRT is reported separately |
| C9 | Snapdragon edge target through ONNX Runtime QNN | In progress on Galaxy S25 (SM8750, Hexagon v79). The arm64 build, the strict NPU placement gate (no CPU fallback), on-device QNN self-conformance, and a 65-frame CLI encode/decode at 176x144 pass. A diagnostic NPU sweep gives 22.7/30.7 fps encode/decode at QCIF, 8.8/10.7 at CIF and 4.0/4.9 at 360p. Open: QNN context artifacts (graphs compile on every launch), 540p and above, the CPU baseline, and a controlled measurement; see [the C9 evidence](evidence/phase-c9-snapdragon-qnn-20261101.md) | ARM64 build, static-shape QNN context artifacts per profile and SoC/QAIRT version, an NPU partition-coverage gate, and on-device self-conformance and measurements |
| C10 | MLVC codec through ONNX Runtime | Experimental; upstream-runtime parity passes 2026-11-02 on ONNX Runtime 1.26.0 CPU. Prioritized by user direction on 2026-11-02. Microsoft's published libmlvc MLVC-S bundles (generic and qualcomm) run unchanged. Every NVCR record equals the record from MLVC's own conversion runtime, and decoded YUV is byte-identical across six cases: LTR recovery, I-period override, transposed canvas, edge padding, the qualcomm graph on CPU, and a rate-controlled stream with dropped frames. DCVC-RT ONNX Runtime streams are byte-identical to `main`. On 2026-11-03 both parts ran on a Galaxy S25 NPU through `onnxruntime-qnn` with CPU fallback disabled, and streams decoded across NPU, phone CPU and x86 CPU with bounded drift that recovers at LTR recovery; the phone CPU does not reproduce the x86 references, so parity holds on x86 only. On the S25 with MLVC's QNN settings and the compiled-graph cache, 640x360 runs at about 300 fps encode and decode after a 0.35 s cached setup, and 960x540 at 111–138 fps. DCVC-RT keeps ONNX Runtime's QNN defaults and reproduces its C9 digests with the cache. Open: arm64 parity, DCVC-RT burst mode, ONNX Runtime CUDA, the standard MLVC model, NHWC bundles and encoder rate control; see the [C10 evidence](evidence/phase-c10-mlvc-codec-20261102.md) | Byte-identical records and reconstructions against MLVC's conversion runtime for the same bundle and execution provider; reset, malformed-unit and sanitizer gates; DCVC-RT unchanged; QNN device gate as documented in the integration guide |
| C11 | DCVC-UF on the TensorRT provider | Phases 0-3 complete 2026-11-02: pinned upstream references and approved gates; codec-agnostic TensorRT session with DCVC-RT parity; 20 FP16 DCVC-UF plans (hwc8, upstream stage boundaries, builder level 5); glue kernels bit-identical to upstream in 219 checks. TensorRT FP16 stages alone are 1.5x/1.7x slower than upstream's fused group encode/decode at 1080p on RTX 4070. Phase 4 next: execution path behind a codec-agnostic mode choice with fused CUDA as default; directed by user 2026-11-02 | Port DCVC-UF from the second fused-CUDA provider to the existing TensorRT provider while staying as close as possible to pinned upstream test_video.py. Phases: 0 reference vectors and gates; 1 codec-agnostic TensorRT session (stage contracts validated against engines) with DCVC-RT byte parity; 2 per-stage ONNX/TensorRT export; 3 codec-local glue kernels; 4 adapter path, engine bundle, gates and performance. See [the C11 evidence](evidence/phase-c11-dcvcuf-tensorrt-20261102.md) |

C0 confirmed that C1 could proceed without a new chunk API. The existing
session verbs were sufficient. C1 now routes the production path through
codec-owned sessions instead of preserving shared fixed-GOP and one-to-one
backend semantics. C2 proves DCVC-UF-shaped lifecycle behavior with
deterministic fixtures before a second codec is integrated.

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, profile, digest, and
redistribution status.

## Current evaluation priority

RTX 3050 Laptop Windows/WSL 2 controlled timing campaign audited on 2026-10-13:
1,440 NVCR-only throughput operations across six resolutions, four QPs, three
GOPs and ten repetitions, plus an eight-operation compatible reset smoke.
The [current compact result](results/rtx3050/measurement/summary.md) records
both process and synchronized codec intervals, input/engine identities, and
576 regenerated condition statistics. It supersedes the legacy portability
publication. The packaged release executable is bound to its OCI identity;
the mounted measurement harness has a separate captured identity. Full matrix
quality/memory and matched pinned Python reference gates remain open.

Matched NVCR/Python measurements are complete on RTX 4070 and Jetson Orin.
Process throughput is the primary comparison; codec-interval measurements are
reported separately. See the [result inventory](results/README.md) for coverage and retained
records.

The measurement contract and execution commands are documented in
[measurement-contract.md](docs/experiments/measurement-contract.md) and
[measurement-runbook.md](docs/experiments/measurement-runbook.md).

RTX 5060 Laptop now retains the corrected NVCR 1.0.7 Docker/WSL2 timing
campaign under results/rtx5060/measurement: all 1,440 throughput executions and
576 condition statistics, with common process/codec timers and same-session
warm-up/reset. The eight-operation smoke and 17 preflight checks pass.
Input hashes, exact engine identities, the OCI/executable receipt and measured
harness state are retained. No full memory or quality campaign was run.
The earlier separate-process-warm-up diagnostics are superseded in Git history.

The laptop launcher and selected-pass runner retain the existing release binary
and engines; only the measurement image adds NumPy for bounded smoke quality
checks. RTX 3050 consolidation and paper integration are complete; broader
release and submission checks remain open. No new GPU campaign was launched
during export.
C1-C4 are verified by the clean CUDA 12.8/TensorRT 10.9 Release suite, the
registered grouped-session contracts, and exact B7 byte/reconstruction parity.
See [the Phase C1-C3 evidence](evidence/phase-c1-c3-runtime-ownership-20260914.md).
The C4 prototype shows that `PacketIO` can preserve per-frame timing for a
grouped access unit; it does not define a standard-container mapping. C5 tested
the public MLVC-S checkpoint with the upstream default generic ONNX FP16 export.
A real ONNX Runtime CUDA session ran, but the pair exceeded the predeclared
strict reference-consistency limits at both rate points. C6 traced the first
stream-relevant difference to FP16 encoder raw symbols before entropy coding,
so the tested pair is not byte/payload interchangeable with the reference path.
MLVC self-conformance, MLVC's general suitability, and ONNX Runtime CUDA's
suitability as an NVCR provider remain unresolved.

C10 was started by user direction on 2026-11-02 with a different gate from
C5/C6. C5/C6 compared an exported MLVC graph with MLVC's PyTorch model; that
result is unchanged and remains an export property. C10 asks whether NVCR
reproduces MLVC's own deployment runtime for the same published bundle and
execution provider. It does, byte for byte, on ONNX Runtime CPU. The
adapter's sessions own MLVC's I/LTR/feature-reset schedule, reference store,
dropped-frame handling and entropy coding. Each `NVAU` access unit carries one
upstream `.mlvc` record. The only provider change is codec-neutral: the ONNX
Runtime session binds float16, float32 and int32 tensors checked against the
model. C10 also gives a first answer to the open capability-vocabulary
question: the registry calls `mlvc` with `tensorrt` compatible because both
are registered, but only the adapter can reject the pair. That admission
decision is shared by two real codecs and is recorded for a later admission
test.

C6.5 brings the CLI into line with the existing session contract. C8 holds
DCVC-RT constant and tests ONNX Runtime as the second provider. It was
prioritized ahead of C7 on 2026-11-01 so that one provider implementation can
cover CPU development, CUDA comparison, and a Snapdragon NPU through the QNN
execution provider (C9). The TensorRT backend's inter-stage CUDA kernels are
not reusable on those targets, so C8 adds host implementations of the same
transforms behind a provider-neutral DCVC-RT backend rather than changing the
accepted TensorRT path. 
On 2026-10-31,
the user prioritized starting DCVC-UF integration, making C7 the active bounded
experiment. The pinned upstream fused shuffle/clamp kernel matches exactly on
a synthetic vector; one HT-S frame branch exports and passes ONNX CPU and
TensorRT FP32 checks, including its unclamped reset feature. This is stage
feasibility evidence, not a supported codec. Manually supplied checkpoints now
pass the upstream 17-frame I/P smoke. The upstream Python/fused head comparison
fails its unchanged numerical limit; the first divergence occurs in the fused
1x1 convolution's FP16 accumulation. Exact stage buffer/reset checks pass. On 2026-11-01, the
experimental provider session connects UF synthesis and reconstruction using a
shared opaque device buffer and an ordinary completion dependency. Both decoder
features, 16 frames, and two reset features match exactly on the native QCIF
vector. Two separate processes under normal asynchronous reference execution
preserve all 23 tensor hashes; six provider-only pipeline replays with normal
CUDA launches also pass exactly. The pinned upstream setup's disabled empty-
tensor initialization was restored in the UF execution setup, closing the
native I-frame replay gate. On 2026-11-02 the full pinned DMCI/HT-S native
path was connected to registered codec/provider sessions and the normal NVCR
CLI. The pinned Python/native golden then exposed and fixed the step-based
frame-type schedule, the HT-S feature-memory reset, and FP16 input
normalization; streams and reconstructions are now byte-identical to upstream at
QCIF and 1080p. Moving pixel conversion onto the GPU and keeping models across
resets raised 1080p codec FPS to 183.8/135.1. The user then directed a port to
the existing TensorRT provider instead of the second fused-CUDA provider. Earlier stream rejections do not establish
stream corruption or an environment mismatch.
Shared runtime/provider APIs remain unchanged. C11 continues that port.
The minimum capability vocabulary remains open in
[the cross-codec audit](docs/cross-codec-requirements.md#open-questions-requiring-prototypes).

C6.6 makes the pre-1.1 session contract explicit: successful sends consume
input, and receive-side `try_again` reports that output is not ready. The
registered grouped decoder now delays its first access unit until another unit
arrives, then preserves order and timestamps through flush, final drain, reset,
and reuse. DCVC-RT behavior and the C6.5 CLI driver remain unchanged.

C6.7 rebases the completed provider/session work onto the current v1.0.8 main
line and audits it against the adopted minor-release API policy. The original
flat configuration and component-construction seams remain available as source
compatibility shims; scoped configuration and sessions remain the preferred
architecture. All compatible gates pass. Two 1080p TensorRT cases were omitted
by user direction because the active training process left insufficient VRAM;
that process was not interrupted.

Energy measurement remains optional downstream evidence, not a release gate.

## Later codec integrations

Additional codec adapters may be added after the adapter/session/access-unit
contracts and compatibility evidence are maintained. A future codec must pass
its own parser, round-trip, lifecycle, artifact, and reference gates before it
is described as production-supported.

## Later provider integrations

Additional execution providers may be added after independent provider,
artifact, tensor-binding, synchronization, and cross-provider conformance
tests pass. The current provider API alone is not evidence of a second
production provider.

## Media-framework and container integration

FFmpeg bindings, standard-container mapping, indexing, and application-level
audio/subtitle integration remain future work. `NVAU` is an elementary
access-unit contract; `NVCR`, `NVCS`, and `.nvcr` are development/application
wrappers.

## Explicit non-commitments

The current release does not commit to CPU neural-codec inference, Windows or
macOS production support, universal TensorRT-plan portability, INT8 release
support, a stable C ABI, upstream Python payload interchangeability, an
industry-standard neural bitstream, or unrestricted model/checkpoint/engine
redistribution. The ONNX Runtime CPU path added in C8 is a development and
conformance path, not a production CPU-inference commitment. The same applies
to MLVC on ONNX Runtime CPU (C10).

## v1.x maintenance and release gates

The v1.x line is released. These remain ongoing maintenance and release gates;
they do not imply that v1 is pending.

1. Clean source and package builds.
2. Complete parser, reset, flush, malformed-input, delayed-output, and I/P tests.
3. Validated target-local artifacts and runtime round trips.
4. Complete reproducible evaluation results.
5. Final dependency, model, dataset, and package-license review.
6. Installation and external-consumer verification from published artifacts.

## Historical NVCR measurement evidence (2026-09-20)

Historical record dated 2026-09-20. For the completed matched comparison,
see the [current result inventory](results/README.md).

The completed NVCR-only matrix in `output/peer-review/` covers six
resolutions, four configured QPs, three GOPs and ten throughput/memory repetitions.
All 3,024 operations passed; the preceding eight-operation smoke/reset gate
passed. The [current review](results/jetson-orin/measurement/summary.md) records the
independent numerical audit. NVCR and Python compact observations now
regenerate 720 condition statistics each: n, mean, median, sample SD, extrema,
CV and descriptive Student-t intervals. Pooled process throughput remains
distinct from the condition means. No raw runs are committed. Timing variability
remains explicit (maximum completed-codec FPS CV 9.74%, process FPS CV 3.07%
for NVCR); every operation contains a TensorRT device-model warning. The paired
Python/NVCR quality and entropy-rate comparison is recorded; matched
process-speedup and memory-ratio conclusions, the warning-free artifact gate
and the overall evaluation milestone remain open. No new GPU execution was
launched during review.

The separate RTX 4070 readiness gate uses source `a444fb6` plus the captured
working diff, a fresh SM-8.9/CUDA-12.8 Release build and the exact catalog
bundles. Preflight, the 3,024-operation NVCR-only dry-run plan and all eight
bounded smoke operations pass, including cold versus warmed/reset stream and
reconstruction hashes. The later direct full matched campaign passed all
6,048 operations; its fork-source and no-matched-smoke limitations are
recorded in [the RTX result](results/rtx4070/measurement/summary.md). Preflight
again requires a clean pinned reference source and a passing matched smoke
before a full `run`; that campaign predates this stricter gate.
