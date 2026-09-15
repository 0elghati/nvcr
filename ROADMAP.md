# NVCR roadmap

Last reviewed: 2026-09-14

## Product direction

NVCR is a native C++ runtime architecture for neural video codecs. The product
direction is a reusable systems layer for stateful codec sessions, execution
providers, target-aware artifacts, bounded access units, and reproducible
validation. It is codec-extensible, but the current supported
end-to-end integration is DCVC-RT through TensorRT FP16 on Linux/NVIDIA targets.

## Completed runtime foundations

- C++20 `Frame`, `Packet`, `Runtime`, session, error, statistics, and reset/
  flush contracts.
- Codec descriptors, adapter boundary, provider API, static registry, and
  provider-mediated `RuntimeServices`.
- Artifact descriptors, catalog parsing, resolver ranking, version/digest/
  license checks, and target compatibility classes.
- Bounded `NVAU` v1 and generalized sectioned `NVAU` v2 parsing/serialization.
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
| Binary and container packaging | Implemented | Complete license, provenance, clean-package, and version-first container-tag checks |
| Public documentation and onboarding | Implemented | Keep the project identity, contributions, reference comparisons, latest-release examples, and platform workflows current |
| CLI and artifact-client build identity | Implemented | Keep `version.txt`, build metadata, both version commands, citation metadata, and package checks consistent |
| Linux container GPU injection | Follow-up | Validate and document configured `nvidia` runtime, Docker `--gpus`, and CDI paths across supported Docker and NVIDIA Container Toolkit versions |
| Exact-target artifacts | In progress | Produce current warning-free profile sets and target-local evidence |
| Reproducible evaluation | In progress | Complete exact native and direct-Docker matrices plus pinned Python comparisons |
| Compatibility classes | Experimental | Compare against complete exact baselines |
| Public C++ API/ABI | Transitional | Freeze only after ownership and compatibility contracts are accepted |

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
| C5 | MLVC reference/export feasibility | Complete; candidate rejected 2026-09-14 | The pinned public MLVC-S model was compared through its PyTorch CPU reference and a proven ONNX Runtime CUDA session; the default FP16 export exceeded the predeclared PSNR and BPP limits; see [the C5 evidence](evidence/phase-c5-mlvc-reference-export-20260914.md) |
| C6 | MLVC CUDA equivalence diagnosis | Complete; candidate rejected 2026-09-14 | The first stream-relevant divergence is in FP16 encoder raw symbols before entropy coding; the tested MLVC/ORT-CUDA pair is rejected as the next production axis; see [the C6 evidence](evidence/phase-c6-mlvc-cuda-equivalence-20260914.md) |
| C7 | DCVC-UF fused-operator feasibility | Next | One fused operator and one frame-specific high-throughput branch match a pinned upstream encode/decode vector without generic runtime code owning UF branch semantics |

C0 confirmed that C1 could proceed without a new chunk API. The existing
session verbs were sufficient. C1 now routes the production path through
codec-owned sessions instead of preserving shared fixed-GOP and one-to-one
backend semantics. C2 proves DCVC-UF-shaped lifecycle behavior with
deterministic fixtures before a second codec is integrated.

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, profile, digest, and
redistribution status.

C1-C4 are verified by the clean CUDA 12.8/TensorRT 10.9 Release suite, the
registered grouped-session contracts, and exact B7 byte/reconstruction parity.
See [the Phase C1-C3 evidence](evidence/phase-c1-c3-runtime-ownership-20260914.md).
The C4 prototype shows that `PacketIO` can preserve per-frame timing for a
grouped access unit; it does not define a standard-container mapping. C5 tested
the public MLVC-S checkpoint with the upstream default generic ONNX FP16 export.
A real ONNX Runtime CUDA session ran, but the result exceeded the predeclared
numerical limits at both rate points. C6 traced the first stream-relevant
difference to FP16 encoder raw symbols before entropy coding. The tested
MLVC/ORT-CUDA pair is rejected as the next production axis, and no NVCR provider
or codec adapter was started. C7 now tests DCVC-UF export feasibility. The
minimum capability vocabulary remains open in
[the cross-codec audit](docs/cross-codec-requirements.md#open-questions-requiring-prototypes).

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
redistribution.

## v1.x maintenance and release gates

The v1.x line is released. These remain ongoing maintenance and release gates;
they do not imply that v1 is pending.

1. Clean source and package builds.
2. Complete parser, reset, flush, malformed-input, delayed-output, and I/P tests.
3. Validated target-local artifacts and runtime round trips.
4. Complete reproducible evaluation results.
5. Final dependency, model, dataset, and package-license review.
6. Installation and external-consumer verification from published artifacts.
