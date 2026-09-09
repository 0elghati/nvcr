# NVCR roadmap

Last reviewed: 2026-09-09

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

The intended boundary assigns GOP/reference semantics, native entropy coding,
and codec-private payload syntax to the codec adapter, while TensorRT owns
target-local plans, execution, synchronization, and provider errors. The v1.x
production implementation does not yet realize that split: its provider factory
creates a monolithic `TensorRTBackend` that also owns DCVC-RT orchestration,
entropy, payload, and device-DPB behavior. The proposed v2 extraction is
documented in [the provider-boundary RFC](docs/provider-boundary-v2.md).

The deterministic test codec and CPU provider are conformance fixtures. They do
not count as additional production codecs/providers or performance baselines.

## Current work

| Area | Status | Remaining gate |
|---|---|---|
| Runtime and stream contracts | Implemented | Maintain parser, reset, flush, delayed-output, and I/P coverage |
| TensorRT provider path | v1.x monolith shipped; v2 boundary design accepted | Proceed through the staged RFC gates without combining contract, extraction, and orchestration changes |
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
| B1 | Vision and execution governance | Recorded | Vision tracked; roadmap and RFC cross-linked; performance limits remain an explicit pre-extraction decision |
| B2 | Provider-session contracts and deterministic fixtures | Complete in [PR #148](https://github.com/0elghati/nvcr/pull/148) | Experimental contracts and deterministic tests cover ownership, bounds, dependencies, completion, reset, and errors without switching production |
| B3 | Pre-refactor target baseline | In progress; RTX 4070 preflight passed | Exact inputs and bundles, per-run values, variance, copies/synchronization, peak memory, context policy, and approved acceptance limits recorded |
| B4 | TensorRT execution session behind the facade | Pending | Engine/context, binding, allocation, stream/event, graph-cache, and enqueue ownership move behind the session contract while bundle validation and context policy remain intact |
| B5 | I-frame codec orchestration | Pending | DCVC-RT owns stage order, quantization, entropy, and payload assembly with byte and reconstructed-frame parity |
| B6 | P-frame state and orchestration | Pending | DCVC-RT owns reference semantics; GOP, reset, flush, repeated-session, and malformed-input gates pass |
| B7 | Production construction cleanup | Pending | A real adapter factory selects codec and provider; direct registration, component factory, and the unused stub are removed after their last callers |

B2 was completed in PR #148. Its clean CPU Release build, test, install,
sanitizer, and TensorRT Release target gates passed. B3 must be captured and
approved before TensorRT extraction starts in B4. A second provider, a second
codec, FFmpeg, and stream changes remain separate approved work.

B3 recording now retains raw clean samples, per-profile and pooled summaries,
provider copy/synchronization counters, context policy, peak memory, and bounded
CUDA graph-cache behavior. The clean RTX 4070 all-profile preflight passes with
current exact catalog bundles; the idle-device measurement and numeric limit
approval remain.

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, profile, digest, and
redistribution status.

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
