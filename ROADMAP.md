# NVCR roadmap

Last reviewed: 2026-10-13

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
- Bounded `NVAU` v1 and experimental sectioned `NVAU` v2 parsing/serialization.
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
| Binary and container packaging | Implemented | The engine-bundle packager now carries the NVCR and upstream notices inside the installed bundle; republish existing catalog archives and verify manifests/digests before the release gate. Complete provenance, clean-package, and version-first container-tag checks |
| TensorRT provider path | v1.x monolith shipped; v2 boundary design accepted | Proceed through the staged RFC gates without combining contract, extraction, and orchestration changes |
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
