# NVCR roadmap

Last reviewed: 2026-09-21

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

The adapter owns GOP/reference semantics, native entropy coding, and
codec-private payload syntax. TensorRT owns target-local plans, CUDA/TensorRT
execution, synchronization, and provider errors. TensorRT currently creates a
provider-owned monolithic DCVC-RT backend; independent model-stage loading
through `IExecutionProvider::load` remains transitional.

The deterministic test codec and CPU provider are conformance fixtures. They do
not count as additional production codecs/providers or performance baselines.

## Current work

| Area | Status | Remaining gate |
|---|---|---|
| Runtime and stream contracts | Implemented | Maintain parser, reset, flush, delayed-output, and I/P coverage |
| TensorRT provider path | Implemented | Keep provider-owned boundary explicit; split model stages only with a new production integration |
| Binary and container packaging | Implemented | Complete license, provenance, clean-package, and version-first container-tag checks |
| Public documentation and onboarding | Implemented | Keep the project identity, contributions, reference comparisons, latest-release examples, and platform workflows current |
| CLI and artifact-client build identity | Follow-up | Add `nvcr --version` and `nvcr-artifacts --version`, and reconcile the legacy `current_software_version` constant; until then use package manifests, source revisions, or OCI metadata |
| Linux container GPU injection | Follow-up | Validate and document configured `nvidia` runtime, Docker `--gpus`, and CDI paths across supported Docker and NVIDIA Container Toolkit versions |
| Exact-target artifacts | In progress | RTX 4070 catalog bundles and bounded execution pass exact CUDA 12.8/TensorRT 10.9 checks; retain the Jetson warning investigation and complete matched target evidence |
| Reproducible evaluation | In progress | RTX 4070 matched NVCR/Python campaign is complete and audited with explicit fork source-state identity, but had no matched smoke. Resolve pinned-source/reset gates before publication claims |
| Compatibility classes | Experimental | Compare against complete exact baselines |
| Public C++ API/ABI | Transitional | Freeze only after ownership and compatibility contracts are accepted |

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, profile, digest, and
redistribution status.

## Current evaluation priority

The common measurement contract and executable campaign are in
[measurement-contract.md](docs/experiments/measurement-contract.md) and
[measurement-runbook.md](docs/experiments/measurement-runbook.md). Quality now
uses common decoded bytes; new observations separate timing, memory, actual
stream components and statistical units. The configured QPs and repetition
count are retained. The [assessment](docs/experiments/measurement-readiness/assessment.md)
and [validation](docs/experiments/measurement-readiness/validation.md) distinguish
software checks from pending hardware gates. No evaluation milestone or v1 exit
criterion is marked complete by these changes.

`scripts/measurement_jetson.sh` provides logged preparation, smoke, campaign,
resume and analysis commands. The focused software suite passes; the current
NVCR execution evidence is summarized below. Jetson Python validation remains
pending.
`scripts/measurement_x86_64.sh` provides the corresponding exact RTX 4070
desktop NVCR-only launcher using the local x86_64 engine set. Its bounded
NVCR smoke passed. The full RTX 4070 matched NVCR/Python campaign was
subsequently run directly and audited in
[the current RTX result](results/rtx4070/measurement/summary.md). Its current
compact CSV/JSONL exports retain 6,048 full-matrix and 640 labelled targeted
repeat operations. Process-level 100-frame job FPS is reported alongside
completed-frame codec FPS; the targeted repeats remain separate from the
balanced full-matrix aggregates. Raw runs remain local; matched smoke/reset
and pinned-source publication gates remain pending.

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

## v1 exit criteria

1. Clean source and package builds.
2. Complete parser, reset, flush, malformed-input, delayed-output, and I/P tests.
3. Validated target-local artifacts and runtime round trips.
4. Complete reproducible evaluation results.
5. Final dependency, model, dataset, and package-license review.
6. Installation and external-consumer verification from published artifacts.

## Current NVCR measurement evidence (2026-09-20)

The completed NVCR-only matrix in `output/peer-review/` covers six
resolutions, four configured QPs, three GOPs and ten throughput/memory repetitions.
All 3,024 operations passed; the preceding eight-operation smoke/reset gate
passed. The [current review](results/jetson-orin/measurement/summary.md) records the
independent numerical audit and exports mean/sample SD and descriptive intervals.
No raw runs are committed. Timing variability remains explicit (maximum FPS
CV 9.74%); every operation contains a TensorRT device-model warning. The Python
comparison, warning-free artifact gate and overall evaluation milestone remain
open. No new GPU execution was launched during review.

The separate RTX 4070 readiness gate uses source `a444fb6` plus the captured
working diff, a fresh SM-8.9/CUDA-12.8 Release build and the exact catalog
bundles. Preflight, the 3,024-operation NVCR-only dry-run plan and all eight
bounded smoke operations pass, including cold versus warmed/reset stream and
reconstruction hashes. The later direct full matched campaign passed all
6,048 operations; its fork-source and no-matched-smoke limitations are
recorded in [the RTX result](results/rtx4070/measurement/summary.md).
Smoke evidence remains supported but is no longer mandatory when the operator
explicitly authorizes a direct campaign run.
