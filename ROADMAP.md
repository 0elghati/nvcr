# NVCR roadmap

Last reviewed: 2026-08-11

## Product direction

NVCR is a native C++ runtime architecture for neural video codecs. The product
direction is a reusable systems layer for stateful codec sessions, execution
providers, target-aware artifacts, bounded access units, and reproducible
validation. It is codec-extensible, but the current production vertical is
DCVC-RT through TensorRT FP16 on Linux/NVIDIA targets.

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

## Current production vertical

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

## Current release-readiness work

| Area | Status | Remaining gate |
|---|---|---|
| Runtime and stream contracts | Implemented | Maintain parser, reset, flush, delayed-output, and I/P coverage |
| TensorRT provider path | Implemented | Keep provider-owned boundary explicit; split model stages only with a new production integration |
| Binary and container packaging | Implemented | Complete license, provenance, and clean-package checks |
| Exact-target artifacts | In progress | Produce current warning-free profile sets and target-local evidence |
| Reproducible evaluation | In progress | Complete exact matrices and pinned Python comparisons |
| Compatibility classes | Experimental | Compare against complete exact baselines |
| Public C++ API/ABI | Transitional | Freeze only after ownership and compatibility contracts are accepted |

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, profile, digest, and
redistribution status.

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
