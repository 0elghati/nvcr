# NVCR roadmap

Last reviewed: 2026-08-06

## Current position

NVCR is a Linux C++20 runtime architecture for neural video codecs. The first complete path is DCVC-RT as the codec adapter and TensorRT as the execution provider.

The current scope is intentionally small:

- DCVC-RT model profile `dcvcrt-cvpr2025`;
- NVIDIA Linux targets;
- TensorRT FP16;
- target-local model and engine bundles;
- native I/P encode/decode, reset, flush, and bounded `NVAU` access units.

This is development software, not a supported v1 release. A capability is implemented when code and contract tests exist; it is supported only after the relevant target evidence passes.

## Architecture work

The extensible-runtime refactor is implemented in the current `main` baseline and is ready for the next validation run.

Completed:

- codec adapter and execution provider boundaries;
- static codec/provider registry;
- session-oriented contracts with compatibility one-shot calls;
- versioned `NVAU` format and stream specification;
- artifact provenance, catalog loading, and typed resolver;
- explicit separation of model components from provider engine profiles;
- deterministic test codec/provider contracts;
- CPU and TensorRT contract coverage.

## Next gate

The next gate is CI, release hardening, and the SoftwareX evaluation protocol:

1. Run the core GCC/Clang sanitizer and parser jobs.
2. Test CMake installation with an external consumer.
3. Validate the release package, license files, and excluded model/engine assets.
4. Rebuild the target-local engine collection from the current baseline.
5. Run a clean Release evaluation on the reference target and replace the retained diagnostic matrix.
6. Follow [the SoftwareX experiment protocol](docs/experiments/README.md) and publish one clean machine-readable result package.

The next evaluation should cover correctness, I/P GOP behavior, reset/reuse, malformed input rejection, runtime, memory, payload size, reconstruction quality, and pinned-Python comparison where available.

## Evidence policy

Only the latest machine-readable matrix is retained at [evidence/live-release-20260806/resolution-matrix.jsonl](evidence/live-release-20260806/resolution-matrix.jsonl). It is a diagnostic record, not a post-refactor release result. Raw streams and older run bundles were removed; the next clean run replaces this file.

No performance, energy, cross-runtime, or target-support claim should be made from the retained file alone.

The experiment protocol currently has two explicit blockers: the RTX 3050 engine assets are staged but still need a checked-in `rtx3050-laptop-ubuntu2404` target profile and validation, and no driver yet converts the existing benchmark JSONL into the complete SoftwareX result schema.

## Release boundary

A v1 release must include a clean source build, target-local artifact validation, complete I/P round trips, format and negative-path tests, a reproducible evaluation record, and final licensing/package review.

The following remain outside v1:

- additional production codecs or providers;
- INT8 release support;
- a public C ABI;
- FFmpeg and standard container integration;
- universal TensorRT-plan portability;
- redistribution of checkpoints, exported model assets, or TensorRT plans.

See [Scope and support](docs/scope-and-support.md), [Performance](docs/performance.md), [SoftwareX experiments](docs/experiments/README.md), and [Release policy](docs/releasing.md) for the current contracts.
