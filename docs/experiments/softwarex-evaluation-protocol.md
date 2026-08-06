# SoftwareX evaluation protocol

## Purpose

The evaluation validates the first complete NVCR runtime implementation:

```text
DCVC-RT codec semantics -> TensorRT execution -> C++20 runtime -> NVAU access units
```

It must show native encode/decode, stateful I/P GOP operation, reproducible artifact preparation and validation, target-aware selection, containerized build/test/run workflows, correctness and robustness, useful runtime performance, and quality comparable to the pinned Python DCVC-RT reference.

Energy is optional downstream research. It is not a SoftwareX v1 gate.

## Scope

Use:

- model profile `dcvcrt-cvpr2025`;
- TensorRT FP16;
- Linux NVIDIA targets;
- raw planar YUV420P8;
- target-local or explicitly classified compatibility bundles;
- the pinned upstream source and checkpoint hashes in [dcvcrt-artifacts.md](../dcvcrt-artifacts.md).

Do not describe the result as a universal runtime, a multi-codec product, or a Python-compatible bitstream unless a separate test proves that claim.

## Required matrix

### Exact targets

Run the primary tables with exact bundles for:

- RTX 3050 exact, once a target profile and bundle exist;
- RTX 4070 exact;
- RTX 5060 exact, currently represented by `rtx5060-laptop-ubuntu2404`;
- Jetson Orin Nano exact, when the target is available.

### Profiles and coding structure

Use `qcif`, `cif`, `360p`, `720p`, and `1080p`. Add `540p` when the target has a validated bundle.

Run:

- `GOP=1` for all-intra diagnostics;
- normal I/P operation, using a fixed documented GOP such as the sequence length or the repository default;
- `QP=32` as the minimum common point;
- `QP=22,32,42,52` when time and reference support allow.

Use one fixed frame count per sequence and record it in every row. The repository matrix defaults to 97 frames, one warm-up phase, and three measured repetitions.

### Operations

Record encode, decode, and end-to-end round-trip correctness separately. Codec timing excludes file I/O and quality calculation. State the initialization boundary and whether engine loading is included.

## Metrics

Every result row must include status, target identity, GPU/SM, compatibility class, engine profile, sequence, dimensions, frame count, QP, GOP, operation, throughput, payload bytes, quality, memory, commit, image/build identity, and artifact digests. The complete field list is in [result-schema.md](result-schema.md).

Recommended additions are startup time, engine-load time, first-frame latency, median and p95 frame latency, driver/CUDA/TensorRT versions, OS, compiler, clocks/power mode, and GPU-utilization summary.

## Python reference

The mandatory reference comparison is RTX 4070 exact. Use the same source, dimensions, frame count, QP, and GOP pattern where the Python path allows it. Record Python and NVCR reconstruction quality, payload sizes, throughput, and Python-to-NVCR reconstruction similarity.

The acceptable claim is:

> NVCR preserves the rate-distortion behavior of the pinned Python DCVC-RT reference within the recorded tolerance.

Do not claim bit-exact or payload interchangeability without bidirectional cross-runtime golden tests.

## Acceptance gates

An exact target is ready for a SoftwareX table only when source build, artifact validation, engine contract tests, I-frame round trip, I/P round trip, reset/reuse, malformed-input rejection, and the requested metrics all pass.

A same-compute bundle additionally needs representative same-SM correctness and sanity-performance validation. Ampere-plus is a desktop fallback and needs separate correctness and performance-ratio measurements. Jetson is exact-only.

A CPU-only pass, a successful TensorRT build, or a catalog match by itself is not end-to-end evidence.

## Evidence package

Use one directory per clean run:

```text
evidence/softwarex-YYYYMMDD-<shortcommit>/
  README.md
  commands.md
  environment.json
  hardware-targets.json
  artifact-catalog.json
  artifact-digests.json
  test-summary.json
  exact-results.jsonl
  same-compute-results.jsonl
  ampere-plus-results.jsonl
  python-reference-results.jsonl
  failures.jsonl
  summary.md
```

Keep only metadata, hashes, commands, summaries, and small result rows in Git or the external evidence store. Raw streams and plans stay local.
