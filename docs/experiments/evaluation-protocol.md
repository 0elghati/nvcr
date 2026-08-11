# Reproducible evaluation protocol

## Purpose

The evaluation has two linked parts. Architecture-level tests validate the
runtime contracts; the production matrix validates the first complete codec /
provider vertical:

```text
architecture contracts -> sessions, registry/services, artifacts, NVAU parser
production vertical    -> DCVC-RT adapter -> TensorRT FP16 -> Linux/NVIDIA
```

The production evaluation validates:

```text
DCVC-RT codec semantics -> TensorRT execution -> C++20 runtime -> NVAU access units
```

It must show native encode/decode, stateful I/P GOP operation, reproducible
artifact preparation and validation, target-aware selection, containerized
build/test/run workflows, correctness and robustness, useful runtime
performance, and quality comparable to the pinned Python DCVC-RT reference.

Energy is optional downstream research and is not a release gate.

## Scope

Use:

- model profile `dcvcrt-cvpr2025`;
- TensorRT FP16;
- Linux NVIDIA targets;
- raw planar YUV420P8;
- target-local or explicitly classified compatibility bundles;
- the pinned upstream source and checkpoint hashes in [dcvcrt-artifacts.md](../dcvcrt-artifacts.md).

The deterministic test codec and CPU provider are used for architecture
conformance only. They are not performance baselines or additional production
verticals. Energy is optional downstream research data unless a separate
controlled protocol elevates it.

Do not describe the result as a universal runtime, a multi-codec product, or a Python-compatible bitstream unless a separate test proves that claim.

## Automation contract

Use `scripts/benchmark_softwarex_matrix.py` for publication rows. Its input is
a local JSON manifest using `nvcr.softwarex.inputs.v1`; start from
[the example](inputs.example.json). For every selected profile the
driver:

1. detects the GPU/runtime and validates it against the test target profile;
2. validates every engine bundle and its current model, engine, and build-target
   profile digests;
3. distinguishes source build, core tests, GPU test registration, GPU test
   execution, I/P roundtrip, and matrix completion;
4. runs isolated warm-up and measured encode/decode round trips without
   optional profiling instrumentation;
5. with `--profile`, runs separate additional round trips for verbose
   per-frame latency, PSNR-Y/U/V/YUV, and sampled process/GPU memory;
6. records clean-pass codec throughput, process wall time, payload/BPP,
   profile-pass metrics, commands, and identities;
7. writes pass, fail, and skipped states without silently promoting a partial
   run.

The output directory must be new or empty. Raw streams and reconstructions are
created in a temporary work directory and removed after each case. Use
`--work-dir` only when failed raw products need local debugging; that directory
remains outside the evidence package.

Profiling is opt-in because verbose output, reconstruction-quality calculation,
`/proc` polling, and `nvidia-smi` sampling can perturb execution. Without
`--profile`, the driver produces performance-only case rows: throughput, wall
time, payload, and BPP are valid, profile fields are `null`, and the package
remains `partial`. A publication run uses `--profile`; its extra repetitions
never contribute to throughput means, standard deviations, compatibility
ratios, or `total_wall_time_ms`.

## Required matrix

### Exact targets

Run the primary tables with exact bundles for:

- RTX 3050 exact;
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

Use one fixed frame count per sequence and record it in every row. The input
manifest fixes the frame count. The required driver settings are one warm-up
run and three measured repetitions.

### Operations

Each aggregate case row records paired encode/decode commands, separate encode
and decode throughput/latency, and end-to-end reconstruction quality. The
registered tests separately establish I-frame, I/P, reset/reuse, and malformed
input behavior. Codec timing excludes file I/O and quality calculation. The
primary process wall time includes normal CLI and file-I/O costs, but never the
verbose, quality, or memory-sampling costs of the separate profile pass.

## Metrics

Every result row must include status, target identity, GPU/SM, compatibility
class, engine profile, sequence, dimensions, frame count, QP, GOP, operation,
throughput, payload bytes, profiling mode, quality, memory, commit, image/build
identity, and artifact digests. Quality, latency, and memory are populated only
by `--profile`. The complete field list is in
[result-schema.md](result-schema.md).

Recommended additions are startup time, engine-load time, first-frame latency, median and p95 frame latency, driver/CUDA/TensorRT versions, OS, compiler, clocks/power mode, and GPU-utilization summary.

## Python reference

The mandatory reference comparison is RTX 4070 exact. Use the same source, dimensions, frame count, QP, and GOP pattern where the Python path allows it. Record Python and NVCR reconstruction quality, payload sizes, throughput, and Python-to-NVCR reconstruction similarity.

The driver accepts precomputed, pinned-reference rows through
`--python-reference-jsonl`. Those rows use
`nvcr.softwarex.python-reference.v1` and are joined by sequence, dimensions,
frame count, QP, and GOP. Each row binds the input digest, pinned source commit,
checkpoint hashes, and exact command. Every requested RTX 4070 exact case
requires a matching comparison row before the package can be marked complete.
The existing `nvcr_dcvcrt_i_frame_golden` test remains a useful one-frame
conformance gate, but does not replace the publication rate-distortion
comparison.

When a local wrapper can produce that schema, invoke it before ingestion with
`--python-reference-command '...'` and provide its output path through
`--python-reference-jsonl`. The driver parses the command without a shell,
records it, requires a zero exit status, and then applies the same pin/digest
checks. Precomputed rows remain valid when they were produced and retained by
an equivalent pinned process.

The acceptable claim is:

> NVCR preserves the rate-distortion behavior of the pinned Python DCVC-RT reference within the recorded tolerance.

Do not claim bit-exact or payload interchangeability without bidirectional cross-runtime golden tests.

## Acceptance gates

An exact target is ready for a publication table only when source build, artifact
validation, engine contract tests, I-frame round trip, I/P round trip,
reset/reuse, malformed-input rejection, the clean performance pass, and the
separate `--profile` metrics all pass.

A same-compute bundle additionally needs representative same-SM correctness,
sanity-performance validation, and an exact baseline row for each case.
Ampere-plus is a desktop fallback and needs separate correctness and
performance-ratio measurements against exact rows. Jetson is exact-only.

A CPU-only pass, a successful TensorRT build, or a catalog match by itself is not end-to-end evidence.

The generated `run-summary.json` and `summary.md` make these states explicit.
Only `status: complete` is publication-ready. `planned` and `partial`
packages are diagnostic records.

## Evidence package

Use one directory per clean run:

```text
evidence/evaluation-YYYYMMDD-<shortcommit>/
  README.md
  commands.md
  environment.json
  hardware-targets.json
  artifact-catalog.json
  artifact-digests.json
  test-summary.json
  run-summary.json
  exact-results.jsonl
  same-compute-results.jsonl
  ampere-plus-results.jsonl
  python-reference-results.jsonl
  failures.jsonl
  summary.md
```

Keep only metadata, hashes, commands, summaries, and small result rows in Git or the external evidence store. Raw streams and plans stay local.
