# Performance

This page defines the measurement protocol. It is intentionally short: numbers belong in the latest machine-readable run, not in a growing diary.

The SoftwareX matrix, target records, result schema, and run commands live in [docs/experiments](experiments/README.md).

## Rules

Run a Release build only after the configured correctness and artifact tests pass. Record:

- commit, target profile, engine/profile digests, and command;
- OS, compiler, driver or JetPack, CUDA, TensorRT, GPU, power mode, and clocks;
- sequence, format, dimensions, frame count, QP, GOP pattern, and reset pattern;
- warm-up count, repetitions, timing boundaries, and statistic;
- encode/decode throughput, latency, memory, payload size, and reconstruction quality;
- the matching pinned-Python command and tolerance when comparing runtimes.

Use the same input, profile, QP, frame count, and timing boundary for both runtimes. Do not turn one machine or one sequence into a general performance claim.

## Current evidence

The retained file [resolution-matrix.jsonl](../evidence/live-release-20260806/resolution-matrix.jsonl) is the latest local diagnostic summary. It predates this documentation cleanup and is not a clean v1 release result. The next run will replace it after the refactor branch is merged and the target is rebuilt.

Raw streams are local run products and are not part of the documentation set.

## Run the matrix

The repository script is the source of the matrix command and options:

```bash
scripts/benchmark_resolution_matrix.sh --help
```

Its current output uses `nvcr.benchmark.resolution-matrix.v1`. The complete publication schema is `nvcr.softwarex.result.v1`; use the adapter described in [Result schema](experiments/result-schema.md) before treating rows as SoftwareX evidence.

The publication run should cover QCIF, CIF, 360p, 540p, 720p, and 1080p; all-intra and normal I/P GOP cases; three measured repetitions; warm-up frames; payload size; PSNR-YUV; and a clean pinned-Python comparison where available.

Energy is optional investigation data, not a v1 release blocker.
