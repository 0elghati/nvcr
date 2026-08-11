# Performance

This page defines the measurement protocol. Reported numbers come from the
machine-readable result set rather than an accumulating narrative log.

The publication matrix, target records, result schema, and run commands live in [docs/experiments](experiments/README.md). The matrix is evidence for both
architecture-level gates and the DCVC-RT/TensorRT production vertical; CPU
fixture results do not establish production performance.

## Rules

Run a Release build only after the configured correctness and artifact tests pass. Record:

- commit, target profile, engine/profile digests, and command;
- OS, compiler, driver or JetPack, CUDA, TensorRT, GPU, power mode, and clocks;
- sequence, format, dimensions, frame count, QP, GOP pattern, and reset pattern;
- warm-up count, repetitions, timing boundaries, and statistic;
- encode/decode throughput, latency, memory, payload size, and reconstruction quality;
- the matching pinned-Python command and tolerance when comparing runtimes.

Use the same input, profile, QP, frame count, and timing boundary for both runtimes. Do not turn one machine or one sequence into a general performance claim.

Keep optional instrumentation outside primary timing. The publication-matrix driver
always derives FPS, FPS variation, compatibility ratios, and total process wall
time from repetitions without verbose per-frame output, quality calculation,
process polling, or `nvidia-smi` polling. `--profile` adds separate repetitions
for latency, PSNR, and memory. Never substitute profile-pass throughput for the
clean-pass values.

## Current evidence

The retained file
[resolution-matrix.jsonl](../evidence/live-release-20260806/resolution-matrix.jsonl)
is a diagnostic matrix, not a complete publication result package. Use its
recorded commit, artifact identity, status, and error fields when interpreting
the rows; do not promote it to a release baseline.

Raw streams are local run products and are not part of the documentation set.

## Run the matrix

The publication driver is the source of the matrix command, result schema, and
completion state:

```bash
python3 scripts/benchmark_softwarex_matrix.py --help
```

It writes `nvcr.softwarex.result.v1` rows and refuses to mark a package complete
when `--profile`, a build/test gate, required metric, target/artifact identity,
Python reference, or compatibility baseline is missing. Omitting `--profile`
is the low-overhead performance-only mode and always leaves the package
`partial`. The older
`benchmark_resolution_matrix.sh` output uses
`nvcr.benchmark.resolution-matrix.v1` and remains diagnostic.

The publication run must cover QCIF, CIF, 360p, 720p, and 1080p, with 540p
optional; all-intra and normal I/P GOP cases; three measured repetitions; one
warm-up run; payload/BPP; PSNR-Y/U/V/YUV; latency; host/GPU memory; and a clean
pinned-Python comparison for every RTX 4070 exact case.

Energy is optional investigation data, not a v1 release blocker.

## Direct Docker benchmark

The native diagnostic matrix can be run inside a pulled x86_64 runtime image
without Docker Compose:

```bash
scripts/benchmark_docker.sh \
  --image omarelghati/nvcr:0.19.1-amd64-cuda12.8-trt10.9 \
  --input-dir /data/nvcr/yuv \
  --engine-volume nvcr-engines \
  --results-dir evidence/performance/rtx4070-docker \
  --hardware rtx4070-docker \
  -- \
  --resolutions "qcif cif 360p 720p 1080p" \
  --frames 300 --qp 32 --gops "1 299" --repetitions 3
```

The launcher performs `docker pull`, mounts the input directory read-only, the
engine volume read-only, and evidence/output directories as writable mounts.
Rows contain codec-reported throughput plus process wall-time throughput,
which includes container process startup and is the metric to compare against
the native run. The launcher defaults to container user `0:0` for NVIDIA
device compatibility; pull an immutable image tag when retaining evidence.
