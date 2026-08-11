# Reproducible experiments

This directory defines the reproducible NVCR evaluation protocol. It
is separate from the shorter product and usage documentation and evaluates both
architecture-level runtime claims and the first production vertical.

The evaluation covers the first complete NVCR vertical:

```text
codec adapter:       DCVC-RT
execution provider:  TensorRT
runtime:             C++20
artifacts:           target-local TensorRT engine bundles
stream layer:        bounded NVAU access units
pixels:              planar YUV420P8
```

The production claim is narrow: NVCR provides a reproducible native runtime
around this one codec/provider pair. Architecture-level evidence additionally
covers session lifecycle, delayed output, registration, provider mediation,
artifact resolution, bounded parsing, and malformed rejection. This is not a
claim of universal NVIDIA support, multi-codec production support, or Python
bitstream compatibility.

## Read in this order

1. [Evaluation protocol](evaluation-protocol.md)
2. [Hardware targets](hardware-targets.md)
3. [Compatibility levels](compatibility-levels.md)
4. [Result schema](result-schema.md)
5. [Runbook](runbook.md)
6. [Missing-data checklist](missing-data-checklist.md)
7. [Input manifest example](inputs.example.json)

The reviewer-facing claim matrix is in [../reproducibility/claims-and-evidence.md](../reproducibility/claims-and-evidence.md).

## Current state

The repository has device detection, target profiles, artifact validation,
target-local engine building, Docker/Compose workflows, the lower-level
`benchmark_resolution_matrix.sh`, and the publication driver
`scripts/benchmark_softwarex_matrix.py`. The driver validates profile digests
and target identity before execution, runs registered build/test gates, captures
the complete result schema, and writes the evidence package described below.

Checked-in target profiles cover RTX 3050 Laptop, RTX 4070, RTX 5060 Laptop,
and Jetson Orin Nano. A target row becomes publication evidence only after live
identity detection, current artifact validation, registered GPU gates, and the
complete protocol pass. The RTX 4070 profile uses TensorRT 10.9; its exact
bundles must be rebuilt from the current model-profile digest before a clean
publication run.

The driver is tested automation, not evidence by itself. A result package is
complete only when it records a clean commit, current artifacts, registered and
passing GPU gates, the requested matrix, a separate `--profile` collection
pass, and the mandatory RTX 4070 Python reference comparison. Primary
throughput and wall-time values always come from repetitions without optional
profiling instrumentation.

## Evidence rule

A clean evidence package contains metadata, hashes, commands, summaries, and small JSONL rows. Keep checkpoints, TensorRT plans, raw YUV, encoded streams, reconstructed video, and other large products outside Git.
