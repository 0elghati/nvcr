# Reproducible evaluation

This directory defines NVCR's detailed evaluation protocol. It supplements the
end-user guides and evaluates both runtime architecture contracts and the
supported end-to-end integration:

```text
DCVC-RT codec adapter -> TensorRT FP16 provider -> Linux/NVIDIA target
```

The supported scope is narrow. Architecture checks cover session
lifecycle, delayed output, registration, provider mediation, artifact
resolution, bounded parsing, and malformed-input rejection. Target evaluation
adds current target-local artifacts, native I/P execution, quality, rate,
performance, memory, and reference comparison. It does not establish universal
NVIDIA support, multiple supported codecs/providers, or Python payload
interchangeability.

## Read in this order

1. [Evaluation protocol](evaluation-protocol.md)
2. [Hardware targets](hardware-targets.md)
3. [Compatibility levels](compatibility-levels.md)
4. [Result schema](result-schema.md)
5. [Runbook](runbook.md)
6. [Readiness checklist](missing-data-checklist.md)
7. [Input manifest example](inputs.example.json)

For the higher-level distinctions between functional validation, diagnostic,
and controlled runs, start with
[Performance and benchmarking](../performance.md).

## Current state

The repository provides device detection, target profiles, artifact validation,
target-local engine building, Docker/Compose workflows, the lower-level
`benchmark_resolution_matrix.sh`, and the strict evaluation driver
`scripts/benchmark_softwarex_matrix.py`. The legacy filename is retained for
compatibility. The driver validates profile digests and target identity, runs
registered build/test gates, captures the complete result schema, and writes an
evidence package.

Checked-in target profiles cover RTX 3050 Laptop, RTX 4070, RTX 5060 Laptop,
and Jetson Orin Nano. A profile is an artifact request, not a support claim. A
row becomes controlled evidence only after live identity detection, current
artifact validation, registered GPU gates, and the complete protocol pass.

The driver is tested automation, not evidence by itself. A package is complete
only when it records a clean commit, current artifacts, passing registered
gates, the requested matrix, separate profiling repetitions, and required
pinned-reference comparisons. Primary throughput and wall-time values always
come from repetitions without optional profiling instrumentation.

## Retention rule

A clean evidence package contains metadata, hashes, commands, summaries, and
small JSONL rows. Keep checkpoints, TensorRT plans, raw YUV, encoded streams,
reconstructed video, and other large products outside Git. Index every retained
package and its limitations in [`results/README.md`](../../results/README.md).
