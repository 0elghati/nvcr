# Retained results

These files preserve diagnostic measurements and Python-reference comparisons.
None is a current release or controlled-performance baseline.

| Package | Category | Recorded scope | Principal limitation |
|---|---|---|---|
| [`rtx3050/`](rtx3050/) | Historical diagnostic | Six resolutions, QP 63, GOP 1/8/265, 266 frames, three repetitions | Exact platform, artifact digests, process timing, and test gates were not recorded; summary and rows disagree about dirty state |
| [`rtx5060/`](rtx5060/) | Historical diagnostic | Same matrix shape as RTX 3050 | Exact platform, artifact digests, process timing, and test gates were not recorded; summary and rows disagree about dirty state |
| [`rtx4070/results.jsonl`](rtx4070/results.jsonl) | Superseded diagnostic | Four resolutions, QP 63, GOP 1/8/265 | Older orphaned matrix; use the nested comparison below for the maintained RTX 4070 report |
| [`rtx4070/`](rtx4070/) ([summary](rtx4070/summary.md)) | NVCR/Python comparison | Six resolutions, QP 32, GOP 1/30/100, 100 frames | Environment, artifact, and Python source/checkpoint identities are incomplete; timing boundaries differ, so the report does not establish a speedup |
| [`jetson-orin/`](jetson-orin/) ([summary](jetson-orin/summary.md)) | NVCR/Python comparison | Six resolutions, QP 32, GOP 1/30/100, 100 frames | Exact Jetson/L4T and artifact identities are incomplete; R=32 source rows are absent and only derived deltas remain |
| [`../evidence/live-release-20260806/resolution-matrix.jsonl`](../evidence/live-release-20260806/resolution-matrix.jsonl) | Historical diagnostic | Six resolutions, QP 32, GOP 1/97, 97 frames | Target, artifact, environment, status, and error identities were not recorded |
| [`../evidence/performance/rtx5060-docker/`](../evidence/performance/rtx5060-docker/) ([summary](../evidence/performance/rtx5060-docker/summary.md)) | Failed container diagnostic | Docker image identity and a partial resolution matrix | 94 of 108 expected rows exist; several GOP-1 cases and generated headers are missing |

## Interpretation

- A hardware label or target-profile name is not an exact device or engine
  identity.
- Where a summary says the repository was dirty but JSONL says otherwise,
  treat the package as dirty or unresolved.
- Diagnostic `payload_bytes` counts serialized `NVAU` access-unit bytes. It is
  neither an inner entropy/rANS count nor the complete `.nvcr` file size.
- Codec-loop FPS and complete-process FPS are different timing boundaries.
- Machine-local paths in historical JSONL are retained as provenance, not as
  runnable instructions.
- JSONL is canonical; CSV and Markdown summaries are derived views.

For the measurement contract, see [Performance](../docs/performance.md) and the
[result schema](../docs/experiments/result-schema.md).
