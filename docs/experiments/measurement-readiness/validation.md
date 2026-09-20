# Measurement validation record

Date: 2026-09-20. The NVCR-only campaign is complete: all 3,024 configured
operations and the preceding eight-operation smoke passed. The
[current result](../../../results/jetson-orin/measurement/summary.md),
[aggregate statistics](../../../results/jetson-orin/measurement/condition-statistics.csv)
and [audit](../../../results/jetson-orin/measurement/audit.json) retain compact
evidence. Raw runs and logs remain local under `output/peer-review/`.

The complete NVCR-versus-Python evaluation remains open. The pinned Python
reference requires a passing GPU smoke, and persistent TensorRT device-model
warnings prevent a warning-free exact-target qualification.

## Software verification

The Release TensorRT CLI and production C++ wire fixture built successfully.
The original seven-suite selection passed measurement semantics, legacy
consolidation, SoftwareX orchestration, runtime contracts, format contracts,
DCVC-RT payloads and CLI inter-GOP acceptance.

The final focused CPU-only checks passed 3/3 on 2026-09-20:

```bash
ctest --test-dir build-release --output-on-failure \
  -R 'nvcr_measurement_semantics|nvcr_legacy_consolidation|nvcr_softwarex_driver'
```

The measurement suite contains 23 tests, covering independently known PSNR,
exact reconstruction, malformed dimensions/lengths, real production-writer byte
reconciliation, sample SD, duplicate/incomplete observations, child failures and
timeouts, coding schedules, NVCR-only selection, bound profile line endings,
device-name alias constraints and RD validation. The two shell launchers pass
`bash -n`. No GPU experiment was repeated for these software checks.

## Completed NVCR campaign

The matrix has six resolutions, QPs 0/21/42/63, GOPs 1/30/100 and 100 measured
frames after ten same-session warm-up frames. Each condition has ten independent
encode processes and ten independent decode processes in each of the throughput
and memory modes. One separate quality pass per condition gives 72 quality
results. The smoke checks cold versus warmed/reset stream and reconstruction
hashes for encode and decode.

The offline audit found no missing, failed or duplicated observations. It
verified all 3,024 retained metric files, recalculated 576 mean/sample-SD
summaries and recomputed quality from per-frame plane SSE/sample counts. All
repeated encoded streams within each condition have identical recorded hashes.
All 18 four-point RD curves are monotonic. Descriptive Student-t intervals are
included in the CSV, with their assumptions recorded in the audit.

The maximum throughput CV is 9.74% (QCIF, QP42, GOP100 decode). Every operation
contains a TensorRT device-model warning. These observations remain visible;
no outliers were excluded. Memory values are whole-process RSS high-water in
MiB, not isolated GPU allocation. Before/after snapshots cannot establish fixed
clocks or absence of throttling throughout the experiment.

## Artifact and source identity

The device reports Orin, AArch64, compute capability 8.7, eight SMs, CUDA 12.6
and TensorRT 10.3.0. The six published bundles from the
[engine-assets release](https://github.com/0elghati/nvcr/releases/tag/engine-assets)
passed inventory/digest and engine/build-target profile validation.

The model-profile hash discrepancy was entirely CRLF versus LF. The normalized
bytes match the published digest
`3aaa4e2f3d309090e2947e065ce9d6be8456b01d14fde8a05f87f0daed348246` exactly.
The checker accepts only that exact match and saves a normalized evidence copy;
it does not rewrite the original profile or published engine manifest. The
registered `Jetson Orin Nano` marketing name is mapped explicitly to CUDA `Orin`
for its eight-SM, SM-8.7 target; numeric hardware/runtime checks remain enforced.

The measured source was `fbcc19b1bc12e74c3d89517b1941ef68ddf75af4` plus the
measurement changes recorded by the run's source diff and file hashes. The
result audit preserves run and source-result SHA256 identities. A later commit
must not be represented as the measured source revision without these records.

## Reference and historical comparison limits

The clean `assets` checkout is pinned at
`1feb52a592a9ff2c4e4ba2e5122e2da49a211466`; checkpoint hashes passed. The selected
reference interpreter is `/home/oelghati/DCVC-RT/.venv-jetson/bin/python`.
CUDA PyTorch and the required inference/entropy extensions imported successfully,
but the bounded reference attempt failed during warm-up with
`CUDA error: misaligned address`. The captured stderr remains local at
`evidence/measurement-readiness/python-smoke/encode.stderr`. A rebuilt extension
is a diagnostic option, not an established fix.

Historical RTX4070 Python data contain QP32 summaries using
`ave_all_frame_psnr`, without the common decoded-output quality contract or
source hashes. The pinned upstream evaluator computes mean-frame PSNR before
8-bit serialization. Those historical records cannot be converted to the new
pooled decoded-output metric from averages alone. They must not be compared
against the new QP0/21/42/63 results as matched quality observations.

See the [contract](../measurement-contract.md), [assessment](assessment.md) and
[runbook](../measurement-runbook.md) for definitions and execution commands.
