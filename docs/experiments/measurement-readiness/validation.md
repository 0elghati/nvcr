# Measurement validation record

Date: 2026-09-21. The NVCR campaign and the matched Python reference campaign
are complete: each has 3,024 latest configured operations and a passing
eight-operation smoke. The [NVCR result](../../../results/jetson-orin/measurement/summary.md)
and [Python result](../../../results/jetson-orin/measurement/python-reference/summary.md)
retain compact evidence. Raw runs and logs remain local under the evidence
directories.

The execution gates pass, while direct speedup/memory interpretation remains
subject to the recorded implementation identities and the persistent TensorRT
device-model warnings in the NVCR run.

## Software verification

The Release TensorRT CLI and production C++ wire fixture built successfully.
The original seven-suite selection passed measurement semantics, legacy
consolidation, SoftwareX orchestration, runtime contracts, format contracts,
DCVC-RT payloads and CLI inter-GOP acceptance.

The final focused Release checks passed 7/7 on 2026-09-20:

```bash
ctest --test-dir build-release-rtx4070 --output-on-failure \
  -R 'nvcr_measurement_semantics|nvcr_legacy_consolidation|nvcr_softwarex_driver|nvcr_contract_tests|nvcr_format_contract_tests|nvcr_dcvcrt_payloads|nvcr_cli_accepts_inter_gop'
```

The measurement suite contains 25 tests, covering independently known PSNR,
exact reconstruction, malformed dimensions/lengths, real production-writer byte
reconciliation, sample SD, duplicate/incomplete observations, child failures and
timeouts, coding schedules, NVCR-only selection, bound profile line endings,
device-name alias constraints, portable RTX campaign identity and RD validation.
The three requested Python suites pass 47 tests with two expected optional
skips. The two shell launchers pass `bash -n` and ShellCheck.

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

## Completed Python reference campaign

The Python result package retains 576 condition/operation/metric rows with
n=10, 72 common-contract quality/rate points, a sanitized campaign manifest,
and an audit record. The latest 3,024 operations passed after resuming an
interrupted launch; two failed/skipped attempts remain retained in the local
raw observations. The Python checkout was at commit
`48ab0ac5e5199d78fffb944bfbafafb2b6142f7b` with tracked energy-measurement
edits and differed from the model-profile commit; the explicit source policy
and changed paths are recorded in the package audit.

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

## RTX 4070 NVCR readiness gate

The RTX workstation validation used NVCR source
`a444fb62e5351930d6c7a388b527acb7ad8da46e` plus the captured working-tree diff.
The host is Ubuntu 24.04.4 LTS with kernel 7.0.0-31, an AMD Ryzen 7 5800X,
31 GiB system RAM and an NVIDIA GeForce RTX 4070 with 12,282 MiB memory,
compute capability 8.9 and 46 SMs. The driver is 580.173.02; `nvidia-smi`
reports driver capability CUDA 13.0, while the measured CUDA runtime identity is
12.8. The build uses nvcc 12.8.93, TensorRT 10.9.0, GCC 13.3.0 and CMake
3.28.3. Python 3.13.13 is the campaign-control interpreter, not the deferred
PyTorch reference environment.

`build-release-rtx4070` is a fresh Release build explicitly targeting SM 8.9.
Its CLI SHA256 is
`d22754daa957bc9dfc7aa65dab5732cd698c5bba369014400588c108e7b27596`.
The previous `build-release` cache recorded nvcc 12.6, so it was preserved but
not used. The six catalog-selected bundles all passed inventory, model/profile,
exact target, SM-count, CUDA-runtime and TensorRT validation. They were already
built for RTX 4070, SM 8.9, CUDA 12.8 and TensorRT 10.9.0; no engine rebuild or
compatibility bypass was needed. The exact bundle and plan hashes remain in the
local preflight evidence.

The NVCR-only preflight and dry-run passed and resolved all 3,024 operations in
the local `evidence/measurement-nvcr-check.*` package. The bounded NVCR smoke,
whose package path is retained in
`evidence/measurement-rtx4070.smoke-path.txt`, completed all eight encode/decode
operations for QCIF QP 0, GOP 30, three measured frames and two warm-up frames.
There were no failed, skipped or missing observations. The metric schema,
completed-frame timing contract, per-frame synchronization, reset interval,
byte reconciliation and decoded quality checks passed. Cold versus warmed/reset
encoded-stream and decoded-output SHA256 comparisons both matched, and all
operation stderr files were empty. No full RTX campaign was launched.

This gate selected only NVCR. The portable RTX manifest retains both
implementations and the complete configured matrix, but the clean pinned source,
CUDA-enabled reference interpreter, Python doctor, matched smoke and one-condition
matched sanity gate remain pending.

## Reference and historical comparison limits

The selected reference interpreter is `/home/oelghati/DCVC-RT/.venv-jetson/bin/python`.
CUDA PyTorch and the required inference/entropy extensions imported
successfully. The first interrupted launch had an allocation failure, but the
resumed run completed every latest operation; the retained failed attempt and
its stderr remain local and are counted in the Python audit.

Historical RTX4070 Python data contain QP32 summaries using
`ave_all_frame_psnr`, without the common decoded-output quality contract or
source hashes. The pinned upstream evaluator computes mean-frame PSNR before
8-bit serialization. Those historical records cannot be converted to the new
pooled decoded-output metric from averages alone. They must not be compared
against the new QP0/21/42/63 results as matched quality observations.

See the [contract](../measurement-contract.md), [assessment](assessment.md) and
[runbook](../measurement-runbook.md) for definitions and execution commands.
