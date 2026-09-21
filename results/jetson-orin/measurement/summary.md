# Jetson Orin NVCR repeated measurements — 2026-09-20

The NVCR campaign is complete and its retained numerical records pass an independent offline audit.

The matched Python reference campaign is also complete. Its compact package is
in [python-reference/](python-reference/summary.md); direct implementation
ratios should use both packages while preserving their recorded source and
runtime identities.

The local raw campaign is `output/peer-review/`. This directory contains the
current compact summary and aggregate statistics; raw observations, per-operation
logs and generated products remain outside source control. Historical result
folders retain their original timing and quality definitions and must not be
merged with these observations.

## Coverage and integrity

- Six resolutions: 176×144, 352×288, 640×360, 960×540, 1280×720 and 1920×1080.
- QPs 0, 21, 42, 63; GOPs 1, 30, 100; 100 measured frames with 10 same-session warm-up frames.
- All 3,024 expected operations passed, with no failures, duplicates, missing observations or mixed fingerprints.
- Throughput: 720 encodes + 720 decodes; memory: 720 encodes + 720 decodes. Each condition/operation has n=10 in each mode.
- Quality: 72 encodes + 72 decodes, one separate pass per condition. These are not ten repeated quality observations.
- The preceding eight-operation NVCR smoke passed, including matching cold versus warmed/reset stream and reconstruction hashes.
- Audit checked 3,024 retained metric files, recalculated 576 mean/sample-SD summaries, checked stored byte reconciliation and recomputed 72 quality records from per-frame plane SSE/sample counts.
- All 21 encoded outputs per condition (ten throughput, ten memory, one quality) have the same recorded stream hash. All 18 four-point RD curves increase in both bitrate and PSNR as the model QP index increases.
- The current executable and measurement-source hashes match the recorded run identity. No GPU execution was launched during this review.

## Variability across independent process executions

CV = 100 × sample SD / mean. Values below summarize the CVs of the 72 conditions separately for each operation; they do not pool different QPs, GOPs or scenes.

| Quantity | Operation | Median CV | Maximum CV |
|---|---|---:|---:|
| Process-level FPS | encode | 0.76% | 2.50% |
| Process-level FPS | decode | 1.40% | 3.07% |
| Process RSS high-water | encode | 0.13% | 1.73% |
| Process RSS high-water | decode | 0.10% | 3.96% |

The largest process-level FPS CV is 360p, QP21, GOP100, decode: **18.00 ± 0.55 FPS** (mean ± sample SD, n=10), CV 3.07%. Keep all ten observations; additional repetitions reveal this variability rather than eliminating it.

The archived completed-codec FPS has a larger maximum CV: QCIF, QP42, GOP100, decode is **176.12 ± 17.16 FPS**, CV 9.74%. It remains available in the JSONL and CSV as a secondary implementation metric.

The CSV includes mean, sample SD, CV, minimum/maximum and 95% Student-t intervals for each time, FPS and RSS quantity. Intervals use df=9 and assume independent, stationary executions; they are not adjusted for multiple conditions and do not establish absence of thermal or clock drift. The original paper’s 7.09% within-condition variation should not be compared directly to CV unless its denominator/definition matches.

## Primary narrative: process-level FPS

Process-level FPS is frames divided by isolated process wall time, including model initialization and warm-up. The means below pool the 12 QP/GOP conditions and ten throughput repetitions per resolution.

| Resolution | Encode FPS | Decode FPS |
|---|---:|---:|
| 176×144 | 29.15 | 27.23 |
| 352×288 | 23.35 | 21.66 |
| 640×360 | 15.78 | 15.98 |
| 960×540 | 10.48 | 11.18 |
| 1280×720 | 7.20 | 7.85 |
| 1920×1080 | 3.60 | 4.05 |

## Secondary archive: completed-codec FPS

`metrics.throughput_fps` is the synchronized completed-codec interval, excluding process initialization. Each range below spans the 12 separate condition means (four QPs × three GOPs) for that sequence. It is not a confidence interval or a pooled mean.

| Resolution | Encode FPS range | Decode FPS range |
|---|---:|---:|
| 176×144 | 116.74–221.14 | 72.51–183.82 |
| 352×288 | 45.84–97.60 | 34.14–85.86 |
| 640×360 | 20.15–50.21 | 22.55–49.78 |
| 960×540 | 9.59–22.81 | 11.35–24.64 |
| 1280×720 | 5.55–12.78 | 6.61–14.14 |
| 1920×1080 | 2.43–5.65 | 2.94–6.33 |

## Paired Python-versus-NVCR quality and rate

The 72 common conditions have a paired quality comparison in
[`quality-comparison.csv`](quality-comparison.csv). Both implementations use
the `decoded-yuv420p8-pooled-plane-6-1-1-v1` contract; each condition has one
quality pass, so these values are descriptive paired differences rather than
ten-repetition confidence intervals.

Python minus NVCR pooled decoded-YUV PSNR averages **+0.0286 dB** (median
`+0.0215 dB`, range `−0.1270` to `+0.1294 dB`, mean absolute difference
`0.0347 dB`). The corresponding entropy-BPP difference averages **−0.20%**
(median `−0.15%`, range `−2.03%` to `+1.47%`). Per-resolution means are:

| Resolution | PSNR difference (dB) | Entropy-BPP difference |
|---|---:|---:|
| 176×144 | +0.033 | −0.58% |
| 352×288 | +0.017 | +0.04% |
| 640×360 | +0.016 | −0.08% |
| 960×540 | +0.014 | −0.14% |
| 1280×720 | +0.080 | −0.25% |
| 1920×1080 | +0.011 | −0.19% |

File-BPP differences are not used for the codec-rate conclusion because the
NVCR container and Python stream wrapper have different fixed overheads. The
comparison is empirical evidence under the recorded implementation and source
identities; it does not establish that the two implementations are internally
identical.

## Interpretation limits

1. **TensorRT device-model warning in all 3,024 operations.** Catalog/bundle checks and codec execution passed, but TensorRT still reports use of a plan across different device models. This is a reproducibility caveat requiring investigation before claiming a warning-free exact-target setup. The warning is in each operation’s stderr, not in the main nohup progress output.
2. **Memory is whole-process RSS high-water in MiB**, including initialization and warm-up. It is not isolated CUDA allocation or total Jetson memory. Median memory CV is low, but 1080p QP42 GOP1 decode has mean 1,070.00 MiB and SD 42.39 MiB (3.96% CV); keep that variation visible.
3. Power-mode snapshots agree on MAXN_SUPER. Governors remain CPU `schedutil` / GPU `nvhost_podgov`; snapshots do not establish fixed clocks throughout. Available temperatures rose from roughly 50–52°C to 62–64°C. These readings do not prove the cause of timing variation or exclude throttling.
4. Process-level FPS is the primary narrative metric. Completed-codec FPS is retained as a secondary archive metric; do not mix the two timing definitions.
5. The matched Python campaign is complete and retained in the companion package. Historical input acquisition/preprocessing and some FPS metadata remain unverified as recorded in the manifest; direct speedup and relative-memory conclusions must preserve the separate implementation/source identities.
6. Successful encoded and decoded products were deleted by the original runner after evaluation. The review checks retained hashes, accounting and SSE; it cannot re-decode those deleted products.

## Review artifacts

- `nvcr/results.jsonl`: 3,024 compact latest-operation NVCR observations.
- `condition-statistics.csv`: 576 condition/operation/metric rows with n=10.
- `audit.json`: audit results and SHA256 identities of source result files.
- `quality-comparison.csv`: 72 paired Python/NVCR PSNR and rate observations.
- Original `observations.jsonl`, `analysis.json` and `rd-points.json` remain unchanged in the local raw campaign directory.

No repetition or outlier was excluded, and no further benchmark was run.
