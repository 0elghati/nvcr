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
| Completed codec throughput | encode | 0.71% | 6.18% |
| Completed codec throughput | decode | 1.50% | 9.74% |
| Process RSS high-water | encode | 0.13% | 1.73% |
| Process RSS high-water | decode | 0.10% | 3.96% |

The largest throughput CV is QCIF, QP42, GOP100, decode: **176.12 ± 17.16 FPS** (mean ± sample SD, n=10), range 148.56–197.46 FPS, CV 9.74%. Its descriptive 95% Student-t interval for the mean is 163.85–188.40 FPS. Keep all ten observations; additional repetitions reveal this variability rather than eliminating it.

The CSV includes mean, sample SD, CV, minimum/maximum and 95% Student-t intervals for each time, FPS and RSS quantity. Intervals use df=9 and assume independent, stationary executions; they are not adjusted for multiple conditions and do not establish absence of thermal or clock drift. The original paper’s 7.09% within-condition variation should not be compared directly to CV unless its denominator/definition matches.

## Throughput overview

Each range below spans the 12 separate condition means (four QPs × three GOPs) for that sequence. It is not a confidence interval or a pooled mean.

| Resolution | Encode FPS range | Decode FPS range |
|---|---:|---:|
| 176×144 | 116.74–221.14 | 72.51–183.82 |
| 352×288 | 45.84–97.60 | 34.14–85.86 |
| 640×360 | 20.15–50.21 | 22.55–49.78 |
| 960×540 | 9.59–22.81 | 11.35–24.64 |
| 1280×720 | 5.55–12.78 | 6.61–14.14 |
| 1920×1080 | 2.43–5.65 | 2.94–6.33 |

## Interpretation limits

1. **TensorRT device-model warning in all 3,024 operations.** Catalog/bundle checks and codec execution passed, but TensorRT still reports use of a plan across different device models. This is a reproducibility caveat requiring investigation before claiming a warning-free exact-target setup. The warning is in each operation’s stderr, not in the main nohup progress output.
2. **Memory is whole-process RSS high-water in MiB**, including initialization and warm-up. It is not isolated CUDA allocation or total Jetson memory. Median memory CV is low, but 1080p QP42 GOP1 decode has mean 1,070.00 MiB and SD 42.39 MiB (3.96% CV); keep that variation visible.
3. Power-mode snapshots agree on MAXN_SUPER. Governors remain CPU `schedutil` / GPU `nvhost_podgov`; snapshots do not establish fixed clocks throughout. Available temperatures rose from roughly 50–52°C to 62–64°C. These readings do not prove the cause of timing variation or exclude throttling.
4. FPS refers to the new synchronized completed-codec interval. Process elapsed time is exported separately. Do not mix these values with older codec-loop timing definitions.
5. The matched Python campaign is complete and retained in the companion package. Historical input acquisition/preprocessing and some FPS metadata remain unverified as recorded in the manifest; direct speedup and relative-memory conclusions must preserve the separate implementation/source identities.
6. Successful encoded and decoded products were deleted by the original runner after evaluation. The review checks retained hashes, accounting and SSE; it cannot re-decode those deleted products.

## Review artifacts

- `nvcr/results.jsonl`: 3,024 compact latest-operation NVCR observations.
- `condition-statistics.csv`: 576 condition/operation/metric rows with n=10.
- `audit.json`: audit results and SHA256 identities of source result files.
- Original `observations.jsonl`, `analysis.json` and `rd-points.json` remain unchanged in the local raw campaign directory.

No repetition or outlier was excluded, and no further benchmark was run.
