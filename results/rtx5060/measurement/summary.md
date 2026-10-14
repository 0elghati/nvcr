# RTX 5060 Laptop NVCR 1.0.7 process/codec measurements

This corrected cohort supersedes the separate-process-warm-up diagnostic dataset.
Execution is Linux/amd64 through Windows Docker Desktop/WSL2, not native Windows inference.
The measurement image adds NumPy for smoke quality evaluation to the pinned release image;
the release executable is unchanged and is bound by its SHA256 to the recorded OCI receipt.

## Complete operation-level data

- 1,440 passed operations: six sequences/resolutions, QPs 0/21/42/63, GOPs 1/30/100, encode/decode and ten repetitions.
- Exactly 100 timed frames per execution after ten same-session warm-up frames, reset and rewind.
- Eight smoke operations passed, including cold versus warmed/reset stream and decoded-output hash equality.
- All 17 preflight checks passed; input-prefix/full-file and exact engine identities are retained.
- All 1,440 individual process durations/FPS, initialization durations and synchronized codec durations/FPS are retained in operation-measurements.csv and operation-measurements.jsonl.
- condition-statistics.csv contains 576 summaries: four timing metrics for each of 144 operation conditions, including arithmetic mean, median, sample SD, extrema, CV and descriptive 95% Student-t intervals.
- run.json and manifest.json preserve the resolved plan and measured harness identity; preflight.json and provenance.json preserve artifact and input identities. Binary identity is not inferred from the harness dirty state.
- Memory and full quality passes were intentionally not run. The smoke uses bounded memory/quality checks; throughput-mode RSS fields are not a separate repeated-memory result.
- Exactly 240 GOP-1 all-intra warnings were found in the 1,440 operation stderr logs; no other warning/error lines were found.

## Primary process throughput

Process FPS is 100 divided by the full isolated-process wall interval, including startup, initialization, warm-up/reset, file I/O and teardown. Codec FPS is reported separately using host-yuv420p8-completed-frame-v1, with a device barrier before and after each timed frame. Offline quality checks are excluded from throughput jobs.

Pooled rates below are total timed frames divided by total process time over equal-work runs (120 executions per resolution and operation). They are not arithmetic condition means, device rankings or Python speedups.

| Sequence | Encode process FPS | Decode process FPS |
|---|---:|---:|
| akiyo-qcif | 70.782 | 83.338 |
| waterfall-cif | 59.553 | 72.604 |
| basketball-360p | 49.433 | 59.845 |
| basketball-540p | 31.717 | 42.596 |
| fourpeople-720p | 22.550 | 30.681 |
| basketball-1080p | 12.056 | 15.340 |

## Reproduction and provenance

Run python3 scripts/export_measurement_summary.py <campaign-directory> results/rtx5060/measurement to regenerate the operation tables and all timing statistics. The original source records are bound by SHA256 in provenance.json. Statistics use each execution's FPS; they are not computed as 100 divided by the mean process duration.

The recorded measurement-harness working diff is preserved in run.json/provenance.json. Later exporter changes do not change the measured executable or timing observations. This cohort must not be merged with the superseded diagnostic cohort, whose timers and warm-up policy differ. The prior dataset remains available in repository history at commit 8f30bb4.

The paper and RTX 3050 integration remain pending. No matched Python, latency-distribution, memory-efficiency or native-Windows claim is made.
