# Jetson Orin measurement packages

These compact packages retain the matched six-resolution execution matrix used
for the peer-review measurements: QPs 0, 21, 42 and 63; GOPs 1, 30 and 100;
100 frames; ten throughput and memory repetitions; and one quality pass per
condition.

The primary FPS field is `process_fps` (`frames / process_seconds`, including
process initialization and warm-up). The archived `metrics.throughput_fps`
field reports the synchronized completed-codec interval separately.

- [NVCR results](summary.md): native TensorRT execution, condition statistics,
  audit and the compact [results.jsonl](nvcr/results.jsonl) dataset.
- [Python reference results](python-reference/summary.md): DCVC-RT reference
  execution, condition statistics, common decoded-output quality/rate points,
  source manifest, audit and the compact
  [results.jsonl](python-reference/results.jsonl) dataset.

Raw per-operation observations, logs, bitstreams and reconstructions remain in
the operator's local evidence directories. The committed files are the compact
publication datasets and their provenance records.

## Regenerate the statistical exports

Each file has one row per sequence/QP/GOP/implementation/operation/mode/metric
in `campaign=full`: 720 rows per implementation, including 144 process-FPS
rows from throughput-mode observations and 144 process RSS rows from memory-mode
observations. The process execution is the statistical unit. Both packages
retain all four QPs and ten repetitions per condition. The exporter checks
all 144 encode/decode conditions and ten distinct repeats in each mode.

From the repository root, regenerate both files from the committed compact
latest-operation observations (no benchmark run):

```bash
python3 scripts/export_observation_statistics.py --campaign full --expected-conditions 144 \
  results/jetson-orin/measurement/nvcr/results.jsonl \
  results/jetson-orin/measurement/condition-statistics.csv
python3 scripts/export_observation_statistics.py --campaign full --expected-conditions 144 \
  results/jetson-orin/measurement/python-reference/results.jsonl \
  results/jetson-orin/measurement/python-reference/condition-statistics.csv
```

The CSV reports n, arithmetic mean, median, sample SD, extrema, CV and a
two-sided Student-t 95% interval for each condition mean. It assumes
independent, stationary fresh-process repetitions; intervals are descriptive
and unadjusted for multiple conditions. Resolution overview tables use pooled
throughput (`sum(frames) / sum(process_seconds)`), which is separate from the
per-condition arithmetic FPS means and their intervals. Completed-frame codec
FPS remains secondary evidence.
