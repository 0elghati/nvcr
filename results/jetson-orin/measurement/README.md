# Jetson Orin measurement packages

These compact packages retain the matched six-resolution execution matrix used
for the peer-review measurements: QPs 0, 21, 42 and 63; GOPs 1, 30 and 100;
100 frames; ten throughput and memory repetitions; and one quality pass per
condition.

- [NVCR results](summary.md): native TensorRT execution, condition statistics
  and audit.
- [Python reference results](python-reference/summary.md): DCVC-RT reference
  execution, condition statistics, common decoded-output quality/rate points,
  source manifest and audit.

Raw per-operation observations, logs, bitstreams and reconstructions remain in
the operator's local evidence directories. The committed files are the compact
publication datasets and their provenance records.
