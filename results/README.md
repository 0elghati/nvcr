# NVCR execution results

Matched NVCR/Python measurements are complete on RTX 4070 and Jetson Orin.
Process throughput is the primary comparison; codec-interval measurements are
reported separately. See the [measurement guide](../docs/performance.md) for coverage and retained
records.

| Target | Evaluation | Coverage |
|---|---|---|
| [RTX 4070](https://github.com/0elghati/nvcr/blob/c2480e8fdbf6b06be481c33d27d09c4c55e95f24/results/rtx4070/measurement/summary.md) | Matched NVCR/Python | 6 resolutions; QPs 0/21/42/63; GOPs 1/30/100; 10 repetitions |
| [Jetson Orin](https://github.com/0elghati/nvcr/blob/c2480e8fdbf6b06be481c33d27d09c4c55e95f24/results/jetson-orin/measurement/summary.md) | Matched NVCR/Python | Same matrix |
| [RTX 3050](rtx3050/summary.md) | Deployment | 6 resolutions; QP 32; GOPs 1/30/100; 2 repetitions |
| [RTX 5060](rtx5060/summary.md) | Deployment | Same deployment matrix |

The matched records above are pinned to source revision
`c2480e8fdbf6b06be481c33d27d09c4c55e95f24`.
The [Orin Python records](https://github.com/0elghati/nvcr/blob/c2480e8fdbf6b06be481c33d27d09c4c55e95f24/results/jetson-orin/measurement/python-reference/summary.md)
complete the Orin comparison.

## Historical reports

The earlier [Jetson Orin](jetson-orin/summary.md) and
[RTX 4070](rtx4070/summary.md) reports retain their original codec-loop
timing and quality definitions. They are separate from the matched process
comparison above. JSONL records retain the source observations; CSV and
Markdown summaries are derived views.

For the measurement method, see [Performance](../docs/performance.md).
