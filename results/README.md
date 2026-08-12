# NVCR execution results

NVCR has completed execution results on Jetson Orin and RTX 4070, with
portability completed on RTX 5060. Each package includes the recorded data and
per-resolution, per-GOP NVCR encode/decode FPS.

| Target | Status | Coverage | Results |
|---|---|---|---|
| [Jetson Orin](jetson-orin/) | Complete | Six resolutions, QP 32, GOP 1/30/100, 100 frames | [Rate, quality, and throughput](jetson-orin/summary.md) |
| [RTX 4070](rtx4070/) | Complete | Six resolutions, QP 32, GOP 1/30/100, 100 frames | [DCVC-RT comparison and throughput](rtx4070/summary.md) |
| [RTX 5060](rtx5060/) | Portability complete | QCIF through 1080p, QP 63, GOP 1/8/265 | [Execution results](rtx5060/summary.md) |
| [RTX 3050](rtx3050/) | Placeholder | Future completion target | [Retained configuration reference](rtx3050/summary.md) |

## Reading the results

- Throughput is the recorded NVCR codec-loop encode/decode FPS. It is shown in
  every completed target report.
- Jetson Orin and RTX 4070 also compare NVCR's inner entropy rate with
  DCVC-RT's Python reference at matching settings.
- JSONL is the recorded source data; CSV and Markdown summaries are derived
  views.

For the measurement method, see [Performance](../docs/performance.md).
