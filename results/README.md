# NVCR execution results

NVCR has completed execution results on Jetson Orin and RTX 4070, with
portability completed on RTX 5060. Each package includes the recorded data and
per-resolution, per-GOP NVCR encode/decode FPS.

| Target | Status | Coverage | What the report shows |
|---|---|---|---|
| [Jetson Orin](jetson-orin/) | Complete | Six resolutions, QP 32, GOP 1/30/100, 100 frames | Inter-coded comparison and throughput |
| [RTX 4070](rtx4070/) | Complete | Six resolutions, QP 32, GOP 1/30/100, 100 frames | All-intra comparison and throughput |
| [RTX 5060](rtx5060/) | Portability complete | QCIF through 1080p, QP 63, GOP 1/8/265 | Portability coverage and throughput |
| [RTX 3050](rtx3050/) | Placeholder | Future completion target | [Retained configuration reference](rtx3050/summary.md) |

## Reading the results

- Throughput is the recorded NVCR codec-loop encode/decode FPS. The target
  reports call out the strongest FPS per resolution and keep the full per-GOP
  tables below them.
- Jetson Orin and RTX 4070 both keep the inter-coded throughput visible.
- JSONL is the recorded source data; CSV and Markdown summaries are derived
  views.

For the measurement method, see [Performance](../docs/performance.md).
