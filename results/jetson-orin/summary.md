# Jetson Orin Python DCVC-RT vs NVCR

This completed execution report is generated from the recorded Python DCVC-RT and NVCR JSONL datasets.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [results.jsonl](python/data/results.jsonl) | `22b98a7d84857dd570f82b331b67ca0d419a7c6a242b64ca255861b192638f5a` | 37 | 36 comparison cases |
| NVCR | [results.jsonl](nvcr/results.jsonl) | `fecc463d5bc0ea0f8683c699456f77161e1098763f840f6e2d612e5d3c69d974` | 108 | 36 average cases plus retained repetition rows |

Both datasets use the hardware label `jetson-orin`, QP values 32, and frame counts 100. Coverage is QCIF, CIF, 360p, 540p, 720p, 1080p with GOPs 1, 30, 100 and encode/decode operations.

## Comparable inner-entropy rate

Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × 58) × 8 / (width × height × frames)`.
The derivation removes the fixed 58-byte non-entropy overhead in every retained NVCR access unit.
`Relative difference` is `(NVCR inner-entropy BPP / Python inner-bitstream BPP) - 1`.

| Resolution | GOP | Python BPP | NVCR BPP | BPP delta | NVCR PSNR-YUV | Python PSNR-YUV | PSNR delta (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QCIF | 1 | 0.187743 | 0.187437 | -0.16% | 35.786 | 35.919 | -0.133 |
| QCIF | 30 | 0.023561 | 0.023576 | +0.07% | 38.099 | 38.324 | -0.225 |
| QCIF | 100 | 0.013516 | 0.013570 | +0.40% | 37.922 | 38.241 | -0.319 |
| CIF | 1 | 0.285699 | 0.286054 | +0.12% | 31.961 | 31.989 | -0.029 |
| CIF | 30 | 0.042682 | 0.042715 | +0.08% | 34.447 | 34.589 | -0.141 |
| CIF | 100 | 0.021428 | 0.021362 | -0.31% | 34.742 | 34.872 | -0.130 |
| 360p | 1 | 0.140051 | 0.140287 | +0.17% | 36.194 | 36.326 | -0.132 |
| 360p | 30 | 0.030624 | 0.030699 | +0.24% | 34.961 | 35.138 | -0.177 |
| 360p | 100 | 0.025091 | 0.025044 | -0.19% | 34.706 | 34.931 | -0.226 |
| 540p | 1 | 0.106861 | 0.107083 | +0.21% | 37.310 | 37.466 | -0.156 |
| 540p | 30 | 0.023637 | 0.023662 | +0.10% | 35.983 | 36.174 | -0.191 |
| 540p | 100 | 0.019452 | 0.019503 | +0.26% | 35.704 | 35.935 | -0.231 |
| 720p | 1 | 0.097984 | 0.098309 | +0.33% | 38.648 | 38.851 | -0.203 |
| 720p | 30 | 0.011496 | 0.011562 | +0.57% | 40.450 | 40.760 | -0.310 |
| 720p | 100 | 0.006081 | 0.006167 | +1.42% | 40.357 | 40.702 | -0.345 |
| 1080p | 1 | 0.065264 | 0.065426 | +0.25% | 37.710 | 37.878 | -0.168 |
| 1080p | 30 | 0.015414 | 0.015463 | +0.32% | 36.535 | 36.679 | -0.144 |
| 1080p | 100 | 0.012919 | 0.012928 | +0.07% | 36.321 | 36.488 | -0.167 |

Across these 18 explicitly selected cases, the relative-difference range is -0.31% to +1.42%; the median is +0.19%.

### Aggregate comparison

| GOP | BPP delta range | Median BPP delta | Median PSNR delta (dB) |
|---:|---:|---:|---:|
| 1 | -0.16% to +0.33% | +0.19% | -0.144 |
| 30 | +0.07% to +0.57% | +0.17% | -0.184 |
| 100 | -0.31% to +1.42% | +0.17% | -0.228 |

Across the selected cases, the PSNR delta range is -0.345 to -0.029 dB; the median is -0.173 dB.

Python and NVCR are identified with the same 64-frame feature-reference reset interval. Only the explicitly selected GOPs are reported.

## Throughput

Python and NVCR `throughput_fps` values are shown side by side using the recorded encode and decode rows.

| Resolution | GOP | Python encode FPS | NVCR encode FPS | Python decode FPS | NVCR decode FPS |
|---|---:|---:|---:|---:|---:|
| QCIF | 1 | 8.136 | 111.407 | 14.432 | 77.600 |
| QCIF | 30 | 8.544 | 200.791 | 19.080 | 126.831 |
| QCIF | 100 | 8.106 | 215.107 | 19.291 | 132.542 |
| CIF | 1 | 7.859 | 43.699 | 9.483 | 35.832 |
| CIF | 30 | 8.239 | 87.830 | 14.323 | 60.895 |
| CIF | 100 | 8.269 | 89.085 | 13.533 | 62.682 |
| 360p | 1 | 6.290 | 20.326 | 6.267 | 22.905 |
| 360p | 30 | 7.599 | 46.112 | 9.497 | 39.594 |
| 360p | 100 | 7.466 | 48.145 | 10.080 | 41.066 |
| 540p | 1 | 4.279 | 9.535 | 3.499 | 11.378 |
| 540p | 30 | 5.818 | 21.622 | 4.724 | 22.154 |
| 540p | 100 | 5.920 | 22.462 | 4.909 | 23.345 |
| 720p | 1 | 2.965 | 5.303 | 1.915 | 6.520 |
| 720p | 30 | 4.411 | 11.994 | 2.959 | 13.483 |
| 720p | 100 | 4.456 | 12.486 | 3.226 | 13.841 |
| 1080p | 1 | — | 2.428 | 1.053 | 3.027 |
| 1080p | 30 | 2.715 | 5.382 | 1.645 | 5.989 |
| 1080p | 100 | — | 5.620 | 1.659 | 6.177 |

### Strongest NVCR inter-coded codec throughput

The strongest NVCR codec-loop result among the recorded inter-coded GOPs is shown per resolution.

| Resolution | Best encode GOP | Encode FPS | Best decode GOP | Decode FPS |
|---|---:|---:|---:|---:|
| QCIF | 100 | 215.107 | 100 | 132.542 |
| CIF | 100 | 89.085 | 100 | 62.682 |
| 360p | 100 | 48.145 | 100 | 41.066 |
| 540p | 100 | 22.462 | 100 | 23.345 |
| 720p | 100 | 12.486 | 100 | 13.841 |
| 1080p | 100 | 5.620 | 100 | 6.177 |

## Additional recorded metrics

- Wrapper-inclusive NVCR payload BPP remains available in the source JSONL; the rate table above uses inner-entropy BPP.
