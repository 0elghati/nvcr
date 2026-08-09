# RTX 4070 Python DCVC-RT vs NVCR

This report is generated from the current canonical Python and NVCR JSONL
datasets. It is diagnostic evidence, not a release or target-support claim.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [python/data/results.jsonl](python/data/results.jsonl) | `ce4485182f2b0e172ef92b120109efd94a921329bae5f44b6f8e5d24a3a2057d` | 36 | Six resolutions, GOP 1/30/100, encode/decode, one row per case |
| NVCR | [nvcr/data/results.jsonl](nvcr/data/results.jsonl) | `114b229555fba937ba9f7cd1617ad676abd195c07918cd20f7f4b5657ded0bc5` | 144 | Six resolutions, GOP 1/30/100, encode/decode, three repetitions plus averages |

Both datasets use RTX 4070, QP 32, and 100 frames. The NVCR run uses 10
warm-up frames, three measured repetitions, and memory profiling. Its recorded
commit is `1381ff73ec863376f8c39fcdd8871ef73d68d496`; the checkout was dirty.
The Python source is the DCVC-RT `runner/dcvc_rt/gop_fps` result set and has
one row per comparison case.

## Decode comparison

Python reports process FPS and average all-frame quality. NVCR reports
codec-loop FPS and YUV PSNR. These timing boundaries differ, so the FPS values
are descriptive and are not a controlled speedup measurement.

`BPP delta` is `(NVCR BPP / Python BPP) - 1`. `PSNR delta` is NVCR PSNR minus
Python PSNR. NVCR rows are the average of the three measured repetitions.

| Resolution | GOP | Python FPS | NVCR FPS | Python BPP | NVCR BPP | BPP delta | Python PSNR | NVCR PSNR | PSNR delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QCIF | 1 | 151.625 | 599.280 | 0.187746 | 0.205836 | +9.64% | 35.916 | 35.789 | -0.127 |
| QCIF | 30 | 152.002 | 979.151 | 0.023346 | 0.041929 | +79.60% | 38.306 | 38.156 | -0.150 |
| QCIF | 100 | 157.740 | 1030.706 | 0.013185 | 0.031872 | +141.73% | 38.209 | 38.018 | -0.191 |
| CIF | 1 | 112.560 | 271.043 | 0.285711 | 0.290680 | +1.74% | 31.989 | 31.961 | -0.028 |
| CIF | 30 | 126.920 | 579.565 | 0.042651 | 0.047277 | +10.85% | 34.576 | 34.445 | -0.131 |
| CIF | 100 | 128.290 | 605.420 | 0.021358 | 0.026065 | +22.04% | 34.858 | 34.740 | -0.118 |
| 360p | 1 | 74.371 | 160.804 | 0.140025 | 0.142358 | +1.67% | 36.320 | 36.192 | -0.127 |
| 360p | 30 | 96.360 | 379.877 | 0.030723 | 0.032739 | +6.56% | 35.133 | 34.940 | -0.193 |
| 360p | 100 | 97.707 | 398.613 | 0.025131 | 0.027043 | +7.61% | 34.915 | 34.690 | -0.225 |
| 540p | 1 | 39.201 | 77.642 | 0.106875 | 0.108026 | +1.08% | 37.460 | 37.301 | -0.158 |
| 540p | 30 | 43.577 | 199.733 | 0.023644 | 0.024566 | +3.90% | 36.159 | 35.975 | -0.184 |
| 540p | 100 | 44.335 | 211.146 | 0.019431 | 0.020366 | +4.81% | 35.920 | 35.700 | -0.220 |
| 720p | 1 | 21.071 | 40.646 | 0.097990 | 0.098790 | +0.82% | 38.853 | 38.648 | -0.205 |
| 720p | 30 | 27.774 | 101.856 | 0.011474 | 0.012060 | +5.10% | 40.765 | 40.456 | -0.309 |
| 720p | 100 | 28.629 | 106.922 | 0.006080 | 0.006667 | +9.65% | 40.710 | 40.371 | -0.339 |
| 1080p | 1 | 10.173 | 18.087 | 0.065259 | 0.065682 | +0.65% | 37.878 | 37.710 | -0.168 |
| 1080p | 30 | 13.395 | 48.576 | 0.015416 | 0.015676 | +1.69% | 36.679 | 36.531 | -0.148 |
| 1080p | 100 | 13.384 | 51.641 | 0.012898 | 0.013162 | +2.04% | 36.484 | 36.325 | -0.159 |

Across the 18 decode cases, the mean PSNR delta is `-0.177 dB`, the median
delta is `-0.164 dB`, and 13 cases are within `0.2 dB`. The largest quality
gaps are 720p GOP 100 (`-0.339 dB`) and 720p GOP 30 (`-0.309 dB`).

## Encode comparison

Values are ordered GOP 1 / 30 / 100.

| Resolution | Python FPS | NVCR FPS |
|---|---:|---:|
| QCIF | 18.010 / 18.130 / 18.382 | 539.024 / 890.267 / 915.816 |
| CIF | 17.462 / 17.799 / 17.968 | 260.632 / 559.654 / 580.303 |
| 360p | 11.769 / 17.445 / 17.515 | 160.617 / 398.325 / 417.027 |
| 540p | 13.365 / 15.809 / 16.008 | 77.230 / 213.928 / 227.894 |
| 720p | 11.064 / 13.615 / 13.861 | 38.215 / 97.197 / 102.091 |
| 1080p | 7.163 / 10.002 / 10.125 | 16.539 / 47.361 / 50.323 |

The median coefficient of variation across the three NVCR repetitions is 0.66%
for encode throughput and 0.28% for decode throughput.

## Assessment

The current run is structurally clean and fully covered, but the BPP gap grows
with long GOPs at the smaller resolutions. That gap is approximately `+4.81%`
to `+141.73%` for GOP 100, so this dataset does not support a claim of
cross-runtime compression equivalence. The remaining investigation should
focus on long-GOP state, reference-feature evolution, and entropy payload
accounting.

Memory measurements are available in the NVCR rows. They are omitted from the
numeric comparison because the Python and NVCR samplers and timing boundaries
are not identical.
