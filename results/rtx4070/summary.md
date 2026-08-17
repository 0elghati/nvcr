# RTX 4070 Python DCVC-RT vs NVCR

This completed execution report is generated from the recorded Python DCVC-RT and NVCR JSONL datasets.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [results.jsonl](python/data/results.jsonl) | `b0022c28f4a743aa6215ba1790630f4c8e97d4c3cbf8fe3796c49e4a9b83661a` | 36 | 36 comparison cases |
| NVCR | [results.jsonl](nvcr/results.jsonl) | `25c82bacc4959389d5d278e9cad91b22dc092a96838fa0b9468a5045ffb517cb` | 108 | 36 average cases plus retained repetition rows |

Both datasets use the hardware label `rtx4070`, QP values 32, and frame counts 100. Coverage is QCIF, CIF, 360p, 540p, 720p, 1080p with GOPs 1, 30, 100 and encode/decode operations.

## Comparable inner-entropy rate

Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × 58) × 8 / (width × height × frames)`.
The derivation removes the fixed 58-byte non-entropy overhead in every retained NVCR access unit.
`Relative difference` is `(NVCR inner-entropy BPP / Python inner-bitstream BPP) - 1`.

| Resolution | GOP | Python BPP | NVCR BPP | BPP delta | NVCR PSNR-YUV | Python PSNR-YUV | PSNR delta (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QCIF | 1 | 0.187746 | 0.187528 | -0.12% | 35.789 | 35.916 | -0.127 |
| QCIF | 30 | 0.023346 | 0.023621 | +1.18% | 38.156 | 38.306 | -0.150 |
| QCIF | 100 | 0.013185 | 0.013564 | +2.87% | 38.018 | 38.209 | -0.191 |
| CIF | 1 | 0.285711 | 0.286103 | +0.14% | 31.961 | 31.989 | -0.028 |
| CIF | 30 | 0.042651 | 0.042700 | +0.11% | 34.445 | 34.576 | -0.131 |
| CIF | 100 | 0.021358 | 0.021488 | +0.61% | 34.740 | 34.858 | -0.118 |
| 360p | 1 | 0.140025 | 0.140344 | +0.23% | 36.192 | 36.320 | -0.127 |
| 360p | 30 | 0.030723 | 0.030725 | +0.01% | 34.940 | 35.133 | -0.193 |
| 360p | 100 | 0.025131 | 0.025029 | -0.41% | 34.690 | 34.915 | -0.225 |
| 540p | 1 | 0.106875 | 0.107131 | +0.24% | 37.301 | 37.460 | -0.158 |
| 540p | 30 | 0.023644 | 0.023671 | +0.11% | 35.975 | 36.159 | -0.184 |
| 540p | 100 | 0.019431 | 0.019471 | +0.21% | 35.700 | 35.920 | -0.220 |
| 720p | 1 | 0.097990 | 0.098286 | +0.30% | 38.648 | 38.853 | -0.205 |
| 720p | 30 | 0.011474 | 0.011556 | +0.72% | 40.456 | 40.765 | -0.309 |
| 720p | 100 | 0.006080 | 0.006164 | +1.37% | 40.371 | 40.710 | -0.339 |
| 1080p | 1 | 0.065259 | 0.065458 | +0.30% | 37.710 | 37.878 | -0.168 |
| 1080p | 30 | 0.015416 | 0.015452 | +0.24% | 36.531 | 36.679 | -0.148 |
| 1080p | 100 | 0.012898 | 0.012938 | +0.31% | 36.325 | 36.484 | -0.159 |

Across these 18 explicitly selected cases, the relative-difference range is -0.41% to +2.87%; the median is +0.24%.

### Aggregate comparison

| GOP | BPP delta range | Median BPP delta | Median PSNR delta (dB) |
|---:|---:|---:|---:|
| 1 | -0.12% to +0.30% | +0.23% | -0.143 |
| 30 | +0.01% to +1.18% | +0.17% | -0.167 |
| 100 | -0.41% to +2.87% | +0.46% | -0.205 |

Across the selected cases, the PSNR delta range is -0.339 to -0.028 dB; the median is -0.164 dB.

Python and NVCR are identified with the same 64-frame feature-reference reset interval. Only the explicitly selected GOPs are reported.

## Throughput

Python and NVCR `throughput_fps` values are shown side by side using the recorded encode and decode rows.

| Resolution | GOP | Python encode FPS | NVCR encode FPS | Python decode FPS | NVCR decode FPS |
|---|---:|---:|---:|---:|---:|
| QCIF | 1 | 18.010 | 536.178 | 151.625 | 602.651 |
| QCIF | 30 | 18.130 | 898.100 | 152.002 | 1032.281 |
| QCIF | 100 | 18.382 | 914.651 | 157.740 | 1024.494 |
| CIF | 1 | 17.462 | 260.120 | 112.560 | 272.565 |
| CIF | 30 | 17.799 | 526.318 | 126.920 | 582.228 |
| CIF | 100 | 17.968 | 550.071 | 128.290 | 604.520 |
| 360p | 1 | 11.769 | 160.286 | 74.371 | 161.998 |
| 360p | 30 | 17.445 | 389.682 | 96.360 | 381.536 |
| 360p | 100 | 17.515 | 416.428 | 97.707 | 400.243 |
| 540p | 1 | 13.365 | 77.552 | 39.201 | 77.822 |
| 540p | 30 | 15.809 | 213.869 | 43.577 | 200.224 |
| 540p | 100 | 16.008 | 227.510 | 44.335 | 211.503 |
| 720p | 1 | 11.064 | 38.391 | 21.071 | 42.407 |
| 720p | 30 | 13.615 | 98.539 | 27.774 | 102.195 |
| 720p | 100 | 13.861 | 102.555 | 28.629 | 107.463 |
| 1080p | 1 | 7.163 | 16.590 | 10.173 | 18.230 |
| 1080p | 30 | 10.002 | 47.399 | 13.395 | 48.962 |
| 1080p | 100 | 10.125 | 50.458 | 13.384 | 51.947 |

### Strongest NVCR inter-coded codec throughput

The strongest NVCR codec-loop result among the recorded inter-coded GOPs is shown per resolution.

| Resolution | Best encode GOP | Encode FPS | Best decode GOP | Decode FPS |
|---|---:|---:|---:|---:|
| QCIF | 100 | 914.651 | 30 | 1032.281 |
| CIF | 100 | 550.071 | 100 | 604.520 |
| 360p | 100 | 416.428 | 100 | 400.243 |
| 540p | 100 | 227.510 | 100 | 211.503 |
| 720p | 100 | 102.555 | 100 | 107.463 |
| 1080p | 100 | 50.458 | 100 | 51.947 |

## Additional recorded metrics

- Wrapper-inclusive NVCR payload BPP remains available in the source JSONL; the rate table above uses inner-entropy BPP.
