# RTX 4070: inter-coded comparison and FPS

This completed report compares NVCR against the Python DCVC-RT reference at
every recorded GOP and keeps the FPS results visible.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [results.jsonl](python/data/results.jsonl) | `ce4485182f2b0e172ef92b120109efd94a921329bae5f44b6f8e5d24a3a2057d` | 36 | 36 comparison cases |
| NVCR R=64 | [results.jsonl](nvcr/results.jsonl) | `fc0c53bc0f2cee173f213d49995ed97db1e88c15832872644b8b8db0cad99f88` | 108 | 36 average cases plus retained repetition rows |

Both datasets use the hardware label `rtx4070`, QP values 32, and frame counts 100. Coverage is QCIF, CIF, 360p, 540p, 720p, 1080p with GOPs 1, 30, 100 and encode/decode operations. Python and NVCR are both identified with a 64-frame feature-reference reset interval, so all three GOPs are directly comparable.

## Comparable inner-entropy rate and quality

Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × 58) × 8 / (width × height × frames)`.
The derivation removes the fixed 58-byte non-entropy overhead in every retained NVCR access unit.
`BPP delta` is `(NVCR inner-entropy BPP / Python inner-bitstream BPP) - 1`. `PSNR delta` is NVCR PSNR-YUV minus Python average all-frame PSNR.

| Resolution | GOP | NVCR BPP | Python BPP | BPP delta | NVCR PSNR | Python PSNR | PSNR delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| QCIF | 1 | 0.187528 | 0.187746 | -0.12% | 35.789 | 35.916 | -0.127 dB |
| QCIF | 30 | 0.023621 | 0.023346 | +1.18% | 38.156 | 38.306 | -0.150 dB |
| QCIF | 100 | 0.013564 | 0.013185 | +2.87% | 38.018 | 38.209 | -0.191 dB |
| CIF | 1 | 0.286103 | 0.285711 | +0.14% | 31.961 | 31.989 | -0.028 dB |
| CIF | 30 | 0.042700 | 0.042651 | +0.11% | 34.445 | 34.576 | -0.131 dB |
| CIF | 100 | 0.021488 | 0.021358 | +0.61% | 34.740 | 34.858 | -0.118 dB |
| 360p | 1 | 0.140344 | 0.140025 | +0.23% | 36.192 | 36.320 | -0.127 dB |
| 360p | 30 | 0.030725 | 0.030723 | +0.01% | 34.940 | 35.133 | -0.193 dB |
| 360p | 100 | 0.025029 | 0.025131 | -0.41% | 34.690 | 34.915 | -0.225 dB |
| 540p | 1 | 0.107131 | 0.106875 | +0.24% | 37.301 | 37.460 | -0.158 dB |
| 540p | 30 | 0.023671 | 0.023644 | +0.11% | 35.975 | 36.159 | -0.184 dB |
| 540p | 100 | 0.019471 | 0.019431 | +0.21% | 35.700 | 35.920 | -0.220 dB |
| 720p | 1 | 0.098286 | 0.097990 | +0.30% | 38.648 | 38.853 | -0.205 dB |
| 720p | 30 | 0.011556 | 0.011474 | +0.72% | 40.456 | 40.765 | -0.309 dB |
| 720p | 100 | 0.006164 | 0.006080 | +1.37% | 40.371 | 40.710 | -0.339 dB |
| 1080p | 1 | 0.065458 | 0.065259 | +0.30% | 37.710 | 37.878 | -0.168 dB |
| 1080p | 30 | 0.015452 | 0.015416 | +0.24% | 36.531 | 36.679 | -0.148 dB |
| 1080p | 100 | 0.012938 | 0.012898 | +0.31% | 36.325 | 36.484 | -0.159 dB |

### Aggregate comparison

| GOP | NVCR BPP delta range | Median BPP delta | Median PSNR delta |
|---:|---:|---:|---:|
| 1 | -0.12% to +0.30% | +0.24% | -0.143 dB |
| 30 | +0.01% to +1.18% | +0.18% | -0.167 dB |
| 100 | -0.41% to +2.87% | +0.46% | -0.206 dB |

Across all 18 cases, NVCR entropy BPP is within -0.41% to +2.87% of Python, and NVCR PSNR is within -0.34 dB of Python at every case.

## Strongest inter-coded FPS by resolution

The strongest recorded encode FPS in the inter-coded runs comes from GOP 100 at
every resolution. Decode FPS also peaks at GOP 100 at every resolution except
QCIF, where GOP 30 is marginally faster (1033.758 vs 1021.886, +1.2%, within
the two-repetition measurement noise). GOP 100 is used below for a GOP that is
consistent across resolutions.

| Resolution | Best inter-coded GOP | Encode FPS | Decode FPS |
|---|---:|---:|---:|
| QCIF | 100 | 957.692 | 1021.886 |
| CIF | 100 | 576.553 | 606.987 |
| 360p | 100 | 408.543 | 399.723 |
| 540p | 100 | 225.741 | 211.098 |
| 720p | 100 | 102.380 | 107.388 |
| 1080p | 100 | 50.303 | 51.689 |

## FPS results

NVCR's measured codec-loop FPS is shown below. Values are the average rows from the recorded repetitions.

| Resolution | GOP | Encode FPS | Decode FPS |
|---|---:|---:|---:|
| QCIF | 1 | 511.567 | 601.672 |
| QCIF | 30 | 934.139 | 1033.758 |
| QCIF | 100 | 957.692 | 1021.886 |
| CIF | 1 | 258.295 | 271.468 |
| CIF | 30 | 553.868 | 582.268 |
| CIF | 100 | 576.553 | 606.987 |
| 360p | 1 | 159.469 | 162.194 |
| 360p | 30 | 391.979 | 381.815 |
| 360p | 100 | 408.543 | 399.723 |
| 540p | 1 | 77.246 | 78.123 |
| 540p | 30 | 213.126 | 200.988 |
| 540p | 100 | 225.741 | 211.098 |
| 720p | 1 | 38.329 | 42.376 |
| 720p | 30 | 97.266 | 102.276 |
| 720p | 100 | 102.380 | 107.388 |
| 1080p | 1 | 16.591 | 18.174 |
| 1080p | 30 | 47.307 | 49.032 |
| 1080p | 100 | 50.303 | 51.689 |

## Additional recorded metrics

- Memory and wrapper-inclusive payload BPP remain available in the source JSONL.

