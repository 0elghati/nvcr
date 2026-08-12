# RTX 4070: all-intra comparison and FPS

This completed report compares NVCR against the Python DCVC-RT reference and keeps the all-intra FPS results visible.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [results.jsonl](python/data/results.jsonl) | `ce4485182f2b0e172ef92b120109efd94a921329bae5f44b6f8e5d24a3a2057d` | 36 | 36 comparison cases |
| NVCR | [results.jsonl](nvcr/data/results.jsonl) | `6a1d77cc197f17d90930a2dab9fb291eb41c0547d6ab929bea626a3b16165cdb` | 144 | 36 average cases plus retained repetition rows |

Both datasets use the hardware label `rtx4070`, QP values 32, and frame counts 100. Coverage is QCIF, CIF, 360p, 540p, 720p, 1080p with GOPs 1, 30, 100 and encode/decode operations.

## Comparable inner-entropy rate

Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × 58) × 8 / (width × height × frames)`.
The derivation removes the fixed 58-byte non-entropy overhead in every retained NVCR access unit.
`Relative difference` is `(NVCR inner-entropy BPP / Python inner-bitstream BPP) - 1`.

| Resolution | GOP | Python inner-bitstream BPP | NVCR derived inner-entropy BPP | Relative difference |
|---|---:|---:|---:|---:|
| QCIF | 1 | 0.187746 | 0.187528 | -0.12% |
| CIF | 1 | 0.285711 | 0.286103 | +0.14% |
| 360p | 1 | 0.140025 | 0.140344 | +0.23% |
| 540p | 1 | 0.106875 | 0.107131 | +0.24% |
| 720p | 1 | 0.097990 | 0.098286 | +0.30% |
| 1080p | 1 | 0.065259 | 0.065458 | +0.30% |

Across these 6 explicitly selected cases, the relative-difference range is -0.12% to +0.30%; the median is +0.23%.

Python records a 64-frame feature-reference reset interval; the caller identifies NVCR as 32 frames. Because the inter-reference policies differ, inter-coded GOPs 30, 100 are excluded. GOP 1 is all-intra and does not exercise that policy.

## Strongest inter-coded FPS by resolution

The strongest recorded FPS in the inter-coded runs comes from GOP 100 at every resolution. GOP 30 remains visible in the full table below.

| Resolution | Best inter-coded GOP | Encode FPS | Decode FPS |
|---|---:|---:|---:|
| QCIF | 100 | 915.816 | 1030.706 |
| CIF | 100 | 580.303 | 605.420 |
| 360p | 100 | 417.027 | 398.613 |
| 540p | 100 | 227.894 | 211.146 |
| 720p | 100 | 102.091 | 106.922 |
| 1080p | 100 | 50.323 | 51.641 |

## FPS results

NVCR's measured codec-loop FPS is shown below. Values are the average rows from the recorded repetitions.

| Resolution | GOP | Encode FPS | Decode FPS |
|---|---:|---:|---:|
| QCIF | 1 | 539.024 | 599.280 |
| QCIF | 30 | 890.267 | 979.151 |
| QCIF | 100 | 915.816 | 1030.706 |
| CIF | 1 | 260.632 | 271.043 |
| CIF | 30 | 559.654 | 579.565 |
| CIF | 100 | 580.303 | 605.420 |
| 360p | 1 | 160.617 | 160.804 |
| 360p | 30 | 398.325 | 379.877 |
| 360p | 100 | 417.027 | 398.613 |
| 540p | 1 | 77.230 | 77.642 |
| 540p | 30 | 213.928 | 199.733 |
| 540p | 100 | 227.894 | 211.146 |
| 720p | 1 | 38.215 | 40.646 |
| 720p | 30 | 97.197 | 101.856 |
| 720p | 100 | 102.091 | 106.922 |
| 1080p | 1 | 16.539 | 18.087 |
| 1080p | 30 | 47.361 | 48.576 |
| 1080p | 100 | 50.323 | 51.641 |

## Additional recorded metrics

- PSNR, memory, and wrapper-inclusive payload BPP remain available in the source JSONL.
