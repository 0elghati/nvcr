# RTX 4070 Python DCVC-RT vs NVCR

This report is generated from the supplied canonical Python and NVCR JSONL datasets. It is historical diagnostic evidence, not a release, support, or current-performance claim.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Rows | Coverage |
|---|---|---|---:|---|
| Python DCVC-RT | [results.jsonl](python/data/results.jsonl) | `ce4485182f2b0e172ef92b120109efd94a921329bae5f44b6f8e5d24a3a2057d` | 36 | 36 comparison cases |
| NVCR | [results.jsonl](nvcr/data/results.jsonl) | `6a1d77cc197f17d90930a2dab9fb291eb41c0547d6ab929bea626a3b16165cdb` | 144 | 36 average cases plus retained repetition rows |

Both datasets use the hardware label `rtx4070`, QP values 32, and frame counts 100. Coverage is QCIF, CIF, 360p, 540p, 720p, 1080p with GOPs 1, 30, 100 and encode/decode operations.
Matching labels, dimensions, QP, GOP, and frame counts do not establish byte-identical run-time inputs because input hashes were not retained.
The NVCR recorded commit is `1381ff73ec863376f8c39fcdd8871ef73d68d496`. Python duplicate keys are resolved using the latest `source_timestamp`; NVCR rows are selected from `run_index=average`.
Retained run-level provenance classifies the NVCR checkout as dirty. The canonical JSONL records `nvcr_dirty=false`; treat the run as dirty where those sources disagree.
Exact GPU and target identity, CUDA/TensorRT and engine digests, Python source revision, and checkpoint digest are not recorded, so the matrix is not a controlled baseline.

## Comparable inner-entropy rate

Python BPP is the decode row's inner `bit_stream` value. NVCR inner-entropy BPP is derived as `(payload_bytes - frames × 58) × 8 / (width × height × frames)`.
The derivation assumes a fixed 58-byte non-entropy overhead in every retained NVCR access unit. The supplied JSONL does not retain streams or inner-byte counts, so this assumption cannot be verified from the report inputs.
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

## Metrics intentionally not compared

- Throughput is omitted because Python records process-level timing while NVCR records codec-loop timing.
- PSNR is omitted because the retained runtimes use different temporal aggregation boundaries.
- Memory is omitted because the samplers and measurement scopes are not identical.
- Wrapper-inclusive NVCR `payload_bpp` is not compared with Python inner-bitstream BPP.
