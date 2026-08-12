# RTX 4070 results

This completed RTX 4070 run compares NVCR with Python DCVC-RT over six
resolutions, GOP 1/30/100, and 100 frames. The [summary](summary.md) includes
the rate comparison and NVCR encode/decode FPS for every recorded case.

- [python/data/results.jsonl](python/data/results.jsonl) is the canonical Python DCVC-RT dataset.
- [nvcr/results.jsonl](nvcr/results.jsonl) is the canonical NVCR dataset.
- Both datasets cover QP 32, 100 frames, six resolutions, GOP sizes 1/30/100, and encode/decode operations.
- Both runs use a 64-frame feature-reference reset interval, so all three GOPs are directly comparable.
