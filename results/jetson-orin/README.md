# Jetson Orin results

This completed Jetson Orin run compares NVCR with Python DCVC-RT over six
resolutions, GOP 1/30/100, and 100 frames. The [summary](summary.md) includes
rate, quality, and NVCR encode/decode throughput.

- `nvcr/data/`: current NVCR R=64 run, with three measured repetitions.
- `python/data/`: pinned Python reference dataset.
