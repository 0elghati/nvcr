# Jetson Orin results

The retained Jetson Orin results include the historical comparison folders and
the new matched repeated-measurement packages. The [NVCR measurement
summary](measurement/summary.md) and [Python reference measurement
summary](measurement/python-reference/summary.md) use six resolutions, QPs
0/21/42/63, GOP 1/30/100, 100 frames and ten throughput/memory repetitions.

- `nvcr/data/`: current NVCR R=64 run, with three measured repetitions.
- `python/data/`: pinned Python reference dataset.
- `measurement/`: compact NVCR and Python repeated-measurement results; these
  are the peer-review evidence packages and use the common quality contract.
