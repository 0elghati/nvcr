# RTX 3050 controlled timing experiment

Date: 2026-10-13. NVCR-only throughput campaign through Docker Desktop/WSL 2.

All 1,440 operations pass: six sequences/resolutions, QPs 0/21/42/63,
GOPs 1/30/100, encode/decode, ten independent repetitions. Each operation
times 100 frames after ten warm-up frames with same-session reset/rewind
and device synchronization around each frame. The eight-operation smoke
passes, including cold versus warmed/reset stream and reconstruction hashes.

Process throughput is the primary metric; codec interval throughput is
reported separately. Process intervals include initialization and file I/O.
Codec intervals exclude initialization and warm-up. Condition statistics
include n, mean, median, sample SD, CV, extrema and descriptive 95% intervals.

The measured executable is the pinned NVCR 1.0.7 release binary, separately
identified from the mounted measurement harness. The measurement image adds
NumPy for smoke checks without replacing the NVCR executable. Image, executable,
engine, input and harness identities are retained in provenance.json. The
container release receipt replaces unavailable native CMake build evidence;
this is not a new source build or native Windows execution.

Only throughput is measured across the full matrix. Quality and memory are
exercised in the bounded smoke, not across all conditions. There is no Python
reference comparison or RD claim. Host power and platform limitations remain
environment-specific. This supersedes the earlier legacy portability summary.

## Regenerate statistics

```bash
python3 scripts/export_observation_statistics.py --campaign full --expected-conditions 144 --mode throughput results/rtx3050/measurement/observations.jsonl results/rtx3050/measurement/condition-statistics.csv
```

Compact observations retain timing identities and measurements only. Raw
process outputs, commands, input frames and streams remain in ignored local
evidence. The original campaign and smoke were reanalyzed before publication.

## Process throughput condition means

| Sequence | QP | GOP | Encode frames/s | Decode frames/s |
|---|---:|---:|---:|---:|
| akiyo-qcif | 0 | 1 | 24.788 | 26.460 |
| akiyo-qcif | 0 | 30 | 25.766 | 27.764 |
| akiyo-qcif | 0 | 100 | 26.314 | 27.228 |
| akiyo-qcif | 21 | 1 | 24.552 | 26.186 |
| akiyo-qcif | 21 | 30 | 26.816 | 26.700 |
| akiyo-qcif | 21 | 100 | 25.857 | 27.259 |
| akiyo-qcif | 42 | 1 | 24.359 | 26.100 |
| akiyo-qcif | 42 | 30 | 25.536 | 26.857 |
| akiyo-qcif | 42 | 100 | 25.889 | 27.175 |
| akiyo-qcif | 63 | 1 | 23.928 | 24.895 |
| akiyo-qcif | 63 | 30 | 25.112 | 26.765 |
| akiyo-qcif | 63 | 100 | 26.239 | 27.139 |
| basketball-1080p | 0 | 1 | 3.899 | 4.886 |
| basketball-1080p | 0 | 30 | 7.452 | 8.813 |
| basketball-1080p | 0 | 100 | 7.757 | 9.212 |
| basketball-1080p | 21 | 1 | 3.918 | 4.859 |
| basketball-1080p | 21 | 30 | 7.561 | 8.909 |
| basketball-1080p | 21 | 100 | 7.779 | 9.216 |
| basketball-1080p | 42 | 1 | 3.901 | 4.823 |
| basketball-1080p | 42 | 30 | 7.601 | 8.979 |
| basketball-1080p | 42 | 100 | 7.745 | 9.202 |
| basketball-1080p | 63 | 1 | 3.886 | 4.796 |
| basketball-1080p | 63 | 30 | 7.487 | 8.861 |
| basketball-1080p | 63 | 100 | 7.696 | 9.154 |
| basketball-360p | 0 | 1 | 16.423 | 18.045 |
| basketball-360p | 0 | 30 | 21.235 | 22.737 |
| basketball-360p | 0 | 100 | 21.056 | 22.956 |
| basketball-360p | 21 | 1 | 16.421 | 18.007 |
| basketball-360p | 21 | 30 | 21.008 | 22.532 |
| basketball-360p | 21 | 100 | 20.758 | 22.681 |
| basketball-360p | 42 | 1 | 16.301 | 17.684 |
| basketball-360p | 42 | 30 | 20.301 | 22.103 |
| basketball-360p | 42 | 100 | 20.763 | 22.513 |
| basketball-360p | 63 | 1 | 16.198 | 17.644 |
| basketball-360p | 63 | 30 | 20.248 | 21.918 |
| basketball-360p | 63 | 100 | 20.869 | 22.230 |
| basketball-540p | 0 | 1 | 11.263 | 12.702 |
| basketball-540p | 0 | 30 | 16.398 | 18.110 |
| basketball-540p | 0 | 100 | 16.626 | 18.051 |
| basketball-540p | 21 | 1 | 10.979 | 12.517 |
| basketball-540p | 21 | 30 | 16.633 | 18.330 |
| basketball-540p | 21 | 100 | 16.793 | 18.554 |
| basketball-540p | 42 | 1 | 11.107 | 12.457 |
| basketball-540p | 42 | 30 | 16.558 | 18.276 |
| basketball-540p | 42 | 100 | 16.657 | 18.270 |
| basketball-540p | 63 | 1 | 11.270 | 12.536 |
| basketball-540p | 63 | 30 | 16.484 | 17.938 |
| basketball-540p | 63 | 100 | 16.506 | 17.276 |
| fourpeople-720p | 0 | 1 | 7.536 | 8.940 |
| fourpeople-720p | 0 | 30 | 12.826 | 14.613 |
| fourpeople-720p | 0 | 100 | 12.985 | 14.732 |
| fourpeople-720p | 21 | 1 | 7.496 | 8.915 |
| fourpeople-720p | 21 | 30 | 12.879 | 14.568 |
| fourpeople-720p | 21 | 100 | 13.230 | 14.776 |
| fourpeople-720p | 42 | 1 | 7.477 | 8.845 |
| fourpeople-720p | 42 | 30 | 12.772 | 14.573 |
| fourpeople-720p | 42 | 100 | 12.731 | 14.749 |
| fourpeople-720p | 63 | 1 | 7.430 | 8.813 |
| fourpeople-720p | 63 | 30 | 12.788 | 14.486 |
| fourpeople-720p | 63 | 100 | 12.914 | 14.285 |
| waterfall-cif | 0 | 1 | 20.871 | 22.323 |
| waterfall-cif | 0 | 30 | 23.400 | 25.245 |
| waterfall-cif | 0 | 100 | 23.566 | 25.578 |
| waterfall-cif | 21 | 1 | 20.981 | 22.627 |
| waterfall-cif | 21 | 30 | 24.296 | 24.937 |
| waterfall-cif | 21 | 100 | 23.849 | 25.998 |
| waterfall-cif | 42 | 1 | 20.722 | 22.186 |
| waterfall-cif | 42 | 30 | 24.529 | 25.306 |
| waterfall-cif | 42 | 100 | 24.100 | 25.195 |
| waterfall-cif | 63 | 1 | 21.142 | 21.406 |
| waterfall-cif | 63 | 30 | 23.884 | 25.549 |
| waterfall-cif | 63 | 100 | 24.594 | 25.400 |
