# RTX 5060 Laptop portability — NVCR 1.0.7

Published Linux/amd64 image executed through Windows Docker Desktop/WSL2. These are NVCR-only deployment diagnostics, not matched Python comparisons or native Windows inference.

- Six resolutions; QPs 0/21/42/63; GOPs 1/30/100; 100 frames.
- Ten measured repetitions per encode/decode condition: 144 conditions, 1,440 measured operations.
- All 24 measured smoke operations are present. Separate-process warm-up does not test reset equivalence.
- Memory profiling was disabled. Process throughput was not retained in the diagnostic JSONL.
- Image: `omarelghati/nvcr@sha256:e8b04d3efc3c4048167aef1bc9ff1c72c3b748f8f53584650f0cad761e2a57e9`.
- Binary release/revision are from OCI metadata; the original dirty flag describes the local harness.

## Codec throughput

Mean ± sample SD over ten independent process executions. Individual rounded codec FPS values are retained in results.jsonl. CI95 fields in results.csv are descriptive Student-t intervals; they do not establish independence/stationarity or compare devices.

| Resolution | QP | GOP | Encode FPS | Decode FPS |
|---|---:|---:|---:|---:|
| qcif | 0 | 1 | 218.365 ± 7.722 | 261.210 ± 7.724 |
| qcif | 0 | 30 | 342.189 ± 19.337 | 368.341 ± 26.519 |
| qcif | 0 | 100 | 374.441 ± 21.517 | 396.955 ± 18.887 |
| qcif | 21 | 1 | 220.067 ± 8.034 | 256.101 ± 7.938 |
| qcif | 21 | 30 | 351.712 ± 21.984 | 382.182 ± 16.688 |
| qcif | 21 | 100 | 366.138 ± 24.639 | 404.856 ± 15.814 |
| qcif | 42 | 1 | 219.305 ± 6.644 | 266.896 ± 5.922 |
| qcif | 42 | 30 | 350.101 ± 20.912 | 394.287 ± 17.493 |
| qcif | 42 | 100 | 407.058 ± 39.313 | 420.798 ± 10.368 |
| qcif | 63 | 1 | 217.879 ± 7.248 | 261.621 ± 9.503 |
| qcif | 63 | 30 | 350.704 ± 19.681 | 400.280 ± 15.703 |
| qcif | 63 | 100 | 357.741 ± 20.519 | 403.832 ± 19.947 |
| cif | 0 | 1 | 149.717 ± 1.386 | 168.407 ± 1.138 |
| cif | 0 | 30 | 307.552 ± 18.592 | 294.185 ± 10.285 |
| cif | 0 | 100 | 328.457 ± 16.218 | 313.600 ± 5.593 |
| cif | 21 | 1 | 149.651 ± 1.226 | 167.244 ± 1.591 |
| cif | 21 | 30 | 319.147 ± 18.472 | 303.689 ± 5.863 |
| cif | 21 | 100 | 334.692 ± 16.653 | 312.715 ± 4.075 |
| cif | 42 | 1 | 147.756 ± 3.469 | 162.544 ± 1.183 |
| cif | 42 | 30 | 326.387 ± 7.971 | 303.430 ± 4.677 |
| cif | 42 | 100 | 345.752 ± 12.343 | 313.087 ± 3.143 |
| cif | 63 | 1 | 149.708 ± 0.967 | 152.369 ± 0.938 |
| cif | 63 | 30 | 327.522 ± 8.446 | 299.845 ± 4.419 |
| cif | 63 | 100 | 345.284 ± 7.387 | 309.814 ± 7.821 |
| 360p | 0 | 1 | 91.017 ± 0.376 | 103.247 ± 0.738 |
| 360p | 0 | 30 | 236.832 ± 2.931 | 203.213 ± 2.613 |
| 360p | 0 | 100 | 254.848 ± 3.951 | 212.876 ± 1.046 |
| 360p | 21 | 1 | 90.753 ± 0.350 | 102.016 ± 0.326 |
| 360p | 21 | 30 | 239.565 ± 7.873 | 205.187 ± 1.185 |
| 360p | 21 | 100 | 249.272 ± 10.956 | 211.994 ± 1.780 |
| 360p | 42 | 1 | 90.876 ± 0.296 | 101.215 ± 0.491 |
| 360p | 42 | 30 | 232.534 ± 12.542 | 204.757 ± 0.885 |
| 360p | 42 | 100 | 248.100 ± 17.433 | 209.755 ± 2.495 |
| 360p | 63 | 1 | 90.614 ± 0.576 | 97.664 ± 0.606 |
| 360p | 63 | 30 | 236.280 ± 5.879 | 199.858 ± 1.035 |
| 360p | 63 | 100 | 239.669 ± 13.619 | 199.162 ± 9.472 |
| 540p | 0 | 1 | 46.283 ± 0.088 | 52.679 ± 0.100 |
| 540p | 0 | 30 | 139.734 ± 0.661 | 116.620 ± 0.354 |
| 540p | 0 | 100 | 149.689 ± 0.350 | 122.120 ± 0.334 |
| 540p | 21 | 1 | 46.075 ± 0.069 | 52.121 ± 0.091 |
| 540p | 21 | 30 | 139.706 ± 1.318 | 117.272 ± 0.391 |
| 540p | 21 | 100 | 149.149 ± 1.730 | 122.118 ± 0.305 |
| 540p | 42 | 1 | 46.207 ± 0.065 | 51.762 ± 0.087 |
| 540p | 42 | 30 | 139.677 ± 1.060 | 116.258 ± 0.422 |
| 540p | 42 | 100 | 144.680 ± 6.823 | 120.261 ± 0.304 |
| 540p | 63 | 1 | 46.106 ± 0.064 | 50.261 ± 0.094 |
| 540p | 63 | 30 | 138.588 ± 0.722 | 113.880 ± 0.299 |
| 540p | 63 | 100 | 146.411 ± 5.666 | 119.028 ± 0.565 |
| 720p | 0 | 1 | 25.229 ± 0.032 | 30.000 ± 0.042 |
| 720p | 0 | 30 | 79.419 ± 1.153 | 65.148 ± 0.195 |
| 720p | 0 | 100 | 84.558 ± 0.559 | 68.022 ± 0.221 |
| 720p | 21 | 1 | 25.143 ± 0.022 | 29.946 ± 0.060 |
| 720p | 21 | 30 | 78.635 ± 0.828 | 65.583 ± 0.095 |
| 720p | 21 | 100 | 84.888 ± 0.805 | 68.320 ± 0.149 |
| 720p | 42 | 1 | 25.205 ± 0.058 | 29.837 ± 0.061 |
| 720p | 42 | 30 | 78.837 ± 0.420 | 65.592 ± 0.143 |
| 720p | 42 | 100 | 84.838 ± 0.604 | 67.714 ± 0.293 |
| 720p | 63 | 1 | 25.212 ± 0.046 | 29.497 ± 0.054 |
| 720p | 63 | 30 | 78.907 ± 0.575 | 65.335 ± 0.178 |
| 720p | 63 | 100 | 84.941 ± 0.892 | 68.103 ± 0.422 |
| 1080p | 0 | 1 | 11.023 ± 0.069 | 13.276 ± 0.118 |
| 1080p | 0 | 30 | 33.873 ± 0.509 | 28.615 ± 0.034 |
| 1080p | 0 | 100 | 36.162 ± 0.368 | 29.694 ± 0.094 |
| 1080p | 21 | 1 | 11.069 ± 0.057 | 13.296 ± 0.202 |
| 1080p | 21 | 30 | 33.934 ± 0.430 | 28.643 ± 0.045 |
| 1080p | 21 | 100 | 36.155 ± 0.274 | 29.841 ± 0.063 |
| 1080p | 42 | 1 | 11.104 ± 0.041 | 13.351 ± 0.027 |
| 1080p | 42 | 30 | 33.963 ± 0.389 | 28.583 ± 0.145 |
| 1080p | 42 | 100 | 36.103 ± 0.211 | 29.544 ± 0.128 |
| 1080p | 63 | 1 | 11.088 ± 0.040 | 13.156 ± 0.023 |
| 1080p | 63 | 30 | 33.763 ± 0.719 | 28.175 ± 0.096 |
| 1080p | 63 | 100 | 36.232 ± 0.752 | 29.525 ± 0.073 |

## Reproduction and limits

Run `python3 scripts/consolidate_portability_results.py --source <original-campaign-directory> --output results/rtx5060`. The original source files remain local; provenance.json records their hashes. The compact condition observations retain all repetitions needed to regenerate the statistics.

The prior QP32/two-repetition result is superseded. These four QPs contain no QP32 observation; do not reuse the earlier Figure 5 values as new measurements. The paper awaits RTX 3050 integration.

The recorded bundle checks passed, but at-execution engine/input digests and full operation logs were not supplied. No warning-free, common-quality, memory-efficiency or cross-device speed claim is made.
