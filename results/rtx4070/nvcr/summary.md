# NVCR Performance Run

- Commit: fe8d891fb9fc4b68bbcd74dc00ade17783e0e386
- Dirty: true
- Hardware: rtx4070
- Execution mode: native
- Container image: none
- Container digest: none
- Resolutions: qcif cif 360p 540p 720p 1080p
- QP: 32
- GOPs: 1 30 100
- Frames: 100
- Warm-up frames: 10
- Measured repetitions: 2
- Memory profiling: false
- Memory sample interval ms: 100

The CSV and JSONL files in this directory contain individual repetitions and
average rows. Throughput is the runtime-reported encode/decode FPS. Payload BPP
is codec payload bits divided by width * height * frames. When memory profiling
is enabled, peak memory is the profiled NVCR RAM peak from /proc; Jetson system
RAM and largest-free-block values come from tegrastats when available.

## Results

| Operation | Resolution | Run | GOP | Payload bytes | Payload BPP | Codec seconds | Codec FPS | PSNR-YUV | Peak RAM MB | System RAM MB | Min LFB MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| encode | qcif | 1 | 1 | 65209 | 0.205836490 | 0.192 | 521.255 | null | null | null | null |
| decode | qcif | 1 | 1 | 65209 | 0.205836490 | 0.166 | 603.869 | 35.788776 | null | null | null |
| encode | qcif | 2 | 1 | 65209 | 0.205836490 | 0.181 | 551.102 | null | null | null | null |
| decode | qcif | 2 | 1 | 65209 | 0.205836490 | 0.166 | 601.433 | 35.788776 | null | null | null |
| encode | qcif | average | 1 | 65209 | 0.205836490 | 0.186500 | 536.178500 | null | null | null | null |
| decode | qcif | average | 1 | 65209 | 0.205836490 | 0.166000 | 602.651000 | 35.788776 | null | null | null |
| encode | cif | 1 | 1 | 368350 | 0.290680240 | 0.389 | 256.955 | null | null | null | null |
| decode | cif | 1 | 1 | 368350 | 0.290680240 | 0.366 | 273.448 | 31.961096 | null | null | null |
| encode | cif | 2 | 1 | 368350 | 0.290680240 | 0.380 | 263.286 | null | null | null | null |
| decode | cif | 2 | 1 | 368350 | 0.290680240 | 0.368 | 271.683 | 31.961096 | null | null | null |
| encode | cif | average | 1 | 368350 | 0.290680240 | 0.384500 | 260.120500 | null | null | null | null |
| decode | cif | average | 1 | 368350 | 0.290680240 | 0.367000 | 272.565500 | 31.961096 | null | null | null |
| encode | 360p | 1 | 1 | 409991 | 0.142357986 | 0.629 | 158.957 | null | null | null | null |
| decode | 360p | 1 | 1 | 409991 | 0.142357986 | 0.617 | 162.185 | 36.192285 | null | null | null |
| encode | 360p | 2 | 1 | 409991 | 0.142357986 | 0.619 | 161.616 | null | null | null | null |
| decode | 360p | 2 | 1 | 409991 | 0.142357986 | 0.618 | 161.811 | 36.192285 | null | null | null |
| encode | 360p | average | 1 | 409991 | 0.142357986 | 0.624000 | 160.286500 | null | null | null | null |
| decode | 360p | average | 1 | 409991 | 0.142357986 | 0.617500 | 161.998000 | 36.192285 | null | null | null |
| encode | 540p | 1 | 1 | 700010 | 0.108026235 | 1.289 | 77.586 | null | null | null | null |
| decode | 540p | 1 | 1 | 700010 | 0.108026235 | 1.285 | 77.819 | 37.301491 | null | null | null |
| encode | 540p | 2 | 1 | 700010 | 0.108026235 | 1.290 | 77.518 | null | null | null | null |
| decode | 540p | 2 | 1 | 700010 | 0.108026235 | 1.285 | 77.824 | 37.301491 | null | null | null |
| encode | 540p | average | 1 | 700010 | 0.108026235 | 1.289500 | 77.552000 | null | null | null | null |
| decode | 540p | average | 1 | 700010 | 0.108026235 | 1.285000 | 77.821500 | 37.301491 | null | null | null |
| encode | 720p | 1 | 1 | 1138058 | 0.098789757 | 2.605 | 38.389 | null | null | null | null |
| decode | 720p | 1 | 1 | 1138058 | 0.098789757 | 2.355 | 42.459 | 38.648291 | null | null | null |
| encode | 720p | 2 | 1 | 1138058 | 0.098789757 | 2.605 | 38.392 | null | null | null | null |
| decode | 720p | 2 | 1 | 1138058 | 0.098789757 | 2.361 | 42.354 | 38.648291 | null | null | null |
| encode | 720p | average | 1 | 1138058 | 0.098789757 | 2.605000 | 38.390500 | null | null | null | null |
| decode | 720p | average | 1 | 1138058 | 0.098789757 | 2.358000 | 42.406500 | 38.648291 | null | null | null |
| encode | 1080p | 1 | 1 | 1702465 | 0.065681520 | 6.024 | 16.600 | null | null | null | null |
| decode | 1080p | 1 | 1 | 1702465 | 0.065681520 | 5.476 | 18.261 | 37.709501 | null | null | null |
| encode | 1080p | 2 | 1 | 1702465 | 0.065681520 | 6.031 | 16.580 | null | null | null | null |
| decode | 1080p | 2 | 1 | 1702465 | 0.065681520 | 5.495 | 18.198 | 37.709501 | null | null | null |
| encode | 1080p | average | 1 | 1702465 | 0.065681520 | 6.027500 | 16.590000 | null | null | null | null |
| decode | 1080p | average | 1 | 1702465 | 0.065681520 | 5.485500 | 18.229500 | 37.709501 | null | null | null |
| encode | qcif | 1 | 30 | 13283 | 0.041928662 | 0.110 | 907.495 | null | null | null | null |
| decode | qcif | 1 | 30 | 13283 | 0.041928662 | 0.097 | 1031.164 | 38.156101 | null | null | null |
| encode | qcif | 2 | 30 | 13283 | 0.041928662 | 0.113 | 888.705 | null | null | null | null |
| decode | qcif | 2 | 30 | 13283 | 0.041928662 | 0.097 | 1033.397 | 38.156101 | null | null | null |
| encode | qcif | average | 30 | 13283 | 0.041928662 | 0.111500 | 898.100000 | null | null | null | null |
| decode | qcif | average | 30 | 13283 | 0.041928662 | 0.097000 | 1032.280500 | 38.156101 | null | null | null |
| encode | cif | 1 | 30 | 59909 | 0.047276673 | 0.189 | 529.809 | null | null | null | null |
| decode | cif | 1 | 30 | 59909 | 0.047276673 | 0.172 | 581.854 | 34.444886 | null | null | null |
| encode | cif | 2 | 30 | 59909 | 0.047276673 | 0.191 | 522.827 | null | null | null | null |
| decode | cif | 2 | 30 | 59909 | 0.047276673 | 0.172 | 582.602 | 34.444886 | null | null | null |
| encode | cif | average | 30 | 59909 | 0.047276673 | 0.190000 | 526.318000 | null | null | null | null |
| decode | cif | average | 30 | 59909 | 0.047276673 | 0.172000 | 582.228000 | 34.444886 | null | null | null |
| encode | 360p | 1 | 30 | 94287 | 0.032738542 | 0.270 | 370.496 | null | null | null | null |
| decode | 360p | 1 | 30 | 94287 | 0.032738542 | 0.262 | 381.263 | 34.940017 | null | null | null |
| encode | 360p | 2 | 30 | 94287 | 0.032738542 | 0.245 | 408.867 | null | null | null | null |
| decode | 360p | 2 | 30 | 94287 | 0.032738542 | 0.262 | 381.810 | 34.940017 | null | null | null |
| encode | 360p | average | 30 | 94287 | 0.032738542 | 0.257500 | 389.681500 | null | null | null | null |
| decode | 360p | average | 30 | 94287 | 0.032738542 | 0.262000 | 381.536500 | 34.940017 | null | null | null |
| encode | 540p | 1 | 30 | 159188 | 0.024566049 | 0.473 | 211.500 | null | null | null | null |
| decode | 540p | 1 | 30 | 159188 | 0.024566049 | 0.501 | 199.467 | 35.974793 | null | null | null |
| encode | 540p | 2 | 30 | 159188 | 0.024566049 | 0.462 | 216.238 | null | null | null | null |
| decode | 540p | 2 | 30 | 159188 | 0.024566049 | 0.498 | 200.980 | 35.974793 | null | null | null |
| encode | 540p | average | 30 | 159188 | 0.024566049 | 0.467500 | 213.869000 | null | null | null | null |
| decode | 540p | average | 30 | 159188 | 0.024566049 | 0.499500 | 200.223500 | 35.974793 | null | null | null |
| encode | 720p | 1 | 30 | 138928 | 0.012059722 | 1.015 | 98.556 | null | null | null | null |
| decode | 720p | 1 | 30 | 138928 | 0.012059722 | 0.979 | 102.151 | 40.455605 | null | null | null |
| encode | 720p | 2 | 30 | 138928 | 0.012059722 | 1.015 | 98.522 | null | null | null | null |
| decode | 720p | 2 | 30 | 138928 | 0.012059722 | 0.978 | 102.238 | 40.455605 | null | null | null |
| encode | 720p | average | 30 | 138928 | 0.012059722 | 1.015000 | 98.539000 | null | null | null | null |
| decode | 720p | average | 30 | 138928 | 0.012059722 | 0.978500 | 102.194500 | 40.455605 | null | null | null |
| encode | 1080p | 1 | 30 | 406322 | 0.015676003 | 2.110 | 47.400 | null | null | null | null |
| decode | 1080p | 1 | 30 | 406322 | 0.015676003 | 2.047 | 48.863 | 36.531272 | null | null | null |
| encode | 1080p | 2 | 30 | 406322 | 0.015676003 | 2.110 | 47.398 | null | null | null | null |
| decode | 1080p | 2 | 30 | 406322 | 0.015676003 | 2.038 | 49.060 | 36.531272 | null | null | null |
| encode | 1080p | average | 30 | 406322 | 0.015676003 | 2.110000 | 47.399000 | null | null | null | null |
| decode | 1080p | average | 30 | 406322 | 0.015676003 | 2.042500 | 48.961500 | 36.531272 | null | null | null |
| encode | qcif | 1 | 100 | 10097 | 0.031871843 | 0.109 | 916.069 | null | null | null | null |
| decode | qcif | 1 | 100 | 10097 | 0.031871843 | 0.101 | 988.171 | 38.018328 | null | null | null |
| encode | qcif | 2 | 100 | 10097 | 0.031871843 | 0.110 | 913.233 | null | null | null | null |
| decode | qcif | 2 | 100 | 10097 | 0.031871843 | 0.094 | 1060.817 | 38.018328 | null | null | null |
| encode | qcif | average | 100 | 10097 | 0.031871843 | 0.109500 | 914.651000 | null | null | null | null |
| decode | qcif | average | 100 | 10097 | 0.031871843 | 0.097500 | 1024.494000 | 38.018328 | null | null | null |
| encode | cif | 1 | 100 | 33030 | 0.026065341 | 0.184 | 542.799 | null | null | null | null |
| decode | cif | 1 | 100 | 33030 | 0.026065341 | 0.165 | 606.296 | 34.739865 | null | null | null |
| encode | cif | 2 | 100 | 33030 | 0.026065341 | 0.179 | 557.343 | null | null | null | null |
| decode | cif | 2 | 100 | 33030 | 0.026065341 | 0.166 | 602.745 | 34.739865 | null | null | null |
| encode | cif | average | 100 | 33030 | 0.026065341 | 0.181500 | 550.071000 | null | null | null | null |
| decode | cif | average | 100 | 33030 | 0.026065341 | 0.165500 | 604.520500 | 34.739865 | null | null | null |
| encode | 360p | 1 | 100 | 77884 | 0.027043056 | 0.247 | 404.708 | null | null | null | null |
| decode | 360p | 1 | 100 | 77884 | 0.027043056 | 0.250 | 400.122 | 34.690251 | null | null | null |
| encode | 360p | 2 | 100 | 77884 | 0.027043056 | 0.234 | 428.148 | null | null | null | null |
| decode | 360p | 2 | 100 | 77884 | 0.027043056 | 0.250 | 400.365 | 34.690251 | null | null | null |
| encode | 360p | average | 100 | 77884 | 0.027043056 | 0.240500 | 416.428000 | null | null | null | null |
| decode | 360p | average | 100 | 77884 | 0.027043056 | 0.250000 | 400.243500 | 34.690251 | null | null | null |
| encode | 540p | 1 | 100 | 131972 | 0.020366049 | 0.445 | 224.671 | null | null | null | null |
| decode | 540p | 1 | 100 | 131972 | 0.020366049 | 0.473 | 211.478 | 35.700143 | null | null | null |
| encode | 540p | 2 | 100 | 131972 | 0.020366049 | 0.434 | 230.349 | null | null | null | null |
| decode | 540p | 2 | 100 | 131972 | 0.020366049 | 0.473 | 211.528 | 35.700143 | null | null | null |
| encode | 540p | average | 100 | 131972 | 0.020366049 | 0.439500 | 227.510000 | null | null | null | null |
| decode | 540p | average | 100 | 131972 | 0.020366049 | 0.473000 | 211.503000 | 35.700143 | null | null | null |
| encode | 720p | 1 | 100 | 76804 | 0.006667014 | 0.975 | 102.552 | null | null | null | null |
| decode | 720p | 1 | 100 | 76804 | 0.006667014 | 0.928 | 107.721 | 40.371141 | null | null | null |
| encode | 720p | 2 | 100 | 76804 | 0.006667014 | 0.975 | 102.559 | null | null | null | null |
| decode | 720p | 2 | 100 | 76804 | 0.006667014 | 0.933 | 107.204 | 40.371141 | null | null | null |
| encode | 720p | average | 100 | 76804 | 0.006667014 | 0.975000 | 102.555500 | null | null | null | null |
| decode | 720p | average | 100 | 76804 | 0.006667014 | 0.930500 | 107.462500 | 40.371141 | null | null | null |
| encode | 1080p | 1 | 100 | 341152 | 0.013161728 | 1.979 | 50.543 | null | null | null | null |
| decode | 1080p | 1 | 100 | 341152 | 0.013161728 | 1.925 | 51.942 | 36.325376 | null | null | null |
| encode | 1080p | 2 | 100 | 341152 | 0.013161728 | 1.985 | 50.373 | null | null | null | null |
| decode | 1080p | 2 | 100 | 341152 | 0.013161728 | 1.925 | 51.951 | 36.325376 | null | null | null |
| encode | 1080p | average | 100 | 341152 | 0.013161728 | 1.982000 | 50.458000 | null | null | null | null |
| decode | 1080p | average | 100 | 341152 | 0.013161728 | 1.925000 | 51.946500 | 36.325376 | null | null | null |
