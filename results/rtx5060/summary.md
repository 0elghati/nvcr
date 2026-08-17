# NVCR Performance Run

- Commit: fba7d311d016cadc367cab3f4980ecb445750920
- Dirty: true
- Hardware: rtx5060
- Execution mode: docker
- Container image: omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9
- Container digest: omarelghati/nvcr@sha256:40f215cd15d424f2f86971523ffff9c5a321081b9ffe68c898116254dcf971de
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
| encode | qcif | 1 | 1 | 65281 | 0.206063763 | 0.982 | 101.845 | null | null | null | null |
| decode | qcif | 1 | 1 | 65281 | 0.206063763 | 0.336 | 297.609 | 35.794646 | null | null | null |
| encode | qcif | 2 | 1 | 65281 | 0.206063763 | 0.770 | 129.821 | null | null | null | null |
| decode | qcif | 2 | 1 | 65281 | 0.206063763 | 0.639 | 156.529 | 35.794646 | null | null | null |
| encode | qcif | average | 1 | 65281 | 0.206063763 | 0.876000 | 115.833000 | null | null | null | null |
| decode | qcif | average | 1 | 65281 | 0.206063763 | 0.487500 | 227.069000 | 35.794646 | null | null | null |
| encode | cif | 1 | 1 | 368308 | 0.290647096 | 0.730 | 136.926 | null | null | null | null |
| decode | cif | 1 | 1 | 368308 | 0.290647096 | 0.583 | 171.481 | 31.961102 | null | null | null |
| encode | cif | 2 | 1 | 368308 | 0.290647096 | 0.609 | 164.090 | null | null | null | null |
| decode | cif | 2 | 1 | 368308 | 0.290647096 | 0.576 | 173.493 | 31.961102 | null | null | null |
| encode | cif | average | 1 | 368308 | 0.290647096 | 0.669500 | 150.508000 | null | null | null | null |
| decode | cif | average | 1 | 368308 | 0.290647096 | 0.579500 | 172.487000 | 31.961102 | null | null | null |
| encode | 360p | 1 | 1 | 409795 | 0.142289931 | 1.109 | 90.155 | null | null | null | null |
| decode | 360p | 1 | 1 | 409795 | 0.142289931 | 0.937 | 106.772 | 36.195557 | null | null | null |
| encode | 360p | 2 | 1 | 409795 | 0.142289931 | 1.029 | 97.152 | null | null | null | null |
| decode | 360p | 2 | 1 | 409795 | 0.142289931 | 1.076 | 92.953 | 36.195557 | null | null | null |
| encode | 360p | average | 1 | 409795 | 0.142289931 | 1.069000 | 93.653500 | null | null | null | null |
| decode | 360p | average | 1 | 409795 | 0.142289931 | 1.006500 | 99.862500 | 36.195557 | null | null | null |
| encode | 540p | 1 | 1 | 699731 | 0.107983179 | 2.300 | 43.483 | null | null | null | null |
| decode | 540p | 1 | 1 | 699731 | 0.107983179 | 2.209 | 45.275 | 37.305031 | null | null | null |
| encode | 540p | 2 | 1 | 699731 | 0.107983179 | 2.053 | 48.704 | null | null | null | null |
| decode | 540p | 2 | 1 | 699731 | 0.107983179 | 1.931 | 51.780 | 37.305031 | null | null | null |
| encode | 540p | average | 1 | 699731 | 0.107983179 | 2.176500 | 46.093500 | null | null | null | null |
| decode | 540p | average | 1 | 699731 | 0.107983179 | 2.070000 | 48.527500 | 37.305031 | null | null | null |
| encode | 720p | 1 | 1 | 1137623 | 0.098751997 | 4.745 | 21.074 | null | null | null | null |
| decode | 720p | 1 | 1 | 1137623 | 0.098751997 | 3.383 | 29.563 | 38.654498 | null | null | null |
| encode | 720p | 2 | 1 | 1137623 | 0.098751997 | 5.278 | 18.946 | null | null | null | null |
| decode | 720p | 2 | 1 | 1137623 | 0.098751997 | 4.809 | 20.795 | 38.654498 | null | null | null |
| encode | 720p | average | 1 | 1137623 | 0.098751997 | 5.011500 | 20.010000 | null | null | null | null |
| decode | 720p | average | 1 | 1137623 | 0.098751997 | 4.096000 | 25.179000 | 38.654498 | null | null | null |
| encode | 1080p | 1 | 1 | 1702484 | 0.065682253 | 14.339 | 6.974 | null | null | null | null |
| decode | 1080p | 1 | 1 | 1702484 | 0.065682253 | 11.036 | 9.061 | 37.712054 | null | null | null |
| encode | 1080p | 2 | 1 | 1702484 | 0.065682253 | 10.449 | 9.570 | null | null | null | null |
| decode | 1080p | 2 | 1 | 1702484 | 0.065682253 | 7.551 | 13.243 | 37.712054 | null | null | null |
| encode | 1080p | average | 1 | 1702484 | 0.065682253 | 12.394000 | 8.272000 | null | null | null | null |
| decode | 1080p | average | 1 | 1702484 | 0.065682253 | 9.293500 | 11.152000 | 37.712054 | null | null | null |
| encode | qcif | 1 | 30 | 13243 | 0.041802399 | 0.292 | 341.992 | null | null | null | null |
| decode | qcif | 1 | 30 | 13243 | 0.041802399 | 0.244 | 409.410 | 38.041375 | null | null | null |
| encode | qcif | 2 | 30 | 13243 | 0.041802399 | 0.253 | 395.164 | null | null | null | null |
| decode | qcif | 2 | 30 | 13243 | 0.041802399 | 0.230 | 435.670 | 38.041375 | null | null | null |
| encode | qcif | average | 30 | 13243 | 0.041802399 | 0.272500 | 368.578000 | null | null | null | null |
| decode | qcif | average | 30 | 13243 | 0.041802399 | 0.237000 | 422.540000 | 38.041375 | null | null | null |
| encode | cif | 1 | 30 | 60129 | 0.047450284 | 0.339 | 295.137 | null | null | null | null |
| decode | cif | 1 | 30 | 60129 | 0.047450284 | 0.309 | 323.671 | 34.456341 | null | null | null |
| encode | cif | 2 | 30 | 60129 | 0.047450284 | 0.436 | 229.252 | null | null | null | null |
| decode | cif | 2 | 30 | 60129 | 0.047450284 | 0.408 | 244.821 | 34.456341 | null | null | null |
| encode | cif | average | 30 | 60129 | 0.047450284 | 0.387500 | 262.194500 | null | null | null | null |
| decode | cif | average | 30 | 60129 | 0.047450284 | 0.358500 | 284.246000 | 34.456341 | null | null | null |
| encode | 360p | 1 | 30 | 94312 | 0.032747222 | 0.399 | 250.907 | null | null | null | null |
| decode | 360p | 1 | 30 | 94312 | 0.032747222 | 0.458 | 218.262 | 34.952783 | null | null | null |
| encode | 360p | 2 | 30 | 94312 | 0.032747222 | 0.408 | 245.242 | null | null | null | null |
| decode | 360p | 2 | 30 | 94312 | 0.032747222 | 0.466 | 214.366 | 34.952783 | null | null | null |
| encode | 360p | average | 30 | 94312 | 0.032747222 | 0.403500 | 248.074500 | null | null | null | null |
| decode | 360p | average | 30 | 94312 | 0.032747222 | 0.462000 | 216.314000 | 34.952783 | null | null | null |
| encode | 540p | 1 | 30 | 159379 | 0.024595525 | 1.187 | 84.225 | null | null | null | null |
| decode | 540p | 1 | 30 | 159379 | 0.024595525 | 1.394 | 71.729 | 35.978429 | null | null | null |
| encode | 540p | 2 | 30 | 159379 | 0.024595525 | 1.502 | 66.571 | null | null | null | null |
| decode | 540p | 2 | 30 | 159379 | 0.024595525 | 1.060 | 94.377 | 35.978429 | null | null | null |
| encode | 540p | average | 30 | 159379 | 0.024595525 | 1.344500 | 75.398000 | null | null | null | null |
| decode | 540p | average | 30 | 159379 | 0.024595525 | 1.227000 | 83.053000 | 35.978429 | null | null | null |
| encode | 720p | 1 | 30 | 138551 | 0.012026997 | 1.200 | 83.320 | null | null | null | null |
| decode | 720p | 1 | 30 | 138551 | 0.012026997 | 2.874 | 34.791 | 40.441241 | null | null | null |
| encode | 720p | 2 | 30 | 138551 | 0.012026997 | 2.380 | 42.016 | null | null | null | null |
| decode | 720p | 2 | 30 | 138551 | 0.012026997 | 1.748 | 57.204 | 40.441241 | null | null | null |
| encode | 720p | average | 30 | 138551 | 0.012026997 | 1.790000 | 62.668000 | null | null | null | null |
| decode | 720p | average | 30 | 138551 | 0.012026997 | 2.311000 | 45.997500 | 40.441241 | null | null | null |
| encode | 1080p | 1 | 30 | 406179 | 0.015670486 | 5.681 | 17.602 | null | null | null | null |
| decode | 1080p | 1 | 30 | 406179 | 0.015670486 | 3.289 | 30.405 | 36.532072 | null | null | null |
| encode | 1080p | 2 | 30 | 406179 | 0.015670486 | 2.675 | 37.385 | null | null | null | null |
| decode | 1080p | 2 | 30 | 406179 | 0.015670486 | 3.250 | 30.773 | 36.532072 | null | null | null |
| encode | 1080p | average | 30 | 406179 | 0.015670486 | 4.178000 | 27.493500 | null | null | null | null |
| decode | 1080p | average | 30 | 406179 | 0.015670486 | 3.269500 | 30.589000 | 36.532072 | null | null | null |
| encode | qcif | 1 | 100 | 10015 | 0.031613005 | 0.260 | 384.223 | null | null | null | null |
| decode | qcif | 1 | 100 | 10015 | 0.031613005 | 0.236 | 423.284 | 37.955549 | null | null | null |
| encode | qcif | 2 | 100 | 10015 | 0.031613005 | 0.257 | 388.444 | null | null | null | null |
| decode | qcif | 2 | 100 | 10015 | 0.031613005 | 0.240 | 416.251 | 37.955549 | null | null | null |
| encode | qcif | average | 100 | 10015 | 0.031613005 | 0.258500 | 386.333500 | null | null | null | null |
| decode | qcif | average | 100 | 10015 | 0.031613005 | 0.238000 | 419.767500 | 37.955549 | null | null | null |
| encode | cif | 1 | 100 | 32953 | 0.026004577 | 0.267 | 374.215 | null | null | null | null |
| decode | cif | 1 | 100 | 32953 | 0.026004577 | 0.304 | 329.015 | 34.725773 | null | null | null |
| encode | cif | 2 | 100 | 32953 | 0.026004577 | 0.276 | 361.718 | null | null | null | null |
| decode | cif | 2 | 100 | 32953 | 0.026004577 | 0.294 | 339.791 | 34.725773 | null | null | null |
| encode | cif | average | 100 | 32953 | 0.026004577 | 0.271500 | 367.966500 | null | null | null | null |
| decode | cif | average | 100 | 32953 | 0.026004577 | 0.299000 | 334.403000 | 34.725773 | null | null | null |
| encode | 360p | 1 | 100 | 77960 | 0.027069444 | 0.361 | 276.687 | null | null | null | null |
| decode | 360p | 1 | 100 | 77960 | 0.027069444 | 0.432 | 231.696 | 34.692829 | null | null | null |
| encode | 360p | 2 | 100 | 77960 | 0.027069444 | 0.368 | 271.514 | null | null | null | null |
| decode | 360p | 2 | 100 | 77960 | 0.027069444 | 0.445 | 224.571 | 34.692829 | null | null | null |
| encode | 360p | average | 100 | 77960 | 0.027069444 | 0.364500 | 274.100500 | null | null | null | null |
| decode | 360p | average | 100 | 77960 | 0.027069444 | 0.438500 | 228.133500 | 34.692829 | null | null | null |
| encode | 540p | 1 | 100 | 132289 | 0.020414969 | 0.612 | 163.495 | null | null | null | null |
| decode | 540p | 1 | 100 | 132289 | 0.020414969 | 0.757 | 132.071 | 35.697533 | null | null | null |
| encode | 540p | 2 | 100 | 132289 | 0.020414969 | 0.616 | 162.233 | null | null | null | null |
| decode | 540p | 2 | 100 | 132289 | 0.020414969 | 0.781 | 128.086 | 35.697533 | null | null | null |
| encode | 540p | average | 100 | 132289 | 0.020414969 | 0.614000 | 162.864000 | null | null | null | null |
| decode | 540p | average | 100 | 132289 | 0.020414969 | 0.769000 | 130.078500 | 35.697533 | null | null | null |
| encode | 720p | 1 | 100 | 76746 | 0.006661979 | 1.058 | 94.538 | null | null | null | null |
| decode | 720p | 1 | 100 | 76746 | 0.006661979 | 1.354 | 73.849 | 40.359799 | null | null | null |
| encode | 720p | 2 | 100 | 76746 | 0.006661979 | 1.081 | 92.536 | null | null | null | null |
| decode | 720p | 2 | 100 | 76746 | 0.006661979 | 1.352 | 73.957 | 40.359799 | null | null | null |
| encode | 720p | average | 100 | 76746 | 0.006661979 | 1.069500 | 93.537000 | null | null | null | null |
| decode | 720p | average | 100 | 76746 | 0.006661979 | 1.353000 | 73.903000 | 40.359799 | null | null | null |
| encode | 1080p | 1 | 100 | 340907 | 0.013152276 | 2.518 | 39.716 | null | null | null | null |
| decode | 1080p | 1 | 100 | 340907 | 0.013152276 | 3.122 | 32.033 | 36.319967 | null | null | null |
| encode | 1080p | 2 | 100 | 340907 | 0.013152276 | 2.483 | 40.269 | null | null | null | null |
| decode | 1080p | 2 | 100 | 340907 | 0.013152276 | 3.096 | 32.299 | 36.319967 | null | null | null |
| encode | 1080p | average | 100 | 340907 | 0.013152276 | 2.500500 | 39.992500 | null | null | null | null |
| decode | 1080p | average | 100 | 340907 | 0.013152276 | 3.109000 | 32.166000 | 36.319967 | null | null | null |
