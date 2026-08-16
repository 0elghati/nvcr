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
average rows. Codec throughput is the runtime-reported encode/decode FPS.
Process throughput uses the full command wall time, including process startup
and container overhead, and is the comparison metric for native versus Docker
runs. Payload BPP is codec payload bits divided by width * height * frames.
When memory profiling is enabled, peak memory is the profiled nvcr process
RAM peak from /proc; Jetson system RAM and largest-free-block values come from
tegrastats when available.

## Results

| Operation | Resolution | Run | GOP | Payload bytes | Payload BPP | Codec seconds | Codec FPS | Process seconds | Process FPS | PSNR-YUV | Peak RAM MB | System RAM MB | Min LFB MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| encode | qcif | 1 | 1 | 65281 | 0.206063763 | 0.982 | 101.845 | 3.327074 | 30.056440 | null | null | null | null |
| decode | qcif | 1 | 1 | 65281 | 0.206063763 | 0.336 | 297.609 | 2.486357 | 40.219486 | 35.794646 | null | null | null |
| encode | qcif | 2 | 1 | 65281 | 0.206063763 | 0.770 | 129.821 | 2.900463 | 34.477254 | null | null | null | null |
| decode | qcif | 2 | 1 | 65281 | 0.206063763 | 0.639 | 156.529 | 3.048367 | 32.804449 | 35.794646 | null | null | null |
| encode | qcif | average | 1 | 65281 | 0.206063763 | 0.876000 | 115.833000 | 3.113768 | 32.115431 | null | null | null | null |
| decode | qcif | average | 1 | 65281 | 0.206063763 | 0.487500 | 227.069000 | 2.767362 | 36.135497 | 35.794646 | null | null | null |
| encode | cif | 1 | 1 | 368308 | 0.290647096 | 0.730 | 136.926 | 2.820885 | 35.449868 | null | null | null | null |
| decode | cif | 1 | 1 | 368308 | 0.290647096 | 0.583 | 171.481 | 2.830830 | 35.325329 | 31.961102 | null | null | null |
| encode | cif | 2 | 1 | 368308 | 0.290647096 | 0.609 | 164.090 | 2.819979 | 35.461257 | null | null | null | null |
| decode | cif | 2 | 1 | 368308 | 0.290647096 | 0.576 | 173.493 | 2.845588 | 35.142122 | 31.961102 | null | null | null |
| encode | cif | average | 1 | 368308 | 0.290647096 | 0.669500 | 150.508000 | 2.820432 | 35.455561 | null | null | null | null |
| decode | cif | average | 1 | 368308 | 0.290647096 | 0.579500 | 172.487000 | 2.838209 | 35.233487 | 31.961102 | null | null | null |
| encode | 360p | 1 | 1 | 409795 | 0.142289931 | 1.109 | 90.155 | 3.329426 | 30.035207 | null | null | null | null |
| decode | 360p | 1 | 1 | 409795 | 0.142289931 | 0.937 | 106.772 | 3.427600 | 29.174933 | 36.195557 | null | null | null |
| encode | 360p | 2 | 1 | 409795 | 0.142289931 | 1.029 | 97.152 | 3.354166 | 29.813671 | null | null | null | null |
| decode | 360p | 2 | 1 | 409795 | 0.142289931 | 1.076 | 92.953 | 3.484459 | 28.698860 | 36.195557 | null | null | null |
| encode | 360p | average | 1 | 409795 | 0.142289931 | 1.069000 | 93.653500 | 3.341796 | 29.924029 | null | null | null | null |
| decode | 360p | average | 1 | 409795 | 0.142289931 | 1.006500 | 99.862500 | 3.456030 | 28.934934 | 36.195557 | null | null | null |
| encode | 540p | 1 | 1 | 699731 | 0.107983179 | 2.300 | 43.483 | 5.060351 | 19.761475 | null | null | null | null |
| decode | 540p | 1 | 1 | 699731 | 0.107983179 | 2.209 | 45.275 | 5.012947 | 19.948346 | 37.305031 | null | null | null |
| encode | 540p | 2 | 1 | 699731 | 0.107983179 | 2.053 | 48.704 | 4.594316 | 21.766026 | null | null | null | null |
| decode | 540p | 2 | 1 | 699731 | 0.107983179 | 1.931 | 51.780 | 4.606617 | 21.707904 | 37.305031 | null | null | null |
| encode | 540p | average | 1 | 699731 | 0.107983179 | 2.176500 | 46.093500 | 4.827333 | 20.715372 | null | null | null | null |
| decode | 540p | average | 1 | 699731 | 0.107983179 | 2.070000 | 48.527500 | 4.809782 | 20.790963 | 37.305031 | null | null | null |
| encode | 720p | 1 | 1 | 1137623 | 0.098751997 | 4.745 | 21.074 | 7.536965 | 13.267940 | null | null | null | null |
| decode | 720p | 1 | 1 | 1137623 | 0.098751997 | 3.383 | 29.563 | 6.338891 | 15.775630 | 38.654498 | null | null | null |
| encode | 720p | 2 | 1 | 1137623 | 0.098751997 | 5.278 | 18.946 | 8.043801 | 12.431934 | null | null | null | null |
| decode | 720p | 2 | 1 | 1137623 | 0.098751997 | 4.809 | 20.795 | 7.862733 | 12.718224 | 38.654498 | null | null | null |
| encode | 720p | average | 1 | 1137623 | 0.098751997 | 5.011500 | 20.010000 | 7.790383 | 12.836339 | null | null | null | null |
| decode | 720p | average | 1 | 1137623 | 0.098751997 | 4.096000 | 25.179000 | 7.100812 | 14.082896 | 38.654498 | null | null | null |
| encode | 1080p | 1 | 1 | 1702484 | 0.065682253 | 14.339 | 6.974 | 17.487233 | 5.718458 | null | null | null | null |
| decode | 1080p | 1 | 1 | 1702484 | 0.065682253 | 11.036 | 9.061 | 15.078489 | 6.631964 | 37.712054 | null | null | null |
| encode | 1080p | 2 | 1 | 1702484 | 0.065682253 | 10.449 | 9.570 | 13.618526 | 7.342939 | null | null | null | null |
| decode | 1080p | 2 | 1 | 1702484 | 0.065682253 | 7.551 | 13.243 | 11.574676 | 8.639551 | 37.712054 | null | null | null |
| encode | 1080p | average | 1 | 1702484 | 0.065682253 | 12.394000 | 8.272000 | 15.552879 | 6.429678 | null | null | null | null |
| decode | 1080p | average | 1 | 1702484 | 0.065682253 | 9.293500 | 11.152000 | 13.326583 | 7.503799 | 37.712054 | null | null | null |
| encode | qcif | 1 | 30 | 13243 | 0.041802399 | 0.292 | 341.992 | 2.536691 | 39.421435 | null | null | null | null |
| decode | qcif | 1 | 30 | 13243 | 0.041802399 | 0.244 | 409.410 | 2.325236 | 43.006387 | 38.041375 | null | null | null |
| encode | qcif | 2 | 30 | 13243 | 0.041802399 | 0.253 | 395.164 | 2.401841 | 41.634729 | null | null | null | null |
| decode | qcif | 2 | 30 | 13243 | 0.041802399 | 0.230 | 435.670 | 2.245686 | 44.529823 | 38.041375 | null | null | null |
| encode | qcif | average | 30 | 13243 | 0.041802399 | 0.272500 | 368.578000 | 2.469266 | 40.497865 | null | null | null | null |
| decode | qcif | average | 30 | 13243 | 0.041802399 | 0.237000 | 422.540000 | 2.285461 | 43.754849 | 38.041375 | null | null | null |
| encode | cif | 1 | 30 | 60129 | 0.047450284 | 0.339 | 295.137 | 2.548577 | 39.237582 | null | null | null | null |
| decode | cif | 1 | 30 | 60129 | 0.047450284 | 0.309 | 323.671 | 2.562231 | 39.028487 | 34.456341 | null | null | null |
| encode | cif | 2 | 30 | 60129 | 0.047450284 | 0.436 | 229.252 | 2.628330 | 38.046973 | null | null | null | null |
| decode | cif | 2 | 30 | 60129 | 0.047450284 | 0.408 | 244.821 | 2.619036 | 38.181988 | 34.456341 | null | null | null |
| encode | cif | average | 30 | 60129 | 0.047450284 | 0.387500 | 262.194500 | 2.588453 | 38.633114 | null | null | null | null |
| decode | cif | average | 30 | 60129 | 0.047450284 | 0.358500 | 284.246000 | 2.590634 | 38.600590 | 34.456341 | null | null | null |
| encode | 360p | 1 | 30 | 94312 | 0.032747222 | 0.399 | 250.907 | 2.664024 | 37.537199 | null | null | null | null |
| decode | 360p | 1 | 30 | 94312 | 0.032747222 | 0.458 | 218.262 | 2.769813 | 36.103520 | 34.952783 | null | null | null |
| encode | 360p | 2 | 30 | 94312 | 0.032747222 | 0.408 | 245.242 | 2.758476 | 36.251901 | null | null | null | null |
| decode | 360p | 2 | 30 | 94312 | 0.032747222 | 0.466 | 214.366 | 2.949153 | 33.908041 | 34.952783 | null | null | null |
| encode | 360p | average | 30 | 94312 | 0.032747222 | 0.403500 | 248.074500 | 2.711250 | 36.883356 | null | null | null | null |
| decode | 360p | average | 30 | 94312 | 0.032747222 | 0.462000 | 216.314000 | 2.859483 | 34.971357 | 34.952783 | null | null | null |
| encode | 540p | 1 | 30 | 159379 | 0.024595525 | 1.187 | 84.225 | 3.893708 | 25.682460 | null | null | null | null |
| decode | 540p | 1 | 30 | 159379 | 0.024595525 | 1.394 | 71.729 | 4.171340 | 23.973112 | 35.978429 | null | null | null |
| encode | 540p | 2 | 30 | 159379 | 0.024595525 | 1.502 | 66.571 | 3.966149 | 25.213374 | null | null | null | null |
| decode | 540p | 2 | 30 | 159379 | 0.024595525 | 1.060 | 94.377 | 3.812023 | 26.232790 | 35.978429 | null | null | null |
| encode | 540p | average | 30 | 159379 | 0.024595525 | 1.344500 | 75.398000 | 3.929928 | 25.445759 | null | null | null | null |
| decode | 540p | average | 30 | 159379 | 0.024595525 | 1.227000 | 83.053000 | 3.991681 | 25.052102 | 35.978429 | null | null | null |
| encode | 720p | 1 | 30 | 138551 | 0.012026997 | 1.200 | 83.320 | 3.667071 | 27.269720 | null | null | null | null |
| decode | 720p | 1 | 30 | 138551 | 0.012026997 | 2.874 | 34.791 | 6.145780 | 16.271328 | 40.441241 | null | null | null |
| encode | 720p | 2 | 30 | 138551 | 0.012026997 | 2.380 | 42.016 | 4.970625 | 20.118194 | null | null | null | null |
| decode | 720p | 2 | 30 | 138551 | 0.012026997 | 1.748 | 57.204 | 4.667636 | 21.424121 | 40.441241 | null | null | null |
| encode | 720p | average | 30 | 138551 | 0.012026997 | 1.790000 | 62.668000 | 4.318848 | 23.154323 | null | null | null | null |
| decode | 720p | average | 30 | 138551 | 0.012026997 | 2.311000 | 45.997500 | 5.406708 | 18.495543 | 40.441241 | null | null | null |
| encode | 1080p | 1 | 30 | 406179 | 0.015670486 | 5.681 | 17.602 | 8.855109 | 11.292916 | null | null | null | null |
| decode | 1080p | 1 | 30 | 406179 | 0.015670486 | 3.289 | 30.405 | 7.218067 | 13.854125 | 36.532072 | null | null | null |
| encode | 1080p | 2 | 30 | 406179 | 0.015670486 | 2.675 | 37.385 | 5.747397 | 17.399181 | null | null | null | null |
| decode | 1080p | 2 | 30 | 406179 | 0.015670486 | 3.250 | 30.773 | 7.047313 | 14.189805 | 36.532072 | null | null | null |
| encode | 1080p | average | 30 | 406179 | 0.015670486 | 4.178000 | 27.493500 | 7.301253 | 13.696279 | null | null | null | null |
| decode | 1080p | average | 30 | 406179 | 0.015670486 | 3.269500 | 30.589000 | 7.132690 | 14.019956 | 36.532072 | null | null | null |
| encode | qcif | 1 | 100 | 10015 | 0.031613005 | 0.260 | 384.223 | 2.183208 | 45.804156 | null | null | null | null |
| decode | qcif | 1 | 100 | 10015 | 0.031613005 | 0.236 | 423.284 | 2.280030 | 43.859072 | 37.955549 | null | null | null |
| encode | qcif | 2 | 100 | 10015 | 0.031613005 | 0.257 | 388.444 | 2.205892 | 45.333135 | null | null | null | null |
| decode | qcif | 2 | 100 | 10015 | 0.031613005 | 0.240 | 416.251 | 2.313496 | 43.224626 | 37.955549 | null | null | null |
| encode | qcif | average | 100 | 10015 | 0.031613005 | 0.258500 | 386.333500 | 2.194550 | 45.567428 | null | null | null | null |
| decode | qcif | average | 100 | 10015 | 0.031613005 | 0.238000 | 419.767500 | 2.296763 | 43.539538 | 37.955549 | null | null | null |
| encode | cif | 1 | 100 | 32953 | 0.026004577 | 0.267 | 374.215 | 2.318868 | 43.124490 | null | null | null | null |
| decode | cif | 1 | 100 | 32953 | 0.026004577 | 0.304 | 329.015 | 2.309915 | 43.291636 | 34.725773 | null | null | null |
| encode | cif | 2 | 100 | 32953 | 0.026004577 | 0.276 | 361.718 | 2.219313 | 45.058989 | null | null | null | null |
| decode | cif | 2 | 100 | 32953 | 0.026004577 | 0.294 | 339.791 | 2.240679 | 44.629329 | 34.725773 | null | null | null |
| encode | cif | average | 100 | 32953 | 0.026004577 | 0.271500 | 367.966500 | 2.269090 | 44.070530 | null | null | null | null |
| decode | cif | average | 100 | 32953 | 0.026004577 | 0.299000 | 334.403000 | 2.275297 | 43.950306 | 34.725773 | null | null | null |
| encode | 360p | 1 | 100 | 77960 | 0.027069444 | 0.361 | 276.687 | 2.437891 | 41.019061 | null | null | null | null |
| decode | 360p | 1 | 100 | 77960 | 0.027069444 | 0.432 | 231.696 | 2.753348 | 36.319419 | 34.692829 | null | null | null |
| encode | 360p | 2 | 100 | 77960 | 0.027069444 | 0.368 | 271.514 | 2.457009 | 40.699892 | null | null | null | null |
| decode | 360p | 2 | 100 | 77960 | 0.027069444 | 0.445 | 224.571 | 2.597230 | 38.502558 | 34.692829 | null | null | null |
| encode | 360p | average | 100 | 77960 | 0.027069444 | 0.364500 | 274.100500 | 2.447450 | 40.858853 | null | null | null | null |
| decode | 360p | average | 100 | 77960 | 0.027069444 | 0.438500 | 228.133500 | 2.675289 | 37.379139 | 34.692829 | null | null | null |
| encode | 540p | 1 | 100 | 132289 | 0.020414969 | 0.612 | 163.495 | 2.804770 | 35.653547 | null | null | null | null |
| decode | 540p | 1 | 100 | 132289 | 0.020414969 | 0.757 | 132.071 | 3.186646 | 31.380957 | 35.697533 | null | null | null |
| encode | 540p | 2 | 100 | 132289 | 0.020414969 | 0.616 | 162.233 | 2.870072 | 34.842331 | null | null | null | null |
| decode | 540p | 2 | 100 | 132289 | 0.020414969 | 0.781 | 128.086 | 3.217125 | 31.083654 | 35.697533 | null | null | null |
| encode | 540p | average | 100 | 132289 | 0.020414969 | 0.614000 | 162.864000 | 2.837421 | 35.243272 | null | null | null | null |
| decode | 540p | average | 100 | 132289 | 0.020414969 | 0.769000 | 130.078500 | 3.201885 | 31.231603 | 35.697533 | null | null | null |
| encode | 720p | 1 | 100 | 76746 | 0.006661979 | 1.058 | 94.538 | 3.368712 | 29.684936 | null | null | null | null |
| decode | 720p | 1 | 100 | 76746 | 0.006661979 | 1.354 | 73.849 | 4.030854 | 24.808639 | 40.359799 | null | null | null |
| encode | 720p | 2 | 100 | 76746 | 0.006661979 | 1.081 | 92.536 | 3.461667 | 28.887816 | null | null | null | null |
| decode | 720p | 2 | 100 | 76746 | 0.006661979 | 1.352 | 73.957 | 4.016996 | 24.894224 | 40.359799 | null | null | null |
| encode | 720p | average | 100 | 76746 | 0.006661979 | 1.069500 | 93.537000 | 3.415189 | 29.280956 | null | null | null | null |
| decode | 720p | average | 100 | 76746 | 0.006661979 | 1.353000 | 73.903000 | 4.023925 | 24.851358 | 40.359799 | null | null | null |
| encode | 1080p | 1 | 100 | 340907 | 0.013152276 | 2.518 | 39.716 | 5.617377 | 17.801903 | null | null | null | null |
| decode | 1080p | 1 | 100 | 340907 | 0.013152276 | 3.122 | 32.033 | 6.971215 | 14.344702 | 36.319967 | null | null | null |
| encode | 1080p | 2 | 100 | 340907 | 0.013152276 | 2.483 | 40.269 | 5.603766 | 17.845142 | null | null | null | null |
| decode | 1080p | 2 | 100 | 340907 | 0.013152276 | 3.096 | 32.299 | 6.907516 | 14.476984 | 36.319967 | null | null | null |
| encode | 1080p | average | 100 | 340907 | 0.013152276 | 2.500500 | 39.992500 | 5.610571 | 17.823498 | null | null | null | null |
| decode | 1080p | average | 100 | 340907 | 0.013152276 | 3.109000 | 32.166000 | 6.939366 | 14.410538 | 36.319967 | null | null | null |
