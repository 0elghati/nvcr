# NVCR Performance Run

- Commit: b4d0f8b5e2eb8b6d09261d498effc3ef7bc01d0b
- Dirty: true
- Hardware: rtx3050
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
average rows. Payload BPP is codec payload bits divided by width * height * frames.
When memory profiling is enabled, peak memory is the profiled nvcr process
RAM peak from /proc; Jetson system RAM and largest-free-block values come from
tegrastats when available.

## Results

| Operation | Resolution | Run | GOP | Payload bytes | Payload BPP | Codec seconds | FPS | PSNR-YUV | Peak RAM MB | System RAM MB | Min LFB MB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| encode | qcif | 1 | 1 | 65261 | 0.206000631 | 0.847 | 117.999 | null | null | null | null |
| decode | qcif | 1 | 1 | 65261 | 0.206000631 | 0.639 | 156.510 | 35.792242 | null | null | null |
| encode | qcif | 2 | 1 | 65261 | 0.206000631 | 0.700 | 142.884 | null | null | null | null |
| decode | qcif | 2 | 1 | 65261 | 0.206000631 | 0.664 | 150.505 | 35.792242 | null | null | null |
| encode | qcif | average | 1 | 65261 | 0.206000631 | 0.773500 | 130.441500 | null | null | null | null |
| decode | qcif | average | 1 | 65261 | 0.206000631 | 0.651500 | 153.507500 | 35.792242 | null | null | null |
| encode | cif | 1 | 1 | 368364 | 0.290691288 | 1.548 | 64.592 | null | null | null | null |
| decode | cif | 1 | 1 | 368364 | 0.290691288 | 1.336 | 74.852 | 31.960618 | null | null | null |
| encode | cif | 2 | 1 | 368364 | 0.290691288 | 1.290 | 77.546 | null | null | null | null |
| decode | cif | 2 | 1 | 368364 | 0.290691288 | 1.121 | 89.202 | 31.960618 | null | null | null |
| encode | cif | average | 1 | 368364 | 0.290691288 | 1.419000 | 71.069000 | null | null | null | null |
| decode | cif | average | 1 | 368364 | 0.290691288 | 1.228500 | 82.027000 | 31.960618 | null | null | null |
| encode | 360p | 1 | 1 | 409727 | 0.142266319 | 2.432 | 41.118 | null | null | null | null |
| decode | 360p | 1 | 1 | 409727 | 0.142266319 | 2.222 | 44.997 | 36.177398 | null | null | null |
| encode | 360p | 2 | 1 | 409727 | 0.142266319 | 2.408 | 41.533 | null | null | null | null |
| decode | 360p | 2 | 1 | 409727 | 0.142266319 | 2.177 | 45.931 | 36.177398 | null | null | null |
| encode | 360p | average | 1 | 409727 | 0.142266319 | 2.420000 | 41.325500 | null | null | null | null |
| decode | 360p | average | 1 | 409727 | 0.142266319 | 2.199500 | 45.464000 | 36.177398 | null | null | null |
| encode | 540p | 1 | 1 | 700081 | 0.108037191 | 4.757 | 21.022 | null | null | null | null |
| decode | 540p | 1 | 1 | 700081 | 0.108037191 | 4.131 | 24.205 | 37.293340 | null | null | null |
| encode | 540p | 2 | 1 | 700081 | 0.108037191 | 4.757 | 21.024 | null | null | null | null |
| decode | 540p | 2 | 1 | 700081 | 0.108037191 | 4.121 | 24.266 | 37.293340 | null | null | null |
| encode | 540p | average | 1 | 700081 | 0.108037191 | 4.757000 | 21.023000 | null | null | null | null |
| decode | 540p | average | 1 | 700081 | 0.108037191 | 4.126000 | 24.235500 | 37.293340 | null | null | null |
| encode | 720p | 1 | 1 | 1137492 | 0.098740625 | 8.722 | 11.465 | null | null | null | null |
| decode | 720p | 1 | 1 | 1137492 | 0.098740625 | 7.270 | 13.756 | 38.649586 | null | null | null |
| encode | 720p | 2 | 1 | 1137492 | 0.098740625 | 8.772 | 11.399 | null | null | null | null |
| decode | 720p | 2 | 1 | 1137492 | 0.098740625 | 7.333 | 13.638 | 38.649586 | null | null | null |
| encode | 720p | average | 1 | 1137492 | 0.098740625 | 8.747000 | 11.432000 | null | null | null | null |
| decode | 720p | average | 1 | 1137492 | 0.098740625 | 7.301500 | 13.697000 | 38.649586 | null | null | null |
| encode | 1080p | 1 | 1 | 1701636 | 0.065649537 | 19.272 | 5.189 | null | null | null | null |
| decode | 1080p | 1 | 1 | 1701636 | 0.065649537 | 15.981 | 6.258 | 37.708906 | null | null | null |
| encode | 1080p | 2 | 1 | 1701636 | 0.065649537 | 20.121 | 4.970 | null | null | null | null |
| decode | 1080p | 2 | 1 | 1701636 | 0.065649537 | 16.400 | 6.098 | 37.708906 | null | null | null |
| encode | 1080p | average | 1 | 1701636 | 0.065649537 | 19.696500 | 5.079500 | null | null | null | null |
| decode | 1080p | average | 1 | 1701636 | 0.065649537 | 16.190500 | 6.178000 | 37.708906 | null | null | null |
| encode | qcif | 1 | 30 | 13284 | 0.041931818 | 0.535 | 186.953 | null | null | null | null |
| decode | qcif | 1 | 30 | 13284 | 0.041931818 | 0.487 | 205.545 | 38.116869 | null | null | null |
| encode | qcif | 2 | 30 | 13284 | 0.041931818 | 0.602 | 165.998 | null | null | null | null |
| decode | qcif | 2 | 30 | 13284 | 0.041931818 | 0.588 | 169.964 | 38.116869 | null | null | null |
| encode | qcif | average | 30 | 13284 | 0.041931818 | 0.568500 | 176.475500 | null | null | null | null |
| decode | qcif | average | 30 | 13284 | 0.041931818 | 0.537500 | 187.754500 | 38.116869 | null | null | null |
| encode | cif | 1 | 30 | 60270 | 0.047561553 | 0.555 | 180.339 | null | null | null | null |
| decode | cif | 1 | 30 | 60270 | 0.047561553 | 0.659 | 151.675 | 34.458990 | null | null | null |
| encode | cif | 2 | 30 | 60270 | 0.047561553 | 0.570 | 175.301 | null | null | null | null |
| decode | cif | 2 | 30 | 60270 | 0.047561553 | 0.699 | 142.967 | 34.458990 | null | null | null |
| encode | cif | average | 30 | 60270 | 0.047561553 | 0.562500 | 177.820000 | null | null | null | null |
| decode | cif | average | 30 | 60270 | 0.047561553 | 0.679000 | 147.321000 | 34.458990 | null | null | null |
| encode | 360p | 1 | 30 | 94264 | 0.032730556 | 1.077 | 92.826 | null | null | null | null |
| decode | 360p | 1 | 30 | 94264 | 0.032730556 | 1.089 | 91.866 | 34.943035 | null | null | null |
| encode | 360p | 2 | 30 | 94264 | 0.032730556 | 0.927 | 107.873 | null | null | null | null |
| decode | 360p | 2 | 30 | 94264 | 0.032730556 | 1.115 | 89.678 | 34.943035 | null | null | null |
| encode | 360p | average | 30 | 94264 | 0.032730556 | 1.002000 | 100.349500 | null | null | null | null |
| decode | 360p | average | 30 | 94264 | 0.032730556 | 1.102000 | 90.772000 | 34.943035 | null | null | null |
| encode | 540p | 1 | 30 | 159063 | 0.024546759 | 1.640 | 60.959 | null | null | null | null |
| decode | 540p | 1 | 30 | 159063 | 0.024546759 | 2.007 | 49.818 | 35.982801 | null | null | null |
| encode | 540p | 2 | 30 | 159063 | 0.024546759 | 1.660 | 60.246 | null | null | null | null |
| decode | 540p | 2 | 30 | 159063 | 0.024546759 | 1.987 | 50.320 | 35.982801 | null | null | null |
| encode | 540p | average | 30 | 159063 | 0.024546759 | 1.650000 | 60.602500 | null | null | null | null |
| decode | 540p | average | 30 | 159063 | 0.024546759 | 1.997000 | 50.069000 | 35.982801 | null | null | null |
| encode | 720p | 1 | 30 | 138858 | 0.012053646 | 2.817 | 35.502 | null | null | null | null |
| decode | 720p | 1 | 30 | 138858 | 0.012053646 | 3.419 | 29.249 | 40.439735 | null | null | null |
| encode | 720p | 2 | 30 | 138858 | 0.012053646 | 2.860 | 34.963 | null | null | null | null |
| decode | 720p | 2 | 30 | 138858 | 0.012053646 | 3.396 | 29.446 | 40.439735 | null | null | null |
| encode | 720p | average | 30 | 138858 | 0.012053646 | 2.838500 | 35.232500 | null | null | null | null |
| decode | 720p | average | 30 | 138858 | 0.012053646 | 3.407500 | 29.347500 | 40.439735 | null | null | null |
| encode | 1080p | 1 | 30 | 406137 | 0.015668866 | 6.202 | 16.123 | null | null | null | null |
| decode | 1080p | 1 | 30 | 406137 | 0.015668866 | 7.342 | 13.620 | 36.531961 | null | null | null |
| encode | 1080p | 2 | 30 | 406137 | 0.015668866 | 6.573 | 15.214 | null | null | null | null |
| decode | 1080p | 2 | 30 | 406137 | 0.015668866 | 7.345 | 13.615 | 36.531961 | null | null | null |
| encode | 1080p | average | 30 | 406137 | 0.015668866 | 6.387500 | 15.668500 | null | null | null | null |
| decode | 1080p | average | 30 | 406137 | 0.015668866 | 7.343500 | 13.617500 | 36.531961 | null | null | null |
| encode | qcif | 1 | 100 | 10019 | 0.031625631 | 0.614 | 162.887 | null | null | null | null |
| decode | qcif | 1 | 100 | 10019 | 0.031625631 | 0.511 | 195.505 | 37.898262 | null | null | null |
| encode | qcif | 2 | 100 | 10019 | 0.031625631 | 0.596 | 167.664 | null | null | null | null |
| decode | qcif | 2 | 100 | 10019 | 0.031625631 | 0.502 | 199.075 | 37.898262 | null | null | null |
| encode | qcif | average | 100 | 10019 | 0.031625631 | 0.605000 | 165.275500 | null | null | null | null |
| decode | qcif | average | 100 | 10019 | 0.031625631 | 0.506500 | 197.290000 | 37.898262 | null | null | null |
| encode | cif | 1 | 100 | 32947 | 0.025999842 | 0.904 | 110.621 | null | null | null | null |
| decode | cif | 1 | 100 | 32947 | 0.025999842 | 0.761 | 131.336 | 34.737855 | null | null | null |
| encode | cif | 2 | 100 | 32947 | 0.025999842 | 0.836 | 119.658 | null | null | null | null |
| decode | cif | 2 | 100 | 32947 | 0.025999842 | 0.648 | 154.261 | 34.737855 | null | null | null |
| encode | cif | average | 100 | 32947 | 0.025999842 | 0.870000 | 115.139500 | null | null | null | null |
| decode | cif | average | 100 | 32947 | 0.025999842 | 0.704500 | 142.798500 | 34.737855 | null | null | null |
| encode | 360p | 1 | 100 | 78324 | 0.027195833 | 0.873 | 114.574 | null | null | null | null |
| decode | 360p | 1 | 100 | 78324 | 0.027195833 | 1.102 | 90.762 | 34.700202 | null | null | null |
| encode | 360p | 2 | 100 | 78324 | 0.027195833 | 0.895 | 111.672 | null | null | null | null |
| decode | 360p | 2 | 100 | 78324 | 0.027195833 | 0.999 | 100.074 | 34.700202 | null | null | null |
| encode | 360p | average | 100 | 78324 | 0.027195833 | 0.884000 | 113.123000 | null | null | null | null |
| decode | 360p | average | 100 | 78324 | 0.027195833 | 1.050500 | 95.418000 | 34.700202 | null | null | null |
| encode | 540p | 1 | 100 | 131825 | 0.020343364 | 1.557 | 64.210 | null | null | null | null |
| decode | 540p | 1 | 100 | 131825 | 0.020343364 | 1.918 | 52.133 | 35.697964 | null | null | null |
| encode | 540p | 2 | 100 | 131825 | 0.020343364 | 1.612 | 62.043 | null | null | null | null |
| decode | 540p | 2 | 100 | 131825 | 0.020343364 | 1.966 | 50.859 | 35.697964 | null | null | null |
| encode | 540p | average | 100 | 131825 | 0.020343364 | 1.584500 | 63.126500 | null | null | null | null |
| decode | 540p | average | 100 | 131825 | 0.020343364 | 1.942000 | 51.496000 | 35.697964 | null | null | null |
| encode | 720p | 1 | 100 | 76800 | 0.006666667 | 2.693 | 37.132 | null | null | null | null |
| decode | 720p | 1 | 100 | 76800 | 0.006666667 | 3.284 | 30.454 | 40.352804 | null | null | null |
| encode | 720p | 2 | 100 | 76800 | 0.006666667 | 2.700 | 37.041 | null | null | null | null |
| decode | 720p | 2 | 100 | 76800 | 0.006666667 | 3.275 | 30.530 | 40.352804 | null | null | null |
| encode | 720p | average | 100 | 76800 | 0.006666667 | 2.696500 | 37.086500 | null | null | null | null |
| decode | 720p | average | 100 | 76800 | 0.006666667 | 3.279500 | 30.492000 | 40.352804 | null | null | null |
| encode | 1080p | 1 | 100 | 341400 | 0.013171296 | 6.027 | 16.592 | null | null | null | null |
| decode | 1080p | 1 | 100 | 341400 | 0.013171296 | 7.082 | 14.120 | 36.325771 | null | null | null |
| encode | 1080p | 2 | 100 | 341400 | 0.013171296 | 6.144 | 16.275 | null | null | null | null |
| decode | 1080p | 2 | 100 | 341400 | 0.013171296 | 7.126 | 14.034 | 36.325771 | null | null | null |
| encode | 1080p | average | 100 | 341400 | 0.013171296 | 6.085500 | 16.433500 | null | null | null | null |
| decode | 1080p | average | 100 | 341400 | 0.013171296 | 7.104000 | 14.077000 | 36.325771 | null | null | null |
