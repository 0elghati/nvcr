# RTX 3050 placeholder

- Commit: 27f60770890c3c86dd7f0076604b5563404cf006
- Dirty: true
- Hardware: rtx3050
- Execution mode: docker
- Container image: omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9
- Container digest: omarelghati/nvcr@sha256:de9160ceda49fed788655a568d75d79ef7542bfa65a926e765b83e7b10c52339
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
| encode | qcif | 1 | 1 | 65261 | 0.206000631 | 0.870 | 115.002 | 3.464347 | 28.865469 | null | null | null | null |
| decode | qcif | 1 | 1 | 65261 | 0.206000631 | 0.695 | 143.903 | 3.521165 | 28.399692 | 35.792242 | null | null | null |
| encode | qcif | 2 | 1 | 65261 | 0.206000631 | 0.879 | 113.825 | 6.814689 | 14.674184 | null | null | null | null |
| decode | qcif | 2 | 1 | 65261 | 0.206000631 | 0.734 | 136.224 | 3.473619 | 28.788419 | 35.792242 | null | null | null |
| encode | qcif | average | 1 | 65261 | 0.206000631 | 0.874500 | 114.413500 | 5.139518 | 19.457077 | null | null | null | null |
| decode | qcif | average | 1 | 65261 | 0.206000631 | 0.714500 | 140.063500 | 3.497392 | 28.592734 | 35.792242 | null | null | null |
| encode | cif | 1 | 1 | 368364 | 0.290691288 | 1.198 | 83.444 | 4.069925 | 24.570477 | null | null | null | null |
| decode | cif | 1 | 1 | 368364 | 0.290691288 | 0.992 | 100.798 | 3.948298 | 25.327369 | 31.960618 | null | null | null |
| encode | cif | 2 | 1 | 368364 | 0.290691288 | 1.267 | 78.923 | 4.216469 | 23.716527 | null | null | null | null |
| decode | cif | 2 | 1 | 368364 | 0.290691288 | 1.087 | 92.033 | 5.912065 | 16.914564 | 31.960618 | null | null | null |
| encode | cif | average | 1 | 368364 | 0.290691288 | 1.232500 | 81.183500 | 4.143197 | 24.135951 | null | null | null | null |
| decode | cif | average | 1 | 368364 | 0.290691288 | 1.039500 | 96.415500 | 4.930181 | 20.283231 | 31.960618 | null | null | null |
| encode | 360p | 1 | 1 | 409727 | 0.142266319 | 2.091 | 47.834 | 5.336536 | 18.738747 | null | null | null | null |
| decode | 360p | 1 | 1 | 409727 | 0.142266319 | 1.992 | 50.213 | 5.211077 | 19.189891 | 36.177398 | null | null | null |
| encode | 360p | 2 | 1 | 409727 | 0.142266319 | 2.061 | 48.532 | 4.985654 | 20.057549 | null | null | null | null |
| decode | 360p | 2 | 1 | 409727 | 0.142266319 | 1.891 | 52.892 | 5.290613 | 18.901401 | 36.177398 | null | null | null |
| encode | 360p | average | 1 | 409727 | 0.142266319 | 2.076000 | 48.183000 | 5.161095 | 19.375733 | null | null | null | null |
| decode | 360p | average | 1 | 409727 | 0.142266319 | 1.941500 | 51.552500 | 5.250845 | 19.044554 | 36.177398 | null | null | null |
| encode | 540p | 1 | 1 | 700081 | 0.108037191 | 4.128 | 24.222 | 7.403212 | 13.507650 | null | null | null | null |
| decode | 540p | 1 | 1 | 700081 | 0.108037191 | 3.667 | 27.272 | 7.396680 | 13.519579 | 37.293340 | null | null | null |
| encode | 540p | 2 | 1 | 700081 | 0.108037191 | 4.292 | 23.298 | 7.565348 | 13.218163 | null | null | null | null |
| decode | 540p | 2 | 1 | 700081 | 0.108037191 | 3.792 | 26.372 | 9.302209 | 10.750135 | 37.293340 | null | null | null |
| encode | 540p | average | 1 | 700081 | 0.108037191 | 4.210000 | 23.760000 | 7.484280 | 13.361339 | null | null | null | null |
| decode | 540p | average | 1 | 700081 | 0.108037191 | 3.729500 | 26.822000 | 8.349445 | 11.976844 | 37.293340 | null | null | null |
| encode | 720p | 1 | 1 | 1137492 | 0.098740625 | 7.660 | 13.055 | 11.662671 | 8.574365 | null | null | null | null |
| decode | 720p | 1 | 1 | 1137492 | 0.098740625 | 6.264 | 15.964 | 12.454358 | 8.029318 | 38.649586 | null | null | null |
| encode | 720p | 2 | 1 | 1137492 | 0.098740625 | 7.751 | 12.901 | 11.334733 | 8.822440 | null | null | null | null |
| decode | 720p | 2 | 1 | 1137492 | 0.098740625 | 6.493 | 15.400 | 10.776473 | 9.279474 | 38.649586 | null | null | null |
| encode | 720p | average | 1 | 1137492 | 0.098740625 | 7.705500 | 12.978000 | 11.498702 | 8.696634 | null | null | null | null |
| decode | 720p | average | 1 | 1137492 | 0.098740625 | 6.378500 | 15.682000 | 11.615415 | 8.609249 | 38.649586 | null | null | null |
| encode | 1080p | 1 | 1 | 1701636 | 0.065649537 | 17.173 | 5.823 | 24.138025 | 4.142841 | null | null | null | null |
| decode | 1080p | 1 | 1 | 1701636 | 0.065649537 | 13.791 | 7.251 | 19.998237 | 5.000441 | 37.708906 | null | null | null |
| encode | 1080p | 2 | 1 | 1701636 | 0.065649537 | 17.857 | 5.600 | 24.170843 | 4.137216 | null | null | null | null |
| decode | 1080p | 2 | 1 | 1701636 | 0.065649537 | 14.263 | 7.011 | 22.616217 | 4.421606 | 37.708906 | null | null | null |
| encode | 1080p | average | 1 | 1701636 | 0.065649537 | 17.515000 | 5.711500 | 24.154434 | 4.140027 | null | null | null | null |
| decode | 1080p | average | 1 | 1701636 | 0.065649537 | 14.027000 | 7.131000 | 21.307227 | 4.693243 | 37.708906 | null | null | null |
| encode | qcif | 1 | 30 | 13284 | 0.041931818 | 0.660 | 151.547 | 4.147254 | 24.112340 | null | null | null | null |
| decode | qcif | 1 | 30 | 13284 | 0.041931818 | 0.485 | 206.133 | 3.952369 | 25.301281 | 38.116869 | null | null | null |
| encode | qcif | 2 | 30 | 13284 | 0.041931818 | 0.516 | 193.663 | 3.994066 | 25.037143 | null | null | null | null |
| decode | qcif | 2 | 30 | 13284 | 0.041931818 | 0.471 | 212.252 | 4.188045 | 23.877489 | 38.116869 | null | null | null |
| encode | qcif | average | 30 | 13284 | 0.041931818 | 0.588000 | 172.605000 | 4.070660 | 24.566041 | null | null | null | null |
| decode | qcif | average | 30 | 13284 | 0.041931818 | 0.478000 | 209.192500 | 4.070207 | 24.568775 | 38.116869 | null | null | null |
| encode | cif | 1 | 30 | 60270 | 0.047561553 | 0.798 | 125.241 | 3.837176 | 26.060832 | null | null | null | null |
| decode | cif | 1 | 30 | 60270 | 0.047561553 | 0.628 | 159.212 | 3.797784 | 26.331145 | 34.458990 | null | null | null |
| encode | cif | 2 | 30 | 60270 | 0.047561553 | 0.597 | 167.477 | 3.539888 | 28.249481 | null | null | null | null |
| decode | cif | 2 | 30 | 60270 | 0.047561553 | 0.538 | 185.727 | 3.792512 | 26.367748 | 34.458990 | null | null | null |
| encode | cif | average | 30 | 60270 | 0.047561553 | 0.697500 | 146.359000 | 3.688532 | 27.111057 | null | null | null | null |
| decode | cif | average | 30 | 60270 | 0.047561553 | 0.583000 | 172.469500 | 3.795148 | 26.349434 | 34.458990 | null | null | null |
| encode | 360p | 1 | 30 | 94264 | 0.032730556 | 0.882 | 113.400 | 5.883496 | 16.996697 | null | null | null | null |
| decode | 360p | 1 | 30 | 94264 | 0.032730556 | 0.941 | 106.279 | 4.498091 | 22.231653 | 34.943035 | null | null | null |
| encode | 360p | 2 | 30 | 94264 | 0.032730556 | 0.849 | 117.752 | 4.076763 | 24.529265 | null | null | null | null |
| decode | 360p | 2 | 30 | 94264 | 0.032730556 | 0.907 | 110.299 | 4.335299 | 23.066460 | 34.943035 | null | null | null |
| encode | 360p | average | 30 | 94264 | 0.032730556 | 0.865500 | 115.576000 | 4.980130 | 20.079797 | null | null | null | null |
| decode | 360p | average | 30 | 94264 | 0.032730556 | 0.924000 | 108.289000 | 4.416695 | 22.641364 | 34.943035 | null | null | null |
| encode | 540p | 1 | 30 | 159063 | 0.024546759 | 1.555 | 64.313 | 5.150305 | 19.416326 | null | null | null | null |
| decode | 540p | 1 | 30 | 159063 | 0.024546759 | 1.705 | 58.656 | 7.709202 | 12.971511 | 35.982801 | null | null | null |
| encode | 540p | 2 | 30 | 159063 | 0.024546759 | 1.652 | 60.540 | 5.377265 | 18.596815 | null | null | null | null |
| decode | 540p | 2 | 30 | 159063 | 0.024546759 | 1.701 | 58.790 | 5.519322 | 18.118167 | 35.982801 | null | null | null |
| encode | 540p | average | 30 | 159063 | 0.024546759 | 1.603500 | 62.426500 | 5.263785 | 18.997736 | null | null | null | null |
| decode | 540p | average | 30 | 159063 | 0.024546759 | 1.703000 | 58.723000 | 6.614262 | 15.118845 | 35.982801 | null | null | null |
| encode | 720p | 1 | 30 | 138858 | 0.012053646 | 2.719 | 36.780 | 6.339448 | 15.774244 | null | null | null | null |
| decode | 720p | 1 | 30 | 138858 | 0.012053646 | 3.024 | 33.071 | 9.364587 | 10.678528 | 40.439735 | null | null | null |
| encode | 720p | 2 | 30 | 138858 | 0.012053646 | 2.838 | 35.238 | 6.542861 | 15.283834 | null | null | null | null |
| decode | 720p | 2 | 30 | 138858 | 0.012053646 | 3.070 | 32.568 | 7.477901 | 13.372737 | 40.439735 | null | null | null |
| encode | 720p | average | 30 | 138858 | 0.012053646 | 2.778500 | 36.009000 | 6.441154 | 15.525168 | null | null | null | null |
| decode | 720p | average | 30 | 138858 | 0.012053646 | 3.047000 | 32.819500 | 8.421244 | 11.874730 | 40.439735 | null | null | null |
| encode | 1080p | 1 | 30 | 406137 | 0.015668866 | 6.206 | 16.113 | 12.822836 | 7.798587 | null | null | null | null |
| decode | 1080p | 1 | 30 | 406137 | 0.015668866 | 6.516 | 15.347 | 12.795524 | 7.815233 | 36.531961 | null | null | null |
| encode | 1080p | 2 | 30 | 406137 | 0.015668866 | 6.302 | 15.869 | 12.821001 | 7.799703 | null | null | null | null |
| decode | 1080p | 2 | 30 | 406137 | 0.015668866 | 6.641 | 15.058 | 13.058148 | 7.658054 | 36.531961 | null | null | null |
| encode | 1080p | average | 30 | 406137 | 0.015668866 | 6.254000 | 15.991000 | 12.821919 | 7.799145 | null | null | null | null |
| decode | 1080p | average | 30 | 406137 | 0.015668866 | 6.578500 | 15.202500 | 12.926836 | 7.735845 | 36.531961 | null | null | null |
| encode | qcif | 1 | 100 | 10019 | 0.031625631 | 0.347 | 288.446 | 3.476488 | 28.764661 | null | null | null | null |
| decode | qcif | 1 | 100 | 10019 | 0.031625631 | 0.431 | 232.238 | 3.471480 | 28.806158 | 37.898262 | null | null | null |
| encode | qcif | 2 | 100 | 10019 | 0.031625631 | 0.524 | 190.751 | 5.655724 | 17.681202 | null | null | null | null |
| decode | qcif | 2 | 100 | 10019 | 0.031625631 | 0.488 | 204.747 | 3.691828 | 27.086852 | 37.898262 | null | null | null |
| encode | qcif | average | 100 | 10019 | 0.031625631 | 0.435500 | 239.598500 | 4.566106 | 21.900499 | null | null | null | null |
| decode | qcif | average | 100 | 10019 | 0.031625631 | 0.459500 | 218.492500 | 3.581654 | 27.920062 | 37.898262 | null | null | null |
| encode | cif | 1 | 100 | 32947 | 0.025999842 | 0.578 | 173.009 | 3.689714 | 27.102372 | null | null | null | null |
| decode | cif | 1 | 100 | 32947 | 0.025999842 | 0.557 | 179.682 | 3.765995 | 26.553408 | 34.737855 | null | null | null |
| encode | cif | 2 | 100 | 32947 | 0.025999842 | 0.523 | 191.229 | 3.586767 | 27.880261 | null | null | null | null |
| decode | cif | 2 | 100 | 32947 | 0.025999842 | 0.570 | 175.547 | 3.706656 | 26.978495 | 34.737855 | null | null | null |
| encode | cif | average | 100 | 32947 | 0.025999842 | 0.550500 | 182.119000 | 3.638241 | 27.485810 | null | null | null | null |
| decode | cif | average | 100 | 32947 | 0.025999842 | 0.563500 | 177.614500 | 3.736325 | 26.764267 | 34.737855 | null | null | null |
| encode | 360p | 1 | 100 | 78324 | 0.027195833 | 0.802 | 124.717 | 4.020424 | 24.872998 | null | null | null | null |
| decode | 360p | 1 | 100 | 78324 | 0.027195833 | 0.848 | 117.861 | 4.286815 | 23.327342 | 34.700202 | null | null | null |
| encode | 360p | 2 | 100 | 78324 | 0.027195833 | 0.842 | 118.722 | 4.308742 | 23.208630 | null | null | null | null |
| decode | 360p | 2 | 100 | 78324 | 0.027195833 | 0.890 | 112.338 | 4.454817 | 22.447611 | 34.700202 | null | null | null |
| encode | 360p | average | 100 | 78324 | 0.027195833 | 0.822000 | 121.719500 | 4.164583 | 24.012008 | null | null | null | null |
| decode | 360p | average | 100 | 78324 | 0.027195833 | 0.869000 | 115.099500 | 4.370816 | 22.879023 | 34.700202 | null | null | null |
| encode | 540p | 1 | 100 | 131825 | 0.020343364 | 1.478 | 67.660 | 7.448614 | 13.425316 | null | null | null | null |
| decode | 540p | 1 | 100 | 131825 | 0.020343364 | 1.699 | 58.850 | 5.703278 | 17.533776 | 35.697964 | null | null | null |
| encode | 540p | 2 | 100 | 131825 | 0.020343364 | 1.489 | 67.148 | 5.116102 | 19.546131 | null | null | null | null |
| decode | 540p | 2 | 100 | 131825 | 0.020343364 | 1.680 | 59.528 | 6.089064 | 16.422885 | 35.697964 | null | null | null |
| encode | 540p | average | 100 | 131825 | 0.020343364 | 1.483500 | 67.404000 | 6.282358 | 15.917590 | null | null | null | null |
| decode | 540p | average | 100 | 131825 | 0.020343364 | 1.689500 | 59.189000 | 5.896171 | 16.960159 | 35.697964 | null | null | null |
| encode | 720p | 1 | 100 | 76800 | 0.006666667 | 2.634 | 37.971 | 8.357375 | 11.965480 | null | null | null | null |
| decode | 720p | 1 | 100 | 76800 | 0.006666667 | 2.927 | 34.166 | 7.439151 | 13.442394 | 40.352804 | null | null | null |
| encode | 720p | 2 | 100 | 76800 | 0.006666667 | 2.667 | 37.492 | 6.486527 | 15.416570 | null | null | null | null |
| decode | 720p | 2 | 100 | 76800 | 0.006666667 | 2.916 | 34.290 | 7.410725 | 13.493956 | 40.352804 | null | null | null |
| encode | 720p | average | 100 | 76800 | 0.006666667 | 2.650500 | 37.731500 | 7.421951 | 13.473546 | null | null | null | null |
| decode | 720p | average | 100 | 76800 | 0.006666667 | 2.921500 | 34.228000 | 7.424938 | 13.468126 | 40.352804 | null | null | null |
| encode | 1080p | 1 | 100 | 341400 | 0.013171296 | 5.810 | 17.212 | 10.321253 | 9.688746 | null | null | null | null |
| decode | 1080p | 1 | 100 | 341400 | 0.013171296 | 6.192 | 16.149 | 12.693822 | 7.877848 | 36.325771 | null | null | null |
| encode | 1080p | 2 | 100 | 341400 | 0.013171296 | 5.965 | 16.764 | 12.951167 | 7.721312 | null | null | null | null |
| decode | 1080p | 2 | 100 | 341400 | 0.013171296 | 6.248 | 16.005 | 12.565876 | 7.958060 | 36.325771 | null | null | null |
| encode | 1080p | average | 100 | 341400 | 0.013171296 | 5.887500 | 16.988000 | 11.636210 | 8.593863 | null | null | null | null |
| decode | 1080p | average | 100 | 341400 | 0.013171296 | 6.220000 | 16.077000 | 12.629849 | 7.917751 | 36.325771 | null | null | null |
