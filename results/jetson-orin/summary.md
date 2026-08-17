# Jetson Orin: inter-coded comparison and FPS

This completed Jetson Orin report compares NVCR against the Python DCVC-RT
reference and keeps the inter-coded throughput results visible.

## Inputs and coverage

| Runtime | Canonical data | SHA-256 | Coverage |
|---|---|---|---|
| NVCR R=64 | [`nvcr/data/results.jsonl`](nvcr/data/results.jsonl) | `c82f26cb76ad273008943e814fb587853e7ede3c67b83a898e32620699215b9e` | 144 rows: six resolutions, GOP 1/30/100, encode/decode, three measured repetitions plus average rows |
| Python | [`python/data/results.jsonl`](python/data/results.jsonl) | `baf8467c73efd2cbe0bc2a99b7fb959fbb76ad6872bb9d0f04990de92d736fc5` | 37 rows: the same six sequences, GOPs, operations, and 100-frame count; one duplicate 360p GOP-1 encode key |

The report selects Python's latest `source_timestamp` for duplicate keys. Both
runtimes use the same sequence basename, dimensions, QP 32, GOP, and 100-frame
count. Python has complete payload, quality, and decode timing coverage. Its
1080p GOP-1 and GOP-100 encode rows lack throughput values.

The NVCR run used commit `0a21a749e9b945006de7403b6782a0970b78f7b3`, with 10
warm-up frames, three measured repetitions, and memory profiling.
## Entropy payload and reconstruction quality
NVCR entropy BPP is derived from each measured `.nvcr` stream, removing the
sequence container, record length, PacketIO, NVAU, and inner DCVC-RT payload
headers. All three NVCR repetitions record identical rANS byte counts in every
case. Python BPP is its decode row's inner `bit_stream` BPP.

`BPP delta` is `(NVCR rANS BPP / Python bitstream BPP) - 1`. `R=64 vs R=32`
shows the change from the earlier native cadence. PSNR delta is NVCR PSNR-YUV
minus Python average all-frame PSNR.

| Resolution | GOP | NVCR rANS BPP | Python BPP | BPP delta | R=64 vs R=32 | NVCR PSNR | Python PSNR | PSNR delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QCIF | 1 | 0.187436869 | 0.187743000 | -0.2% | 0.0% | 35.786 | 35.919 | -0.133 dB |
| QCIF | 30 | 0.023576389 | 0.023561000 | +0.1% | -6.7% | 38.099 | 38.324 | -0.225 dB |
| QCIF | 100 | 0.013570076 | 0.013516000 | +0.4% | -13.8% | 37.922 | 38.241 | -0.319 dB |
| CIF | 1 | 0.286053504 | 0.285699000 | +0.1% | 0.0% | 31.961 | 31.989 | -0.029 dB |
| CIF | 30 | 0.042934817 | 0.042682000 | +0.6% | -10.0% | 34.478 | 34.589 | -0.111 dB |
| CIF | 100 | 0.021574337 | 0.021428000 | +0.7% | -21.7% | 34.775 | 34.872 | -0.097 dB |
| 360p | 1 | 0.140318056 | 0.140051000 | +0.2% | 0.0% | 36.203 | 36.326 | -0.124 dB |
| 360p | 30 | 0.030691319 | 0.030624000 | +0.2% | -2.3% | 34.962 | 35.138 | -0.176 dB |
| 360p | 100 | 0.025051042 | 0.025091000 | -0.2% | -5.8% | 34.697 | 34.931 | -0.235 dB |
| 540p | 1 | 0.107045370 | 0.106861000 | +0.2% | 0.0% | 37.311 | 37.466 | -0.155 dB |
| 540p | 30 | 0.023697222 | 0.023637000 | +0.3% | -2.3% | 35.989 | 36.174 | -0.185 dB |
| 540p | 100 | 0.019466975 | 0.019452000 | +0.1% | -5.0% | 35.692 | 35.935 | -0.243 dB |
| 720p | 1 | 0.098262066 | 0.097984000 | +0.3% | 0.0% | 38.649 | 38.851 | -0.202 dB |
| 720p | 30 | 0.011553646 | 0.011496000 | +0.5% | -8.4% | 40.452 | 40.760 | -0.308 dB |
| 720p | 100 | 0.006145920 | 0.006081000 | +1.1% | -17.9% | 40.358 | 40.702 | -0.345 dB |
| 1080p | 1 | 0.065417670 | 0.065264000 | +0.2% | 0.0% | 37.709 | 37.878 | -0.170 dB |
| 1080p | 30 | 0.015440278 | 0.015414000 | +0.2% | -2.2% | 36.530 | 36.679 | -0.149 dB |
| 1080p | 100 | 0.012926582 | 0.012919000 | +0.1% | -4.8% | 36.325 | 36.488 | -0.163 dB |

### Aggregate comparison

| GOP | NVCR BPP delta range | Median BPP delta | Median PSNR delta |
|---:|---:|---:|---:|
| 1 | -0.2% to +0.3% | +0.2% | -0.144 dB |
| 30 | +0.1% to +0.6% | +0.2% | -0.181 dB |
| 100 | -0.2% to +1.1% | +0.2% | -0.239 dB |

The reset correction removed the earlier long-GOP rate divergence. Across all
18 cases, NVCR entropy BPP is within -0.2% to +1.1% of Python. The old R=32
run used 2.2% to 10.0% more entropy at GOP 30 and 4.8% to 21.7% more at GOP
100.

The quality table records NVCR and Python values for every resolution and GOP.

## Strongest inter-coded throughput by resolution

The strongest recorded throughput in the inter-coded runs comes from GOP 100 at every resolution. GOP 30 remains visible in the full table below.

| Resolution | Best inter-coded GOP | Encode throughput FPS | Decode throughput FPS |
|---|---:|---:|---:|
| QCIF | 100 | 248.052 | 254.457 |
| CIF | 100 | 103.946 | 106.835 |
| 360p | 100 | 49.607 | 54.628 |
| 540p | 100 | 22.527 | 25.112 |
| 720p | 100 | 12.749 | 14.467 |
| 1080p | 100 | 5.830 | 6.566 |

## Throughput results

The table reports the recorded NVCR codec-loop throughput alongside Python throughput for each case. NVCR values are means of three measured repetitions.

| Resolution | GOP | NVCR encode throughput FPS | Python encode throughput FPS | NVCR decode throughput FPS | Python decode throughput FPS |
|---|---:|---:|---:|---:|---:|
| QCIF | 1 | 123.327 | 8.136 | 131.185 | 14.432 |
| QCIF | 30 | 238.666 | 8.544 | 239.606 | 19.080 |
| QCIF | 100 | 248.052 | 8.106 | 254.457 | 19.291 |
| CIF | 1 | 48.159 | 7.859 | 53.849 | 9.483 |
| CIF | 30 | 99.679 | 8.239 | 104.270 | 14.323 |
| CIF | 100 | 103.946 | 8.269 | 106.835 | 13.533 |
| 360p | 1 | 20.596 | 6.290 | 25.697 | 6.267 |
| 360p | 30 | 47.387 | 7.599 | 52.395 | 9.497 |
| 360p | 100 | 49.607 | 7.466 | 54.628 | 10.080 |
| 540p | 1 | 9.619 | 4.279 | 11.787 | 3.499 |
| 540p | 30 | 21.518 | 5.818 | 24.104 | 4.724 |
| 540p | 100 | 22.527 | 5.920 | 25.112 | 4.909 |
| 720p | 1 | 5.582 | 2.965 | 7.027 | 1.915 |
| 720p | 30 | 12.217 | 4.411 | 13.993 | 2.959 |
| 720p | 100 | 12.749 | 4.456 | 14.467 | 3.226 |
| 1080p | 1 | 2.500 | — | 3.177 | 1.053 |
| 1080p | 30 | 5.599 | 2.715 | 6.353 | 1.645 |
| 1080p | 100 | 5.830 | — | 6.566 | 1.659 |

The three NVCR repetitions are stable: median coefficient of variation is
0.17% for encode and 0.42% for decode. The largest variation is 4.94% for
QCIF GOP-100 decode; all other cases are below that value.

## Memory

| Operation | NVCR process peak RSS | Python process peak RSS | Python peak GPU memory |
|---|---:|---:|---:|
| Encode | 1057–1124 MB; median 1079 MB | 1437–1924 MB; median 1558 MB | 138–434 MB; median 200 MB |
| Decode | 1053–1132 MB; median 1081 MB | 1418–1866 MB; median 1488 MB | 137–434 MB; median 179 MB |

NVCR also records 2161–2444 MB peak Jetson system RAM and a 4 MB minimum
largest free block in every case. NVCR per-process GPU memory is unavailable.

## Assessment

The 64-frame reference cadence produces close entropy-rate agreement across all
18 recorded cases. The throughput, quality, and memory tables above are the
completed Jetson Orin execution record.
