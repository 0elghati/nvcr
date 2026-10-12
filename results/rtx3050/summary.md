# RTX 3050 Laptop portability result

Current result: 2026-10-12, NVCR 1.0.7 through Docker Desktop/WSL 2 on Windows.

All 1,440 retained source observations cover six resolutions, QPs 0/21/42/63,
GOPs 1/30/100, encode/decode, and ten repetitions of 100 frames after ten
warm-up frames. A separate 24-operation smoke covers three frames at QP 0,
GOPs 1/3 and two warm-up frames. The audit verifies the complete condition
and repetition grids and recomputes every retained average.

The launcher label and local folder names originally said RTX 5060. The user
confirmed the naming mistake; live detection records RTX 3050 Laptop, SM 8.6,
16 multiprocessors, CUDA runtime 12080 and TensorRT 10.9.0. Local folders and
publication metadata now say RTX 3050. Numeric observations are unchanged.

Source revision: `6a66db7a98081f9210dedb49d768899bd4e9f2b7`; source capture
marked dirty. Image: `omarelghati/nvcr:1.0.7-amd64-cuda12.8-trt10.9`.
Digest: `sha256:e8b04d3efc3c4048167aef1bc9ff1c72c3b748f8f53584650f0cad761e2a57e9`.
Recorded Docker Desktop: 4.49.0; engine: 28.5.1; default WSL distribution:
Ubuntu-22.04, WSL version 2. Storage: Windows repository bind mount.

## Scope and limits

This is an environment-specific diagnostic result under [the performance
protocol](../../docs/performance.md), not a matched Python comparison.
Retained values use the legacy codec interval and quality definitions.
FPS and wrapper wall time are omitted from the publication tables.
Memory was not profiled. Input hashes, driver, Windows build, WSL kernel,
and power/clock state were not recorded in this matrix. Those missing fields
prevent a controlled cross-host performance claim. The diagnostic audit
does not independently prove decoded size or warning-free execution for every
matrix operation; separate artifact admission smokes provide that evidence.

One current publication result replaces the earlier two-repetition QP-32
report. Raw repetitions, streams and long logs remain local. JSONL/CSV retain
144 condition summaries, including sample SD and extrema for codec duration.
See [validation metadata](validation.json) for source hashes and audit counts,
and [engine admission/publication evidence](../../evidence/rtx3050-trt109-local-20261012/summary.md)
for six exact bundles, 84 warning-free plan loads and published-bundle checks.

## Condition means

| Resolution | QP | GOP | Repetitions | Encode codec seconds | Decode codec seconds | Payload BPP | Legacy PSNR-YUV |
|---|---:|---:|---:|---:|---:|---:|---:|
| qcif | 0 | 1 | 10 | 0.654800 | 0.574100 | 0.067102273 | 28.611048 |
| qcif | 0 | 30 | 10 | 0.520700 | 0.520300 | 0.024154040 | 30.630047 |
| qcif | 0 | 100 | 10 | 0.554800 | 0.503400 | 0.021395202 | 30.433957 |
| qcif | 21 | 1 | 10 | 0.851800 | 0.718200 | 0.140612374 | 33.407347 |
| qcif | 21 | 30 | 10 | 0.526200 | 0.511500 | 0.033421717 | 35.698299 |
| qcif | 21 | 100 | 10 | 0.524000 | 0.535700 | 0.026511995 | 35.501795 |
| qcif | 42 | 1 | 10 | 0.947900 | 0.752100 | 0.287389520 | 38.137590 |
| qcif | 42 | 30 | 10 | 0.479000 | 0.504000 | 0.051963384 | 39.886153 |
| qcif | 42 | 100 | 10 | 0.501800 | 0.527800 | 0.039034091 | 39.776692 |
| qcif | 63 | 1 | 10 | 0.806700 | 0.652700 | 0.585350379 | 42.440415 |
| qcif | 63 | 30 | 10 | 0.449200 | 0.460900 | 0.091849747 | 42.908545 |
| qcif | 63 | 100 | 10 | 0.522800 | 0.445600 | 0.069144571 | 42.769138 |
| cif | 0 | 1 | 10 | 1.266500 | 1.072200 | 0.040138100 | 26.735250 |
| cif | 0 | 30 | 10 | 0.630300 | 0.675300 | 0.009405777 | 28.038554 |
| cif | 0 | 100 | 10 | 0.622500 | 0.661100 | 0.006801610 | 27.992179 |
| cif | 21 | 1 | 10 | 1.405100 | 1.210000 | 0.135912247 | 29.661527 |
| cif | 21 | 30 | 10 | 0.688300 | 0.735800 | 0.028344381 | 32.273723 |
| cif | 21 | 100 | 10 | 0.773100 | 0.773300 | 0.016152146 | 32.585016 |
| cif | 42 | 1 | 10 | 1.589400 | 1.276200 | 0.529448390 | 34.418435 |
| cif | 42 | 30 | 10 | 0.711100 | 0.697400 | 0.073141572 | 36.348929 |
| cif | 42 | 100 | 10 | 0.683700 | 0.712800 | 0.040660511 | 36.545292 |
| cif | 63 | 1 | 10 | 1.247100 | 1.169600 | 1.337533933 | 39.935721 |
| cif | 63 | 30 | 10 | 0.604400 | 0.670400 | 0.169650410 | 39.585372 |
| cif | 63 | 100 | 10 | 0.557700 | 0.609500 | 0.110456124 | 39.599134 |
| 360p | 0 | 1 | 10 | 2.330600 | 2.065000 | 0.035064236 | 29.917590 |
| 360p | 0 | 30 | 10 | 1.042300 | 1.048100 | 0.006757986 | 28.741219 |
| 360p | 0 | 100 | 10 | 0.996800 | 1.057700 | 0.005417708 | 28.399101 |
| 360p | 21 | 1 | 10 | 2.467300 | 2.151200 | 0.086714236 | 33.963038 |
| 360p | 21 | 30 | 10 | 1.017800 | 1.061000 | 0.018193056 | 32.752984 |
| 360p | 21 | 100 | 10 | 1.331000 | 1.209500 | 0.014783333 | 32.461748 |
| 360p | 42 | 1 | 10 | 3.148700 | 2.471000 | 0.220078472 | 38.262583 |
| 360p | 42 | 30 | 10 | 0.994900 | 1.018300 | 0.054842361 | 36.847769 |
| 360p | 42 | 100 | 10 | 0.932600 | 1.000200 | 0.046429861 | 36.638590 |
| 360p | 63 | 1 | 10 | 2.391400 | 2.120600 | 0.516954514 | 42.439919 |
| 360p | 63 | 30 | 10 | 0.944400 | 1.000100 | 0.152606944 | 40.317682 |
| 360p | 63 | 100 | 10 | 0.880900 | 0.965900 | 0.134894444 | 40.181927 |
| 540p | 0 | 1 | 10 | 4.716000 | 4.084400 | 0.028262500 | 31.163930 |
| 540p | 0 | 30 | 10 | 1.749100 | 1.927800 | 0.004700772 | 29.707668 |
| 540p | 0 | 100 | 10 | 2.645700 | 2.360900 | 0.003673920 | 29.253265 |
| 540p | 21 | 1 | 10 | 4.951500 | 4.238300 | 0.066887346 | 35.122960 |
| 540p | 21 | 30 | 10 | 1.750900 | 1.921800 | 0.013606327 | 33.861093 |
| 540p | 21 | 100 | 10 | 2.623500 | 2.312200 | 0.010947994 | 33.498962 |
| 540p | 42 | 1 | 10 | 6.633500 | 5.077200 | 0.163604784 | 39.140174 |
| 540p | 42 | 30 | 10 | 1.666300 | 1.851700 | 0.041545525 | 37.788771 |
| 540p | 42 | 100 | 10 | 1.648500 | 1.854800 | 0.035231173 | 37.570214 |
| 540p | 63 | 1 | 10 | 4.939600 | 4.239800 | 0.387089506 | 42.799791 |
| 540p | 63 | 30 | 10 | 1.757100 | 1.947400 | 0.119485185 | 41.084712 |
| 540p | 63 | 100 | 10 | 1.667700 | 1.877700 | 0.106022377 | 40.959016 |
| 720p | 0 | 1 | 10 | 8.876800 | 7.366000 | 0.027208507 | 31.872486 |
| 720p | 0 | 30 | 10 | 3.146500 | 3.359400 | 0.003362587 | 33.822028 |
| 720p | 0 | 100 | 10 | 2.853900 | 3.152000 | 0.001769705 | 33.615389 |
| 720p | 21 | 1 | 10 | 9.035600 | 7.490000 | 0.065003125 | 36.389199 |
| 720p | 21 | 30 | 10 | 3.208200 | 3.363400 | 0.007883594 | 38.510909 |
| 720p | 21 | 100 | 10 | 4.334900 | 3.923600 | 0.004145486 | 38.412397 |
| 720p | 42 | 1 | 10 | 11.630600 | 8.952200 | 0.140898438 | 40.492360 |
| 720p | 42 | 30 | 10 | 3.063400 | 3.255400 | 0.017927431 | 41.863639 |
| 720p | 42 | 100 | 10 | 2.961700 | 3.193000 | 0.010515625 | 41.775835 |
| 720p | 63 | 1 | 10 | 9.093400 | 7.565600 | 0.301465972 | 43.730588 |
| 720p | 63 | 30 | 10 | 3.186500 | 3.356100 | 0.045857986 | 43.914015 |
| 720p | 63 | 100 | 10 | 3.013800 | 3.241600 | 0.031645052 | 43.862515 |
| 1080p | 0 | 1 | 10 | 20.471700 | 16.017900 | 0.018915162 | 32.888915 |
| 1080p | 0 | 30 | 10 | 7.036900 | 7.216800 | 0.002916821 | 31.109725 |
| 1080p | 0 | 100 | 10 | 6.538400 | 6.890800 | 0.002230710 | 30.633139 |
| 1080p | 21 | 1 | 10 | 20.211200 | 16.043700 | 0.042442130 | 36.256224 |
| 1080p | 21 | 30 | 10 | 7.074100 | 7.246700 | 0.008688426 | 34.875423 |
| 1080p | 21 | 100 | 10 | 9.470100 | 8.684000 | 0.007049807 | 34.544858 |
| 1080p | 42 | 1 | 10 | 25.640100 | 19.066100 | 0.098500694 | 38.834132 |
| 1080p | 42 | 30 | 10 | 6.834400 | 7.059500 | 0.026772724 | 37.820530 |
| 1080p | 42 | 100 | 10 | 6.532800 | 6.889100 | 0.023017863 | 37.682091 |
| 1080p | 63 | 1 | 10 | 20.226200 | 16.220000 | 0.281832755 | 40.920196 |
| 1080p | 63 | 30 | 10 | 7.089700 | 7.244700 | 0.086436574 | 39.815458 |
| 1080p | 63 | 100 | 10 | 6.679900 | 7.000100 | 0.076259606 | 39.742647 |
