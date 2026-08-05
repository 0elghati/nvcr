# Orin Docker sequence-length GOP benchmark

Commit: `0ac426fe5ecb213d33b7005abb31e3784bb74224` (dirty: `true`)

Runner status: `complete`

Each sequence has one initial I-frame followed by inter frames; GOP equals frame count.

| Resolution | Frames/GOP | Payload bytes | Encode fps | Decode fps | PSNR-YUV |
|---|---:|---:|---:|---:|---:|
| qcif | 300 | 30244 | 255.849333 | 271.963000 | 38.342767 dB |
| cif | 266 | 381952 | 105.074000 | 109.397333 | 26.030571 dB |
| 360p | 500 | 388547 | 51.668333 | 56.078333 | 34.765268 dB |
| 540p | 500 | 664997 | 20.966333 | 23.847000 | 35.722156 dB |
| 720p | 601 | 495231 | 10.494333 | 11.999333 | 40.492927 dB |
| 1080p | 500 | 1689457 | 5.738333 | 6.455333 | 36.385035 dB |
