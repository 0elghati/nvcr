# Jetson Orin Python DCVC-RT repeated measurements — 2026-09-21

The Python reference campaign completed the same matrix as the NVCR campaign. The latest observation for every configured operation passed; two failed/skipped attempts from the interrupted first pass remain retained in the local raw campaign for audit.

## Coverage and identity

- Six resolutions: 176×144, 352×288, 640×360, 960×540, 1280×720 and 1920×1080.
- QPs 0, 21, 42, 63; GOPs 1, 30, 100; 100 measured frames after 10 same-session warm-up frames.
- Throughput: 720 encode and 720 decode processes; memory: 720 encode and 720 decode processes; n=10 per condition and operation.
- Quality: 72 encode and 72 decode processes, one pass per condition; common decoded-output quality contract.
- All 3024 latest operations passed. The raw local campaign retains 3026 records because 2 earlier failed/skipped attempt records were preserved.
- Reference commit: `48ab0ac5e5199d78fffb944bfbafafb2b6142f7b`; expected model-profile commit: `1feb52a592a9ff2c4e4ba2e5122e2da49a211466`; source policy: `explicit-dirty-source-and-fork-override`.

## Variability across independent process executions

CV = 100 × sample SD / mean. Values summarize the 72 conditions separately for each operation.

| Quantity | Operation | Median CV | Maximum CV |
|---|---|---:|---:|
| Completed codec throughput | encode | 1.13% | 3.59% |
| Completed codec throughput | decode | 0.75% | 6.18% |
| Process RSS high-water | encode | 0.18% | 0.29% |
| Process RSS high-water | decode | 0.17% | 0.30% |

The largest throughput CV is waterfall-cif, QP63, GOP1, decode: **24.91 ± 1.54 FPS** (n=10), CV 6.18%, descriptive 95% interval 23.81–26.01 FPS.

## Throughput overview

| Resolution | Encode FPS range | Decode FPS range |
|---|---:|---:|
| 176×144 | 37.18–60.15 | 40.16–62.22 |
| 352×288 | 33.57–48.98 | 24.91–58.99 |
| 640×360 | 17.33–31.46 | 19.06–40.59 |
| 960×540 | 7.44–14.83 | 10.19–21.99 |
| 1280×720 | 4.12–8.19 | 6.21–13.48 |
| 1920×1080 | 1.92–4.14 | 2.91–6.18 |

## Interpretation limits

1. This result uses the updated local Python checkout with tracked energy-measurement edits. The exact commit, expected commit and changed paths are recorded in `manifest.json` and `audit.json`; this is an explicit source-state deviation from the model-profile pin.
2. Process RSS high-water includes initialization and warm-up. It is not isolated GPU allocation or total Jetson memory.
3. The quality points use `decoded-yuv420p8-pooled-plane-6-1-1-v1`; rate is exported as both `file_bpp` and `entropy_bpp` in `rd-points.json`.
4. Two latest successful high-resolution encode records contain `NvMapMemAlloc` allocation warnings in stderr; they completed and are retained as nonempty-stderr observations rather than being hidden.
5. The campaign retains no raw bitstreams or decoded products in Git. Raw observations and logs remain in the local evidence directory, with source hashes recorded in `audit.json`.
6. Direct NVCR/Python speedup and memory-ratio claims should use the matched NVCR compact statistics and this package, while preserving the different implementation/source identities.

## Review artifacts

- `condition-statistics.csv`: 576 condition/operation/metric rows with n=10.
- `rd-points.json`: 72 quality/rate points using the common decoded-output contract.
- `manifest.json`: sanitized campaign controls and Python source identity.
- `audit.json`: compact audit, source hashes and interpretation limits.
