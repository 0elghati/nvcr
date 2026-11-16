# `nvcr.fx.v1` engine bundles for DCVC-RT and DCVC-UF (RTX 4070, 2026-11-16)

Branch `feat/fx-engine-assets` (#256) at `6c480312`, based on `origin/main` `ae22b870` (1.6.0). Roadmap C14: "fx in
published bundles". The programs, plans and conformance of the profile itself are in the
[DCVC-RT](fx-dcvcrt-rtx4070-20261112.md) and [DCVC-UF](fx-dcvcuf-rtx4070-20261112.md) evidence; this file covers the
bundles built for each registered resolution, their packaging, and their staging for the rolling releases.

## Scope

- **Bundles:** one TensorRT bundle per registered engine profile (`qcif`, `cif`, `360p`, `540p`, `720p`, `1080p`) for
  each codec, every one carrying the native plans, the `nvcr.fx.v1` plans and their programs in `normative/`.
- **Release separation:** DCVC-RT fx bundles go to `engine-assets-dcvcrt-fx`, because installers up to 1.6.0 check
  DCVC-RT's exact native file set (`v1.6.0:scripts/nvcr_artifacts.py`); the upload workflow refuses them on
  `engine-assets`. DCVC-UF bundles go to `engine-assets-dcvcuf`.
- **DCVC-UF profiles:** `scripts/backends/dcvcuf/export_tensorrt_stages.py --profile` sizes each context to the profile's
  largest frame, padded to 16 as both DCVC-UF sessions pad; the hyper axes round up (720p latent 45, padded 48, hyper
  12). Before, only 1080p existed and smaller frames ran on 1080p plans.
- **DCVC-UF display:** `add_fx_plans.py` folds the chunk-add after each of the display's eight per-frame branches into
  the following 1x1 convolution (FP16, not in the profile).
- **Not included:** catalog publication (the workflow is not dispatched; it runs from `main`, which lacks this branch's
  release guard and DCVC-UF profiles), QNN, any second GPU.

## Environment

RTX 4070 (SM 8.9, driver 580.178.04), Ubuntu 24.04, CUDA runtime 12.8, TensorRT 10.9.0, target profile
`rtx4070-ubuntu2404` (`277e2a15…9db0`). NVCR Release build `build-release` at `6c480312`. DCVC-UF plans built
`--from-export` from the RTX 4070 export of NVCR `388c45e5`, builder optimization level 5, `tables.bin` `9c73455a…1616`;
fx plans with Triton 3.5.1 (PyTorch 2.6.0). DCVC-RT bundles are those staged 2026-11-14 (built 2026-11-12).

Programs, the same in every bundle of a codec (SHA-256 of the sorted `sha256sum` list of `normative/`): DCVC-UF 27
files `88afd49c…`, DCVC-RT 43 files `eff55b6f…`. The [DCVC-UF evidence](fx-dcvcuf-rtx4070-20261112.md) quotes
`61516f95…` for the same 27 files under a list format not recorded there; it is not reproduced here.

## Gates

Per bundle (`build-bundles/uf6/verify.sh`, kept out of Git):

1. `tests/nvcr_fx_tensorrt_tests <programs> <bundle> <size>...`: every fx stage's TensorRT plan against the CPU
   reference, bit-for-bit. Sizes: the visible size and 64x64, except DCVC-RT `360p` and `540p`, whose registered
   profiles are fixed-size (`configs/engine-profiles/360p.json`, `540p.json`) and are checked at their size only.
2. `scripts/package_engine_bundle.sh`, then `nvcr_artifacts.extract_engine_archive` (validates manifest, digests and
   the fx file set).
3. `nvcr encode --numerics nvcr.fx.v1 --frames 17 --qp 32` and `nvcr decode`, on the source bundle and on the extracted
   one (DCVC-UF `--gop-size 65528`, one I-frame and two chunks; DCVC-RT default GOP).

| Codec | Profile | Clip | fx stages | Archive SHA-256 | Stream / decoded YUV, source = extracted | PSNR-Y |
|---|---|---|---|---|---|---|
| DCVC-UF | qcif | Akiyo 176x144 | 13/13 at 176x144, 64x64 | `94d1e1f2…c16e` | `73158690…ab52` / `cdd05267…0818` | 37.46 dB |
| DCVC-UF | cif | Crew 352x288 | 13/13 at 352x288, 64x64 | `1855a849…8f41` | `e864c68f…77f5` / `513c5015…3b20` | 32.72 dB |
| DCVC-UF | 360p | BasketballDrive 640x360 | 13/13 at 640x360, 64x64 | `542d8f88…88ec` | `12c244eb…1035` / `098a9dad…e8de` | 32.29 dB |
| DCVC-UF | 540p | BasketballDrive 960x540 | 13/13 at 960x540, 64x64 | `49934798…4ded` | `6cff6713…bd71` / `004d1290…3012` | 32.99 dB |
| DCVC-UF | 720p | Park Joy 1280x720 | 13/13 at 1280x720, 64x64 | `e3d9345d…86b3` | `1c1b3792…5f66` / `2ab99fad…2b73` | 27.05 dB |
| DCVC-UF | 1080p | BasketballDrive 1920x1080 | 13/13 at 1920x1080, 64x64 | `60de6cd0…bdb6` | `e114f11f…131d` / `851faca9…4fd5` | 33.52 dB |
| DCVC-RT | qcif | Akiyo 176x144 | 11/11 at 176x144, 64x64 | `d55fca92…078f` | `0cf40d5c…f2ee` / `ebd2de6c…c1ef` | 37.18 dB |
| DCVC-RT | cif | Crew 352x288 | 11/11 at 352x288, 64x64 | `65734ce5…30b5` | `f149e35f…486f` / `b5cbdc73…ae45` | 34.51 dB |
| DCVC-RT | 360p | BasketballDrive 640x360 | 11/11 at 640x360 | `14d1e5d0…e469` | `3ae5f525…efcd` / `5f8085f0…e344` | 34.67 dB |
| DCVC-RT | 540p | BasketballDrive 960x540 | 11/11 at 960x540 | `762422d2…3f94` | `45137487…42e1` / `4735d8fe…2a93` | 35.65 dB |
| DCVC-RT | 720p | Park Joy 1280x720 | 11/11 at 1280x720, 64x64 | `aefc1b06…1ab3` | `ab49a9ca…7ee2` / `0fbd85d4…efb1` | 29.33 dB |
| DCVC-RT | 1080p | BasketballDrive 1920x1080 | 11/11 at 1920x1080, 64x64 | `df2a2c96…bc02` | `4cb7a8bf…7a1b` / `ab29aadc…dd4d` | 35.75 dB |

The first verification run called the fixed-size DCVC-RT 360p and 540p plans at 64x64 as well; TensorRT rejected the
shape and the test process crashed (CUDA illegal memory access). Every stage was already identical at the registered
size; rerun at that size only, both pass. Packaging is deterministic: the six DCVC-RT archives are byte-identical to
those staged on 2026-11-14.

Other gates:

| Gate | Result |
|---|---|
| DCVC-UF display fold: one stream (17 frames, BasketballDrive 1080p, QP 32) decoded with the pre-fold 1080p bundle the 2026-11-12 evidence used and with the new one | I picture byte-identical; 16 P pictures within 1 code value, 62.4-64.6 dB between the decodes (the FP16 display; the earlier TensorRT-vs-CPU decode spread was 63.4-64.7 dB). The two bundles' own encodes differ (rebuilt FP16 analysis plans) |
| DCVC-UF profile shapes (`tests/dcvcuf_tensorrt_stage_tests.py`) | 12/12; the 1080p `stages.json` equals the 2026-11-12 bundle's |
| Release guard and fx bundle validation (`tests/engine_catalog_tests.py`) | 28/28 |
| `nvcr_artifacts.py validate` on all twelve bundles | pass |

## Staged

`scripts/release_engine_assets.sh --skip-dispatch` to `s3://nvcr-release-assets-…-eu-west-1/<release>/`:

- `engine-assets-dcvcrt-fx`: the six DCVC-RT archives (2026-11-14, unchanged).
- `engine-assets-dcvcuf`: the six DCVC-UF archives (2026-11-16); the 1080p object replaces the pre-fold one staged on
  2026-11-14.

The rolling catalogs are not updated until the upload workflow runs after #256 merges; staging alone is not catalog
admission (see `docs/releasing.md`).

## Not verified

- Catalog admission and an `nvcr-artifacts install` from either release.
- A second GPU (RTX 5060) for any of these bundles; cross-provider (ONNX Runtime CPU) decodes at the new profiles.
- DCVC-UF native (non-fx) encode/decode on the new profile-specific plans, including whether the QCIF nondeterminism
  recorded under C13 for the 1080p bundle's spatial priors is gone.
- BD-rate at the new profiles, and the speed effect of the display fold under `docs/performance.md`.
- QNN: `nvcr.fx.v1` remains `unsupported` there.
