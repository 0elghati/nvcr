# `nvcr.fx.v1` CPU executor: tiled kernels (issue #294, 2026-11-23)

Diagnostic, stage-level timing. This is not a `docs/performance.md` process or
codec-loop measurement, and the laptop ran on battery. Ratios between the two
kernels hold; absolute times are pessimistic.

## What changed

- `gemm_tiled` and `depthwise3x3_tiled` (`src/deterministic/fx.cpp`) compute the
  reference kernels' integers:
  - `gemm` quantises each input value once per tile instead of once per output
    channel;
  - it multiplies `int16` values in pairs (two products always fit `int32`,
    because `2 · 32639² < 2^31`) and sums them exactly in `int64`;
  - work is split over 128-pixel tiles and output-group ranges, so small latent
    planes still use every worker;
  - on any range failure, the reference kernel reruns and reports its own error,
    so both kernels fail on the same inputs with the same message.
- `fx::run` uses the tiled kernels by default. `Kernels::reference` keeps the
  plain loops, and the TensorRT conformance test now names them explicitly.
- `parallel_for` moved unchanged from `program.cpp` to `src/deterministic/parallel.hpp`,
  and now hands out indexes dynamically instead of round-robin. Each index
  writes disjoint outputs, so `nvcr.intconv.v1` results cannot change.

## Environment

- Intel Core Ultra 9 275HX (24 cores), Windows 11 build 26300, WSL 2 kernel
  6.6.87.2, Ubuntu 24.04.4.
- On battery at 17%, Windows Balanced power plan.
- GCC 13.3.0, CMake 3.28.3, Ninja 1.12.1; Release build for baseline x86-64
  (no `-march`), `NVCR_WARNINGS_AS_ERRORS=ON`.
- Programs: the `normative/` directory of
  `nvcr-engines-rtx5060-laptop-ubuntu2404-dcvcrt-cvpr2025-qcif.tar.gz` from
  `engine-assets-dcvcrt-fx` (SHA-256 `18552caf…532d`, verified), 11 DCVC-RT
  stages.
- Inputs: the seeded generator shared with the TensorRT conformance test
  (`tests/fx_inputs.hpp`, seed 7).
- Command:
  `nvcr_fx_cpu_benchmark <normative> --repeat 3 [--reference] 176x144 640x360`
  (median of 3 runs after one warm-up).

## Bit-identity

`nvcr_fx_executor_tests <normative> 176x144 72x40 640x360` passes:

- 180 synthetic gemms: K of 1, 2, 9, 37 and 64; groups of 1, 2 and 4; zero to
  two residuals; planes from 1x1 to 13x21; weights at ±32639; inputs up to
  2^30. Each is also forced to fail through a pre-activation above the limit,
  and each with residuals through a stored value above 2^30. Both kernels
  return the same error.
- Synthetic depthwise kernels at 1x1, 3x5 and 17x9, including a stored-bound
  failure.
- All 11 DCVC-RT stages at all three sizes: outputs bit-identical.

CTest: 32 of 35 pass. The three failures come from this environment and none
of them exercises the executor:

- `nvcr_artifact_profiles` and `nvcr_native_installer`: the Windows working
  copy gives the shell scripts CRLF line endings (`set: pipefail: invalid
  option name`).
- `nvcr_measurement_semantics`: this WSL Python has no `numpy`.

## Timing (ms per stage, median of 3)

| Stage | 176x144 reference | 176x144 tiled | 640x360 tiled |
|---|---|---|---|
| `i_prior` | 2833.0 | 66.7 | 145.8 |
| `i_spatial_prior_1` | 2246.7 | 57.9 | 140.0 |
| `i_spatial_prior_2` | 605.6 | 58.4 | 135.4 |
| `i_spatial_prior_3` | 2441.3 | 54.8 | 148.6 |
| `i_synthesis` | 5002.8 | 161.7 | 1128.2 |
| `p_prior` | 1262.8 | 48.7 | 104.4 |
| `p_reference_feature` | 3271.8 | 49.9 | 262.0 |
| `p_reference_frame` | 3766.7 | 60.4 | 308.0 |
| `p_reset_frame` | 3477.9 | 47.6 | 250.7 |
| `p_spatial_prior` | 669.2 | 23.4 | 49.6 |
| `p_synthesis` | 740.6 | 38.8 | 184.7 |

The reference kernels were not run at 640x360. Pixel count alone puts them at
about 9 times their 176x144 times.

Decode work per frame (the exact stages only; the FP16 `p_display` is not
included):

| Frame | Stages | 176x144 reference | 176x144 tiled | 640x360 tiled |
|---|---|---|---|---|
| I | `i_prior`, `i_spatial_prior_1`–`3`, `i_synthesis` | 13,130 | 400 (33×) | 1,698 |
| P | `p_reference_feature`, `p_prior`, `p_spatial_prior`, `p_synthesis` | 5,945 | 161 (37×) | 601 |

## Where the time goes

Temporary per-op timing (not committed) showed `gemm` at about 93% of the
tiled time and `depthwise3x3` at about 11% before it was threaded. Across one
timed pass of every stage at both sizes, the gemms performed 306 G
multiply-accumulates in 8.6 s, about 35 G/s, or roughly 0.5 per cycle per
core. The inner loop accumulates through memory into `int64` and is not
register-blocked. A build with `-march=native` was no faster, so the limit is
the loop structure, not the instruction set.

## Not measured

- The FP16 stages on ONNX Runtime CPU (`p_display`, and the encoder's analysis
  transforms). The published fx bundle contains no ONNX graphs for them.
- Full-codec or process timing under `docs/performance.md`, mains power, and
  1080p.
- Encode, and any TensorRT cross-check of the tiled kernels' outputs beyond
  their identity with the reference kernels.
