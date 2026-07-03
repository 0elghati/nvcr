# Performance

Return to the [docs index](README.md) or the [project overview](../README.md).

## Target

NVCR's steady-state encode and decode latency must be no slower than the pinned
DCVC-RT Python implementation on the same GPU, sequence, resolution, QP, entropy
coder count, and frame type. Comparisons use release builds, discard the first ten
frames, and report encode and decode separately.

The entropy microbenchmark is a narrower regression signal. Build it with
`NVCR_BUILD_BENCHMARKS=ON` and run `nvcr_rans_benchmark`; it models the I-frame
symbol counts and wide CDFs used at 1080p.

## Current baseline

Normal-GOP BQTerrace 1920×1080, QP 32, 97 frames, GOP 97, RTX 4070:

| Native revision | Encode | Decode |
|---|---:|---:|
| Host-staged P path | 4.532 fps (≈21.40 s) | 7.318 fps (≈13.26 s) |
| Device-chained P stages | 5.321 fps (18.230 s) | 8.294 fps (11.695 s) |

The reported Python encode result for the same 97-frame workload was about
18.70 s (5.19 fps), so the current native encoder has crossed that observed
end-to-end target. A same-protocol Python decode result and automated warmed
stage gate are still required before declaring general parity.

The earlier all-I point-in-time baseline from 2026-07-02 on an NVIDIA GeForce RTX 4070, using
`FourPeople_1280x720_60.yuv`, QP 32, 15 I-frames, and five measured frames after
ten warm-up frames:

| Runtime | Encode | Decode |
|---|---:|---:|
| Pinned Python DCVC-RT | 23.976 ms | 21.294 ms |
| Native NVCR release build | 153.674 ms | 101.174 ms |

The end-to-end target is therefore not met yet. These numbers are a development
baseline, not portable performance claims.

The rANS optimization in this tree reduced its synthetic 1080p-sized round trip
from 91.6 to 46.6 ms with one coder and from 48.0 to 25.2 ms with two coders. The
encoded sizes and upstream golden streams remain unchanged.

## Dominant remaining work

The P-frame reference, analysis, hyper-analysis, prior, and synthesis stages now
chain through `DeviceTensor` buffers on one CUDA stream. The remaining P spatial
prior, mask/reduction/restore/index work, entropy boundary, and serialized feature
DPB are host-staged. Device allocations are still per-frame rather than arena-backed.
The I-frame path also remains host-staged.

The next performance milestone is a GPU-resident I-frame graph:

1. Introduce reusable device tensor storage sized once per resolution.
2. Bind TensorRT outputs directly to the next stage's inputs.
3. Use the existing CUDA operators for prior, mask, quantization, and index work.
4. Transfer only entropy symbols/indexes and the final reconstructed frame.
5. Add CUDA-event stage timings and an automated post-warmup comparison gate.

## See also

- [Architecture](architecture.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
- [Native command-line interface](cli.md)
