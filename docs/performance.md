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

## Required protocol and evidence

Performance evidence is valid only from a Release build after correctness and
bundle validation pass. Record:

- NVCR commit/tag, model/engine manifest digests, target/engine profile, and command;
- OS, compiler, driver/JetPack, CUDA, TensorRT, GPU, power mode, and clocks;
- input identity, pixel format, resolution, frame count, QP, GOP/reset pattern;
- warm-up/discard count, repetitions, initialization inclusion, and statistic;
- encode/decode latency, throughput, host/GPU peak memory, allocation/transfer counts;
- payload bytes/bitrate and reconstructed PSNR or the declared distortion metric;
- on Orin, rail, sample interval, idle method, raw and idle-adjusted energy;
- matching pinned-Python command/protocol and any numerical tolerance.

Preserve failed and superseded runs as labeled history. A point result on one
sequence or a quality-mismatched x265 setting is not a support or parity claim.
symbol counts and wide CDFs used at 1080p.

## Historical development baselines

Normal-GOP BQTerrace 1920×1080, QP 32, 97 frames, GOP 97, RTX 4070:

| Native revision | Encode | Decode |
|---|---:|---:|
| Host-staged P path | 4.532 fps (≈21.40 s) | 7.318 fps (≈13.26 s) |
| Device-chained P stages | 5.321 fps (18.230 s) | 8.294 fps (11.695 s) |

The reported Python encode result for the same 97-frame workload was about
18.70 s (5.19 fps), so this particular native P-path run crossed that observed
encode baseline. It is a historical development measurement, not a general
performance or support claim: a same-protocol Python decode result, repeated
release runs, and an automated warmed stage gate are still required.

The earlier all-I point-in-time baseline from 2026-07-02 on an NVIDIA GeForce RTX 4070, using
`FourPeople_1280x720_60.yuv`, QP 32, 15 I-frames, and five measured frames after
ten warm-up frames:

| Runtime | Encode | Decode |
|---|---:|---:|
| Pinned Python DCVC-RT | 23.976 ms | 21.294 ms |
| Native NVCR release build | 153.674 ms | 101.174 ms |

The complete end-to-end target is therefore not met yet. These numbers are a
development baseline, not portable performance claims or v1 release evidence.

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

## Jetson energy profiling

On Orin-class targets, measure encode and decode energy with
`scripts/profile_energy.py`. It wraps an `nvcr` command, samples `tegrastats`,
integrates the selected power rail, and writes raw plus idle-subtracted
joules/frame:

```bash
./scripts/profile_energy.py --idle-seconds 10 --interval-ms 100 \
  --output-json /tmp/nvcr-energy.json \
  -- ./build-orin-release/cli/nvcr encode ... --profile
```

Use `VDD_IN` when available; otherwise explicitly pass the board rail used by
the target's `tegrastats` output with `--rail`. Treat the raw value as board
energy for the whole command and the idle-subtracted value as the preferred
codec-run estimate. Both still include initialization and file I/O unless the
benchmarked command is structured to exclude them.

## Historical incomplete study — NVCR vs libx265 on Orin Nano

Methodology: both codecs encode the same raw YUV420 source on the same device
under `scripts/profile_energy.py` at idle-adjusted `VDD_IN` board power.
The `--frames` argument is passed to the profiler so j/frame is consistent.
Latency is wall-clock codec time reported by each tool; quality comparison
requires separate PSNR/VMAF measurement (not included here) because
QP and CRF are not the same scale.

### Test conditions

| Parameter | Value |
|---|---|
| Content | BasketballDrive 1920×1080 50 fps |
| Frames | 97 |
| GOP structure | All-I-plus-P single GOP (GOP 97) |
| Device | Jetson Orin Nano, 8 SMs SM 8.7 |
| Power rail | VDD\_IN |
| Idle sample duration | 10 s |
| Sample interval | 100 ms |

### Commands

```bash
# NVCR / DCVC-RT — FP16 TensorRT, performance mode
NVCR_TENSORRT_LOW_MEMORY_MODE=0 \
./scripts/profile_energy.py --idle-seconds 10 --interval-ms 100 \
  --output-json /tmp/nvcr-energy.json --frames 97 \
  -- nvcr encode \
       -i /path/to/BasketballDrive_1920x1080_50.yuv \
       -o /tmp/basketball.nvcr \
       -s 1920x1080 -r 50 --frames 97 --gop-size 97 --qp 32 \
       --engine-dir build/engines/dcvcrt

# libx265 via FFmpeg — software H.265 reference
./scripts/profile_energy.py --idle-seconds 10 --interval-ms 100 \
  --output-json /tmp/x265-energy.json --frames 97 \
  -- ffmpeg -f rawvideo -pix_fmt yuv420p -s:v 1920x1080 -r 50 \
       -i /path/to/BasketballDrive_1920x1080_50.yuv \
       -vframes 97 -c:v libx265 -preset medium -crf 28 \
       /tmp/encoded.mp4
```

### Results

| Codec | Setting | fps (wall) | Active energy (J) | J/frame (idle-adj.) | Active avg power (W) |
|---|---|---:|---:|---:|---:|
| NVCR DCVC-RT | QP 32, GOP 97, FP16 | 5.12 | 390.7 | 2.81 | 20.7 |
| libx265 (FFmpeg) | CRF 28, preset medium | 2.61 | 383.0 | 1.53 | 10.4 |

### Notes

- Quality is not directly comparable: DCVC-RT QP 32 and x265 CRF 28 are
  independent scales. Use PSNR or VMAF on the reconstructed output for a
  fair quality-efficiency comparison.
- NVCR active energy includes TensorRT initialization (~1.2 s) and one I-frame.
  Steady-state P-frame GPU time is ~175 ms/frame (all CUDA kernel execution;
  CPU overhead is < 3 ms/frame after current optimizations).
- libx265 is CPU-only on Orin; its power draw will come from the CPU/DRAM rails
  rather than the GPU, so VDD\_IN captures both.

## See also

- [Architecture](architecture.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
- [Native command-line interface](cli.md)
