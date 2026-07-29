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

## Current optimization state

The TensorRT backend now defaults to persistent execution contexts on discrete
GPUs, avoiding per-frame context recreation in the normal RTX performance path.
Integrated and Jetson-class devices keep the conservative low-memory mode unless
`NVCR_TENSORRT_LOW_MEMORY_MODE=0` explicitly overrides it.

P-frame encode already chains reference, analysis, hyper-analysis, prior, spatial
prior, synthesis, and feature-DPB updates through `DeviceTensor` buffers on one
CUDA stream. Only compact rANS entropy symbols/indexes and optional verification
frames cross back to the host.

I-frame encode now uses the same device-resident strategy for the expensive
front half of the path:

- input YUV420P8/RGB24 conversion and padding happen in CUDA;
- `i_analysis`, `i_hyper_analysis`, and `i_hyper_synthesis` run through
  device-address TensorRT bindings;
- padded hyperprior outputs are cropped on GPU;
- image-prior sigmoid transforms, q-encode multiplication, four-way masks,
  symbol reconstruction, quarter reductions, q-decode multiplication, and
  entropy index construction run in CUDA;
- I-frame z symbols and four compact y-index streams are copied through reusable
  pinned host buffers for CPU rANS.

For conformance testing, I-frame encode can rebuild the encoder DPB by decoding
the just-produced I-frame payload when `verify_encoder_reconstruction` is enabled.
Normal CLI encoding does not pay this bridge; it uses the GPU-produced I-frame
reconstruction directly. The bridge remains a test/debug guard until I-frame
decode is device-resident or the GPU and host reconstruction paths are proven
byte-identical.

Encode-side temporary device tensors are now served from a bounded per-session
CUDA scratch arena. Owned TensorRT outputs, DPB/reference tensors, entropy
download buffers, and decode-side staging still allocate outside the arena where
their lifetimes cross stage or frame boundaries. The final performance gate still
needs warmed QCIF/720p/1080p JSON benchmark evidence.

The next performance milestone is to finish the GPU-resident I-frame graph:

1. Move I-frame decode hyperprior, spatial prior, restore, and synthesis to
   device-address bindings.
2. Replace the encode-time decode-conformance bridge with a shared
   device-resident reconstruction path.
3. Bind TensorRT outputs directly to the next stage's inputs where lifetimes
   permit.
4. Move remaining decode-side staging to reusable device storage.
5. Add CUDA-event stage timings and an automated post-warmup comparison gate.


### 2026-07-29 RTX 4070 paired 720p/1080p smoke baseline

Paired performance smoke is now required before accepting CUDA/TensorRT
optimization direction changes. The helper is diagnostic rather than the final
publication harness, but it prevents accepting a 720p-only win that regresses the
1080p path.

Command:

```bash
./scripts/benchmark_resolution_pair.sh \
  --gops "1 97" \
  --jsonl /tmp/nvcr-paired-baseline-e7de44f.jsonl
```

Inputs: `FourPeople_1280x720_60.yuv` for 720p and
`BQTerrace_1920x1080_60.yuv` for 1080p, 97 frames, QP 32.

| Resolution | GOP | Engine profile | Payload bytes | Codec time | Throughput |
|---|---:|---|---:|---:|---:|
| 720p | 1 | `720p-fp16` | 1,104,333 | 3.178 s | 30.518 fps |
| 1080p | 1 | `1080p-fp16` | 4,421,239 | 7.318 s | 13.254 fps |
| 720p | 97 | `720p-fp16` | 86,297 | 1.030 s | 94.170 fps |
| 1080p | 97 | `1080p-fp16` | 396,285 | 2.199 s | 44.103 fps |

Short 1080p all-I diagnostic profile, collected separately with `--profile`,
reported 10 allocations / 54,477,632 bytes per I frame. Representative hot
TensorRT enqueue stages were `i_synthesis` about 22-23 ms, `i_analysis` about
12 ms, three spatial priors about 9 ms combined, and `i_hyper_synthesis` about
2.7 ms. This makes direct TensorRT binding, exact-shape/tactic inspection, and
I-path graph/partition work higher value than more allocator cleanup alone.

MLVC-inspired interpretation: MLVC's published speed advantage comes largely
from model and entropy-graph choices that reduce accelerator calls and
neural/coder interleaving. For NVCR's faithful DCVC-RT runtime, the safe lessons
to adopt first are paired JSON evidence, fixed-resolution package/profiling
discipline, persistent runtime objects, explicit part metadata, exact-shape
engine experiments, and measured partition ablations. MLVC-style transmitted
scale design, activation redesign, unified I/P graphs, and smaller MLVC-S-style
models are separate codec research unless introduced as a new opt-in backend or
profile with its own fidelity/RD gates.

### 2026-07-29 RTX 4070 encode scratch arena

Change under test: route encode-side manual temporary CUDA allocations through a
bounded per-session scratch arena and skip writing residual tensors that are not
consumed by the normal encode path. TensorRT outputs, DPB/reference tensors, and
CPU entropy download boundaries remain owned allocations because their lifetimes
can outlive a single scratch frame.

Profile command:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_arena_nores_profile_final.nvcr \
  -s 1280x720 -r 30 --frames 3 --gop-size 1 --qp 32 \
  --engine-profile 720p-fp16 --profile
```

Profile result: each sampled I frame reported 10 allocations and 24,408,512
allocated bytes. Before the arena pass, the same path reported about 51
allocations and 92,314,112 allocated bytes per I frame. The remaining hot stages
are TensorRT engine execution, especially `i_synthesis` around 10.1 ms and
`i_analysis` around 5.3 ms, so allocator cleanup is now mostly a prerequisite for
direct binding and CUDA graph work rather than the dominant latency lever.

Release sweep command template:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_arena_nores_final_gop<GOP>.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size <GOP> --qp 32 \
  --engine-profile 720p-fp16
```

| Build state | GOP | Frames | Payload bytes | Codec time | Throughput |
|---|---:|---:|---:|---:|---:|
| scratch arena + no residual write | 1 | 97 | 1,104,333 | 3.215 s | 30.168 fps |
| scratch arena + no residual write | 8 | 97 | 312,696 | 1.316 s | 73.689 fps |
| scratch arena + no residual write | 97 | 97 | 86,297 | 1.029 s | 94.239 fps |

Interpretation: all-I throughput is roughly neutral versus the previous quiet
30.616 fps run, while normal GOP points remain slightly ahead of the `f88c2b2e24be`
reference sweep. The main measurable improvement is the much smaller allocation
footprint, which reduces allocator jitter and prepares the backend for direct
TensorRT output binding.

### 2026-07-29 CLI logging overhead cleanup

Default CLI encode/decode output now prints only the final summary. Per-frame
progress and runtime info logs require `--verbose`; detailed TensorRT/CUDA stage
counters still require `--profile`. This keeps benchmark stdout small and avoids
terminal/log-processing overhead in normal runs.

Quiet 720p all-I check after the change:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_720p_quiet_gop1_final.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size 1 --qp 32 \
  --engine-profile 720p-fp16
```

Result: 97 frames, 1,104,333 payload bytes, 3.168 s codec time, 30.616 fps.

### 2026-07-29 RTX 4070 720p entropy threshold update

Change under test: enable the two-rANS-coder path at exactly 1280x720 instead
of only above 1280x720, and avoid serializing the unused normal encode-side
I-frame latent-state copy.

Command:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_720p_kept_opts_gop1.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size 1 --qp 32 \
  --engine-profile 720p-fp16
```

| Build state | GOP | Payload bytes | Codec time | Throughput |
|---|---:|---:|---:|---:|
| `f88c2b2e24be` reference | 1 | 1,103,993 | 3.327 s | 29.154 fps |
| two-coder 720p + no latent copy | 1 | 1,104,333 | 3.185 s | 30.452 fps |

A larger experiment that skipped normal I-frame host reconstruction entirely was
rejected: it improved GOP-97 slightly but regressed all-I to 24.683 fps, likely
because removing the final synchronization changed frame-to-frame allocation and
stream retirement behavior. Keep that path out until the arena/direct-bind work
provides stable lifetimes and explicit synchronization boundaries.

### 2026-07-29 RTX 4070 720p GOP sweep by commit

Commit: `f88c2b2e24be` (`f88c2b2e24be5f8d28d3cc147bb5c4d1ddb8d44f`)

Command template:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_720p_gop<GOP>_f88c2b2e24be.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size <GOP> --qp 32 \
  --engine-profile 720p-fp16
```

Single-run codec-time summaries:

| Commit | GOP | Frames | Payload bytes | Codec time | Throughput |
|---|---:|---:|---:|---:|---:|
| `f88c2b2e24be` | 1 | 97 | 1,103,993 | 3.327 s | 29.154 fps |
| `f88c2b2e24be` | 2 | 97 | 910,474 | 2.196 s | 44.174 fps |
| `f88c2b2e24be` | 4 | 97 | 500,953 | 1.614 s | 60.082 fps |
| `f88c2b2e24be` | 8 | 97 | 312,353 | 1.329 s | 72.978 fps |
| `f88c2b2e24be` | 16 | 97 | 191,276 | 1.186 s | 81.792 fps |
| `f88c2b2e24be` | 32 | 97 | 125,481 | 1.109 s | 87.487 fps |
| `f88c2b2e24be` | 97 | 97 | 85,951 | 1.033 s | 93.916 fps |

Interpretation: longer GOPs amortize the current I-frame cost and lean on the
faster P-frame path. The GOP-97 run is the best throughput point in this sweep
because it contains one I frame followed by 96 P frames; GOP-1 is the all-I stress
case and remains the main I-frame optimization target. These are reference
single-run numbers, not final repeated release evidence under the full protocol.

### 2026-07-29 RTX 4070 720p all-I profile

Command:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_720p_alli_fast.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size 1 --qp 32 \
  --engine-profile 720p-fp16 --profile
```

Result after making the decode-conformance bridge verification-only:

| Metric | Value |
|---|---:|
| Frames | 97 |
| GOP size | 1 |
| Payload bytes | 1,103,993 |
| Codec time | 3.353 s |
| Throughput | 28.927 fps |
| Representative warmed I-frame latency | ~34.2-34.8 ms |

Interpretation: this fixes the accidental all-I benchmark penalty from running
a full decode-conformance rebuild for every encoded I frame. The remaining hot
stages are `i_analysis` (~5.0 ms), three image spatial-prior engines (~3.1 ms
combined), `i_synthesis` (~9.4 ms), per-frame allocations, entropy, and final
reconstruction download.

### 2026-07-29 RTX 4070 720p GOP-8 profile

Command:

```bash
./build-release/cli/nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople_720p_opt.nvcr \
  -s 1280x720 -r 30 --frames 97 --gop-size 8 --qp 32 \
  --engine-profile 720p-fp16 --profile
```

Result after the encode-side I-frame GPU-residency patch:

| Metric | Value |
|---|---:|
| Frames | 97 |
| GOP size | 8 |
| Payload bytes | 466,023 |
| Codec time | 2.405 s |
| Throughput | 40.331 fps |
| Representative P-frame latency | ~10.3-10.9 ms |
| Representative warmed I-frame latency | ~115 ms |

Interpretation: the P-frame hot path is now close to 10 ms/frame at 720p. This
GOP-8 sample was captured before the all-I bridge fix above, so its I-frame rows
still include verification-style bridge cost and should not be used as the
current I-frame latency point.

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
