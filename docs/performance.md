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

I-frame decode now runs hyperprior, image-prior transforms, four-way
mask/reduction/restore, three spatial priors, and synthesis through
device-address bindings. The reconstructed frame remains device-resident for
the DPB and is downloaded once for the public frame result. The old host
TensorRT and image-prior implementations have been removed.

A bounded per-session CUDA scratch arena serves frame-local tensors and eligible
TensorRT outputs. Image quantization tensors are cached once per QP. I-decode
reuses pinned host buffers at the CPU rANS z/symbol/index boundaries, transfers
symbols as packed int8, and converts them to FP16 on the GPU. The public frame
download remains the other host boundary. The final performance gate still
needs warmed QCIF/720p/1080p JSON benchmark evidence.

The next performance milestone is to tighten the unavoidable entropy boundaries:

1. Establish pinned-Python golden reconstruction for corrected I/P decoding.
2. Isolate the pre-existing P-frame nondeterminism beginning at frame 2.
3. Profile direct rANS decode-into staging to remove the remaining vector copy.
4. Bind TensorRT outputs directly to downstream inputs where lifetimes permit.
5. Add an automated post-warmup comparison gate.


### 2026-07-30 RTX 4070 warmed encode and decode diagnostic

Commit `4b7e181` was measured with the Release CLI on an NVIDIA GeForce RTX
4070, driver 580.159.03, CUDA 12.6, and TensorRT 10.7.0.23. Inputs were
FourPeople 1280x720 and BQTerrace 1920x1080, YUV420P8, 97 frames, QP 32.
Each point used a 10-frame warm-up CLI invocation followed by three measured
CLI invocations. Codec initialization is excluded from the reported codec time.

| Resolution | GOP | Encode payload | Encode time | Encode | Decode time | Decode |
|---|---:|---:|---:|---:|---:|---:|
| 720p | 1 | 1,104,333 bytes | 3.177667 s | 30.524667 fps | 2.744667 s | 35.340000 fps |
| 1080p | 1 | 4,421,239 bytes | 7.371000 s | 13.160000 fps | 6.481000 s | 14.967000 fps |
| 720p | 97 | 86,297 bytes | 1.023333 s | 94.773000 fps | 2.190000 s | 44.292000 fps |
| 1080p | 97 | 396,285 bytes | 2.195667 s | 44.182333 fps | 4.869333 s | 19.920667 fps |

Per-run records: [encode JSONL](evidence/2026-07-30-rtx4070-warmed-encode.jsonl)
and [decode JSONL](evidence/2026-07-30-rtx4070-warmed-decode.jsonl). Both files
contain three measured rows and one aggregate row per resolution/GOP point.

This is diagnostic evidence, not a passed release gate. The warm-up runs in a
separate process and therefore do not implement the required same-session
10-frame discard. Python/native reconstruction conformance is also still
blocked by the frame-1 mismatch. The next benchmark-harness change must measure
post-warm-up frames in one runtime session and record encode and decode together.

### 2026-07-30 RTX 4070 packed I-frame decode staging

This candidate adds reusable pinned index/symbol staging for all four
spatial-prior entropy stages, dependency-scoped CUDA-event waits, packed int8
symbol uploads, and GPU int8-to-FP16 conversion. It also restores the
decode-side image-prior sigmoid transform required before using q-decode values.

A clean temporary Release build passed all 6/6 registered tests. The native I/P
round trip and installed v0.4.1 `720p-fp16` and `1080p-fp16` TensorRT bundle
tests passed. CUDA unit coverage now checks packed symbol conversion and the
decode-only image-prior transform.

The 720p I-frame profile retained one allocation / 5,529,600 bytes, 5 H2D, 5
D2H / 6,451,200 bytes, 6 D2D / 11,059,200 bytes, and 5 causal entropy
synchronizations. Packed uploads halved H2D traffic from 1,904,640 to 952,320
bytes. Representative enqueue times remained about 1.24 ms for
hyper-synthesis, 1.15-1.17 ms for each spatial prior, and 10.03-10.08 ms for
synthesis.

FourPeople 1280x720, 97 frames, QP 32, all-I decode produced 35.434, 35.485,
and 35.511 fps. All three decoded outputs had SHA-256
`3c3da066e973e06352ecc473a9f64e0750ed5075148b43f6105d51bf329fd056`.

For a normal GOP-8 comparison, parent `1407b02` and the candidate encoded
byte-identical access units: 312,696 payload bytes, SHA-256
`76f6bfc6edc16dea56d073c40eb431256a27600291653fecd7a2dc9c67168c08`.
Decoding that exact stream measured 30.673 fps on the parent host path and
43.098 fps on the corrected device path, a 40.5% point improvement. Source
comparison measured 22.872802 dB average PSNR for the parent decoder and
31.569734 dB for the corrected device decoder.

The PSNR result diagnoses a missing image-prior transform in the parent decode
path; it is not a cross-runtime conformance result. Pinned Python DCVC-RT golden
comparison remains required. Repeated normal-GOP decodes also diverge beginning
at P-frame 2 on both parent `1407b02` and the candidate, while the corrected
all-I output is deterministic. The P-path issue therefore remains an explicit
M3 blocker.

Superseded result: reusable pinned staging and CUDA events without packed
uploads averaged 42.109 fps against a 42.264 fps baseline (-0.37%, within run
noise). That partial version was rejected as an independent optimization.

### 2026-07-30 RTX 4070 device-resident I-frame decode

The recovered implementation was rebuilt in Release mode and checked against the
installed v0.4.1 `720p-fp16` and `1080p-fp16` TensorRT bundles. Registered tests
passed 6/6, both bundle contract tests passed, and the native I/P round trip
passed.

The 720p GOP-8 stream used FourPeople, 97 frames, QP 32, and the installed
`720p-fp16` bundle. Encode produced 312,696 payload bytes in 1.313 s
(73.891 fps). Four decode measurements produced 42.257, 42.295, 42.108, and
42.390 fps; the final three-run mean was 42.264 fps.

A nine-frame profile reported one I-frame allocation / 5,529,600 bytes, 5 H2D /
1,904,640 bytes, 5 D2H / 6,451,200 bytes, 6 D2D / 11,059,200 bytes, and 5
synchronizations. No `host` TensorRT stage was present. Representative I-frame
enqueue times were about 1.2 ms for hyper-synthesis, 1.15 ms for each spatial
prior, and 10.0 ms for synthesis.

This is repeated development evidence, not the final publication gate: it does
not yet include the required warm-up discard, QCIF/1080p throughput matrix,
matching pinned-Python run, or cross-runtime golden reconstruction comparison.

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

## Four-resolution DCVC-RT baseline (2026-07-30)

This is the pre-decode-optimization baseline for the M4 resolution-profile gate.
It uses the release build and target-local FP16 TensorRT plans built on the
`rtx4070-ubuntu2404` target profile (RTX 4070, CUDA 12.6, TensorRT 10.7.0).

Protocol:

- Inputs: `akiyo_qcif.yuv`, `paris_cif.yuv`, `FourPeople_1280x720_60.yuv`, and `BQTerrace_1920x1080_60.yuv` from `/home/oelghati/DCVC/datasets/`.
- QP 32, 97 measured frames, and GOP sizes 1 and 97.
- One 10-frame encode warm-up and one 10-frame decode warm-up per case.
- Three measured encode/decode repetitions per point; table values are arithmetic means.
- FPS is codec-reported timing. Weighted YUV PSNR is computed during decode outside codec timing.
- GOP 1 is the all-I development path. GOP 97 is the practical warmed I/P path and the decode-optimization baseline.

| Resolution | GOP | Payload bytes | Encode FPS | Decode FPS | PSNR-YUV (dB) |
|---|---:|---:|---:|---:|---:|
| QCIF (176x144) | 1 | 64,153 | 487.411 | 578.714 | 35.784883 |
| CIF (352x288) | 1 | 1,311,398 | 225.416 | 247.506 | 28.044318 |
| 720p (1280x720) | 1 | 1,104,333 | 35.343 | 42.163 | 38.648032 |
| 1080p (1920x1080) | 1 | 4,421,239 | 15.204 | 17.729 | 34.701453 |
| QCIF (176x144) | 97 | 11,124 | 830.164 | 808.001 | 38.355843 |
| CIF (352x288) | 97 | 155,448 | 536.522 | 377.522 | 25.965082 |
| 720p (1280x720) | 97 | 86,297 | 95.403 | 57.112 | 40.711093 |
| 1080p (1920x1080) | 97 | 396,285 | 44.379 | 26.060 | 35.242946 |

Machine-readable per-run and aggregate evidence is in
`docs/evidence/dcvcrt-resolution-matrix-2026-07-30.jsonl`.

The practical GOP-97 decode/encode ratios are approximately 97.3% at QCIF,
70.4% at CIF, 59.9% at 720p, and 58.7% at 1080p. The increasing gap makes
resolution-scaled decode execution the next optimization target. These values
are the before measurements; any decode change must rerun the same matrix and
report both FPS and PSNR-YUV.

## Device-side decoded-output conversion (2026-07-30)

This candidate moves the visible P-frame `frame_hat` conversion from a full FP16
three-plane host download plus scalar CPU YUV420P8 conversion to CUDA kernels
that emit only visible Y, U, and V bytes. The decoder keeps the FP16
`frame_hat` and feature tensors on device for the DPB, writes compact output
bytes into device scratch, downloads those bytes through a reusable pinned host
buffer, and preserves the existing CPU conversion for reference paths and
I-frame latent-state serialization.

Validation used the same release-build matrix protocol as the baseline above.
Machine-readable evidence is in
`docs/evidence/dcvcrt-resolution-matrix-device-yuv420-2026-07-30.jsonl`.

| Resolution | GOP | Payload bytes | Encode FPS | Decode FPS | Decode change | PSNR-YUV (dB) |
|---|---:|---:|---:|---:|---:|---:|
| QCIF (176x144) | 1 | 64,153 | 481.796 | 577.907 | -0.14% | 35.784883 |
| CIF (352x288) | 1 | 1,311,398 | 224.134 | 247.797 | +0.12% | 28.044318 |
| 720p (1280x720) | 1 | 1,104,333 | 35.026 | 42.174 | +0.03% | 38.648032 |
| 1080p (1920x1080) | 1 | 4,421,239 | 15.074 | 17.717 | -0.07% | 34.701453 |
| QCIF (176x144) | 97 | 11,124 | 860.563 | 944.322 | +16.87% | 38.355843 |
| CIF (352x288) | 97 | 155,448 | 542.374 | 519.126 | +37.51% | 25.965082 |
| 720p (1280x720) | 97 | 86,297 | 94.997 | 90.416 | +58.31% | 40.711093 |
| 1080p (1920x1080) | 97 | 396,285 | 44.379 | 41.843 | +60.56% | 35.242946 |

All payload byte counts and PSNR-YUV values match the baseline matrix. GOP-97
decode clears the +10% acceptance target at both 720p and 1080p, with encode
throughput within normal run variance.

A short 720p GOP-97 Nsight Systems decode trace is retained as
`docs/evidence/dcvcrt-device-yuv420-720p-gop97-decode-2026-07-30.nsys-rep`;
text summaries are in
`docs/evidence/dcvcrt-device-yuv420-720p-gop97-decode-2026-07-30-nsys-stats.txt`
and memcpy byte totals are in
`docs/evidence/dcvcrt-device-yuv420-720p-gop97-decode-2026-07-30-memcpy.csv`.
Under tracing, decode measured 89.157 fps. D2H traffic fell from the baseline
~581.5 MB to 183,398,400 bytes while H2D remained 260,481,152 bytes and D2D
remained 364,985,344 bytes. `cudaMemcpyAsync` host API time fell from the
baseline ~711 ms to 367.728 ms. Total GPU kernel time was 811.426 ms, and the
new visible-output kernels accounted for about 1.018 ms across the 96 P frames.

### Superseded build and validation attempts

The following failures are retained as diagnostic history and are not release
evidence:

- Exporting from the dirty pinned DCVC-RT checkout folded required ONNX inputs away; TensorRT then rejected `i_hyper_synthesis` because `z_hat` was absent. A clean detached pinned source exported the correct inputs.
- PyTorch 2.9 Dynamo export repeatedly failed with FakeTensor recursion in spatial-prior and P-synthesis stages. The deterministic legacy TorchScript ONNX path completed from the clean pinned source.
- The first successful QCIF plan build reached manifest generation but lacked `--target-profile-path`. Artifact preparation now rejects that omission before expensive work.
- The first QCIF contract run rejected the `192x192` warm-up because its I-frame maximum stopped at visible dimensions. QCIF now keeps `176x144` as its optimization point while admitting the padded maximum.
- The first CIF roundtrip rejected the padded `20x24` I hyper-latent. The CIF hyper-analysis profile now matches runtime latent padding.
