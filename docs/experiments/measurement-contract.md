# Comparable implementation measurements

Use `scripts/measurement_campaign.py` for new NVCR-versus-Python measurements.
The configured manifest is `docs/experiments/measurement-campaign.json`. It keeps
QP 0/21/42/63, GOP 1/30/100, six input resolutions, 100 frames starting at frame
zero, ten throughput repetitions and ten independent memory repetitions. One
additional encode/decode execution per implementation and condition supplies
quality and RD data. This deterministic-quality pass is not a timing repetition.
The new schema is `nvcr.measurement.observation.v1`; legacy results are not
converted into observations of this protocol.

The comparison is native TensorRT FP16 DCVC-RT versus pinned PyTorch FP16 DCVC-RT
with required CUDA inference and native rANS extensions. A resulting speedup
compares these complete implementations. It does not isolate a runtime-framework
benefit, a language effect, or a particular GPU optimization.

## Quality

The primary metric contract is `decoded-yuv420p8-pooled-plane-6-1-1-v1`.
Both decoders write actual deployment output. The common offline evaluator reads
planar unsigned 8-bit Y, U, V, at even source dimensions, chroma dimensions W/2
by H/2, and code range 0..255. It performs no further colour conversion, resize,
clipping, rounding or subsampling. It excludes the padding outside source
width/height and evaluates the manifest's first N frames. The source must contain
at least N complete frames; the reconstruction must contain exactly N frames.
Raw files do not encode dimensions: the manifest, parsed encoded headers and
expected decoded byte count jointly establish dimensions in the campaign.

For every frame and plane the evaluator retains integer SSE, sample count, and
PSNR = 10 log10(255² / MSE). The primary result pools SSE and sample count over
time separately for each plane, then weights plane PSNR in the ratio 6:1:1.
`pooled_psnr_yuv_db` and `mean_frame_psnr_yuv_db` are distinct fields; the latter
averages per-frame plane PSNR before the same weighting. Neither is a PSNR of
one pooled YUV MSE. Tests with frame errors 1 and 3 expose the difference.

Exact planes retain the string `"infinity"`, not a cap or a nonstandard JSON
number. A weighted PSNR with an exact plane is consequently infinite even if
other planes differ; `exact_reconstruction` separately requires every SSE to be
zero. No frames are dropped. Truncation, trailing decoded bytes, invalid
dimensions and frame counts fail evaluation. RD integration rejects infinite
quality, NaN, missing values and non-overlapping curves. The Python quality pass
also checks its floating reconstruction for non-finite values before converting
to bytes. Finite checking is excluded from throughput passes.

Serialization remains implementation-specific and is part of deployment quality:
NVCR crops padded YCbCr, rounds clipped luma with `lround`/`roundf`, and truncates
clipped 2x2-averaged chroma. Pinned Python crops first, uses `yuv_444_to_420`
(2x2 average), rounds scaled FP16 luma with PyTorch round-to-even, and truncates
scaled chroma. The evaluator does not modify either internal DPB to force
agreement. These serialization differences are explicitly retained, not hidden
by a second quantization of floating output. Historical upstream float-domain
PSNR stays in its original files and is not relabelled as this metric. New clean
operations record that upstream float diagnostics were not collected.

## Timing and state

`host-yuv420p8-completed-frame-v1` is a host elapsed-time contract: encode starts
with one raw host YUV420P8 frame and ends with complete host serialized packet
bytes; decode starts with host encoded record bytes and ends with host YUV420P8.
It sums these completed-frame intervals over exactly N frames, including frame
zero. It is not a single asynchronous enqueue timer and does not discard the
first ten measured frames. Both paths synchronize the CUDA device before and
after each frame. This explicit policy permits comparison but inhibits overlap;
it must accompany reported throughput. The timer includes the ending barrier.
No additional diagnostic GPU-event timing is added. Existing backend profile
stages are separate diagnostics and may overlap; their sum is not a decomposition
of this host interval.

| Work | Primary completed-frame interval | Initialization-inclusive process time |
|---|---|---|
| Source file read / bitstream file read | Excluded | Included |
| Host-frame construction, YUV420 expansion, normalization, padding, H2D | Encode included | Included |
| Inference, reference updates, entropy coding | Included | Included |
| Native codec-private/NVAU/PacketIO serialization or Python SPS/IP serialization | Included | Included |
| NVCS file header/record length and physical file writes | Excluded | Included |
| PacketIO/NVAU parsing or Python SPS/IP parsing | Decode included | Included |
| Cropping, output chroma conversion, quantization and D2H | Decode included | Included |
| Engine/model loading and runtime construction | Excluded; `initialization_seconds` diagnostic | Included |
| Warm-up and reset | Excluded | Included |
| Common quality evaluation and hashing | Excluded; outside codec process | Excluded |
| Memory polling / verbose per-frame tracing | Disabled | Disabled |

The native sequence reader strips the NVCS length field before the interval;
Python reads its short IP record length within its syntax parser. These are
format-specific framing costs, not the same bytes. Native initialization
includes runtime/engine construction; Python initialization records model
loading/update/half conversion after imports. These diagnostics are labelled
implementation-specific and must not be subtracted from process time to invent
a new common interval. `process_seconds` includes process creation, imports,
loading, warm-up, file I/O, and teardown; the parent observes termination with
10 ms resolution. It is reported separately, never as steady-state codec FPS.

For the fixed 100-measured-frame job, `process_fps = frames / process_seconds`
includes the ten warm-up frames in elapsed time but excludes them from the
numerator. The RTX result reports the arithmetic mean, sample SD and two-sided
Student-t 95% interval of ten independent per-run process FPS values for each
condition. Its pooled throughput is instead `sum(frames) / sum(process_seconds)`
over the stated equal-work runs; that pooled ratio is not the arithmetic mean
or the confidence interval. Full-matrix and targeted-repeat campaigns remain
separate statistical cohorts.

Warm-up runs the first ten frames in the SAME model/runtime session, flushes and
resets reference state, rewinds to frame zero, and retains loaded execution
resources. Encode and decode each have an isolated process and independent
warm-up. Native `Runtime::reset` clears session queues, counters and both DPBs;
the backend keeps engines/quantization caches. Python clears its DPB, resets POC
and resets SPS bookkeeping. The bounded smoke compares cold versus warmed/reset
stream and decoded-output hashes. Until that passes, reset equivalence on the
GPU target is unverified. Existing CPU session tests establish API reset
semantics only.

Both paths use I frames at index 0 and multiples of GOP, the effective P QP shift
pattern 0/8/0/4/0/4/0/4, and feature-reference reset interval 64. The wrapper
explicitly requests two entropy coders at area >=1280x720 to match NVCR. This is
a documented setting difference from pinned `test_video.py` (which uses > at
that boundary), not a patch to the upstream codec. Actual encoded frame type,
effective QP, entropy-coder mode and reference flags are validated per frame.
Input fps metadata retains the archived native configuration, including 30 for
FourPeople and 60 for BasketballDrive 1080p. Those are metadata assertions, not
verified acquisition frame rates. FPS throughput uses the actual timed count.

## Reference source

The new `measure_python_reference.py` wrapper imports the clean source pinned by
`configs/models/dcvcrt-cvpr2025.json` at Microsoft DCVC commit
`1feb52a592a9ff2c4e4ba2e5122e2da49a211466`. Both checkpoint digests must match.
Models use `eval()`, `half()`, `no_grad()`, the upstream deterministic setup,
upstream normalization/nearest-neighbour chroma expansion, 16-pixel replication
padding, rANS compress/decompress, and the original output conversion. CPU
fallback and absence of required extensions fail. Extension paths/hashes and
Python dependencies are recorded; importing an extension is not proof that it
works on all tensor shapes. The bounded QCIF smoke currently exposes a custom
CUDA misaligned-address failure (see validation evidence).

The historical local path is `/home/oelghati/DCVC-RT/runner/load.py`, which builds
commands for `test_video_energy.py` with explicit source dimensions, frame count,
GOP, QP, reset interval and encode-only/decode-only flags, using `.venv-jetson`.
That wrapper also uses FP16/eval/no-grad; it has energy/process/memory measurement
changes and selectable synchronization modes. Its source checkout and CUDA
helpers differ from upstream. Environment capture fingerprints its current
files when available. Retained historical rows do not bind those files to the
exact revision/command that produced each row. Their provenance is therefore
incompletely reconstructable. The new wrapper is not presented as that original
wrapper. It does not reuse historical timing or quality values.

## Memory

The primary common memory quantity is Linux `wait4` **process RSS high-water**, in
MiB (`ru_maxrss` KiB / 1024). The parent reaps the exact operation PID; it does not
use cumulative child usage. Each operation is a separate process. The window
includes imports, engine/model loading, contexts, warm-up, frame buffers and
codec work. This is a reliable process-lifetime high-water measurement, not a
100 ms sampled peak. Ten additional memory executions are kept separate from
throughput. The schema records the method and unit explicitly.

This is not CUDA allocated memory, PyTorch reserved memory, process GPU usage or
whole-system RAM. Those unavailable common counters remain unavailable, not
zero, and are never added to RSS on shared-memory Jetson. Environment snapshots
may contain system RAM/GPU data, but they are not substituted for process memory.
Historical `peak_memory_mb` values from different collectors are not pooled with
these observations. Allocator-specific peak memory is intentionally unsupported
as a cross-framework primary comparison.

## Bytes

`measurement_metrics.nvcr_bytes` walks actual NVCS v1 records, PacketIO v1 metadata
and payload lengths, NVAU v1 or v2, and NVI1/NVP1 private syntax. It reports:

- entropy bytes and 20-byte per-frame private syntax;
- NVAU envelope, identifiers, dependencies/section table, and optional side data;
- outer packet framing separately from variable key/value metadata;
- sequence header and per-record 64-bit lengths;
- inclusive access-unit/packet sizes and complete file length.

Only incremental components enter the reconciliation sum. A successful result
must exactly match file length. The Python wrapper counts SPS bytes, IP syntax
and actual rANS bytes and performs the same reconciliation. Both expose entropy
BPP and complete-file BPP separately. Extra BPP uses width × height × frame
count; overhead bytes per AU use the actual AU count; overhead fraction uses
complete-file bytes; expansion uses entropy bytes. A zero entropy denominator
is explicitly unavailable. The legacy 58-byte rule holds only for NVAU v1 with
a six-byte identifier plus the current 20-byte private syntax; it never includes
PacketIO or NVCS overhead. The production-writer fixture tests both NVAU versions
and variable outer metadata/side sections. One-frame files are supported without
changing the campaign's frame count.

## Observations, provenance and analysis

`plan.json` and `commands.txt` resolve every execution before launch. Within each
repetition, the fixed seed shuffles condition/implementation blocks. Encode then
decode uses the same stream, in separate processes. Throughput, memory and
quality phases are separate. This does not establish paired cross-implementation
samples; analysis reports independent process variability per condition.

Each raw observation has unique execution and attempt IDs, case/job identity,
implementation, operation, mode, status, units, input digests and a common
compatibility fingerprint. Full input bytes and evaluated prefixes are hashed;
pre/post-operation prefix checks reject changing inputs. Reused builds, inputs,
engines, extension binaries, power settings, execution environment or timing
contracts must match the saved fingerprint. Successful resume requires matching
encode/decode attempts. Failures and timed-out/skipped cases remain in the
journal. Missing observations remain explicitly incomplete. New/empty output
directories are required unless `--resume` is requested. A partially written
journal is rejected, not silently repaired. Raw failures are retained locally;
large successful products are removed only after hashes and offline metrics
have been saved. Historical results are untouched.

Environment capture reads source revisions/diffs, binary and helper hashes,
CMake Release flags/link information, compiler, CPU/RAM, kernel/OS, GPU/device,
CUDA/TensorRT, package inventory, checkpoint/engine manifests and hashes,
container digest when applicable, L4T/module, existing nvpmodel/clock settings,
and accessible sensors. Before/after snapshots are outside clean timing. Missing
sensors and absent historical export/build commands are explicitly unavailable;
manifest build/export options and hashes are retained where available. No
privileged setting, service, clock or power mode is changed. Thermal snapshots
cannot prove there was no throttling throughout a run.

`analysis.json` has arithmetic mean and sample standard deviation (ddof=1), n,
minimum and maximum for each implementation/operation/QP/GOP/sequence/mode.
Quality and content variation are never counted as timing repetitions. Raw and
aggregate records use separate files. A complete status requires all planned
observations, not just a zero subprocess return code. Exact quality values stay
explicit. `rd-points.json` provides the common quality definition and both rate
definitions. Optional BD-rate analysis uses version-recorded SciPy PCHIP in log
rate over quality overlap only. Curves must be explicitly supplied in strictly
increasing rate and quality order; nonmonotonicity, duplicate quality, non-finite
values, mismatched definitions and absent overlap fail. No extrapolation, silent
curve sorting or favourable-point deletion is performed.
