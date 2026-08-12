# Performance and benchmarking

NVCR separates functional validation, measurements from one environment,
controlled comparisons, and regeneration of retained summaries. Use the
workflow whose measurement boundary matches the claim being evaluated.

## Functional round-trip validation

Follow [First run](first-run.md) to verify installation, artifact selection,
encode, decode, and raw-output sizing. That short input is a functional check;
it is not a throughput measurement.

## Environment-specific measurements

Use `scripts/benchmark_resolution_matrix.sh` for a native diagnostic matrix.
It writes `nvcr.benchmark.resolution-matrix.v1` JSONL and CSV rows and a
Markdown summary:

```bash
scripts/benchmark_resolution_matrix.sh \
  --resolutions "qcif 720p" \
  --frames 300 \
  --qp 32 \
  --gops "1 299" \
  --repetitions 3 \
  --warmup-frames 10 \
  --engine-root build/engines-rtx4070-exact \
  --results-dir evidence/performance/rtx4070-native \
  --output-dir /tmp/nvcr-performance-streams \
  --hardware rtx4070-native
```

The inputs and engine bundles must already exist. Run the script with
`--help` for the input layout and per-resolution override options.

For a container measurement, set `NVCR_IMAGE` to the current image selected
in [Installation](installation.md), then use the direct Docker launcher:

```bash
scripts/benchmark_docker.sh \
  --image "$NVCR_IMAGE" \
  --input-dir /data/nvcr/yuv \
  --engine-volume nvcr-engines \
  --results-dir evidence/performance/rtx4070-docker \
  --hardware rtx4070-docker \
  -- \
  --resolutions "qcif 720p" \
  --frames 300 \
  --qp 32 \
  --gops "1 299" \
  --repetitions 3 \
  --warmup-frames 10
```

The launcher records the image's resolved digest. Add
`--install-profiles "qcif 720p"` only when the engine volume is empty. The
public engine catalog is anonymous; the artifact client does not implement
authenticated private-catalog downloads.

On Windows, use `scripts/benchmark_docker.ps1` or run the Bash launcher under
WSL 2 as described in [Docker on Windows](docker-windows.md). Identify those
rows as containerized execution on that Windows/WSL2 host. A container fixes
the Linux userspace, but it does not normalize host, GPU, driver, power,
thermal, storage, Docker Desktop, or WSL overhead.

Record at minimum:

- Windows edition and build;
- WSL 2 kernel and Docker Desktop versions;
- GPU and driver;
- image tag and resolved digest;
- CUDA and TensorRT versions;
- storage path and filesystem;
- power and clock mode; and
- whether the measurement uses the process or codec-loop timing boundary.

## Timing boundaries

Diagnostic rows retain two timing boundaries:

| Field | Boundary | Appropriate use |
|---|---|---|
| `throughput_fps` | Codec loop reported by the NVCR CLI | Runtime diagnosis within one setup |
| `process_throughput_fps` | Complete encode or decode command wall time | Native-versus-container or cross-tool comparison |

Do not mix these fields in one ratio or table. For cross-runtime comparisons,
use the same process-level boundary, input prefix, frame count, QP, GOP,
warm-up, and repetition policy.

Profiling and memory sampling can add synchronization or polling overhead.
Keep profiled repetitions separate from clean throughput repetitions, and do
not substitute profile-pass FPS for clean-pass FPS.

The controlled driver uses `encode_fps_mean` and `decode_fps_mean` for the
normal measured repetitions. Its `total_wall_time_ms` includes normal CLI and
file-I/O costs but excludes the separate verbose, quality, and memory-sampling
passes.

## Byte and quality boundaries

In the diagnostic schema, `payload_bytes` is the sum of serialized `NVAU`
access-unit bytes reported by the CLI. It includes the NVAU header and
codec-private payload and excludes outer `PacketIO` or `NVCS` file framing.
`payload_bpp` is:

```text
payload_bytes * 8 / (width * height * frames)
```

An inner entropy or rANS byte count is a different boundary. Compare Python and
NVCR rate values only when both sides use the same byte boundary, and state
that boundary with every cross-runtime result.

## Controlled comparisons

The strict evaluation driver validates source, target, artifact, test, input,
and measurement identities before writing aggregate result rows. Its legacy
filename and schemas remain stable implementation identifiers:

```bash
python3 scripts/benchmark_softwarex_matrix.py --help
```

A complete controlled result requires:

- a clean commit and Release build;
- detected target identity and current target-local artifact digests;
- registered and passing core, engine, and I/P round-trip gates;
- fixed inputs, frame counts, QP/GOP cases, one warm-up, and three measured
  repetitions;
- clean throughput repetitions and separate latency, quality, and memory
  repetitions;
- an immutable image digest or native build identity;
- the required pinned Python-reference rows on the reference target; and
- `run-summary.json` with `status: complete` and no required failed or
  skipped cases.

Exact engine bundles are the baseline. A same-compute-capability or
Ampere-and-newer result also requires matching exact rows and separate
correctness and performance-ratio validation. TensorRT must still match the
engine manifest exactly; a broader hardware class does not make plans portable
across TensorRT versions. Jetson uses exact bundles only.

The full matrix, acceptance logic, and commands are in the
[evaluation protocol](experiments/evaluation-protocol.md),
[result schema](experiments/result-schema.md), and
[runbook](experiments/runbook.md).

## Regenerating retained summaries

Treat JSONL as canonical. Do not hand-edit numbers in generated summaries.

Consolidate Python DCVC-RT runner output:

```bash
python3 scripts/consolidate_dcvc_rt_results.py \
  --source /path/to/dcvc-rt-results \
  --output /tmp/python-data \
  --hardware rtx4070
```

Regenerate a Python-versus-NVCR comparison from two retained JSONL files:

```bash
python3 scripts/consolidate_runtime_results.py \
  --python results/rtx4070/python/data/results.jsonl \
  --nvcr results/rtx4070/nvcr/results.jsonl \
  --comparable-gop 1 \
  --comparable-gop 30 \
  --comparable-gop 100 \
  --nvcr-frame-overhead-bytes 58 \
  --python-reference-reset 64 \
  --nvcr-reference-reset 64 \
  --nvcr-checkout-state dirty \
  --output results/rtx4070/summary.generated.md
```

The retained RTX 4070 comparison subtracts 58 fixed bytes per access unit: the
32-byte NVAU v1 header, six-byte `dcvcrt` codec identifier, and 20-byte
DCVC-RT private payload header. All three GOPs are selected because the
retained Python and NVCR runs both use a 64-frame feature-reference reset
interval. Use other values only when the matching source revision and
retained evidence establish them.

Compare `summary.generated.md` with the retained `summary.md`, then remove the
generated file. Verify the summary and hashes before replacing a retained
result.
Keep raw YUV, reconstructed video, encoded streams, checkpoints, model exports,
and TensorRT plans outside Git.

## Interpreting retained results

[The result inventory](../results/README.md) records the platform, software
identity, dirty state, artifact identity, evidence class, completeness, and
permitted interpretation of each retained result. A profile file, successful
build, TensorRT deserialization, or complete-looking table does not establish
target support by itself.
