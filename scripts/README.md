# Scripts

The runtime does not depend on Python. These scripts prepare artifacts, package releases, and run the checks that are too target-specific for the C++ library.

## Main commands

| Script | Use |
|---|---|
| `nvcr_artifacts.py` | Install, prepare, build, inspect, and validate model/engine bundles |
| `install.sh` | Install a Linux binary and compatible engine profiles |
| `install_from_source.sh` | Build and test a local source checkout |
| `detect_platform.sh` | Inspect CUDA, TensorRT, GPU, and target properties |
| `package_release.sh` | Create an engine-free binary package |
| `package_engine_bundle.sh` | Package one validated target-local engine bundle |
| `release_engine_assets.sh` | Stage and publish rolling engine assets |
| `benchmark_softwarex_matrix.py` | Validate and run the publication matrix; write a complete evidence package |
| `benchmark_resolution_matrix.sh` | Run the lower-level diagnostic resolution matrix |
| `profile_energy.py` | Capture optional Jetson energy measurements |

## Prepare and validate

```bash
./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 720p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python

./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-720p --json
```

The backend helpers under `scripts/backends/dcvcrt/` are implementation details. The front end above is the supported workflow.

## Package

Binary packages do not contain checkpoints, exported model assets, or TensorRT plans:

```bash
./scripts/package_release.sh \
  --version 0.3.0 \
  --platform linux-x86_64-nvidia \
  --install-prefix /path/to/install \
  --output-dir dist
```

Target-local engine bundles are published separately through the rolling `engine-assets` catalog after validation.

## Evaluate

Use the schema-producing driver for publication work:

```bash
python3 scripts/benchmark_softwarex_matrix.py --help
```

It validates target and artifact identities, drives the Release build and
registered CPU/GPU gates, records metrics, and writes the documented evidence
layout. Start with
[the input example](../docs/experiments/softwarex-inputs.example.json) and
[runbook](../docs/experiments/runbook.md).

The default path keeps optional instrumentation out of performance
repetitions. Add `--profile` for a publication package; it collects latency,
PSNR, and memory in separate repetitions while preserving clean FPS and wall
time. A run without the flag is intentionally performance-only and `partial`.

For ordinary performance collection, use the lower-level resolution pipeline:

```bash
scripts/benchmark_resolution_matrix.sh \
  --resolutions "qcif 720p" \
  --frames 300 \
  --gops "1 299" \
  --repetitions 3 \
  --warmup-frames 10 \
  --engine-root build/engines-rtx4070-exact-20260805 \
  --results-dir evidence/performance/rtx4070-qcif-720p \
  --output-dir /tmp/nvcr-performance-streams \
  --hardware rtx4070
```

With the standard sibling layout, `--engine-root` resolves each selected label
to `dcvcrt-<resolution>` under that directory. The existing per-resolution
input and engine options (`--qcif-input`, `--720p-input`, `--qcif-engine-dir`,
and so on) remain available for exceptions. The pipeline writes
`results.jsonl`, `results.csv`, and
`summary.md` under `--results-dir`; the CSV and JSONL contain both every measured
repetition and an `average` row. Each row includes codec payload bytes and
`payload_bpp`, calculated as payload bits divided by `width * height * frames`.
Result directories can be committed as interim
experiment records, while encoded streams remain under `/tmp`.

Use `--resolutions "360p"` or any other supported label to collect one
resolution. Use `--hardware` to keep labels stable across machines; otherwise
the pipeline records the first `nvidia-smi` GPU name, falling back to the host
name. These diagnostic rows are intentionally separate from the SoftwareX
publication driver and can be aggregated after all hardware runs exist.

`benchmark_resolution_matrix.sh` emits the diagnostic schema
`nvcr.benchmark.resolution-matrix.v1`; it is designed for this collection stage
and is not itself a publication-readiness gate.

For target-specific details, read [Getting started](../docs/getting-started.md), [Model and engine preparation](../docs/dcvcrt-artifacts.md), [Performance](../docs/performance.md), and [Release policy](../docs/releasing.md).
