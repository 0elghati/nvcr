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
| `benchmark_resolution_matrix.sh` | Run the current encode/decode evaluation matrix |
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

Run the matrix only after a clean Release build and the configured correctness tests:

```bash
scripts/benchmark_resolution_matrix.sh --help
```

The publication run should write one current JSONL summary under `evidence/`. Raw streams are temporary and should not be committed.

For target-specific details, read [Getting started](../docs/getting-started.md), [Model and engine preparation](../docs/dcvcrt-artifacts.md), [Performance](../docs/performance.md), and [Release policy](../docs/releasing.md).
