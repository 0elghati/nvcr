# SoftwareX runbook

Run from the repository root. Do not invent missing inputs; mark the row skipped and record the missing item instead.

## 1. Record identity

```bash
python3 scripts/nvcr_device.py > /tmp/nvcr-device.json
git rev-parse HEAD
git status --short
```

For a container run, record the image tag and digest as well. A dirty tree is allowed for development diagnostics but not for a publication row.

## 2. CPU-only container

This validates public contracts without pretending to validate TensorRT:

```bash
docker build --platform linux/amd64 --target test \
  -f docker/Dockerfile.x86_64 -t nvcr-test:cpu .
docker run --rm nvcr-test:cpu cpu
```

Record separately: CPU tests passed, GPU tests not registered, engine bundles not validated, and publication matrix not completed.

## 3. x86_64 CUDA/TensorRT container

Build/run on a desktop NVIDIA host. The image contains no plans or checkpoints:

```bash
docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install
NVCR_TEST_ENGINE_PROFILE=qcif \
  docker compose -f docker/compose.x86_64.yaml run --rm --build test gpu
```

For a local engine collection instead of catalog installation:

```bash
export NVCR_ENGINE_ROOT="$PWD/build/engines"
docker compose -f docker/compose.x86_64.yaml run --rm --build test gpu
```

For direct execution, mount engines read-only and inputs read-only:

```bash
docker run --rm --gpus all \
  -e NVCR_ENGINE_ROOT=/opt/nvcr/engines \
  -v "$PWD/build/engines:/opt/nvcr/engines:ro" \
  -v "/path/to/input:/input:ro" -v "/path/to/output:/output" \
  nvcr:rtx4070 --help
```

## 4. Jetson container

Build and run on the native Orin host:

```bash
docker build --platform linux/arm64 --target runtime \
  -f docker/Dockerfile.jetson -t nvcr:orin-nano .
docker compose -f docker/compose.jetson.yaml run --rm --build engine-install
NVCR_TEST_ENGINE_PROFILE=qcif \
  docker compose -f docker/compose.jetson.yaml run --rm --build test gpu
```

Jetson rows are exact-only. QEMU or cross-platform image construction is not target evidence.

## 5. Prepare and validate artifacts

Use the checked-in target profile and build on the final target:

```bash
TARGET_PROFILE=configs/targets/rtx4070-ubuntu2404.json
./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 720p \
  --target-profile "$TARGET_PROFILE" \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-720p --json
```

For a complete set, repeat validation for `qcif`, `cif`, `360p`, `540p`, `720p`, and `1080p`.

## 6. Exact-target rows

### RTX 4070 exact

```bash
export TARGET_PROFILE=configs/targets/rtx4070-ubuntu2404.json
export NVCR_ENGINE_ROOT=/path/to/rtx4070-engines
# Validate all selected bundles, run ctest, then run the matrix:
scripts/benchmark_resolution_matrix.sh \
  --nvcr build-release/cli/nvcr \
  --resolutions "qcif cif 360p 540p 720p 1080p" \
  --gops "1 97" --qp 32 --frames 97 \
  --repetitions 3 --warmup-frames 10 \
  --output-dir /tmp/nvcr-rtx4070 \
  --jsonl evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)/matrix.raw.jsonl
```

The target is mandatory for the Python comparison.

### RTX 3050 exact

Blocked until `configs/targets/rtx3050-ubuntu2404.json` and a complete validated engine set exist. Do not substitute an RTX 4070 or a same-compute bundle for the exact row.

### RTX 5060 exact

Use the actual checked-in profile name until a separate generic target is added:

```bash
export TARGET_PROFILE=configs/targets/rtx5060-laptop-ubuntu2404.json
```

Detect and record the GPU identity before accepting the row. The engine set must be built for that target and TensorRT runtime.

### Jetson Orin Nano exact

```bash
export TARGET_PROFILE=configs/targets/orin-nano-l4t3647.json
export NVCR_ENGINE_ROOT=/path/to/orin-engines
```

Run only exact target-local bundles. Record JetPack/L4T, power mode, clocks, CUDA, and TensorRT.

## 7. Same-compute rows

Build explicitly on a desktop target with the detected SM class:

```bash
./scripts/nvcr_artifacts.py build \
  --profile 720p \
  --target-profile "$TARGET_PROFILE" \
  --hardware-compatibility same_compute_capability
```

Package with the resulting `linux-amd64-sm<CC>` name, validate the bundle, and run correctness plus a short sanity matrix on a representative GPU with the same SM. This is a separate row from exact evidence.

## 8. Ampere-plus rows

Desktop/discrete targets only:

```bash
./scripts/nvcr_artifacts.py build \
  --profile 720p \
  --target-profile "$TARGET_PROFILE" \
  --hardware-compatibility ampere_plus
```

Record correctness, PSNR sanity, payload size, throughput, and the ratio against the exact result on the same target. Do not use Ampere-plus on Jetson or as primary performance evidence.

## 9. Python comparison

On RTX 4070 exact, run the pinned Python DCVC-RT reference with the same source, resolution, frame count, QP, and GOP policy where possible. Record its command and environment beside the NVCR row. Keep payload and reconstruction comparisons separate; non-bit-exact output is not automatically a failure.

## 10. Package the result

Before calling the run complete, assemble the layout in [softwarex-evaluation-protocol.md](softwarex-evaluation-protocol.md), validate every JSON/JSONL file, and write `summary.md` with pass/fail/skipped counts. Do not add plans, checkpoints, raw YUV, streams, or reconstructed video to the repository.
