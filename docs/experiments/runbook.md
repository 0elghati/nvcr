# SoftwareX runbook

Run from a clean repository root on the machine named by the target profile.
Do not substitute another GPU, invent an input, or reuse an engine after any
bound profile changes.

The publication entry point is:

```bash
python3 scripts/benchmark_softwarex_matrix.py --help
```

By default the driver runs only its low-overhead performance repetitions. Add
`--profile` for publication runs: the driver then collects verbose latency,
PSNR, and memory in separate repetitions after all primary timing. Omitting the
flag is useful while tuning, but the resulting package remains `partial`.

`scripts/benchmark_resolution_matrix.sh` remains a lower-level diagnostic. Its
rows are not SoftwareX evidence.

## 1. Supply local inputs

Start from [softwarex-inputs.example.json](softwarex-inputs.example.json) and
create a local manifest using schema `nvcr.softwarex.inputs.v1`. Every entry
must identify a real planar YUV420P8 file, dimensions, frame rate, fixed frame
count, profile, sequence ID, and redistribution status.

An exact package needs at least one sequence for `qcif`, `cif`, `360p`, `720p`,
and `1080p`. `540p` is optional. Keep the manifest and raw YUV outside Git when
the source license requires it.

```bash
export NVCR_SOFTWAREX_INPUTS=/data/nvcr/softwarex-inputs.json
python3 -m json.tool "$NVCR_SOFTWAREX_INPUTS"
python3 scripts/nvcr_device.py > /tmp/nvcr-device.json
git rev-parse HEAD
git status --short
```

A publication run requires a clean commit. A dirty run is retained as
`partial`, even when its cases pass.

## 2. Prepare exact artifacts

Build plans on the final target. This example exports the pinned model once and
builds all six registered profiles:

```bash
export TARGET_PROFILE=configs/targets/rtx4070-ubuntu2404.json
export ENGINE_ROOT="$PWD/build/engines-rtx4070-exact"
export TRTEXEC=/usr/src/tensorrt/bin/trtexec

./scripts/nvcr_artifacts.py prepare --all \
  --target-profile "$TARGET_PROFILE" \
  --engines-root "$ENGINE_ROOT" \
  --hardware-compatibility exact \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --trtexec "$TRTEXEC" \
  --python /path/to/DCVC-RT/src/venv/bin/python3

for profile in qcif cif 360p 540p 720p 1080p; do
  ./scripts/nvcr_artifacts.py validate \
    "$ENGINE_ROOT/dcvcrt-$profile" --json
done
```

The target profile, current model profile, every engine profile, device
identity, CUDA/TensorRT runtime, manifest, and checksum bytes must agree. A
successful TensorRT deserialization alone is insufficient.

## 3. Configure the registered gates

The driver requires a Release tree whose CTest registry contains an engine
contract and native I/P roundtrip test for every selected profile:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIRS="$ENGINE_ROOT/dcvcrt-qcif;$ENGINE_ROOT/dcvcrt-cif;$ENGINE_ROOT/dcvcrt-360p;$ENGINE_ROOT/dcvcrt-720p;$ENGINE_ROOT/dcvcrt-1080p"

cmake --build build-release --parallel
ctest --test-dir build-release -N
```

Add the `540p` directory to `NVCR_TENSORRT_ENGINE_DIRS` when that optional
profile is selected.

## 4. CPU container validation

This is reviewer-accessible contract evidence, not GPU or publication evidence:

```bash
docker build --platform linux/amd64 --target test \
  -f docker/Dockerfile.x86_64 -t nvcr-test:cpu .
docker run --rm nvcr-test:cpu cpu
```

Record it as: CPU/core tests passed; TensorRT engine tests, GPU roundtrips, and
the publication matrix were not run.

## 5. RTX 4070 exact

The current checkout is blocked before this command can complete because the
local 2026-08-05 bundles predate the current model-profile digest. The checked-in
target and detected host now use TensorRT 10.9. First rebuild every bundle.

Use a fresh directory for the strict preflight:

```bash
python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir /tmp/nvcr-softwarex-rtx4070-preflight \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --engine-root "$ENGINE_ROOT" \
  --profiles qcif cif 360p 720p 1080p \
  --qps 32 --gops 1 normal \
  --profile --profile-runs 1 \
  --native-build-id rtx4070-ubuntu2404-release \
  --plan-only
```

After preflight passes, run into a different empty directory. Matching pinned
Python rows are mandatory for every requested case:

```bash
python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-rtx4070" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release \
  --nvcr build-release/cli/nvcr \
  --profiles qcif cif 360p 720p 1080p \
  --qps 32 --gops 1 normal \
  --warmup-runs 1 --measured-runs 3 \
  --profile --profile-runs 1 \
  --python-reference-command '/path/to/pinned-reference-wrapper --output /data/nvcr/python-reference-results.jsonl' \
  --python-reference-jsonl /data/nvcr/python-reference-results.jsonl \
  --native-build-id rtx4070-ubuntu2404-release
```

## 6. RTX 3050 exact

This is blocked until
`configs/targets/rtx3050-laptop-ubuntu2404.json` is created from the actual
machine and exact bundles are rebuilt. Once present:

Configure `build-release-rtx3050` with those five engine directories as in
step 3, then run:

```bash
export ENGINE_ROOT="$PWD/build/engines-rtx3050-exact"
python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-rtx3050" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile configs/targets/rtx3050-laptop-ubuntu2404.json \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release-rtx3050 \
  --nvcr build-release-rtx3050/cli/nvcr \
  --profiles qcif cif 360p 720p 1080p \
  --qps 32 --gops 1 normal \
  --warmup-runs 1 --measured-runs 3 \
  --profile --profile-runs 1 \
  --native-build-id rtx3050-laptop-ubuntu2404-release
```

Do not use an RTX 4070 exact or same-compute bundle for this row.

## 7. RTX 5060 exact

The checked-in profile names the actual Laptop target. Detect the GPU and add a
different profile if the evaluated RTX 5060 is not that device:

Configure `build-release-rtx5060` with the target's five exact engine
directories as in step 3, then run:

```bash
export ENGINE_ROOT="$PWD/build/engines-rtx5060-laptop-exact"
python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-rtx5060" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile configs/targets/rtx5060-laptop-ubuntu2404.json \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release-rtx5060 \
  --nvcr build-release-rtx5060/cli/nvcr \
  --profiles qcif cif 360p 720p 1080p \
  --qps 32 --gops 1 normal \
  --warmup-runs 1 --measured-runs 3 \
  --profile --profile-runs 1 \
  --native-build-id rtx5060-laptop-ubuntu2404-release
```

## 8. Jetson Orin Nano exact

Run natively on the Orin target. QEMU and cross-platform image construction are
not evidence:

Configure `build-release-jetson` with the Orin exact engine directories as in
step 3, then run:

```bash
export ENGINE_ROOT="$PWD/build/engines-orin-nano-exact"
python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-orin" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile configs/targets/orin-nano-l4t3647.json \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release-jetson \
  --nvcr build-release-jetson/cli/nvcr \
  --profiles qcif cif 360p 720p 1080p \
  --qps 32 --gops 1 normal \
  --warmup-runs 1 --measured-runs 3 \
  --profile --profile-runs 1 \
  --native-build-id orin-nano-l4t3647-release
```

Jetson is exact-only. The current driver samples discrete-GPU memory through
`nvidia-smi`; a target-local Jetson GPU-memory sampler must be supplied before a
Jetson package with unavailable memory can become complete.

## 9. Same-compute validation

Build and test on a named desktop target whose detected SM is the intended
class. This RTX 4070 example creates SM 8.9-class bundles:

```bash
export TARGET_PROFILE=configs/targets/rtx4070-ubuntu2404.json
export ENGINE_ROOT="$PWD/build/engines-linux-amd64-sm89"

./scripts/nvcr_artifacts.py build --all \
  --target-profile "$TARGET_PROFILE" \
  --engines-root "$ENGINE_ROOT" \
  --hardware-compatibility same_compute_capability \
  --models build/models/dcvcrt \
  --trtexec "$TRTEXEC"

python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-sm89" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile "$TARGET_PROFILE" \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release-sm89 \
  --nvcr build-release-sm89/cli/nvcr \
  --profiles 720p \
  --qps 32 --gops 1 normal \
  --profile --profile-runs 1 \
  --compatibility-class same_compute_capability \
  --exact-baseline-jsonl /path/to/rtx4070-exact-results.jsonl \
  --native-build-id linux-amd64-sm89-release
```

Configure `build-release-sm89` with the selected same-compute directories as in
step 3. Repeat on a representative GPU for every claimed SM class.

## 10. Ampere-plus validation

Ampere-plus is a desktop fallback, never a Jetson mode:

```bash
export TARGET_PROFILE=configs/targets/rtx4070-ubuntu2404.json
export ENGINE_ROOT="$PWD/build/engines-linux-amd64-ampere-plus"

./scripts/nvcr_artifacts.py build --all \
  --target-profile "$TARGET_PROFILE" \
  --engines-root "$ENGINE_ROOT" \
  --hardware-compatibility ampere_plus \
  --models build/models/dcvcrt \
  --trtexec "$TRTEXEC"

python3 scripts/benchmark_softwarex_matrix.py \
  --output-dir "evidence/softwarex-$(date +%Y%m%d)-$(git rev-parse --short HEAD)-ampere-plus" \
  --inputs "$NVCR_SOFTWAREX_INPUTS" \
  --target-profile "$TARGET_PROFILE" \
  --engine-root "$ENGINE_ROOT" \
  --build-dir build-release-ampere-plus \
  --nvcr build-release-ampere-plus/cli/nvcr \
  --profiles 720p \
  --qps 32 --gops 1 normal \
  --profile --profile-runs 1 \
  --compatibility-class ampere_plus \
  --exact-baseline-jsonl /path/to/rtx4070-exact-results.jsonl \
  --native-build-id linux-amd64-ampere-plus-release
```

Configure the matching Release tree first. Repeat correctness and ratio
measurement on every desktop target where the fallback is claimed.

## 11. Containers and identity

The x86_64 and Jetson test services validate one selected engine profile:

```bash
docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install
NVCR_TEST_ENGINE_PROFILE=qcif \
  docker compose -f docker/compose.x86_64.yaml run --rm --build test gpu
```

Use `docker/compose.jetson.yaml` on the native Jetson. For a driver run inside a
container, pass both `--container-image` and its immutable
`--container-digest`; omit `--native-build-id`. The source/test environment
must mount inputs and engines read-only and evidence output writable. Container
userspace must match the engine TensorRT identity.

## 12. Accept the package

Open `run-summary.json`, `test-summary.json`, `failures.jsonl`, and
`summary.md`. Only package status `complete` is publication-ready. The driver
requires:

- current validated artifacts and target identity;
- a recorded native build ID or immutable container identity;
- source build, core tests, GPU registration, engine tests, and native I/P
  roundtrips;
- the complete exact profile/QP/GOP definition;
- `--profile` enabled, with profile metrics collected in repetitions separate
  from primary timing;
- finite throughput, latency, payload, BPP, PSNR, host memory, and GPU memory;
- matching Python rows for every RTX 4070 exact case;
- matching exact baseline rows for compatibility-class cases;
- no failed/skipped cases and a clean Git tree.

Omitting `--profile` or using `--skip-build`, `--skip-core-tests`,
`--skip-gpu-tests`, `--plan-only`, or `--allow-partial` is diagnostic. These
controls cannot turn missing evidence into a complete package.

Do not commit raw YUV, plans, checkpoints, ONNX files, streams, or
reconstructions.
