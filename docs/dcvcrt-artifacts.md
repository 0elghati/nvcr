# DCVC-RT model and TensorRT engine artifacts

NVCR's DCVC-RT integration uses model assets prepared from a fixed upstream
revision and TensorRT plans built for a recorded target. The generic NVCR
packages do not contain checkpoints, ONNX graphs, entropy assets, or TensorRT
plans.

Runtime commands do not download, export, or build artifacts. Install a
compatible engine bundle before encoding, or prepare one locally from a source
checkout.

## Install from the engine catalog

An installed NVCR package provides `nvcr-artifacts`. The engine catalog uses
anonymous access:

```bash
nvcr-artifacts install \
  --profile qcif \
  --backend dcvcrt \
  --device-id 0 \
  --repo 0elghati/nvcr \
  --asset-release engine-assets \
  --engine-root "${NVCR_ENGINE_ROOT:-$HOME/.local/share/nvcr/engines}"
```

`--profile` accepts one or more values and may be repeated. When it is
omitted, the client installs every compatible profile it finds.

For each profile, catalog selection is automatic:

1. exact target;
2. same numeric compute capability; then
3. Ampere-and-newer desktop compatibility.

The client uses a broader class only when the catalog contains an entry that
passes its checks. The Ampere-and-newer build mode existing in the source does
not imply that an Ampere-and-newer bundle is available for download.

Exact matching includes the operating system, architecture, GPU identity,
numeric compute capability, multiprocessor count, and runtime constraints.
Same-compute-capability matching is desktop x86_64 only and compares both
compute-capability components: SM 8.9 matches SM 8.9, not SM 8.6 or SM 12.0.
Ampere-and-newer is also desktop-only and requires compute-capability major
version 8 or newer. Jetson and other AArch64 targets are exact-only.

All three classes require the exact TensorRT `major.minor.patch` version
recorded in the engine manifest. A broader hardware class does not repair a
TensorRT mismatch. Desktop CUDA compatibility permits an engine built with the
same CUDA major version when its recorded runtime is not newer than the active
runtime; Jetson requires the recorded CUDA runtime exactly.

The installer validates catalog and archive metadata, SHA-256 digests, the
engine manifest, and the bundle file inventory. It then creates:

```text
<engine-root>/bundles/dcvcrt/<target>/<profile>/<archive-sha256>/
<engine-root>/profiles/dcvcrt/<profile> -> immutable bundle
<engine-root>/profiles/default -> dcvcrt
```

Inspect and validate the installed bundle before encoding:

```bash
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
nvcr-artifacts inspect \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
nvcr-artifacts validate \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
```

`inspect` reports manifest identity without executing a plan. Successful
encoding is the execution check. If no catalog entry matches, installation
stops; it never starts an engine build in the background. The public catalog
needs no authentication, and the current client does not send
`GH_TOKEN` or `GITHUB_TOKEN`.

## Bound source and checkpoints

The supported model profile is
[configs/models/dcvcrt-cvpr2025.json](../configs/models/dcvcrt-cvpr2025.json):

```text
Source repository: https://github.com/microsoft/DCVC.git
Source commit:     1feb52a592a9ff2c4e4ba2e5122e2da49a211466
Image checkpoint:  cvpr2025_image.pth.tar
SHA-256:           555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9
Video checkpoint:  cvpr2025_video.pth.tar
SHA-256:           b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6
```

Checkpoint distribution is controlled by the model provider. Obtain the named
files through the
[bound upstream source](https://github.com/microsoft/DCVC/tree/1feb52a592a9ff2c4e4ba2e5122e2da49a211466)
and place them under `<dcvcrt-root>/checkpoints/`. The exporter verifies the
source commit and checkpoint hashes; filenames alone are insufficient.

Preparation is offline. Deployed NVCR does not load Python, PyTorch, source
checkpoints, or ONNX graphs.

## Prepare an exact bundle locally

Local preparation is an advanced workflow and must be run from an NVCR source
checkout. It requires Python with PyTorch, ONNX, and ONNXScript, the bound
DCVC-RT checkout and checkpoints, TensorRT's `trtexec`, and writable model
and engine directories.

First record the live target:

```bash
python3 scripts/nvcr_device.py
```

Then prepare one profile:

```bash
./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 1080p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --hardware-compatibility exact \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines-root build/engines-rtx4070-exact \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --python /path/to/python3
```

The selected target JSON must describe the machine performing the TensorRT
build. Supplying `--target-profile` is an assertion, not hardware emulation;
the build command does not independently prove that the JSON matches the live
host. Compare it with `scripts/nvcr_device.py` before building. Do not select a
different GPU profile to obtain an attractive bundle name.

Use `--all` instead of `--profile 1080p` to build all registered profiles.
The preparation command verifies or obtains the bound source, exports the model
bundle once, and builds the requested plans. Use `--skip-clone` only for an
already verified DCVC-RT checkout and `--skip-engine` only when a portable
model export is the intended result.

## Build from an existing model bundle

Transfer a validated model bundle to the machine that will build the plans,
then run:

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json

./scripts/nvcr_artifacts.py build \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 1080p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --hardware-compatibility exact \
  --models build/models/dcvcrt \
  --engines-root build/engines-rtx4070-exact \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device-id 0
```

The builder validates the model bundle before creating the first plan. Missing
graphs, stale manifests, and digest mismatches are hard failures.

### Broader desktop hardware classes

The same source-checkout command can request one of the two desktop
compatibility modes:

```bash
./scripts/nvcr_artifacts.py build \
  --profile 1080p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --hardware-compatibility same_compute_capability \
  --models build/models/dcvcrt \
  --engines-root build/engines-linux-amd64-sm89 \
  --trtexec /usr/src/tensorrt/bin/trtexec

./scripts/nvcr_artifacts.py build \
  --profile 1080p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --hardware-compatibility ampere_plus \
  --models build/models/dcvcrt \
  --engines-root build/engines-linux-amd64-ampere-plus \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

The target JSON must still match the build host. These options pass TensorRT's
hardware-compatibility mode to every plan and record the selected class in the
manifest. They do not relax the exact TensorRT version requirement. Validate
correctness and performance against an exact bundle before using either class.
Do not use either mode on Jetson.

## Registered FP16 profiles

| Profile | Visible dimensions | Workspace | Builder level |
|---|---|---:|---:|
| `qcif` | 64x64 to 176x144 | 512 MiB | 1 |
| `cif` | 64x64 to 352x288 | 512 MiB | 1 |
| `360p` | fixed 640x360 | 1024 MiB | 4 |
| `540p` | fixed 960x540 | 1024 MiB | 4 |
| `720p` | 64x64 to 1280x720 | 1024 MiB | 2 |
| `1080p` | 64x64 to 1920x1080 | 1024 MiB | 2 |

Internal graph shapes may include codec padding. Manifests record visible
dimensions, and the TensorRT plans record their actual optimization shapes.
FP16 is the supported precision.

## Validate and register integration tests

Validate every completed engine bundle:

```bash
./scripts/nvcr_artifacts.py inspect \
  build/engines-rtx4070-exact/dcvcrt-1080p --json
./scripts/nvcr_artifacts.py validate \
  build/engines-rtx4070-exact/dcvcrt-1080p --json
```

An engine bundle records model, target, engine-profile, CUDA, TensorRT, GPU,
hardware-compatibility, shape, and file-digest identity. A complete bundle
contains the required I/P plans and runtime assets. Validation rejects missing,
extra, modified, stale, or cross-model files.

Register its engine-contract and I/P round-trip tests in a Release build:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$PWD/build/engines-rtx4070-exact/dcvcrt-1080p"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Those integration tests are not registered when no engine directory is given.
A successful plan build or deserialization alone is not an end-to-end
round-trip result.
