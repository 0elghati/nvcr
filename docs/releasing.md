# Release gates and packaging

NVCR uses Release Please for version/changelog/tag creation, but hardware evidence
controls publication. Release Please creates a tagged **draft** release; the asset
workflow publishes it only after the required target gates succeed.

## Release tracks

| Release | Required target gates | Meaning |
|---|---|---|
| v0.3 foundation | RTX 4070 | Scoped development foundation, not v1 product support |
| v1.0 (and later v1 patches) | RTX 4070 and Jetson Orin Nano | Both complete clean-room matrices are mandatory before v1.0 publication |

A failed or skipped mandatory gate leaves the GitHub release as a draft. v1 cannot
be published by disabling the Jetson input: the workflow forces the Orin job for
all `v1.*` tags.

## Exact-tag workflow

[release-please.yml](../.github/workflows/release-please.yml) updates versions and
changelog through its release PR. After that PR is merged, Release Please creates
a forced tag and draft release, then dispatches
[release-assets.yml](../.github/workflows/release-assets.yml) with that tag.

Each target job:

1. checks out the requested tag with full tag history;
2. proves `HEAD`, the tag commit, and `version.txt` agree;
3. starts with new model/engine/build/install directories;
4. runs `nvcr-artifacts prepare` with the declared model, FP16 1080p engine, and
   target profile;
5. validates the model and engine bundles cryptographically, including the
   model/target/engine-profile file digests and every generated engine hash;
6. configures a Release build with `NVCR_TENSORRT_ENGINE_DIR`, so engine and I/P
   tests are registered;
7. runs the complete configured CTest suite;
8. installs and packages code, headers, artifact tooling, versioned profiles,
   configuration example, docs, roadmap, changelog, licenses, and notices;
9. uploads the archive and checksum to the draft release.

The final publication job changes `draft=false` only after its required job results
are successful.

## Required self-hosted runners and variables

Runner labels:

```text
nvcr-release-discrete   # the recorded RTX 4070 target
nvcr-release-jetson     # the recorded Orin Nano target
```

Repository variables:

```text
NVCR_DCVCRT_ROOT_DISCRETE
NVCR_DCVCRT_PYTHON_DISCRETE
NVCR_DCVCRT_ROOT_JETSON
NVCR_DCVCRT_PYTHON_JETSON
```

Each root must be the pinned clean DCVC-RT checkout with the two exact checkpoint
files. Each Python path must be executable and provide PyTorch, ONNX, and
ONNXScript. The runner itself supplies the target profile's CUDA, TensorRT,
compiler, driver/JetPack, device, and direct device access.

For a pre-v1 release, `publish_jetson` is optional. If requested, its failure also
keeps the foundation release in draft. It is always mandatory for v1.

## Package policy

Produced archives are target-specific NVCR packages:

```text
nvcr-vX.Y.Z-linux-x86_64-rtx4070.tar.gz
nvcr-vX.Y.Z-linux-x86_64-rtx4070.tar.gz.sha256
nvcr-vX.Y.Z-linux-aarch64-orin-nano.tar.gz
nvcr-vX.Y.Z-linux-aarch64-orin-nano.tar.gz.sha256
```

Every archive has a top-level versioned directory and an internal
`PACKAGE-MANIFEST.sha256`. `scripts/package_release.sh` checks for required docs,
profiles, license/notice, CLI, and artifact tooling before packaging.

The package script rejects checkpoints, `.pth`/`.pth.tar`, ONNX, entropy/quant
model assets, TensorRT plans, and engine bundles. Generated bundles are release
test inputs, never release outputs. Redistribution requires a separate explicit
rights and technical-portability decision.

## Continuous integration before release

Pull-request CI runs shell/Python/JSON validation and CPU Debug/Release CTest on
x86_64/aarch64 hosted runners. This includes unit, packet/access-unit boundaries,
deterministic parser fuzz, rANS, and manifest/profile tests. CUDA/TensorRT is also
compile-checked on a hosted runner.

A separate trusted `push: main` workflow runs the CPU/parser/manifest/rANS/CUDA
suite on the self-hosted RTX runner. Fork pull requests never execute on
self-hosted hardware. Complete engine generation and I/P integration remain the
release target gates because they depend on locally held checkpoints and long
engine builds.

## Manual dispatch and local packaging test

A maintainer can dispatch a tagged draft build:

```bash
gh workflow run release-assets.yml \
  --ref main \
  -f tag=v0.3.0 \
  -f publish_jetson=false
```

Local package validation operates on an already-installed tree and never accepts
an engine argument:

```bash
./scripts/package_release.sh \
  --version 0.3.0 \
  --platform linux-x86_64-rtx4070 \
  --install-prefix /path/to/install \
  --output-dir dist

(cd dist && sha256sum -c nvcr-v0.3.0-linux-x86_64-rtx4070.tar.gz.sha256)
```

Performance, rate/distortion, memory, and Orin energy evidence are recorded in
[ROADMAP.md](../ROADMAP.md) using [Performance](performance.md). Passing packaging
alone never completes a release milestone.
