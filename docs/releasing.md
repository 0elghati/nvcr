# Releasing

NVCR uses release-please for versioning, changelog generation, tags, and GitHub
Releases. Binary and engine bundle assets are uploaded by a separate workflow.
A third workflow validates every pull request before it can be merged.

## Overview

1. Open a pull request against `main`. The CI workflow builds and tests it.
2. Merge conventional commits into `main`.
3. Release Please opens or updates a release pull request.
4. Merge the release pull request.
5. Release Please creates the tag, updates [CHANGELOG.md](../CHANGELOG.md), and
   publishes the GitHub Release, then explicitly dispatches the release asset
   workflow for that tag (see "Why release assets are dispatched, not
   triggered" below).
6. The release asset workflow builds archives and uploads them to that GitHub
   Release: a portable x86_64 build on a GitHub-hosted runner, plus discrete
   and Jetson builds on self-hosted runners.

## Continuous integration (pull requests)

[.github/workflows/ci.yml](../.github/workflows/ci.yml) runs on every pull
request, using only GitHub-hosted runners:

- `changes`: uses `dorny/paths-filter` to check whether the push/PR touched
  anything that affects the compiled output (`CMakeLists.txt`, `cmake/`,
  `src/`, `include/`, `cli/`, `tests/`, `benchmarks/`, `examples/`,
  `third_party/`, `tools/`, `scripts/`, or the CI workflow file itself). A
  docs-only or `README`/`ROADMAP`-only change reports no match.
- `lint`: shell syntax (`bash -n`) and `shellcheck` on everything under
  `scripts/`, Python syntax check on `scripts/*.py`, and JSON validation of
  the release-please config files. Always runs, regardless of what changed.
- `build-cpu`: a matrix of `{ubuntu-latest, ubuntu-24.04-arm}` x
  `{Debug, Release}` with `-DNVCR_ENABLE_TENSORRT=OFF`, running the full
  CTest suite. This is the hard gate for both host CPU architectures NVCR
  ships for (x86_64 and Jetson's aarch64), without needing GPU hardware.
  Skipped (reported as passing, not failing) when `changes` found nothing
  under the paths above, so a docs-only PR doesn't pay for a full rebuild.
- `build-cuda-portable`: installs the CUDA toolkit and TensorRT development
  packages on an `ubuntu-latest` runner (see
  [scripts/ci/install_cuda_tensorrt.sh](../scripts/ci/install_cuda_tensorrt.sh))
  and configures/builds with `-DNVCR_ENABLE_TENSORRT=ON
  -DNVCR_CUDA_ARCH_SET=portable`. There is no GPU on this runner, so it is a
  compile-only check across every GPU architecture in the portable matrix; it
  cannot run CUDA-executing tests, and it does not block merges
  (`continue-on-error: true`) because NVIDIA's public apt package names and
  versions can change independently of this repository. Also skipped when
  `changes` found nothing under the paths above.

Self-hosted GPU runners (`nvcr-release-discrete`, `nvcr-release-jetson`) are
intentionally never triggered by `pull_request`: a fork could otherwise open a
PR that runs arbitrary code on your hardware. Real GPU execution (CUDA ops
tests, TensorRT engine round-trips) only runs in the release asset workflow or
manually by a maintainer.

## Why release assets are dispatched, not triggered

[.github/workflows/release-assets.yml](../.github/workflows/release-assets.yml)
only listens for `workflow_dispatch`, not `on: release`. That is deliberate:
release-please creates the GitHub Release through the GitHub API using its own
job's token, and GitHub does not let events created that way (`release:
published`, `pull_request` opened, etc.) start further workflow runs, to
prevent recursive automation loops. A repository-wide `RELEASE_PLEASE_TOKEN`
personal access token would work around that, but adds a credential to
manage and rotate. Instead,
[.github/workflows/release-please.yml](../.github/workflows/release-please.yml)
explicitly calls `gh workflow run release-assets.yml -f tag=<tag>` right after
creating a release. `workflow_dispatch` is exempted from the
GITHUB_TOKEN-cascade restriction, so this reliably starts the asset build with
no extra secret required.

If a release's assets are ever missing (for example, the very first release
created before this mechanism existed, or a run that failed for another
reason), rerun it manually. Using the `gh` CLI:

```bash
gh workflow run release-assets.yml --ref main -f tag=v0.1.0
```

or from the GitHub UI: Actions -> release-assets -> Run workflow, and enter
the tag.

## Repository files

Release automation is defined by:

- [release-please-config.json](../release-please-config.json)
- [.release-please-manifest.json](../.release-please-manifest.json)
- [.github/workflows/release-please.yml](../.github/workflows/release-please.yml)
- [.github/workflows/release-assets.yml](../.github/workflows/release-assets.yml)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml)
- [scripts/package_release.sh](../scripts/package_release.sh)
- [scripts/ci/install_cuda_tensorrt.sh](../scripts/ci/install_cuda_tensorrt.sh)

## Required repository settings

Set these once in GitHub:

1. Enable GitHub Actions with permission to create and approve pull requests.
2. Optionally add a `RELEASE_PLEASE_TOKEN` secret (a personal access token or
   GitHub App installation token) if you also want CI to run automatically on
   the release-please pull request itself. This is not required for release
   asset publishing, which uses the `workflow_dispatch` chain described above
   regardless of which token release-please uses.
3. Provide self-hosted runners with these labels when you want discrete or
   Jetson binary assets (the portable release build and all CI checks run on
   GitHub-hosted runners and need no setup):
   - `nvcr-release-discrete`
   - `nvcr-release-jetson`

## Optional repository variables

Set these when a runner should also build and upload an engine bundle:

- `NVCR_RELEASE_DISCRETE=true`
- `NVCR_DCVCRT_ROOT_DISCRETE`
- `NVCR_DCVCRT_PYTHON_DISCRETE`
- `NVCR_RELEASE_JETSON=true`
- `NVCR_DCVCRT_ROOT_JETSON`
- `NVCR_DCVCRT_PYTHON_JETSON`

Each `NVCR_DCVCRT_ROOT_*` value must point to a DCVC-RT checkout containing:

```text
<dcvcrt-root>/checkpoints/cvpr2025_image.pth.tar
<dcvcrt-root>/checkpoints/cvpr2025_video.pth.tar
```

Each `NVCR_DCVCRT_PYTHON_*` value must point to a Python interpreter with
`torch`, `onnx`, and `onnxscript` installed.

## Release flow

### Cut a release

Push conventional commits to `main`. Release Please updates or opens a release
pull request automatically.

When the release pull request is merged:

- the repository version is bumped;
- [CHANGELOG.md](../CHANGELOG.md) is updated;
- a `vX.Y.Z` tag is created;
- a GitHub Release is published.

The current configuration uses a one-time `release-as: 0.1.0` bootstrap to cut
the first tagged release from the existing history. Remove that override after
the first release PR is merged so later releases return to normal
conventional-commit versioning.

### Upload binary and engine assets

The asset workflow runs automatically on `release.published`. It can also be
run manually from GitHub Actions with a tag input.

Produced archives use this naming:

```text
nvcr-vX.Y.Z-linux-x86_64-portable.tar.gz
nvcr-vX.Y.Z-linux-x86_64-discrete.tar.gz
nvcr-vX.Y.Z-linux-aarch64-jetson.tar.gz
dcvcrt-engines-vX.Y.Z-linux-x86_64-discrete.tar.gz
dcvcrt-engines-vX.Y.Z-linux-aarch64-jetson.tar.gz
SHA256SUMS
```

The portable archive covers every GPU architecture in
`NVCR_CUDA_ARCH_SET=portable` in one x86_64 build, runs on a GitHub-hosted
runner, and ships no engine bundle: pair it with a matching engine bundle or
build engines locally (see [dcvcrt-artifacts.md](dcvcrt-artifacts.md)). The
discrete and Jetson archives are built on self-hosted runners for their
specific target GPU and can optionally include a matching engine bundle when
checkpoints are configured.

If a runner does not have checkpoints configured, the workflow still uploads the
binary archive and checksum file. Engine bundles remain optional and target-local.

## Local packaging check

To package an already-installed tree locally:

```bash
./scripts/package_release.sh \
  --version 0.1.0 \
  --platform linux-x86_64-discrete \
  --install-prefix /path/to/install-prefix \
  --engine-dir /path/to/engines/dcvcrt \
  --output-dir dist
```