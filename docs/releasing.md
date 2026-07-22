# Releasing

NVCR uses release-please for versioning, changelog generation, tags, and GitHub
Releases. Binary and engine bundle assets are uploaded by a separate workflow.

## Overview

1. Merge conventional commits into `main`.
2. Release Please opens or updates a release pull request.
3. Merge the release pull request.
4. Release Please creates the tag, updates [CHANGELOG.md](../CHANGELOG.md), and
   publishes the GitHub Release.
5. The release asset workflow builds archives on self-hosted runners and uploads
   them to that GitHub Release.

## Repository files

Release automation is defined by:

- [release-please-config.json](../release-please-config.json)
- [.release-please-manifest.json](../.release-please-manifest.json)
- [.github/workflows/release-please.yml](../.github/workflows/release-please.yml)
- [.github/workflows/release-assets.yml](../.github/workflows/release-assets.yml)
- [scripts/package_release.sh](../scripts/package_release.sh)

## Required repository settings

Set these once in GitHub:

1. Enable GitHub Actions with permission to create and approve pull requests.
2. Add a `RELEASE_PLEASE_TOKEN` secret if you want release-created workflows to
   trigger other workflows such as release asset publishing.
3. Provide self-hosted runners with these labels when you want binary assets:
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

### Upload binary and engine assets

The asset workflow runs automatically on `release.published`. It can also be
run manually from GitHub Actions with a tag input.

Produced archives use this naming:

```text
nvcr-vX.Y.Z-linux-x86_64-discrete.tar.gz
nvcr-vX.Y.Z-linux-aarch64-jetson.tar.gz
dcvcrt-engines-vX.Y.Z-linux-x86_64-discrete.tar.gz
dcvcrt-engines-vX.Y.Z-linux-aarch64-jetson.tar.gz
SHA256SUMS
```

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