# Release gates and packaging

NVCR uses Release Please for version/changelog/tag creation, but publication still
requires external target evidence. With the repository's default GitHub-hosted
runner setup, GitHub Actions can build and upload the generic x86_64 package
family. It does not automate the RTX 4070 / Jetson Orin Nano validation gates or
the Jetson package deliverable.

## Release tracks

| Release | Required target gates before publication | Meaning |
|---|---|---|
| v0.3 foundation | RTX 4070 clean-room matrix recorded in the roadmap | Scoped development foundation, not v1 product support |
| v1.0 (and later v1 patches) | RTX 4070 and Jetson Orin Nano clean-room matrices recorded in the roadmap | First supported product release |

A tagged GitHub release should remain a draft until the required target evidence,
manual deliverables, and roadmap updates are complete.

## Draft release workflow on standard GitHub-hosted runners

[release-please.yml](../.github/workflows/release-please.yml) updates versions and
changelog through its release PR. After that PR is merged, Release Please creates
a tagged draft release and dispatches
[release-assets.yml](../.github/workflows/release-assets.yml).

On GitHub's standard hosted runners, that workflow currently:

1. checks out the exact requested tag and verifies it matches `version.txt`;
2. installs minimal CUDA/TensorRT compile dependencies on hosted Ubuntu 24.04 x86_64;
3. configures a portable TensorRT-enabled Release build;
4. runs the hosted-safe test subset (artifact/profile, smoke/access-unit,
   parser-fuzz boundaries, and rANS);
5. installs and packages the generic `linux-x86_64-nvidia` archive;
6. verifies the archive checksum, manifest presence, and forbidden-asset policy;
7. uploads the x86_64 archive and checksum to the draft release.

The workflow intentionally does not publish the release. Standard hosted runners do
not provide the reference GPU/Jetson environments needed for the target-local
engine builds, full registered GPU suite, Jetson package build, or release-track
support evidence.

## What GitHub-hosted runners do not prove

Standard hosted runners do not satisfy any of the following by themselves:

- exact target-local engine generation on the recorded RTX 4070 reference target;
- exact target-local engine generation on the recorded Jetson Orin Nano target;
- full registered CTest with `NVCR_TENSORRT_ENGINE_DIR` on those targets;
- the `linux-aarch64-jetson-l4t36` package build and checksum;
- correctness, performance, memory, bitrate/distortion, or Jetson energy evidence;
- final support/publication approval for v0.3 or v1.x.

## Manual deliverables before publication

Before publishing a draft release, complete and record the following:

1. run the exact-tag clean-room workflow on the recorded RTX 4070 target;
2. run the exact-tag clean-room workflow on the recorded Jetson Orin Nano target
   when required by the release track;
3. build and validate the Jetson archive on the validated Jetson target;
4. upload the Jetson archive and checksum to the draft release;
5. optionally upload package-family reviewer-convenience engine assets as
   separate GitHub Release assets;
6. record all required evidence in [ROADMAP.md](../ROADMAP.md);
7. publish the draft release manually only after the release-track gates pass.

## Package policy

Produced archives are public package families, not hardware-support claims:

```text
nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz
nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz.sha256
nvcr-vX.Y.Z-linux-aarch64-jetson-l4t36.tar.gz
nvcr-vX.Y.Z-linux-aarch64-jetson-l4t36.tar.gz.sha256
```

These names distinguish the public deliverable family from the exact validated
reference targets. The current support claim still depends on the recorded RTX 4070
and Orin Nano target profiles and their roadmap evidence.

Every archive has a top-level versioned directory and an internal
`PACKAGE-MANIFEST.sha256`. `scripts/package_release.sh` checks for required docs,
profiles, license/notice, CLI, the generic artifact entry point, and backend-local
DCVC-RT artifact helpers before packaging.

The package script rejects checkpoints, `.pth`/`.pth.tar`, ONNX, entropy/quant
model assets, TensorRT plans, and engine bundles. Generated bundles are release
test inputs, never release outputs.

## Optional engine assets

Public binary packages stay engine-free. If reviewers need prebuilt engines, ship
them as separate package-family GitHub Release assets so downloads still come
from the release page rather than a staging service.

Create each archive on the machine where the engine bundle was built and
validated:

```bash
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-v2 --json
./scripts/package_engine_bundle.sh \
  --version 0.3.0 \
  --engine-dir build/engines/dcvcrt-v2 \
  --output-dir dist
```

For the desktop RTX bundle that already exists in this workspace, use
`build/engines/dcvcrt-v2`; the older `build/engines/dcvcrt` and
`build/engines/dcvcrt-1080p` directories are not upload-ready v2 bundles.

The archive filename is derived from `engine_manifest.json`, but uses the generic public package family instead of the exact target profile:

```text
nvcr-v0.3.0-linux-x86_64-nvidia-dcvcrt-cvpr2025-1080p-fp16-engines.tar.gz
nvcr-v0.3.0-linux-aarch64-jetson-l4t36-dcvcrt-cvpr2025-1080p-fp16-engines.tar.gz
```

Stage the `.tar.gz` files in the private S3 release-assets bucket and use a
presigned URL as the temporary workflow input. The staging URL is not the
user-facing distribution channel; it is only used by GitHub Actions before the
asset is copied into GitHub Releases.

First deploy the staging bucket from [AWS CDK release assets](../infra/aws-cdk/README.md).
For the first account setup, bootstrap account `<aws-account-id>` in the selected
region, for example `eu-west-1`:

```bash
cd infra/aws-cdk
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
npm install -g aws-cdk
cdk bootstrap aws://<aws-account-id>/eu-west-1
cdk deploy NvcrReleaseAssetsStack
```

Then package, upload to S3, generate a presigned HTTPS URL, and write the
workflow input file:

```bash
./scripts/stage_engine_release_asset.sh \
  --version 0.3.0 \
  --engine-dir build/engines/dcvcrt-v2 \
  --s3-uri s3://nvcr-release-assets-<aws-account-id>-eu-west-1/v0.3.0 \
  --aws-region eu-west-1 \
  --presign-expires 604800 \
  --asset-manifest dist/nvcr-engine-assets.txt
```

The helper uploads both `.tar.gz` and `.tar.gz.sha256` to S3, but only the
presigned archive URL is passed to the GitHub workflow because the workflow
regenerates the release checksum after validation. Presigned URLs are temporary;
if the workflow is rerun after expiry, regenerate `dist/nvcr-engine-assets.txt`.

If you do not use S3, pass `--download-url` with an HTTPS URL that works with
`curl -fL` from a signed-out machine. Browser preview pages, login pages, and
permission-limited links fail before anything reaches the GitHub Release.

After the workflow in this PR is merged to the default branch, upload staged
engine assets to a draft GitHub Release with one row per asset. Do not run this
against `main` before the PR is merged; GitHub returns 404 when the requested
workflow file is not present on the selected/default branch.

```text
<package-family-engine-asset-file-name> <sha256> <staging-download-url>
```

For example:

```bash
cat > /tmp/nvcr-engine-assets.txt <<'EOF'
nvcr-v0.3.0-linux-x86_64-nvidia-dcvcrt-cvpr2025-1080p-fp16-engines.tar.gz <rtx-archive-sha256> <rtx-staging-https-url>
nvcr-v0.3.0-linux-aarch64-jetson-l4t36-dcvcrt-cvpr2025-1080p-fp16-engines.tar.gz <orin-archive-sha256> <orin-staging-https-url>
EOF

gh workflow run upload-engine-assets.yml \
  --ref main \
  -f tag=v0.3.0 \
  -F engine_assets=@/tmp/nvcr-engine-assets.txt
```

The command above is only a trigger: it sends the text file contents to GitHub as
the `engine_assets` input. The download, SHA-256 check, bundle validation, and
GitHub Release upload all run on the GitHub-hosted Actions runner. If you prefer
not to use the local GitHub CLI, open **Actions → upload-engine-assets → Run
workflow** after the PR is merged, select `main`, paste the contents of
`dist/nvcr-engine-assets.txt` into `engine_assets`, and run it from the browser.

The workflow:

1. checks out the exact release tag separately from the upload automation;
2. downloads each staged archive;
3. verifies the supplied archive SHA-256;
4. rejects unsafe tar paths, source checkpoints, and ONNX files;
5. extracts the bundle and runs the tagged `nvcr-artifacts validate`;
6. confirms the archive filename matches the engine manifest identity;
7. uploads the archive and generated `.sha256` to the GitHub Release.

To publish from the same manual workflow after evidence is recorded, set
`publish_release=true` and set `publish_confirmation` to the exact tag. Leaving
`publish_release` unset keeps the release in draft status.

GitHub Release assets are limited to files under 2 GiB. If a package-family engine archive exceeds that limit, do not push it through Git LFS; split the
engine profile strategy or keep the asset outside the GitHub Release and document
that exception in the release notes.

## Continuous integration before release

Pull-request CI runs on GitHub-hosted standard runners only. It covers:

- shell, Python, JSON, and workflow-YAML validation;
- CPU Release CTest on hosted x86_64 and arm64 Linux runners;
- a required hosted x86_64 CUDA/TensorRT portable build and generic package smoke
  path for release-surface changes.

This hosted CI is useful for packaging discipline and general regressions, but it is
not a substitute for the reference-target gates recorded in the roadmap.

An optional manual self-hosted helper remains in
[gpu-main.yml](../.github/workflows/gpu-main.yml), but it is disabled by default
and not part of the repository's standard hosted-runner contract.

## Manual dispatch and manual Jetson packaging

A maintainer can dispatch the hosted release-asset workflow for a tagged draft:

```bash
gh workflow run release-assets.yml --ref main -f tag=v0.3.0
```

Build and upload the Jetson package manually on the validated Jetson target:

```bash
./scripts/install.sh --build-type Release --build-dir build-release-jetson --prefix "$PWD/install-release-jetson"
./scripts/package_release.sh --version 0.3.0 --platform linux-aarch64-jetson-l4t36 --install-prefix "$PWD/install-release-jetson" --output-dir dist
(cd dist && sha256sum -c nvcr-v0.3.0-linux-aarch64-jetson-l4t36.tar.gz.sha256)
gh release upload v0.3.0 dist/nvcr-v0.3.0-linux-aarch64-jetson-l4t36.tar.gz dist/nvcr-v0.3.0-linux-aarch64-jetson-l4t36.tar.gz.sha256 --clobber
```

Users and reviewers should download both generic packages and optional engine
assets from the GitHub Release. Do not ask reviewers to use S3 presigned staging
URLs directly.

Performance, rate/distortion, memory, and Orin energy evidence are recorded in
[ROADMAP.md](../ROADMAP.md) using [Performance](performance.md). Passing hosted
packaging alone never completes a release milestone.
