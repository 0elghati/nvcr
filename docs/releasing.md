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
2. installs CUDA and TensorRT development packages on hosted Ubuntu 24.04 x86_64;
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
5. record all required evidence in [ROADMAP.md](../ROADMAP.md);
6. publish the draft release manually only after the release-track gates pass.

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
profiles, license/notice, CLI, and artifact tooling before packaging.

The package script rejects checkpoints, `.pth`/`.pth.tar`, ONNX, entropy/quant
model assets, TensorRT plans, and engine bundles. Generated bundles are release
test inputs, never release outputs.

## Continuous integration before release

Pull-request CI runs on GitHub-hosted standard runners only. It covers:

- shell, Python, JSON, and workflow-YAML validation;
- CPU Debug/Release CTest on hosted x86_64 and arm64 Linux runners;
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

Performance, rate/distortion, memory, and Orin energy evidence are recorded in
[ROADMAP.md](../ROADMAP.md) using [Performance](performance.md). Passing hosted
packaging alone never completes a release milestone.
