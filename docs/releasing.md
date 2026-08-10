# Release gates and packaging

NVCR has two independent publication tracks: semver'd application binaries and
a rolling target-local engine catalog. Separating them prevents unchanged plans
from being duplicated or renamed for every application release.

## Application binaries

Release Please remains the version, changelog, and application-tag authority.
The release workflow builds the x86_64 package on the generic self-hosted
X64 runner and builds the AArch64 package natively on the labeled Jetson
self-hosted runner. The Jetson job uses the target's installed JetPack CUDA and
TensorRT stack, targets Orin SM 8.7, and runs the registered test suite before
packaging. The packages are versioned together, but are not interchangeable
binaries.

Binary archive names remain versioned:

```text
nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz
nvcr-vX.Y.Z-linux-aarch64-jetson-l4t36.tar.gz
```

These are separate native ELF families. A portable CUDA architecture set does
not make one host executable run on both x86_64 and AArch64.

Before publishing an application release:

1. validate the exact tag on the RTX 4070 and required Jetson release track;
2. run the registered GPU suites with target-local engines;
3. record correctness/performance/resource evidence in `ROADMAP.md`;
4. build and verify both required binary archives;
5. confirm each archive contains `PACKAGE-MANIFEST.sha256` and no checkpoints,
   ONNX files, runtime model assets, or TensorRT plans.

Publishing the approved GitHub Release triggers `.github/workflows/publish-containers.yml`.
It checks out the exact release tag and publishes the amd64 and Jetson runtime
images to `${DOCKERHUB_USERNAME}/nvcr`, including provenance and SBOM
attestations. Configure the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
repository secrets first. A manual workflow dispatch remains available for an
architecture-specific retry without creating another release.

For a local/manual reproduction, the archive can still be built from a native
Jetson checkout:

```bash
./scripts/install_from_source.sh --build-type Release \
  --build-dir build-release-jetson --prefix "$PWD/install-release-jetson"
./scripts/package_release.sh --version X.Y.Z \
  --platform linux-aarch64-jetson-l4t36 \
  --install-prefix "$PWD/install-release-jetson" --output-dir dist
```

The release workflow selects the Jetson with the
`[self-hosted, Linux, ARM64, Jetson]` labels; after its one-time runner
registration, no sysroot paths or cross-build container are required. Package
creation verifies the ELF machine type against the public platform label, so an
x86 build cannot be mislabeled as the Jetson archive or vice versa. The native
job validates basic CUDA execution, while target-local TensorRT plan validation
and performance remain separate target-evidence gates.

## Rolling engine assets

Validated engines are published under one non-semver GitHub release/tag:
`engine-assets`. Stable filenames contain the portability class, model, and
canonical resolution, but no application version or `fp16` suffix. Exact
bundles keep their registered target ID; compatibility-mode desktop bundles use
generalized names:

```text
nvcr-engines-rtx4070-ubuntu2404-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-orin-nano-l4t3647-dcvcrt-cvpr2025-540p.tar.gz
nvcr-engines-linux-amd64-sm89-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-ampere-plus-dcvcrt-cvpr2025-720p.tar.gz
```

`nvcr-engine-catalog.json` uses `nvcr.engine-catalog.v1`. Every row includes the
archive hash/size, codec/model/target/engine profile, Linux architecture, hardware
compatibility class, GPU identity recorded at build time, CUDA runtime,
TensorRT version, and internal FP16 precision. Catalog installation selects the
best published match per profile: exact device first, then
`same_compute_capability`, then `ampere_plus`. TensorRT matching remains exact.
Desktop CUDA runtime matching allows an active runtime to use an engine recorded
on an older runtime within the same CUDA major family; Jetson remains exact.
NVCR never builds a missing engine automatically during install.

From each target machine, stage a complete six-profile set for a new target or
selected replacements for a target already present in the catalog. Treat each
hardware-compatibility class as its own six-profile set:

```bash
./scripts/release_engine_assets.sh \
  --engine-dir build/engines/dcvcrt-qcif \
  --engine-dir build/engines/dcvcrt-cif \
  --engine-dir build/engines/dcvcrt-360p \
  --engine-dir build/engines/dcvcrt-540p \
  --engine-dir build/engines/dcvcrt-720p \
  --engine-dir build/engines/dcvcrt-1080p \
  --s3-prefix s3://nvcr-release-assets-<account>-eu-west-1/releases \
  --aws-region eu-west-1
```

The helper validates and deterministically packages each bundle, stages the
archives with temporary URLs, and dispatches `upload-engine-assets.yml`. The
workflow:

1. creates `engine-assets` if it does not exist;
2. downloads and verifies every staged archive;
3. rejects unsafe paths and forbidden source assets;
4. validates the v2 engine bundle and registered target-profile digest;
5. merges entries while preserving other targets and compatibility classes;
6. requires all six registered profiles for each new target/class combination;
7. uploads stable archives/checksums with replacement semantics;
8. uploads the merged catalog last.

Uploading the catalog last makes it the authority for the completed update.
The release is already public and rolling; there is no application draft or
`publish_release` input in this workflow.

Use `--skip-dispatch` to stage only. The lower-level helper accepts a direct
HTTPS URL, public URL base, local copy destination, or exact S3 prefix:

```bash
./scripts/stage_engine_release_asset.sh \
  --engine-dir build/engines/dcvcrt-720p \
  --s3-uri s3://bucket/releases/engine-assets \
  --asset-manifest dist/nvcr-engine-assets.txt
```

Semver package assets are immutable once published. The rolling engine catalog is
the source for target-local plans, and new installers use only that catalog.
GitHub's 2 GiB per-asset limit remains enforced.

## SoftwareX evidence package

The publication evaluation uses one clean package per run:

```text
evidence/softwarex-YYYYMMDD-<shortcommit>/
  README.md
  commands.md
  environment.json
  hardware-targets.json
  artifact-catalog.json
  artifact-digests.json
  test-summary.json
  run-summary.json
  exact-results.jsonl
  same-compute-results.jsonl
  ampere-plus-results.jsonl
  python-reference-results.jsonl
  failures.jsonl
  summary.md
```

Keep metadata, hashes, commands, summaries, and small JSONL rows. Do not commit
checkpoints, model exports, TensorRT plans, raw YUV, encoded streams, or
reconstructed video. The field contract is documented in
[docs/experiments/result-schema.md](experiments/result-schema.md).

Generate this layout with `scripts/benchmark_softwarex_matrix.py`. Only
`run-summary.json` status `complete` is publication-ready. Completion requires
the separate `--profile` pass; primary FPS and wall time remain sourced from
the uninstrumented repetitions. Performance-only, plan-only, skipped, dirty,
missing-metric, or otherwise partial packages remain diagnostics.

## CI and evidence

Pull-request CI covers shell/Python/JSON/workflow validation, CPU tests on hosted
x86_64 and arm64, and the hosted x86_64 CUDA/package smoke. Target publication
still requires the registered GPU suites and roadmap evidence. A successful
`trtexec` build or hosted package alone never completes M4.
