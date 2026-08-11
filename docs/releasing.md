# Packaging and release policy

NVCR publishes application packages and target-compatible TensorRT engine
bundles separately. Application versions follow semantic versioning. Engine
bundles use a rolling catalog because unchanged target-local plans do not need
to be duplicated for every application version.

## Application packages

NVCR publishes separate native Linux archives:

```text
nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz
nvcr-vX.Y.Z-linux-aarch64-jetson-l4t36.tar.gz
```

These archives contain different ELF architectures and are not
interchangeable. CUDA architecture coverage does not make an x86_64 executable
run on AArch64 or an AArch64 executable run on x86_64.

Each application archive must:

- be built from the exact release tag;
- pass the registered package and platform tests;
- contain `PACKAGE-MANIFEST.sha256`;
- install and run on its declared architecture;
- exclude checkpoints, ONNX graphs, runtime model assets, TensorRT plans, raw
  datasets, and generated streams;
- retain all required licenses and notices.

A native package can be reproduced with:

```bash
./scripts/install_from_source.sh \
  --build-type Release \
  --build-dir build-release-package \
  --prefix "$PWD/install-release"

./scripts/package_release.sh \
  --version "$(cat version.txt)" \
  --platform linux-x86_64-nvidia \
  --install-prefix "$PWD/install-release" \
  --output-dir dist
```

Use `linux-aarch64-jetson-l4t36` for a package built natively on the supported
Jetson family.

## Engine catalog

Validated engines are separate assets under the rolling `engine-assets`
release. Stable names encode the target or compatibility class, model profile,
and resolution:

```text
nvcr-engines-rtx4070-ubuntu2404-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-orin-nano-l4t3647-dcvcrt-cvpr2025-540p.tar.gz
nvcr-engines-linux-amd64-sm89-dcvcrt-cvpr2025-720p.tar.gz
nvcr-engines-linux-amd64-ampere-plus-dcvcrt-cvpr2025-720p.tar.gz
```

`nvcr-engine-catalog.json` uses schema `nvcr.engine-catalog.v1`. Every entry
records archive hash and size, codec/model/target/engine profiles, architecture,
hardware compatibility, GPU identity, CUDA runtime, TensorRT version, and
precision.

Catalog selection ranks candidates in this order:

1. exact device;
2. same compute capability on supported desktop targets;
3. explicitly published Ampere-plus desktop compatibility.

TensorRT matching remains exact. Jetson engines remain exact target-local
artifacts. The installer validates archive hashes, manifests, required files,
and bundle digests; it never builds a missing engine automatically.

## Engine publication gate

Before an engine archive enters the catalog:

1. detect and record the target identity;
2. validate the current model, engine, and target-profile digests;
3. validate the complete bundle and every required file checksum;
4. independently load every TensorRT plan without warnings;
5. run dimension-matched I/P encode and decode on the target;
6. verify the applicable model, engine, and dependency distribution terms;
7. package deterministically and verify the archive hash;
8. update the catalog only after every selected archive is available.

Uploading the catalog last ensures that installers cannot select an incomplete
update. A local archive, successful upload, or plan-only load does not by itself
establish publication or target support.

## SoftwareX evidence package

The evaluation driver writes a package such as:

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

Commit metadata, hashes, commands, summaries, and compact result rows. Keep
checkpoints, model exports, TensorRT plans, raw YUV, encoded streams, and
reconstructed video outside Git.

Only a `run-summary.json` status of `complete` is publication-ready. Completion
requires the registered build/test gates, current artifact identity, required
metrics, separate profiling repetitions, and pinned reference comparisons.
Dirty, partial, plan-only, skipped, or missing-metric packages remain
diagnostic.

See the [result schema](experiments/result-schema.md) and
[SoftwareX runbook](experiments/runbook.md) for the executable protocol.
