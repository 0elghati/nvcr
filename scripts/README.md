# Scripts

Runtime execution never depends on Python. Scripts provide build detection,
offline pinned-model preparation, release packaging, and focused measurement.

## Supported user entry points

- `install.sh`: native configure/build/install convenience wrapper.
- `nvcr_artifacts.py`: `prepare`, `build`, `inspect`, and `validate` for
  backend artifact bundles and versioned profiles. The current implementation
  supports the DCVC-RT backend.
- `package_release.sh`: package an installed tree; refuses model/engine assets.
- `../docker/publish.sh`: build or publish one architecture-qualified Docker
  Hub runtime image from an exact release tag.
- `package_engine_bundle.sh`: package one already validated engine bundle as a
  separate package-family GitHub Release asset.
- `stage_engine_release_asset.sh`: package an engine bundle, optionally upload
  it to S3 or copy it to another staging folder, and generate the
  `engine_assets.txt` workflow input row.
- `release_engine_assets.sh`: stage one or more validated engine bundles to S3
  and dispatch the GitHub Release upload workflow for a Release Please tag.
- `profile_energy.py`: focused Jetson command/rail measurement helper.

`nvcr_artifacts.py` calls backend-local helpers under
`scripts/backends/dcvcrt/` for DCVC-RT export, TensorRT building, and manifest
writing, and uses the generic `detect_platform.sh` helper for local target
detection. Backend helpers remain available for expert diagnosis but do not
define separate supported workflows.

```bash
./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --engine-profile configs/engine-profiles/1080p-fp16.json \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
```

Engine settings are selected from the profile, with device auto-detection used as
a convenience. TensorRT plans remain tied to the final target and are validated
by hashes plus exact CUDA/TensorRT/GPU/model identities.

```bash
./scripts/nvcr_artifacts.py inspect build/engines/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt --json
```

## Build detection

`detect_platform.sh` reports Jetson/discrete class, CUDA compiler, attached GPU
architecture, TensorRT location, device ID, free memory, and a conservative
workspace/builder budget. Only the versioned RTX 4070 and Orin Nano profiles are
v1 reference targets; detection of another GPU is not a support claim.

```bash
./scripts/detect_platform.sh
./scripts/install_from_source.sh --run-tests
```

Published binary releases use `./scripts/install.sh`, which downloads the release
package and, unless `--engine-profile` is set, every engine profile for the
selected backend. The optional `--arch-set portable` produces a CUDA fat binary
for experimentation.
It does not make the runtime universal or make TensorRT plans portable.

## Packaging

```bash
./scripts/package_release.sh \
  --version 0.3.0 \
  --platform linux-x86_64-nvidia \
  --install-prefix /path/to/install \
  --output-dir dist
```

The archive contains a file-hash manifest and required docs/licenses/profiles. It
never contains checkpoints or derived model/engine assets. The public package
family name is generic; support evidence remains tied to the recorded reference
target profiles.

Package optional reviewer-convenience engines separately after target-local
validation:

```bash
./scripts/package_engine_bundle.sh \
  --version 0.3.0 \
  --engine-dir build/engines/dcvcrt \
  --output-dir dist
```

The generated asset name is derived from the engine manifest and uses the public package family:

```text
nvcr-v0.3.0-<package-family>-dcvcrt-cvpr2025-<engine-profile>-engines.tar.gz
```

Upload these engine archives as separate GitHub Release assets only after their
target evidence is recorded. They are not bundled into the generic binary
packages.

To combine packaging, S3 upload, presigned URL generation, and workflow-input
generation:

```bash
release_tag="v$(cat version.txt)"
./scripts/stage_engine_release_asset.sh \
  --version "${release_tag#v}" \
  --engine-dir build/engines/dcvcrt-720p \
  --s3-uri "s3://nvcr-release-assets-<aws-account-id>-eu-west-1/releases/$release_tag" \
  --aws-region eu-west-1 \
  --asset-manifest dist/nvcr-engine-assets.txt
```

The generated text file is a manual `workflow_dispatch` input, not a file
committed to the repository. If you do not use S3, pass `--download-url` with an
HTTPS URL that works with `curl -fL` from a signed-out machine.

To avoid manual copy/paste after Release Please creates a draft release, stage
and upload every validated local engine bundle in one command:

```bash
./scripts/release_engine_assets.sh \
  --engine-dir build/engines/dcvcrt-720p \
  --engine-dir build/engines/dcvcrt-1080p \
  --s3-prefix s3://nvcr-release-assets-<aws-account-id>-eu-west-1/releases \
  --aws-region eu-west-1
```

The helper validates the bundles, uploads the archives to S3 with temporary
presigned URLs, writes `dist/nvcr-engine-assets.txt`, verifies the draft release
exists, and dispatches `upload-engine-assets.yml` against the exact tag.

## Paired performance smoke

Use `benchmark_resolution_pair.sh` before and after optimization changes. It is a
developer smoke helper, not the final publication harness, but it keeps 720p and
1080p tied together and can append JSONL rows for commit-by-commit comparison.

```bash
./scripts/benchmark_resolution_pair.sh \
  --gops "1 97" \
  --jsonl /tmp/nvcr-paired-benchmark.jsonl
```

Override the default local inputs with `--720p-input`, `--1080p-input`, or the
matching `NVCR_BENCH_720P_INPUT` and `NVCR_BENCH_1080P_INPUT` environment
variables.

## Jetson energy

Use `profile_energy.py` to wrap encode and decode separately. Fix the target
profile, `nvpmodel`, clocks, rail, input, QP, GOP, TensorRT mode, idle method, and
sample interval, and preserve the raw JSON/logs with roadmap evidence.

```bash
./scripts/profile_energy.py --idle-seconds 10 --interval-ms 100 \
  --output-json /tmp/nvcr-energy.json --frames 97 \
  -- nvcr encode ... --profile
```

See [Artifact preparation](../docs/dcvcrt-artifacts.md),
[Performance](../docs/performance.md), and [Releasing](../docs/releasing.md).

## Four-resolution benchmark matrix

Use `benchmark_resolution_matrix.sh` for comparable encode/decode and quality
measurements across QCIF, CIF, 720p, and 1080p. The default protocol runs GOP 1
and GOP 97, performs separate encode/decode warm-ups, and records three measured
repetitions per point.

```bash
scripts/benchmark_resolution_matrix.sh \
  --nvcr build-release/cli/nvcr \
  --qcif-input /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  --cif-input /home/oelghati/DCVC/datasets/cif/paris_cif.yuv \
  --720p-input /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  --1080p-input /home/oelghati/DCVC/datasets/misc/BQTerrace_1920x1080_60.yuv \
  --qcif-engine-dir build/engines/dcvcrt-qcif \
  --cif-engine-dir build/engines/dcvcrt-cif \
  --720p-engine-dir /path/to/720p/dcvcrt \
  --1080p-engine-dir /path/to/1080p/dcvcrt \
  --jsonl docs/evidence/dcvcrt-resolution-matrix-YYYY-MM-DD.jsonl
```

PSNR calculation is deliberately outside codec timing. Keep the emitted JSONL
with the corresponding performance documentation so per-run variance is not
lost behind aggregate FPS values.

## Complete Orin release benchmark

Use `benchmark_orin_release.py` after rebooting the Jetson when `tegrastats`
reports healthy contiguous `lfb` headroom. It validates all selected engine
bundles, builds Release, runs the registered tests, executes every
resolution/GOP as an independent case, collects GOP-97 encode/decode energy,
and writes one self-contained JSON report suitable for an AI documentation
handoff. A failed resolution does not prevent later resolutions from being
attempted.

```bash
sudo nvpmodel -m 2
sudo jetson_clocks

./scripts/benchmark_orin_release.py \
  --execution-mode automatic \
  --output docs/evidence/orin-release-$(date +%F).json
```

For performance diagnosis, compare the same short case in both execution modes
before running the full matrix. `--profile` now includes TensorRT execution-
context acquisition in each engine's CPU `setup` stage; in low-memory mode it
also includes the forced per-engine synchronization and context destruction.

```bash
NVCR_TENSORRT_LOW_MEMORY_MODE=1 build-release/cli/nvcr encode \
  -i /home/oelghati/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/nvcr-orin-low-memory.nvcr -s 1280x720 -r 30 --frames 9 \
  --gop-size 9 --qp 32 --engine-dir build/engines/dcvcrt-cvpr2025/orin-nano-l4t3647/720p-fp16 \
  --profile

NVCR_TENSORRT_LOW_MEMORY_MODE=0 build-release/cli/nvcr encode \
  -i /home/oelghati/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/nvcr-orin-performance.nvcr -s 1280x720 -r 30 --frames 9 \
  --gop-size 9 --qp 32 --engine-dir build/engines/dcvcrt-cvpr2025/orin-nano-l4t3647/720p-fp16 \
  --profile
```

Automatic mode keeps persistent context metadata on integrated GPUs while all
engines reuse one user-managed activation workspace reserved from the session
CUDA arena. Run performance mode only as a diagnostic: it asks TensorRT to give
every context a separate workspace and can exhaust fragmented unified memory.
The existing low-memory release result is not representative of the automatic
device-chained throughput target.

Use `--prepare-system` to have the script invoke the two `sudo` preparation
commands interactively. Use `--skip-energy` only for a diagnostic run; such a
report does not satisfy the Orin release evidence protocol. The final JSON
contains commands, stdout/stderr, platform and clock metadata, input and bundle
hashes, all per-run rows, aggregates, energy records, and explicit failures.
