# Scripts

Runtime execution never depends on Python. Scripts provide build detection,
offline pinned-model preparation, release packaging, and focused measurement.

## Supported user entry points

- `install.sh`: native configure/build/install convenience wrapper.
- `nvcr_artifacts.py`: `prepare`, `build`, `inspect`, and `validate` for the
  pinned model and versioned profiles.
- `package_release.sh`: package an installed tree; refuses model/engine assets.
- `package_engine_bundle.sh`: package one already validated engine bundle as a
  separate package-family GitHub Release asset.
- `stage_engine_release_asset.sh`: package an engine bundle, optionally upload
  it to S3 or copy it to another staging folder, and generate the
  `engine_assets.txt` workflow input row.
- `profile_energy.py`: focused Jetson command/rail measurement helper.

`nvcr_artifacts.py` calls the exporters, TensorRT builder, manifest writer,
`prepare_dcvcrt_artifacts.sh`, and `detect_platform.sh`. Those helpers remain
available for expert diagnosis but do not define separate supported workflows.

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
./scripts/install.sh --run-tests
```

The optional `--arch-set portable` produces a CUDA fat binary for experimentation.
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
./scripts/stage_engine_release_asset.sh \
  --version 0.3.0 \
  --engine-dir build/engines/dcvcrt-v2 \
  --s3-uri s3://nvcr-release-assets-<aws-account-id>-eu-west-1/v0.3.0 \
  --aws-region eu-west-1 \
  --asset-manifest dist/nvcr-engine-assets.txt
```

The generated text file is a manual `workflow_dispatch` input, not a file
committed to the repository. If you do not use S3, pass `--download-url` with an
HTTPS URL that works with `curl -fL` from a signed-out machine.


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
