# Scripts

Runtime execution never depends on Python. Scripts provide build detection,
offline pinned-model preparation, release packaging, and focused measurement.

## Supported user entry points

- `install.sh`: native configure/build/install convenience wrapper.
- `nvcr_artifacts.py`: `prepare`, `build`, `inspect`, and `validate` for the
  pinned model and versioned profiles.
- `package_release.sh`: package an installed tree; refuses model/engine assets.
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
