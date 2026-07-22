# Scripts

Portable build, packaging, and model-artifact preparation scripts may be added
here. Runtime operation must never depend on Python.

## Platform auto-detection

`detect_platform.sh` is the shared detection helper used by `install.sh` and
`prepare_dcvcrt_artifacts.sh`. It identifies Jetson vs. discrete-GPU hosts, the
CUDA compiler, GPU architecture (via `nvidia-smi --query-gpu=compute_cap`),
TensorRT location, and a conservative TensorRT engine-build memory budget.
Run it directly for a diagnostic report:

```bash
./scripts/detect_platform.sh
```

## One-command install

`install.sh` configures, builds, and installs NVCR using the auto-detected
platform settings:

```bash
./scripts/install.sh --run-tests
```

Pass `--no-tensorrt`, `--cuda-arch`, `--cuda-compiler`, or `--tensorrt-root` to
override any auto-detected value; run `./scripts/install.sh --help` for the
full list.

By default (`--arch-set auto`) the build targets only the GPU installed in
the current machine. Pass `--arch-set portable` to instead build one fat
binary covering the curated multi-GPU-architecture list in
[`cmake/NVCRAutodetect.cmake`](../cmake/NVCRAutodetect.cmake) (`75;80;86;87;89;90`),
intended for producing a single build that can be installed on a range of
Jetson/RTX/datacenter machines. This only widens GPU-architecture coverage;
it is orthogonal to host CPU architecture (x86_64 vs. aarch64 still need
separate builds) and it does not bundle any TensorRT `.plan` engine files,
which must always be generated per-target via `prepare_dcvcrt_artifacts.sh`.

## Release packaging

`package_release.sh` turns an installed NVCR prefix, and optionally a prepared
engine directory, into release archives suitable for GitHub Releases:

```bash
./scripts/package_release.sh \
  --version 0.1.0 \
  --platform linux-x86_64-discrete \
  --install-prefix /path/to/install-prefix \
  --engine-dir /path/to/engines/dcvcrt \
  --output-dir dist
```

The archive names match the download commands documented in
[`docs/install-binary.md`](../docs/install-binary.md).

## DCVC-RT artifacts

Use `prepare_dcvcrt_artifacts.sh` for the normal checkpoint-to-engine pipeline.
It auto-tunes `--device-id`, `--workspace-mib`, and `--builder-optimization-level`
from `detect_platform.sh` unless you pass `--no-auto-tune`:

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root ./assets \
  --engines build/engines/dcvcrt \
  --skip-smoke
```

The wrapper verifies the upstream Microsoft DCVC-RT checkout and checkpoints,
exports ONNX/runtime assets, and then calls `build_dcvcrt_tensorrt.sh`. On an
8 GB Orin Nano this auto-tunes to the same tested 1080p settings previously
found by hand (workspace 512 MiB, builder optimization level 1).

Use `build_dcvcrt_tensorrt.sh` directly only when `build/models/dcvcrt` already
contains the exported ONNX files plus entropy, quantization, and manifest assets.
TensorRT `.plan` files must be built on the target runtime and selected CUDA
device. The script writes `engine_manifest.json`, and NVCR rejects an engine
directory whose recorded GPU/TensorRT metadata does not match `--device-id`.

If a known target-local engine directory only lacks `engine_manifest.json`, use
`build_dcvcrt_tensorrt.sh --stamp-only --engines DIR ...` to write metadata
without rebuilding plans. Stamp-only mode asks TensorRT to inspect one existing
plan and fails if TensorRT reports the cross-device plan warning.

## Jetson energy profiling

Use `profile_energy.py` on Jetson targets to wrap an encode or decode command
with `tegrastats` rail sampling. The default rail is `VDD_IN` when present, and
the script records both raw board energy and an idle-subtracted estimate:

```bash
./scripts/profile_energy.py \
  --idle-seconds 10 \
  --interval-ms 100 \
  --output-json /tmp/nvcr-encode-energy.json \
  --tegrastats-log /tmp/nvcr-encode-tegrastats.log \
  --stdout-log /tmp/nvcr-encode.stdout \
  --stderr-log /tmp/nvcr-encode.stderr \
  -- ./build-orin-release/cli/nvcr encode \
    -i /home/oelghati/datasets/hd/BasketballDrive_1920x1080_50.yuv \
    -o /tmp/basketball.nvcr \
    -s 1920x1080 -r 50 --frames 97 --gop-size 97 --qp 32 \
    --engine-dir build/engines/dcvcrt-1080p-orin \
    --profile
```

Decode energy should be measured as a separate command because the CLI encoder
and decoder are separate processes:

```bash
./scripts/profile_energy.py \
  --idle-seconds 10 \
  --interval-ms 100 \
  --output-json /tmp/nvcr-decode-energy.json \
  -- ./build-orin-release/cli/nvcr decode \
    -i /tmp/basketball.nvcr \
    -o /tmp/basketball.decoded.yuv \
    --engine-dir build/engines/dcvcrt-1080p-orin \
    --profile
```

For serious comparisons, keep `nvpmodel`, clocks, engine directory, input
sequence, QP, GOP, TensorRT mode, and sample interval fixed. Prefer runs long
enough to dilute initialization and teardown, then compare the same command with
the same idle baseline method.
