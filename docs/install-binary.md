# Binary Install Guide

This guide covers quick NVCR installation from published release artifacts
without building from source. It assumes release artifacts are published per
platform (CLI binary bundle and optional engine bundles).

## 1) Pick your platform profile

Run this once to identify what package to download:

```bash
uname -m
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
trtexec --version || /usr/src/tensorrt/bin/trtexec --version
```

Typical platform IDs:

| Host | Platform tag | Notes |
|---|---|---|
| x86_64 + RTX/datacenter | `linux-x86_64-discrete` | Fastest build, targets only that GPU's architecture |
| x86_64, any supported NVIDIA GPU | `linux-x86_64-portable` | Single build covering multiple GPU architectures, no bundled engines |
| aarch64 Jetson Orin | `linux-aarch64-jetson` | Build and run on target device |

## 2) Download and install NVCR binary

Set your release coordinates first:

```bash
export NVCR_VERSION="v0.1.0" # x-release-please-version
export NVCR_PLATFORM="linux-x86_64-discrete"   # or linux-aarch64-jetson
export NVCR_PREFIX="$HOME/.local/nvcr"
```

Download and install:

```bash
mkdir -p "$NVCR_PREFIX"
curl -fL "https://github.com/<your-org>/NVCR/releases/download/${NVCR_VERSION}/nvcr-${NVCR_VERSION}-${NVCR_PLATFORM}.tar.gz" -o /tmp/nvcr.tar.gz
tar -xzf /tmp/nvcr.tar.gz -C "$NVCR_PREFIX" --strip-components=1
export PATH="$NVCR_PREFIX/bin:$PATH"
nvcr --help
```

## 3) Use a ready engine bundle (recommended)

If you publish per-architecture/per-runtime engine bundles, install one that
matches the target machine:

```bash
export NVCR_ENGINE_BUNDLE="dcvcrt-engines-${NVCR_VERSION}-${NVCR_PLATFORM}.tar.gz"
mkdir -p "$NVCR_PREFIX/engines"
curl -fL "https://github.com/<your-org>/NVCR/releases/download/${NVCR_VERSION}/${NVCR_ENGINE_BUNDLE}" -o /tmp/nvcr-engines.tar.gz
tar -xzf /tmp/nvcr-engines.tar.gz -C "$NVCR_PREFIX/engines"
```

Expected result:

```text
$NVCR_PREFIX/engines/dcvcrt/
  *.plan
  i_entropy.bin
  i_quant.bin
  i_frame_manifest.json
  p_entropy.bin
  p_quant.bin
  p_frame_manifest.json
  engine_manifest.json
  engine.sha256
```

Run a quick check:

```bash
nvcr decode -h
```

## 4) If no engine bundle exists for this architecture

Build the engines locally on the target machine with your checkpoints.

### 4.1 Prepare checkpoints

Put checkpoints in the DCVC-RT root used for export:

```text
<dcvcrt-root>/checkpoints/cvpr2025_image.pth.tar
<dcvcrt-root>/checkpoints/cvpr2025_video.pth.tar
```

Known hashes:

```text
555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9  cvpr2025_image.pth.tar
b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6  cvpr2025_video.pth.tar
```

### 4.2 Build ONNX assets and TensorRT plans

```bash
cd /path/to/NVCR

./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines "$NVCR_PREFIX/engines/dcvcrt" \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --python /path/to/DCVC-RT/src/venv/bin/python \
  --skip-clone --skip-smoke
```

Use the generated engines:

```bash
nvcr encode -i input.yuv -o output.nvcr -s 176x144 --frames 1 --engine-dir "$NVCR_PREFIX/engines/dcvcrt"
```

## 5) Publisher checklist (for your release process)

For each release, publish:

1. `nvcr-<version>-<platform>.tar.gz` (binary + libs + docs)
2. `dcvcrt-engines-<version>-<platform>.tar.gz` (runtime assets + plans)
3. SHA256 checksums for all published archives
4. A compatibility note: GPU model class, compute capability, TensorRT version

When no prebuilt engine bundle is published for a platform, use
Section 4 of this guide.