# NVCR

NVCR is a Linux C++20 runtime and CLI for DCVC-RT video encoding and decoding
with TensorRT FP16. It is development software, not yet a supported v1 release.

## Install

Requirements: an NVIDIA GPU and driver, Python 3, `curl`, `tar`, and
`sha256sum`. The installer downloads the current binary and the QCIF engine by
default:

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash

export PATH="$HOME/.local/nvcr/bin:$PATH"
nvcr --help
```

Install selected profiles only:

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh \
  | bash -s -- --profile qcif 720p
```

Use `--all-profiles` only when every compatible published profile is needed.

If no compatible engine is published, follow [model and engine
preparation](docs/dcvcrt-artifacts.md). TensorRT plans remain tied to their GPU
and runtime.

## Encode and decode

NVCR reads and writes planar 8-bit YUV420:

```bash
nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

nvcr decode -i output.nvcr -o reconstructed.yuv
```

Raw YUV files do not contain dimensions or frame rate; the encode arguments
must match the input.

## Docker

- [Jetson Orin / L4T 36.4](docs/docker-jetson.md)
- [x86_64 Linux / NVIDIA GPU](docs/docker-x86_64.md)

Both guides cover engine installation, the Akiyo QCIF dataset, volume mapping,
encode/decode, Compose, and local image builds.

## Build from source

Native TensorRT build:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

CPU library and contract tests only:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

## Documentation

- [Getting started](docs/getting-started.md)
- [CLI](docs/cli.md)
- [Compatibility](docs/compatibility.md)
- [C++ API](docs/reference.md)
- [Architecture](docs/architecture.md)
- [Documentation index](docs/README.md)
- [Roadmap](ROADMAP.md)

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. See the [asset policy](ASSET_DISTRIBUTION_POLICY.md) and [model
licensing status](MODEL_LICENSES.md).
