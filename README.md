# NVCR

NVCR is a C++20 runtime for neural video codecs. The current implementation is deliberately narrow: DCVC-RT model semantics running through TensorRT on Linux.

The project is still in development. It is useful software, but it is not yet a supported v1 release or a universal codec framework.

## Current scope

- DCVC-RT CVPR 2025 image/video model pair.
- TensorRT FP16 execution on NVIDIA Linux targets.
- Native I/P encode and decode with explicit sequence state.
- Bounded, versioned `NVAU` access units.
- Offline model export and target-local engine preparation.
- C++ runtime, CLI, artifact validation, and contract tests.

Checkpoints, exported model assets, and TensorRT plans are not distributed. They must be obtained and built under their applicable terms.

## Build

For the CPU-only library and parser/API tests:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

For a native TensorRT build, use a Linux host with CMake 3.24+, a C++20 compiler, CUDA, and TensorRT:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

## Prepare engines

The supported artifact front end verifies the pinned source and checkpoints, exports model assets, and builds engines on the final target:

```bash
./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 720p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
```

Validate a model or engine bundle before using it:

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-720p --json
```

## Run

The CLI reads and writes planar 8-bit YUV420:

```bash
./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

`NVCR` and `NVCS` are development file wrappers. The codec access-unit contract is `NVAU`; it is documented separately from application/container metadata.

## Read next

- [Scope and support](docs/scope-and-support.md)
- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Model and engine preparation](docs/dcvcrt-artifacts.md)
- [CLI](docs/cli.md)
- [Bitstream and access units](docs/bitstream.md)
- [C++ API](docs/reference.md)
- [Compatibility](docs/compatibility.md)
- [Performance protocol](docs/performance.md)
- [SoftwareX experiment protocol](docs/experiments/README.md)
- [Docker](docs/docker.md)
- [Release policy](docs/releasing.md)

The repository's current status and next gate are recorded in [ROADMAP.md](ROADMAP.md). Architectural decisions are in [docs/adr](docs/adr), and the stream format is specified in [docs/spec/nvcr-elementary-stream-v1.md](docs/spec/nvcr-elementary-stream-v1.md).
