# Getting started

The [project overview](../README.md) is the quickest path for installation and
basic use. This page covers source checkouts, model preparation, and native
testing.

## Prebuilt installation

On a Linux NVIDIA system, the installer downloads the binary and compatible
TensorRT engine profiles:

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash
export PATH="$HOME/.local/nvcr/bin:$PATH"
```

See [Compatibility](compatibility.md) if the installer cannot find a matching
engine bundle.

## Build from source

For a native TensorRT build, use Linux with CMake 3.24 or newer, a C++20
compiler, CUDA, and TensorRT:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

For the CPU library and contract tests only:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

## Prepare model and engine files

NVCR does not ship checkpoints, exported model files, or TensorRT plans. The
offline preparation environment needs PyTorch, ONNX, and ONNXScript. Use the
artifact front end with the pinned DCVC-RT checkout and a target profile:

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

Validate both outputs before use:

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-720p --json
```

The complete source, checkpoint, target, and bundle rules are in [Model and
engine preparation](dcvcrt-artifacts.md).

## Run the CLI

The CLI reads and writes planar 8-bit YUV420:

```bash
./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

See the [command line reference](cli.md) for engine installation, quality
metrics, and diagnostic options.

## Full native test run

To run the complete configured test suite, provide a target-local collection of
engine profiles:

```bash
cmake -S . -B build-release-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$NVCR_ENGINE_ROOT/dcvcrt-720p" \
  -DNVCR_TENSORRT_ENGINE_DIRS="$NVCR_ENGINE_ROOT/dcvcrt-qcif;$NVCR_ENGINE_ROOT/dcvcrt-cif;$NVCR_ENGINE_ROOT/dcvcrt-360p;$NVCR_ENGINE_ROOT/dcvcrt-540p;$NVCR_ENGINE_ROOT/dcvcrt-1080p"
cmake --build build-release-clean --parallel
ctest --test-dir build-release-clean --output-on-failure
```

The registered test list is the authority for what that configuration
exercised.
