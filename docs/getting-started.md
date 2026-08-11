# Getting started

NVCR is the runtime architecture; DCVC-RT/TensorRT is the current production
vertical. There are two useful starting points: a CPU build for public
contracts and conformance fixtures, or a native TensorRT build for production
codec execution.

## CPU build

Requirements: CMake 3.24+, a C++20 compiler, and a standard Linux development environment.

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

This covers parsing, rANS, configuration, artifact identity, and the test codec/provider. It does not encode DCVC-RT frames.

## TensorRT build

Use the CUDA and TensorRT versions named by the target profile. Build engines on the machine that will run them; TensorRT plans are not portable between GPU models or TensorRT runtimes.

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

## Install or prepare a target

Generic NVCR packages do not embed checkpoints, exported model assets, or
plans. First try the public rolling engine catalog on the target machine:

```bash
nvcr-artifacts install --profile 720p
```

If no compatible catalog entry exists, obtain the pinned checkpoints under
their applicable terms and build the target-local engine with a DCVC-RT
checkout:

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

The offline Python environment needs `torch`, `onnx`, and `onnxscript`. Python is not a runtime dependency.

Validate both stages before use:

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-720p --json
```

The exact source, checkpoint, target, and bundle rules are in [Model and engine preparation](dcvcrt-artifacts.md).

## Run the CLI

The CLI uses planar 8-bit YUV420:

```bash
./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

The native installer defaults to all compatible profiles. To limit downloads,
select one or more profiles:

```bash
curl -fsSL https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh \
  | bash -s -- --profile qcif 720p
```

Use [Compatibility](compatibility.md) before treating a target as supported.
Docker users should choose the dedicated [Jetson](docker-jetson.md) or
[x86_64 NVIDIA](docker-x86_64.md) guide.

## Full native test run

Configure CMake with a complete target-local engine collection. Then build and run:

```bash
cmake -S . -B build-release-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR=build/engines/dcvcrt-720p \
  -DNVCR_TENSORRT_ENGINE_DIRS="build/engines/dcvcrt-qcif;build/engines/dcvcrt-cif;build/engines/dcvcrt-360p;build/engines/dcvcrt-540p;build/engines/dcvcrt-1080p"
cmake --build build-release-clean --parallel
ctest --test-dir build-release-clean --output-on-failure
```

The registered test list is the authority for what that configuration actually exercised.
