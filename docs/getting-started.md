# Getting started and builds

NVCR is currently a source-built development project. The runtime architecture is
intended for neural video codec backends, but the current release path supports
DCVC-RT only. Published `0.2.x` artifacts are historical snapshots, not
v1-compatible product packages. The supported release procedure is gated in
[Releasing](releasing.md).

## Choose a build

### CPU development build

This build validates public types, bounded packet/access-unit parsing, rANS,
configuration, and artifact profiles. It cannot encode or decode DCVC-RT frames.

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

### Target-profile GPU runtime build

Requirements:

- Linux and a C++20 compiler;
- CMake 3.24 or newer;
- the CUDA and TensorRT versions selected by a reference target profile;
- direct access to the selected NVIDIA device;
- locally generated compatible engines.

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

CMake detects the attached GPU architecture, CUDA compiler, and TensorRT install
on a normal native build. This produces a development build for the selected
profile; it is not support evidence by itself. The source-build helper performs
the same detection and installs to a local platform-specific prefix by default:

```bash
./scripts/install_from_source.sh --run-tests
```

Auto-detection is convenience, not support evidence. v1 is validated only for
[the RTX 4070 and Orin Nano profiles](compatibility.md). Explicit overrides are
available for target bring-up:

```bash
cmake -S . -B build-orin-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DTensorRT_ROOT=/usr
```

A CUDA fat binary (`NVCR_CUDA_ARCH_SET=portable`) is an experimental packaging
option. It does not make a binary universal and never makes TensorRT plans
portable across GPU models or TensorRT runtimes.

## Prepare the pinned model

NVCR ships no checkpoints, ONNX/model assets, or TensorRT plans. Obtain the two
checkpoint files described in [Model and engine preparation](dcvcrt-artifacts.md)
and place them under:

```text
/path/to/DCVC-RT/checkpoints/cvpr2025_image.pth.tar
/path/to/DCVC-RT/checkpoints/cvpr2025_video.pth.tar
```

The Python interpreter used offline must provide `torch`, `onnx`, and
`onnxscript`. Python is not a runtime dependency of `libnvcr` or the CLI.

Use the profile-aware artifact command. This generic front end dispatches to the
current DCVC-RT backend preparation helpers:

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

For Orin, select `configs/targets/orin-nano-l4t3647.json` and run engine
building on the Orin itself. Portable ONNX/assets may be transferred only after
`nvcr-artifacts validate` succeeds; plans must be rebuilt on the final target.

Inspect and validate without executing a bundle:

```bash
./scripts/nvcr_artifacts.py inspect build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py inspect build/engines/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt --json
```

## Run the complete configured suite

The engine and native I/P tests are registered only when CMake receives an engine
bundle. Reconfigure after generating it:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$PWD/build/engines/dcvcrt"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Check `ctest --test-dir build-release -N`: a full GPU configuration includes
artifact/profile, parser/rANS, CUDA ops, engine contracts, and I/P roundtrip
tests. The roundtrip test covers high effective QP, two GOPs, reset/reuse, and
encoder/decoder reconstruction equality. The engine-contract test covers
wrong-model, corrupt-manifest, and corrupt-checksum rejection.

On Jetson, tests require access to `/dev/nvmap` and `/dev/nvhost-*`. A container
that hides those devices can compile successfully but cannot provide GPU test
evidence.

Architecture-specific Docker execution images and named RTX/Jetson Dev
Container configurations are documented in [Docker execution and
development](docker.md). They preserve the same target-local engine rule as a
native build; containerization does not make TensorRT plans portable.

## Encode and decode raw YUV420P8

```bash
export NVCR_ENGINE_ROOT="$PWD/build/engines"

./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

The CLI format is a development sequence format, not MP4/Matroska, FFmpeg, or a
stable upstream DCVC-RT stream. See [CLI](cli.md) and [Bitstream](bitstream.md).

## Install for local development

```bash
cmake --install build-release --prefix "$PWD/install-nvcr"
export PATH="$PWD/install-nvcr/bin:$PATH"
nvcr --help
```

The install contains NVCR code, headers, CMake metadata, configuration examples,
documentation, and notices. It does not contain checkpoints or derived model and
engine assets. Pass the local engine directory explicitly.

## Clean-room target evidence

A release validation starts without pre-existing build/model/engine directories
and records:

1. OS, compiler, CUDA, TensorRT, driver/JetPack, GPU, clocks/power mode;
2. pinned source commit and checkpoint hashes;
3. model preparation and model-bundle validation;
4. target-local engine build and engine-bundle validation;
5. the complete registered CTest suite;
6. native multi-frame encode/decode and reference comparison;
7. release-build latency, memory, throughput, rate/distortion, and applicable
   energy evidence using [Performance](performance.md).

v0.3 requires the RTX 4070 foundation workflow. v1.0 requires separate successful
RTX 4070 and Orin Nano matrices and all applicable M1–M4 roadmap gates.
