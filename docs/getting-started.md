# Getting Started

This guide is the shortest path from a source checkout to a usable NVCR install
on Linux.

## What you need

- CMake 3.24 or newer.
- A C++20 compiler.
- Optional: CUDA and TensorRT when you want the native DCVC-RT backend.

## Build the project

### Quick install (recommended)

`scripts/install.sh` auto-detects the platform (Jetson vs. discrete GPU),
CUDA compiler, GPU architecture (via `nvidia-smi` compute capability), and
TensorRT location, then configures, builds, and installs with one command:

```bash
./scripts/install.sh --run-tests
```

This works unmodified on both a Jetson Orin (auto-detects `sm_87`, `nvcc`
under `/usr/local/cuda`, and system TensorRT under `/usr`) and a discrete-GPU
host (any RTX/datacenter card `nvidia-smi` can see). It installs to
`install-<platform>/` (`install-jetson` or `install-discrete`) by default;
pass `--prefix DIR` to override. Run `./scripts/install.sh --help` for all
options, including `--no-tensorrt` for a CPU-only build and
`--cuda-arch`/`--cuda-compiler`/`--tensorrt-root` to override any
auto-detected value.

By default (`--arch-set auto`) the build targets only the GPU installed in
the current machine, which is the fastest option for local development or a
single-machine install. When you need one binary that works out of the box
across a range of machines (for example, building a package once and
installing it on several different Jetson/RTX/datacenter hosts), pass
`--arch-set portable`:

```bash
./scripts/install.sh --arch-set portable --run-tests
```

This builds a fat binary covering the common architectures in
[cmake/NVCRAutodetect.cmake](../cmake/NVCRAutodetect.cmake)
(`75;80;86;87;89;90` — Turing, Ampere datacenter, Ampere consumer/Jetson
Orin, Ada, Hopper) at the cost of a longer build and a larger `libnvcr.a`.
An explicit `--cuda-arch` (or `-DCMAKE_CUDA_ARCHITECTURES=...`) always wins
over `--arch-set` if you need something else entirely.

**GPU architecture portability is independent of host CPU architecture.**
`--arch-set portable` only widens the range of GPU compute capabilities one
build can run on; it does not make an x86_64 build runnable on aarch64 (or
vice versa). You still need a separate build per host CPU architecture.
TensorRT `.plan` engine files are never part of this portability story either
— they are always tied to one exact GPU model and TensorRT runtime version
and must be generated on (or for) the target machine via
`scripts/prepare_dcvcrt_artifacts.sh`; see
[dcvcrt-artifacts.md](dcvcrt-artifacts.md).

### Manual configure (advanced / CI)

For a normal development build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

For an installable release build on Linux:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
cmake --install build-release --prefix /opt/nvcr
export PATH="/opt/nvcr/bin:$PATH"
```

If you want the CUDA/TensorRT path enabled, `-DNVCR_ENABLE_TENSORRT=ON` alone
is normally enough: CMake auto-detects `CMAKE_CUDA_COMPILER` (falling back to
`/usr/local/cuda*/bin/nvcc` when `nvcc` is not on `PATH`) and
`CMAKE_CUDA_ARCHITECTURES` (via `nvidia-smi --query-gpu=compute_cap`, falling
back to `native`) in [cmake/NVCRAutodetect.cmake](../cmake/NVCRAutodetect.cmake).
Override either when cross-compiling or targeting a GPU that is not attached
to the build host:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DTensorRT_ROOT=/opt/TensorRT
```

For a portable multi-GPU-architecture build (the CMake equivalent of
`install.sh --arch-set portable`), set `NVCR_CUDA_ARCH_SET=portable` instead
of passing `CMAKE_CUDA_ARCHITECTURES` explicitly:

```bash
cmake -S . -B build-portable -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_CUDA_ARCH_SET=portable
```

## Verify the install

Run the test suite after building:

```bash
ctest --test-dir build --output-on-failure
```

On Jetson, run CTest in an environment with direct access to NVIDIA device nodes
such as `/dev/nvmap` and `/dev/nvhost-*`. Without those devices, CUDA runtime
tests can fail with `NvRmMemInitNvmap failed` even though the project built and
installed correctly.

If you used the install prefix path, confirm the CLI is reachable:

```bash
nvcr --help
```

## Generate engine artifacts

The native DCVC-RT backend needs TensorRT plans plus copied entropy,
quantization, and manifest assets. PyTorch checkpoints are used only during
offline export; `nvcr` does not load `.pth.tar` files at runtime.

The shortest path is the artifact wrapper. It auto-detects the platform,
device id, and a safe TensorRT workspace/builder-optimization budget via
[scripts/detect_platform.sh](../scripts/detect_platform.sh), so no manual
memory-tuning flags are required on either an 8 GB Orin Nano or a discrete GPU:

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root ./assets \
  --engines build/engines/dcvcrt \
  --skip-smoke
```

Pass `--no-auto-tune` plus explicit `--workspace-mib`, `--builder-optimization-level`,
and `--device-id` values to reproduce a specific prior configuration instead of
the auto-detected one.

The wrapper expects the public Microsoft DCVC-RT checkout and the two
pretrained checkpoints under `<dcvcrt-root>/checkpoints/`:

```text
assets/checkpoints/cvpr2025_image.pth.tar
assets/checkpoints/cvpr2025_video.pth.tar
```

It exports ONNX/runtime assets to `build/models/dcvcrt` and then builds
target-local TensorRT plans in the engine directory. ONNX files are portable;
TensorRT `.plan` files are not. Rebuild plans on each Jetson/GPU/TensorRT
runtime/device instead of copying them from another platform. The generated
`engine_manifest.json` records the build GPU and TensorRT runtime; `nvcr` rejects
the directory if those fields do not match the selected `--device-id`.

For a release-style install, copy or symlink the generated engine directory to
the path you pass with `--engine-dir`, for example `/opt/nvcr/engines/dcvcrt`.
See [DCVC-RT artifact pipeline](dcvcrt-artifacts.md) for checkpoint download
links, hashes, and advanced skip options.

## Use the CLI

The installed binary is `nvcr`. Encode raw planar YUV420p8 into an NVCR
sequence file:

```bash
nvcr encode \
  -i input.yuv \
  -o output.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir build/engines/dcvcrt
```

Decode the sequence later in a separate process:

```bash
nvcr decode \
  -i output.nvcr \
  -o output.decoded.yuv \
  --engine-dir build/engines/dcvcrt
```

If you are only exploring the docs, the next pages are [Architecture](architecture.md)
for the runtime boundary and [CLI](cli.md) for command details.
