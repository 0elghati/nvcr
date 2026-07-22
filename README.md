# NVCR — Neural Video Codec Runtime

NVCR is a modern C++20 deployment runtime for learned video codecs. Version 0.2.0 <!-- x-release-please-version -->
targets one codec only: **DCVC-RT**.

## About

Deployment-focused C++ runtime for DCVC-RT with deterministic lifecycle
control, stable bitstream framing, and release-ready packaging.

## Documentation

Start with [docs/getting-started.md](docs/getting-started.md) for installation
and first-run steps, or use the quick-start commands below.

Release automation and asset publishing are described in
[docs/releasing.md](docs/releasing.md).

## Contents

- [Quick Start](#quick-start)
- [Why NVCR?](#why-nvcr)
- [Architecture](#architecture)
- [Current scope](#current-scope)
- [Build](#build)
- [Dependencies](#dependencies)
- [Native CLI](#native-cli)
- [Library example](#library-example)
- [Roadmap](#roadmap)
- [License](#license)

The project is being developed to replace DCVC-RT's Python orchestration layer
with a production-shaped C++ library. It does not modify the neural architecture, replace TensorRT, or
reimplement entropy coding. Its purpose is to own lifecycle, state, memory,
bitstream framing, observability, and a stable application boundary that can later
serve FFmpeg.

> Status: the native C++ runtime encodes and decodes configured I/P GOPs through
> fourteen TensorRT plans and native rANS. It is **not yet a complete, production-
> ready DCVC-RT codec**: upstream reconstruction/stream conformance, GPU-resident
> orchestration, performance parity, and FFmpeg integration remain pending.

## Quick Start

### Install a published release

```bash
export NVCR_VERSION="v0.2.0" # x-release-please-version
export NVCR_PLATFORM="linux-x86_64-discrete"   # or linux-x86_64-portable or linux-aarch64-jetson
export NVCR_PREFIX="$HOME/.local/nvcr"

mkdir -p "$NVCR_PREFIX"
curl -fL "https://github.com/<your-org>/NVCR/releases/download/${NVCR_VERSION}/nvcr-${NVCR_VERSION}-${NVCR_PLATFORM}.tar.gz" -o /tmp/nvcr.tar.gz
tar -xzf /tmp/nvcr.tar.gz -C "$NVCR_PREFIX" --strip-components=1
export PATH="$NVCR_PREFIX/bin:$PATH"
nvcr --help
```

Use `linux-x86_64-portable` when you want one x86_64 build that runs across
multiple NVIDIA GPU architecture generations.

If a matching engine bundle is published for your platform:

```bash
curl -fL "https://github.com/<your-org>/NVCR/releases/download/${NVCR_VERSION}/dcvcrt-engines-${NVCR_VERSION}-${NVCR_PLATFORM}.tar.gz" -o /tmp/nvcr-engines.tar.gz
mkdir -p "$NVCR_PREFIX/engines"
tar -xzf /tmp/nvcr-engines.tar.gz -C "$NVCR_PREFIX/engines"
```

### Build from source

```bash
./scripts/install.sh --run-tests
```

### Build local TensorRT engines

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --python /path/to/DCVC-RT/src/venv/bin/python \
  --skip-clone --skip-smoke
```

For the full installation and packaging flow, see
[docs/getting-started.md](docs/getting-started.md),
[docs/install-binary.md](docs/install-binary.md), and
[docs/dcvcrt-artifacts.md](docs/dcvcrt-artifacts.md).

## Why NVCR?

Research code is optimized for changing models quickly. Deployment code has a
different job: deterministic ownership, bounded resources, explicit state,
recoverable errors, stable packaging, and no Python process in the runtime path.
NVCR provides that layer while keeping DCVC-RT's trained model and native entropy
coder authoritative.

## Architecture

```text
Application / future FFmpeg integration
                  |
          nvcr::Runtime API
                  |
     +------------+-------------+
     | lifecycle, config, logs,  |
     | memory, stats, bitstream  |
     +------------+-------------+
                  |
       DCVC-RT orchestration
        /                  \
TensorRT inference     Native C++ entropy coder
(GPU + CUDA stream)    (upstream adapter, CPU)
        \                  /
         explicit sequence state
    (references, latents, GOP, I/P)
```

TensorRT, CUDA tensor operations, and native rANS are owned by one
`dcvcrt::CodecBackend`. This is required because DCVC-RT alternates spatial-prior
inference and entropy decoding; they cannot be independent pre/post-processing
stages.

See [Architecture](docs/architecture.md), [bitstream format](docs/bitstream.md),
[DCVC-RT integration](docs/dcvcrt-integration.md), and
[performance](docs/performance.md) for the contracts and
ownership rules.

The gated implementation plan and current next action are maintained in the
[project roadmap](ROADMAP.md).

## Current scope

Version 0.1 includes:

- a C++20 runtime and explicit lifecycle;
- a codec-level backend boundary that can interleave TensorRT and rANS;
- native I-frame encode/decode through seven TensorRT plans;
- an installable `nvcr encode` / `nvcr decode` command-line tool;
- independent encoder and decoder sequence state;
- frame and compressed-packet ownership types;
- versioned, bounds-checked packet serialization;
- reusable memory resources and a best-fit host memory pool;
- dependency-neutral logging with an spdlog implementation when available;
- atomic encoding/decoding statistics;
- configuration loading and validation;
- optional TensorRT/CUDA, spdlog, CLI11, OpenCV, fmt, and GoogleTest discovery;
- installable CMake package targets.

Not yet included: upstream-compatible I/P bitstreams and reconstruction conformance,
FFmpeg glue, or automated end-to-end performance gates. Python is permitted only for offline model
export and reference conformance; it is not a deployed libnvcr dependency.

## Build

Requirements for the core build:

- CMake 3.24 or newer;
- a C++20 compiler (GCC, Clang, or MSVC).

The fastest path on Linux is `scripts/install.sh`, which auto-detects the
platform (Jetson vs. discrete GPU), CUDA compiler, GPU architecture, and
TensorRT location, then configures/builds/installs with one command:

```bash
./scripts/install.sh --run-tests
```

This works unmodified on a Jetson Orin or a discrete-GPU host and installs to
`install-<platform>/` by default. See [scripts/README.md](scripts/README.md)
and [docs/getting-started.md](docs/getting-started.md) for override flags. The
manual `cmake` steps below remain available for CI pipelines or advanced
cross-compilation cases.

By default the build targets only the GPU installed on the current machine
(fastest for local/dev use). To build a single redistributable binary that
works out of the box across a range of Jetson/RTX/datacenter GPUs, pass
`--arch-set portable`:

```bash
./scripts/install.sh --arch-set portable --run-tests
```

This widens `CMAKE_CUDA_ARCHITECTURES` to a curated multi-arch list
(`75;80;86;87;89;90`, see [cmake/NVCRAutodetect.cmake](cmake/NVCRAutodetect.cmake))
at the cost of a longer build. It is independent of host CPU architecture —
separate builds are still required per CPU arch (e.g. x86_64 vs. aarch64) —
and it never bundles TensorRT `.plan` engine files, which are always tied to
one exact GPU model/TensorRT runtime version and must be generated per-target
with `scripts/prepare_dcvcrt_artifacts.sh`.

On Linux, a release install can be used directly from the install prefix. For
example:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
cmake --install build-release --prefix /opt/nvcr
sudo export PATH="/opt/nvcr/bin:$PATH"
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix /desired/prefix
```

The default build is intentionally useful without a GPU SDK. It builds the public
core, example, and tests. Missing lightweight dependencies are not downloaded
unless requested:

```bash
cmake -S . -B build \
  -DNVCR_FETCH_DEPENDENCIES=ON \
  -DNVCR_ENABLE_TENSORRT=OFF
```

For a machine with CUDA and TensorRT:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/opt/TensorRT
```

On Jetson Orin devices, `nvcc` may not be on `PATH` and CMake may fail to infer
the GPU architecture. `-DNVCR_ENABLE_TENSORRT=ON` alone is normally enough:
[cmake/NVCRAutodetect.cmake](cmake/NVCRAutodetect.cmake) auto-detects
`CMAKE_CUDA_COMPILER` (searching `/usr/local/cuda*/bin/nvcc` when `nvcc` is not
on `PATH`) and `CMAKE_CUDA_ARCHITECTURES` (via `nvidia-smi --query-gpu=compute_cap`,
falling back to `native`). Override explicitly only when cross-compiling or
targeting a GPU not attached to the build host. For a portable multi-arch
build instead, set `-DNVCR_CUDA_ARCH_SET=portable` in place of an explicit
`CMAKE_CUDA_ARCHITECTURES`:

```bash
cmake -S . -B build-orin-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build-orin-install --parallel
cmake --install build-orin-install --prefix /opt/nvcr
export PATH="/opt/nvcr/bin:$PATH"
```

For an unprivileged local install, use a repository-local prefix such as
`--prefix "$PWD/install-orin"`. CUDA runtime tests on Jetson require direct
access to NVIDIA device nodes such as `/dev/nvmap` and `/dev/nvhost-*`; in a
sandbox or container without those devices, CPU tests and the installed CLI can
pass while `nvcr_cuda_ops` fails during CUDA stream creation.

The TensorRT option loads and validates the seven I-frame and seven P-frame plans.
Native multi-frame encoding uses the configured I/P GOP (default 32);
`--gop-size 1` remains an explicit development-only all-intra mode. The current
P-frame path is correctness-first and host-staged, so Release builds are required
for meaningful timing and GPU-resident pipeline optimization is still in progress.
When `NVCR_TENSORRT_ENGINE_DIR` is set, CTest runs an end-to-end 176x144 native
round trip and requires the encoder and decoder reconstructions to match exactly.

Generate the DCVC-RT artifacts before running the native sample or the
`nvcr encode` / `nvcr decode` commands. The wrapper starts from the public
Microsoft DCVC-RT checkout plus the two `cvpr2025_*.pth.tar` checkpoints,
exports ONNX/runtime assets, and builds target-local TensorRT plans. It
auto-tunes `--device-id`, `--workspace-mib`, and `--builder-optimization-level`
from [scripts/detect_platform.sh](scripts/detect_platform.sh):

```bash
./scripts/prepare_dcvcrt_artifacts.sh \
  --dcvcrt-root ./assets \
  --engines build/engines/dcvcrt \
  --skip-smoke
```

On an 8 GB Orin Nano this auto-tunes to a lower workspace and builder
optimization level, reducing TensorRT tactic-search memory pressure; pass
`--no-auto-tune` plus explicit flags to pin a specific configuration instead.

Place the checkpoints at:

```text
assets/checkpoints/cvpr2025_image.pth.tar
assets/checkpoints/cvpr2025_video.pth.tar
```

The lower-level `scripts/build_dcvcrt_tensorrt.sh` script only converts an
existing `build/models/dcvcrt` ONNX/runtime-asset directory into TensorRT plans.
It also writes `engine_manifest.json` with the build GPU, compute capability,
TensorRT version, precision, and profile settings. TensorRT `.plan` files are not
portable across platforms or TensorRT runtimes; NVCR validates the manifest before
loading plans and fails early if the selected `--device-id` does not match the
engine bundle. Rebuild the engine directory on the target machine/device instead
of copying plans across GPU models. For a release-style install, place or symlink
the generated directory at `/opt/nvcr/engines/dcvcrt` and pass it to
`--engine-dir`.

See [DCVC-RT artifact pipeline](docs/dcvcrt-artifacts.md) for checkpoint download
links, expected hashes, skip options, and direct Orin commands.

### Developer-only upstream reference

The optional `dcvcrt-reference` executable launches upstream PyTorch DCVC-RT for
golden-vector generation and conformance work. It is disabled by default, is not
installed, and is never part of the libnvcr runtime path. Enable it explicitly with
`-DNVCR_BUILD_REFERENCE_TOOLS=ON` when maintaining reference data.

Important options:

| Option | Default | Purpose |
|---|---:|---|
| `NVCR_BUILD_TESTS` | `ON` | Build tests; GoogleTest is used when available |
| `NVCR_BUILD_BENCHMARKS` | `OFF` | Build repeatable native microbenchmarks |
| `NVCR_BUILD_EXAMPLES` | `ON` | Build small API examples |
| `NVCR_BUILD_CLI` | `ON` | Build `nvcr` when TensorRT support is enabled |
| `NVCR_BUILD_REFERENCE_TOOLS` | `OFF` | Build the non-installed upstream reference tool |
| `NVCR_DCVCRT_ROOT` | empty | Default working upstream DCVC-RT checkout |
| `NVCR_FETCH_DEPENDENCIES` | `OFF` | Fetch missing lightweight dependencies |
| `NVCR_ENABLE_TENSORRT` | `OFF` | Enable the TensorRT/CUDA integration target |
| `NVCR_ENABLE_OPENCV` | `OFF` | Discover optional OpenCV interop support |
| `NVCR_ENABLE_SANITIZERS` | `OFF` | Enable ASan and UBSan on supported compilers |
| `NVCR_WARNINGS_AS_ERRORS` | `OFF` | Promote project warnings to errors |

Build and run the opt-in entropy benchmark with:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_BUILD_BENCHMARKS=ON
cmake --build build-release --target nvcr_rans_benchmark
./build-release/benchmarks/nvcr_rans_benchmark 10
```

## Dependencies

- **TensorRT + CUDA** — neural inference only; optional for the core build.
- **The pinned DCVC-RT native rANS coder** — vendored with its original license and
  validated against upstream byte streams. Entropy probability tables and complete
  staged orchestration are still being integrated.
- **spdlog** — logging backend when installed or fetched; a small console fallback
  keeps the core independently buildable.
- **GoogleTest** — unit-test framework when installed or fetched; dependency-free
  smoke tests remain available offline.
- **CLI11, fmt, OpenCV** — reserved for tools, formatting, and optional frame
  interop. None leaks into the stable core API.

## Native CLI

After installation, the CLI is available as `nvcr` on `PATH`. Encode raw planar
YUV420p8 into a persistent NVCR sequence bitstream:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

Decode it later in a separate process:

```bash
nvcr decode \
  -i /tmp/akiyo_qcif.nvcr \
  -o /tmp/akiyo_qcif.decoded.yuv \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

The encode command never invokes `Runtime::decode`; the decode command obtains
frame dimensions and timestamps from the bitstream. See [CLI usage](docs/cli.md)
for more examples and current format limitations.

## Documentation map

- [Docs index](docs/README.md)
- [Getting Started](docs/getting-started.md)
- [API Reference](docs/reference.md)
- [Architecture](docs/architecture.md)
- [Bitstream format](docs/bitstream.md)
- [CLI usage](docs/cli.md)
- [DCVC-RT artifact pipeline](docs/dcvcrt-artifacts.md)
- [DCVC-RT integration](docs/dcvcrt-integration.md)
- [Performance](docs/performance.md)

## Library example

Packet framing works without a codec backend:

```cpp
#include <nvcr/nvcr.hpp>

nvcr::Packet packet(
    {std::byte{0x01}, std::byte{0x02}},
    nvcr::Timestamp{41'667},
    nvcr::FrameType::intra);

auto wire = nvcr::PacketIO::serialize(packet);
if (!wire) {
    // wire.error() contains a stable code, subsystem, and message.
}
```

Applications construct `nvcr::Runtime` with one DCVC-RT codec backend that owns
TensorRT execution, CUDA tensor operations, rANS, and sequence-local codec state. This explicit injection prevents global
CUDA, logging, or codec state and makes process-level FFmpeg integration viable.

## Roadmap

The detailed, gated plan lives in [ROADMAP.md](ROADMAP.md). Its active milestone
is GPU-resident I-frame execution, followed by elementary-stream conformance,
P-frame support, a stable C ABI, and the FFmpeg wrapper.

Future codec support remains outside the v0.1 roadmap.

## License

NVCR is licensed under the [MIT License](LICENSE). Integrated third-party code
retains its own license.
