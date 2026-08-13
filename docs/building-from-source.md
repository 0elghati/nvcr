# Building from source

Use a source build when developing NVCR, validating the CPU contracts, or
targeting a Linux/NVIDIA machine that needs a local build. Installing a
packaged build is simpler for ordinary use; see [Installation](installation.md).

## Obtain the source

Clone the current source and record the exact commit used for the build:

```bash
git clone https://github.com/0elghati/nvcr.git
cd nvcr
git rev-parse HEAD
```

Record the commit, build options, and target identity when reporting a result.

## Common prerequisites

- Linux.
- CMake 3.24 or newer.
- A C++20 compiler.
- Git and a standard build tool such as Ninja or Make.
- Python 3 for artifact/install validation tests.

The repository vendors its native rANS dependency. `spdlog`, GoogleTest, and
other lightweight conveniences are optional. With
`NVCR_FETCH_DEPENDENCIES=OFF`, missing GoogleTest causes a dependency-free
basic test to be built instead. Set `NVCR_FETCH_DEPENDENCIES=ON` only when
network downloads during configure are acceptable.

## CPU contract build

This configuration validates the portable source and toolchain contracts:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

Expected result: configure prints
`NVCR CLI skipped because NVCR_ENABLE_TENSORRT=OFF`, the library and tests
build, and CTest passes.

This validates public values, format/parser bounds, entropy, artifact
resolution, registry behavior, and test codec/provider contracts. It does not
build the `nvcr` CLI and does not provide CPU DCVC-RT inference.
`nvcr-artifacts` remains available as the source script
`scripts/nvcr_artifacts.py`.

Install the CPU library and CMake package for a local consumer:

```bash
cmake --install build-cpu --prefix "$PWD/install-cpu"
find "$PWD/install-cpu" -maxdepth 3 -type f | sort
```

The installed library/API is not ABI-frozen. Keep the
consumer and NVCR revision together.

## TensorRT runtime build

Additional prerequisites are:

- an NVIDIA GPU and working driver;
- the CUDA compiler and development headers;
- TensorRT headers and libraries; and
- a target-compatible engine bundle for real codec tests.

For a local machine, the source installer detects Jetson versus a discrete GPU,
`nvcc`, TensorRT, and the current GPU architecture:

```bash
./scripts/install_from_source.sh \
  --build-dir build-release \
  --prefix "$PWD/install-release" \
  --build-type Release \
  --run-tests
```

Expected result: the detection report names the platform, CUDA compiler,
architecture, and TensorRT root; CMake builds the TensorRT backend and CLI;
CTest passes the tests registered for that configuration; and installation
ends with:

```text
NVCR installed to: .../install-release
```

Add the commands to the shell:

```bash
export PATH="$PWD/install-release/bin:$PATH"
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
nvcr codec list
nvcr provider list
```

Building with TensorRT does not create a TensorRT plan and does not guarantee
that an existing plan is compatible. Install the strongest compatible catalog
bundle available for the detected target:

```bash
nvcr-artifacts install \
  --profile qcif \
  --backend dcvcrt \
  --device-id 0 \
  --engine-root "$NVCR_ENGINE_ROOT"
```

For each profile, installation ranks `exact`, then
`same_compute_capability`, then `ampere_plus`. Same-compute matching uses
both numeric compute-capability components: SM 8.9 does not match SM 8.6 or
SM 12.0. Jetson is exact-only. Every class still requires the TensorRT
`major.minor.patch` recorded in the engine manifest.

The public catalog is anonymous. A no-compatible-engine error must be resolved
by creating and validating an appropriate bundle, not by adding a GitHub token.
Local engine preparation is an advanced source-checkout workflow; see
[DCVC-RT model and TensorRT engine artifacts](dcvcrt-artifacts.md). When that
workflow receives `--target-profile`, the JSON must describe the machine
performing the TensorRT build.

### Register the real engine tests

TensorRT engine-contract and I/P round-trip tests are registered only when
CMake receives an engine directory containing `engine_manifest.json`.
Reconfigure the same build after installation:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif"
cmake --build build-release --parallel
ctest --test-dir build-release -N
ctest --test-dir build-release \
  --output-on-failure \
  -R 'nvcr_tensorrt_(engine_contracts|ip_frame_roundtrip)_qcif'
```

Expected result: `ctest -N` lists both `qcif` TensorRT tests and the
filtered run passes on the target. If the tests are absent, inspect the CMake
configure output and the selected bundle path; a plain TensorRT build without
an engine does not register them.

To register additional profiles, pass a semicolon-separated list:

```bash
cmake -S . -B build-release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" \
  -DNVCR_TENSORRT_ENGINE_DIRS="$NVCR_ENGINE_ROOT/profiles/dcvcrt/cif;$NVCR_ENGINE_ROOT/profiles/dcvcrt/720p"
```

Only list bundles that exist on this machine.

## Manual TensorRT configuration

Use explicit CMake arguments when autodetection is wrong or a non-default SDK
is intentional:

```bash
cmake -S . -B build-release-explicit -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTensorRT_ROOT=/opt/TensorRT
cmake --build build-release-explicit --parallel
```

Replace architecture `89` and `TensorRT_ROOT` with values for the actual
target. The `auto` architecture mode is appropriate for a local build.
`./scripts/install_from_source.sh --arch-set portable` builds a larger
multi-architecture host binary for packaging; it does not make TensorRT plans
portable.

TensorRT's same-compute-capability and Ampere-and-newer plan modes are separate
from `CMAKE_CUDA_ARCHITECTURES`. They are requested through the artifact
builder from a source checkout, validated against an exact baseline, and remain
bound to the exact TensorRT version in their manifests. Builder support does
not imply that an Ampere-and-newer catalog bundle is available.

## Troubleshooting

- CMake cannot find `nvcc`: pass
  `--cuda-compiler /absolute/path/to/nvcc` to the source installer or
  `-DCMAKE_CUDA_COMPILER=...` to CMake.
- CMake cannot find TensorRT: pass `--tensorrt-root /absolute/path` or
  `-DTensorRT_ROOT=...`, and confirm that both `NvInfer.h` and
  `libnvinfer` come from the intended installation.
- An engine fails deserialization or reports a runtime mismatch: require the
  manifest's exact TensorRT `major.minor.patch`. A same-compute-capability or
  Ampere-and-newer plan cannot resolve a TensorRT mismatch; install or rebuild
  with the active runtime.
- A reconfigure retains unwanted SDK paths: use a new, explicitly named build
  directory. Do not assume deleting a single cache value changes every
  detected dependency.

For contributor workflow and test layers, continue with
[Contributing](../CONTRIBUTING.md).
