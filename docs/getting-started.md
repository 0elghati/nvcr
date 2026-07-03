# Getting Started

This guide is the shortest path from a source checkout to a usable NVCR install
on Linux.

## What you need

- CMake 3.24 or newer.
- A C++20 compiler.
- Optional: CUDA and TensorRT when you want the native DCVC-RT backend.

## Build the project

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

If you want the CUDA/TensorRT path enabled, configure it explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/opt/TensorRT
```

## Verify the install

Run the test suite after building:

```bash
ctest --test-dir build --output-on-failure
```

If you used the install prefix path, confirm the CLI is reachable:

```bash
nvcr --help
```

## Use the CLI

The installed binary is `nvcr`. Encode raw planar YUV420p8 into an NVCR
sequence file:

```bash
nvcr encode \
  -i input.yuv \
  -o output.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir /opt/nvcr/engines/dcvcrt
```

Decode the sequence later in a separate process:

```bash
nvcr decode \
  -i output.nvcr \
  -o output.decoded.yuv \
  --engine-dir /opt/nvcr/engines/dcvcrt
```

If you are only exploring the docs, the next pages are [Architecture](architecture.md)
for the runtime boundary and [CLI](cli.md) for command details.
