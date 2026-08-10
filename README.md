# NVCR

NVCR is a neural video codec runtime for NVIDIA GPUs. It provides a command
line tool and a C++ library for encoding and decoding DCVC-RT video with
TensorRT.

This page covers the quickest way to install and use NVCR.

## Requirements

NVCR currently runs on:

- Linux
- An NVIDIA GPU with a working NVIDIA driver
- TensorRT-compatible engine files for that GPU and runtime
- Planar 8-bit YUV420 input and output

The prebuilt installer also needs `curl` with HTTPS support, CA certificates,
`tar`, `sha256sum`, and Python 3. On Ubuntu, install them with:

```bash
sudo apt update
sudo apt install -y ca-certificates curl tar coreutils python3
```

For Docker, install Docker with Compose support and the NVIDIA Container
Toolkit. On an x86_64 host, check that Docker can see the GPU before using NVCR:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

NVCR is not a CPU video decoder. Windows, macOS, FFmpeg, and standard video
containers are not supported by the current runtime.

## Installation

### Prebuilt installation

The installer downloads the latest NVCR binary, verifies its SHA-256 checksum,
and installs compatible TensorRT engine profiles for the current machine.

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash
```

The default locations are:

- NVCR programs: `~/.local/nvcr/bin`
- TensorRT engines: `~/.local/share/nvcr/engines`

Add the programs to the current shell's `PATH`:

```bash
export PATH="$HOME/.local/nvcr/bin:$PATH"
nvcr --help
```

To install only selected resolution profiles, add one or more `--profile`
options:

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh \
  | bash -s -- --profile 360p --profile 720p
```

If no compatible engine is available for the machine, follow the manual
preparation steps in [Model and engine preparation](docs/dcvcrt-artifacts.md).

### First encode and decode

NVCR reads planar 8-bit YUV420. The dimensions passed to `-s` must match the
input file.

```bash
mkdir -p output

nvcr encode \
  -i input.yuv \
  -o output/video.nvcr \
  -s 176x144 \
  -r 30 \
  --frames 4 \
  --gop-size 2 \
  --qp 32

nvcr decode \
  -i output/video.nvcr \
  -o output/reconstructed.yuv
```

The encoder selects a compatible engine profile from the video dimensions. The
decoder reads the dimensions from the encoded stream and selects the matching
profile.

## Docker

Docker is useful when you want a fixed CUDA, TensorRT, and compiler environment.
The container still needs access to the host NVIDIA GPU, and TensorRT engines
must match the GPU and TensorRT runtime.

Clone the repository, then run the following on an x86_64 NVIDIA desktop or
workstation:

```bash
git clone https://github.com/0elghati/nvcr.git
cd nvcr

mkdir -p input output
export NVCR_INPUT_DIR="$PWD/input"
export NVCR_OUTPUT_DIR="$PWD/output"

docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install

docker compose -f docker/compose.x86_64.yaml run --rm nvcr encode \
  -i /input/input.yuv \
  -o /output/video.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

docker compose -f docker/compose.x86_64.yaml run --rm nvcr decode \
  -i /output/video.nvcr \
  -o /output/reconstructed.yuv
```

The `engine-install` step downloads compatible engine files into a persistent
Docker volume. The image does not contain checkpoints or TensorRT engines.

On a Jetson Orin, use the Jetson Compose file instead:

```bash
docker compose -f docker/compose.jetson.yaml run --rm --build engine-install
docker compose -f docker/compose.jetson.yaml run --rm nvcr --help
```

The Jetson image must be built and run on the Jetson. See [Docker execution and
development](docs/docker.md) for image options, local engine bundles, WSL2, and
test commands.

## Build from source

For a native TensorRT build, use Linux with CMake 3.24 or newer, a C++20
compiler, CUDA, and TensorRT:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

For the CPU library and format/API tests only:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

TensorRT plans are tied to their target GPU and runtime. Build them on the
machine where they will run. NVCR does not distribute model checkpoints,
exported model files, or TensorRT plans in the source repository.

## Learn more

- [Command line reference](docs/cli.md)
- [Model and engine preparation](docs/dcvcrt-artifacts.md)
- [Compatibility](docs/compatibility.md)
- [Docker execution and development](docs/docker.md)
- [C++ API reference](docs/reference.md)
- [Architecture](docs/architecture.md)
- [License and third-party notices](LICENSE)
