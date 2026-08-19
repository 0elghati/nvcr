# First run

Choose the section for your platform and follow it from top to bottom. Each
procedure installs the current release, records what was installed, creates a
small sample video, and verifies that NVCR can encode and decode it.

The sample contains four 176x144 YUV420 frames. Its expected size is:

```text
176 x 144 x 3 / 2 x 4 = 152064 bytes
```

Keep the resolved release tag or container digest if you later report a problem
or repeat the run.

## Linux x86_64 with an NVIDIA GPU and Docker

### Prerequisites

- Linux x86_64.
- Docker Engine.
- NVIDIA Container Toolkit with the `nvidia` runtime configured, and a driver
  compatible with the image's CUDA userspace. Confirm that `docker info`
  lists the `nvidia` runtime.
- A discrete NVIDIA GPU for which the public catalog contains a compatible
  QCIF engine.
- Python 3 and network access.

Verify the host-to-container GPU path:

```bash
docker version
docker run --rm --runtime=nvidia \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Do not continue until the container lists the intended GPU.

The Linux commands below use the configured NVIDIA runtime because that is the
NVCR container path validated on the documented host. Docker's `--gpus all`
form is an alternative; use one form consistently throughout this procedure.

### Resolve the runtime image

NVCR publishes an immutable versioned tag for each release. Select the release
version, pull that tag, resolve it to an immutable digest, and print its
metadata:

```bash
export NVCR_VERSION="<version>"
export NVCR_IMAGE="omarelghati/nvcr:${NVCR_VERSION}-amd64-cuda12.8-trt10.9"
docker pull "$NVCR_IMAGE"

export NVCR_IMAGE_REF="$(
  docker image inspect "$NVCR_IMAGE" --format '{{ index .RepoDigests 0 }}'
)"
docker image inspect "$NVCR_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Use `NVCR_IMAGE_REF` for the remaining commands.
`latest-amd64-cuda12.8-trt10.9` is only a rolling convenience alias.

### Generate the input

The following standard-library Python program writes the deterministic sample;
it does not require an NVCR source checkout:

```bash
export NVCR_WORK_DIR="$PWD/nvcr-example"
mkdir -p "$NVCR_WORK_DIR"

python3 - "$NVCR_WORK_DIR/input.yuv" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

width, height, frames = 176, 144, 4
path = Path(sys.argv[1])
with path.open("wb") as stream:
    for frame in range(frames):
        stream.write(bytes(
            16 + ((3 * x + 5 * y + 17 * frame) % 220)
            for y in range(height) for x in range(width)
        ))
        stream.write(bytes(
            16 + ((7 * x + 11 * y + 13 * frame + 41) % 225)
            for y in range(height // 2) for x in range(width // 2)
        ))
        stream.write(bytes(
            16 + ((5 * x + 3 * y + 19 * frame + 97) % 225)
            for y in range(height // 2) for x in range(width // 2)
        ))
data = path.read_bytes()
print(f"path={path.resolve()} bytes={len(data)} sha256={sha256(data).hexdigest()}")
PY

test "$(stat -c %s "$NVCR_WORK_DIR/input.yuv")" -eq 152064
touch "$NVCR_WORK_DIR/output.nvcr" "$NVCR_WORK_DIR/reconstructed.yuv"
```

### Install the QCIF engine

The runtime image does not embed TensorRT plans. Install the best compatible
published QCIF bundle into a persistent volume:

```bash
docker volume create nvcr-engines

docker run --rm --runtime=nvidia \
  --volume nvcr-engines:/opt/nvcr/engines \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE_REF" \
  install --profile qcif --engine-root /opt/nvcr/engines

docker run --rm \
  --volume nvcr-engines:/opt/nvcr/engines:ro \
  --entrypoint /opt/nvcr/bin/nvcr-artifacts \
  "$NVCR_IMAGE_REF" \
  inspect /opt/nvcr/engines/profiles/dcvcrt/qcif --json
```

Catalog selection prefers an exact bundle, then a matching compute-capability
bundle, then an applicable Ampere-plus bundle when one is published. It never
uses an SM 8.9 or SM 12.0 bundle on a GPU with a different compute capability.
Jetson is exact-target only.

### Inspect the runtime

The image entrypoint is `nvcr`, so commands after the image name begin with
the NVCR subcommand:

```bash
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" codec list
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" codec describe dcvc-rt
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" provider list
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" provider describe tensorrt
docker run --rm --runtime=nvidia "$NVCR_IMAGE_REF" \
  compatibility check --codec dcvc-rt --provider tensorrt
```

The compatibility command verifies registry wiring. Engine selection and
TensorRT initialization perform the target/runtime checks.

### Encode, decode, and verify

```bash
docker run --rm --runtime=nvidia \
  --volume nvcr-engines:/opt/nvcr/engines:ro \
  --mount "type=bind,source=$NVCR_WORK_DIR,target=/work" \
  "$NVCR_IMAGE_REF" encode \
  -i /work/input.yuv -o /work/output.nvcr \
  -s 176x144 -r 30 -c dcvc-rt --backend dcvcrt \
  --frames 4 --gop-size 2 --qp 32 --verbose

docker run --rm --runtime=nvidia \
  --volume nvcr-engines:/opt/nvcr/engines:ro \
  --mount "type=bind,source=$NVCR_WORK_DIR,target=/work" \
  "$NVCR_IMAGE_REF" decode \
  -i /work/output.nvcr -o /work/reconstructed.yuv \
  -c dcvc-rt --backend dcvcrt \
  --quality-metrics /work/input.yuv

test -s "$NVCR_WORK_DIR/output.nvcr"
test "$(stat -c %s "$NVCR_WORK_DIR/reconstructed.yuv")" -eq 152064
```

Success means discovery reports `dcvc-rt` and `tensorrt`, the selected
bundle is identified during initialization, encode and decode each process four
frames, the encoded file is non-empty, and the reconstructed file is 152064
bytes. Lossy reconstruction is not byte-identical to the input.

Platform setup and failure diagnosis are covered in
[Docker on Linux](docker-x86_64.md) and
[Troubleshooting](troubleshooting.md).

## Windows 11 with Docker Desktop and an NVIDIA GPU

This path runs the Linux/amd64 image through Docker Desktop's WSL 2 backend. It
does not provide a native Windows NVCR executable.

### Prerequisites and preflight

Install Docker Desktop with the WSL 2 engine, enable GPU support, install a
current NVIDIA Windows driver, and make Python 3 available through the `py`
launcher. In PowerShell:

```powershell
wsl --status
docker version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### Resolve the image and generate the input

```powershell
$NvcrImage = "omarelghati/nvcr:latest-amd64-cuda12.8-trt10.9"
docker pull $NvcrImage
$NvcrImageRef = docker image inspect $NvcrImage --format '{{ index .RepoDigests 0 }}'
docker image inspect $NvcrImage --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'

$WorkDir = Join-Path (Get-Location) "nvcr-example"
$InputPath = Join-Path $WorkDir "input.yuv"
$OutputPath = Join-Path $WorkDir "output.nvcr"
$ReconstructedPath = Join-Path $WorkDir "reconstructed.yuv"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

@'
from hashlib import sha256
from pathlib import Path
import sys

width, height, frames = 176, 144, 4
path = Path(sys.argv[1])
with path.open("wb") as stream:
    for frame in range(frames):
        stream.write(bytes(16 + ((3*x + 5*y + 17*frame) % 220) for y in range(height) for x in range(width)))
        stream.write(bytes(16 + ((7*x + 11*y + 13*frame + 41) % 225) for y in range(height//2) for x in range(width//2)))
        stream.write(bytes(16 + ((5*x + 3*y + 19*frame + 97) % 225) for y in range(height//2) for x in range(width//2)))
data = path.read_bytes()
print(f"path={path.resolve()} bytes={len(data)} sha256={sha256(data).hexdigest()}")
'@ | py -3 - $InputPath

if ((Get-Item $InputPath).Length -ne 152064) { throw "Input size is not 152064 bytes" }
New-Item -ItemType File -Force -Path $OutputPath, $ReconstructedPath | Out-Null
```

### Install and inspect the runtime

```powershell
docker volume create nvcr-engines
docker run --rm --gpus all --volume nvcr-engines:/opt/nvcr/engines --entrypoint /opt/nvcr/bin/nvcr-artifacts $NvcrImageRef install --profile qcif --engine-root /opt/nvcr/engines

docker run --rm --gpus all $NvcrImageRef codec list
docker run --rm --gpus all $NvcrImageRef codec describe dcvc-rt
docker run --rm --gpus all $NvcrImageRef provider list
docker run --rm --gpus all $NvcrImageRef provider describe tensorrt
docker run --rm --gpus all $NvcrImageRef compatibility check --codec dcvc-rt --provider tensorrt
```

### Encode, decode, and verify

```powershell
docker run --rm --gpus all --volume nvcr-engines:/opt/nvcr/engines:ro --mount "type=bind,source=$WorkDir,target=/work" $NvcrImageRef encode -i /work/input.yuv -o /work/output.nvcr -s 176x144 -r 30 -c dcvc-rt --backend dcvcrt --frames 4 --gop-size 2 --qp 32 --verbose
docker run --rm --gpus all --volume nvcr-engines:/opt/nvcr/engines:ro --mount "type=bind,source=$WorkDir,target=/work" $NvcrImageRef decode -i /work/output.nvcr -o /work/reconstructed.yuv -c dcvc-rt --backend dcvcrt --quality-metrics /work/input.yuv

if ((Get-Item $OutputPath).Length -le 0) { throw "Encoded output is empty" }
if ((Get-Item $ReconstructedPath).Length -ne 152064) { throw "Decoded size is not 152064 bytes" }
```

The Linux and Windows paths have the same functional success criteria. Report
Windows results as Linux container execution through Docker Desktop/WSL 2, not
as native Windows performance. See [Docker on Windows](docker-windows.md).

## Jetson Orin native installation

The native AArch64 package is the recommended Jetson path. It resolves the
latest stable GitHub release independently of the rolling Jetson container,
which can temporarily trail the native release.

### Prerequisites and preflight

This path requires Jetson Orin, JetPack 6.1 / L4T 36.4, CUDA 12.6, TensorRT
10.3, `curl`, `tar`, `sha256sum`, Python 3, and a compatible exact-target
QCIF catalog entry.

```bash
uname -m
head -n 1 /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
dpkg-query -W 'libnvinfer*' 2>/dev/null | head
```

### Resolve and install the latest stable release

```bash
export NVCR_RELEASE="$(
  curl -fsSL https://api.github.com/repos/0elghati/nvcr/releases/latest |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])'
)"
printf 'NVCR release: %s\n' "$NVCR_RELEASE"

curl -fsSLo /tmp/nvcr-install.sh \
  "https://raw.githubusercontent.com/0elghati/nvcr/${NVCR_RELEASE}/scripts/install.sh"
bash /tmp/nvcr-install.sh \
  --tag "$NVCR_RELEASE" --profile qcif --run-tests

export PATH="$HOME/.local/nvcr/bin:$PATH"
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
```

The installer selects the AArch64 archive, verifies its published checksum,
and installs only the QCIF engine. Retain `NVCR_RELEASE` with the result.

### Generate and inspect

```bash
export NVCR_WORK_DIR="$PWD/nvcr-example"
mkdir -p "$NVCR_WORK_DIR"

python3 - "$NVCR_WORK_DIR/input.yuv" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

width, height, frames = 176, 144, 4
path = Path(sys.argv[1])
with path.open("wb") as stream:
    for frame in range(frames):
        stream.write(bytes(16 + ((3*x + 5*y + 17*frame) % 220) for y in range(height) for x in range(width)))
        stream.write(bytes(16 + ((7*x + 11*y + 13*frame + 41) % 225) for y in range(height//2) for x in range(width//2)))
        stream.write(bytes(16 + ((5*x + 3*y + 19*frame + 97) % 225) for y in range(height//2) for x in range(width//2)))
data = path.read_bytes()
print(f"path={path.resolve()} bytes={len(data)} sha256={sha256(data).hexdigest()}")
PY
test "$(stat -c %s "$NVCR_WORK_DIR/input.yuv")" -eq 152064

nvcr codec list
nvcr codec describe dcvc-rt
nvcr provider list
nvcr provider describe tensorrt
nvcr compatibility check --codec dcvc-rt --provider tensorrt
nvcr-artifacts inspect \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
```

### Encode, decode, and verify

```bash
nvcr encode \
  -i "$NVCR_WORK_DIR/input.yuv" \
  -o "$NVCR_WORK_DIR/output.nvcr" \
  -s 176x144 -r 30 -c dcvc-rt --backend dcvcrt \
  --frames 4 --gop-size 2 --qp 32 --verbose

nvcr decode \
  -i "$NVCR_WORK_DIR/output.nvcr" \
  -o "$NVCR_WORK_DIR/reconstructed.yuv" \
  -c dcvc-rt --backend dcvcrt \
  --quality-metrics "$NVCR_WORK_DIR/input.yuv"

test -s "$NVCR_WORK_DIR/output.nvcr"
test "$(stat -c %s "$NVCR_WORK_DIR/reconstructed.yuv")" -eq 152064
```

Success means the installed release and exact-target bundle are identified,
discovery succeeds, both operations process four frames, and the output checks
pass. See [Jetson](docker-jetson.md) for the target boundary.

## CPU-only contract validation

This path validates runtime, codec/provider, artifact, entropy, access-unit,
parser, and lifecycle contracts. It does not build the production CLI or
execute DCVC-RT/TensorRT inference. It requires Git, curl, Python 3, CMake
3.24 or newer, a C++20 compiler, and a standard build tool.

Resolve and check out the latest stable source:

```bash
export NVCR_RELEASE="$(
  curl -fsSL https://api.github.com/repos/0elghati/nvcr/releases/latest |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])'
)"
git clone --branch "$NVCR_RELEASE" --depth 1 \
  https://github.com/0elghati/nvcr.git nvcr
cd nvcr
git describe --tags --exact-match
```

Configure a Release build without TensorRT:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu -N
ctest --test-dir build-cpu --output-on-failure
```

Success means the checked-out tag equals `NVCR_RELEASE`, configuration
reports that the TensorRT CLI is skipped, the library and registered CPU tests
build, and CTest reports all configured tests passing. The precise test count
can change between releases; `ctest -N` is authoritative for the resolved
tag.

For development on `main`, follow [Building from source](building-from-source.md)
and [Contributing](../CONTRIBUTING.md), and record the exact source commit
rather than describing it as a release.
