# Docker execution and development

NVCR provides separate Docker definitions for its two reference architecture
families. They are deliberately not published as a single multi-platform image:
the CPU architecture, CUDA code generation, NVIDIA userspace, and supported host
runtime differ, while TensorRT plans remain tied to the exact target that built
them.

| Definition | Docker platform | CUDA architecture | Reference target |
|---|---|---:|---|
| `docker/Dockerfile.x86_64` | `linux/amd64` | CUDA 12.8 / TensorRT 10.9, using the compiler-supported architecture set | RTX 4070 / Ubuntu 24.04 and validated desktop targets |
| `docker/Dockerfile.jetson` | `linux/arm64` | SM 8.7 | Jetson Orin Nano / L4T 36.4.x |

Each Dockerfile has a `test` target for one-command validation, a
`development` target for an editor/devcontainer, and a `runtime` target
containing an installed Release CLI. None contains checkpoints, ONNX files,
model assets, TensorRT plans, input video, or generated output.

## Published runtime images

Runtime images use architecture- and NVIDIA-userspace-qualified Docker Hub tags.
There is intentionally no shared `latest` tag:

| Runtime family | Immutable tag | Moving family tag |
|---|---|---|
| amd64 / CUDA 12.8 / TensorRT 10.9 | `VERSION-amd64-cuda12.8-trt10.9` | `amd64-cuda12.8-trt10.9` |
| Jetson / L4T 36.4 / JetPack 6.1 | `VERSION-jetson-l4t36.4` | `jetson-l4t36.4` |

Docker image tags use `amd64`, matching the OCI/Docker `linux/amd64` platform
name. Repository internals may still use `x86_64` for compiler, filesystem, and
target-profile naming.

The default Docker Hub repository is `omarelghati/nvcr`. Override
`NVCR_DOCKERHUB_REPOSITORY` if the repository lives under another namespace.
Build and inspect the current amd64 image locally:

```bash
./docker/publish.sh --load amd64-cuda12.8-trt10.9
```

Publishing is deliberately guarded: the worktree must be clean and HEAD must
exactly match `v$(cat version.txt)`.

```bash
docker login
./docker/publish.sh --push amd64-cuda12.8-trt10.9
```

Run the Jetson form only on the native aarch64 Jetson target:

```bash
docker login
./docker/publish.sh --push jetson
```

The manual `publish-containers` GitHub workflow performs the same guarded
release build. Configure `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository
secrets. The Jetson job additionally requires a self-hosted runner labeled
`jetson`; hosted x86_64 emulation is not used for that image.

To consume a published amd64 image with Compose:

```bash
export NVCR_X86_64_IMAGE="omarelghati/nvcr:0.7.0-amd64-cuda12.8-trt10.9"
export NVCR_INPUT_DIR="/path/to/yuv-input"
export NVCR_OUTPUT_DIR="/path/to/nvcr-output"

docker compose -f docker/compose.x86_64.yaml pull nvcr
docker compose -f docker/compose.x86_64.yaml run --rm nvcr --help
```

Use `NVCR_JETSON_IMAGE` with `docker/compose.jetson.yaml` on Jetson. The
images remain engine-free. Each Compose file provides a one-shot
`engine-install` service that runs the installed `nvcr-artifacts install`
client against the rolling `engine-assets` catalog and stores every
exact-compatible profile in a persistent named volume. For a private catalog
repository, export `GH_TOKEN` with repository read access before running Compose;
the `engine-install` service forwards it to the catalog client:

```bash
docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install
```

To install only selected resolutions, pass profiles after the service command:

```bash
docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install \
  --profile 360p 720p 1080p
```

Run the equivalent command with `docker/compose.jetson.yaml` on Jetson. The
`nvcr` and `test` services mount that volume read-only. Re-running
`engine-install` refreshes the collection from the authoritative catalog.
Set `NVCR_ENGINE_ROOT` to an absolute host collection path to use locally
built bundles instead of the Compose-managed volume.

## Test quick start

The test image is the primary validation path. It validates the mounted
engine bundle, configures the matching CUDA architecture, builds the Release
library/CLI/tests, prints the tests that were actually registered, and runs them
with failure output enabled.

On the RTX/discrete-GPU host, install the exact-compatible rolling assets
and select one canonical profile for the registered integration suite:

```bash
docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install
NVCR_TEST_ENGINE_PROFILE=qcif \
docker compose -f docker/compose.x86_64.yaml run --rm --build test gpu
```

On the Jetson host:

```bash
docker compose -f docker/compose.jetson.yaml run --rm --build engine-install
NVCR_TEST_ENGINE_PROFILE=qcif \
docker compose -f docker/compose.jetson.yaml run --rm --build test gpu
```

To test a locally built collection instead, skip `engine-install` and set
`NVCR_ENGINE_ROOT="$PWD/build/engines"`. The test wrapper resolves
`dcvcrt-$NVCR_TEST_ENGINE_PROFILE` inside either collection. It still accepts
a directly mounted single bundle through `NVCR_ENGINE_DIR` outside Compose.

The Compose-managed build volume makes later test runs incremental. A developer
without a compatible local engine bundle can still compile the library and run
the parser, rANS, configuration, and artifact-profile suite:

```bash
docker build --platform linux/amd64 --target test \
  -f docker/Dockerfile.x86_64 -t nvcr-test:rtx4070 .
docker run --rm nvcr-test:rtx4070 cpu
```

CPU testing is useful, but it does not register or pass the CUDA, engine-contract,
or native I/P roundtrip gates. The `ctest -N` output in the test log makes that
difference visible.

## SoftwareX runs

The experiment runbook is the authoritative sequence for CPU, desktop GPU, and
Jetson evaluation. It keeps container identity separate from engine identity:
containers reproduce userspace, while TensorRT bundles remain exact or explicitly
classified target artifacts. See [SoftwareX experiments](experiments/README.md)
and [the runbook](experiments/runbook.md).

The publication driver lives in the source tree:

```bash
python3 scripts/benchmark_softwarex_matrix.py --help
```

The runtime image contains the installed application only. Run the driver from
a native source build or a matching source/test environment with inputs and
engines mounted read-only and evidence output writable. Container runs must
record both the image tag and immutable digest; native runs record
`--native-build-id`. The image TensorRT userspace must match the selected
engine bundles.

## Host prerequisites

The x86_64 host needs the NVIDIA driver, Docker, and NVIDIA Container Toolkit.
The x86_64 Compose services also map `/dev/nvidia-uvm` and
`/dev/nvidia-uvm-tools` explicitly. This preserves CUDA access on Docker/toolkit
combinations that expose those nodes through `--gpus all` but omit their
device-cgroup rules.
WSL2 exposes GPU compute through Docker Desktop rather than host
`/dev/nvidia-uvm*` nodes. Apply the supplied WSL override after confirming that
`docker run --rm --gpus all ubuntu nvidia-smi` succeeds. For the registered RTX
5060 Laptop migration target, also apply the Blackwell overlay:

```bash
docker compose \
  -f docker/compose.x86_64.yaml \
  -f docker/compose.blackwell.yaml \
  -f docker/compose.wsl.yaml \
  build engine-install
```

This selects CUDA 12.8, TensorRT 10.9, native SM 120 code generation, and the
`rtx5060-laptop-ubuntu2404` target profile. It is an exact-target migration
baseline and does not make existing RTX 4070 plans portable. Evaluate
TensorRT `AMPERE_PLUS` plans only after exact RTX 5060 correctness and
performance evidence exists; edge bundles remain target-local.
The Jetson needs the matching JetPack/L4T installation, Docker, and the NVIDIA
runtime installed by JetPack. Check GPU injection before debugging NVCR:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The Jetson image must be built and run on the Jetson. QEMU/cross-platform image
construction is not a supported substitute because NVCR engines must also be
built and verified on the final GPU/runtime target.

## RTX/discrete-GPU execution image

Build the SM 8.9 image from the repository root:

```bash
docker build \
  --platform linux/amd64 \
  --target runtime \
  -f docker/Dockerfile.x86_64 \
  -t nvcr:rtx4070 .
```

Run with read-only input and engine directories plus a writable output
directory. GPU runtime containers default to root because NVIDIA capability
devices are root-only on some supported hosts. Set `NVCR_HOST_UID` and
`NVCR_HOST_GID` for Compose, or add `--user "$(id -u):$(id -g)"` to a direct
Docker command, only when that user can initialize CUDA on the host:

```bash
docker run --rm --gpus all \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e NVCR_ENGINE_ROOT=/opt/nvcr/engines \
  -v "$PWD/build/engines:/opt/nvcr/engines:ro" \
  -v "/path/to/yuv-input:/input:ro" \
  -v "/path/to/nvcr-output:/output" \
  nvcr:rtx4070 encode \
    -i /input/input.yuv -o /output/output.nvcr \
    -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32
```

With Compose, run `engine-install` once to populate the persistent rolling
catalog volume, then set only the input and output directories. Encoding selects
the matching bundle from `-s`; decoding reads the dimensions embedded in the
first access unit and selects the same bundle automatically:

```bash
export NVCR_INPUT_DIR="/path/to/yuv-input"
export NVCR_OUTPUT_DIR="/path/to/nvcr-output"

docker compose -f docker/compose.x86_64.yaml run --rm --build engine-install
docker compose -f docker/compose.x86_64.yaml run --rm --build nvcr encode \
  -i /input/input.yuv -o /output/output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

docker compose -f docker/compose.x86_64.yaml run --rm nvcr decode \
  -i /output/output.nvcr -o /output/reconstructed.yuv
```

Mount directories rather than individual output files. Docker can then create
both outputs; an individual bind-mounted output file would have to exist on the
host before container startup. Root-default runs may create root-owned output;
change ownership afterward or use the UID/GID override when device permissions
allow it. If the engine collection is elsewhere, set
`NVCR_ENGINE_ROOT` to its host directory. `NVCR_ENGINE_DIR` and
`--engine-dir` remain explicit single-bundle overrides.

## Jetson execution image

Run these commands on the Orin target:

```bash
docker build \
  --platform linux/arm64 \
  --target runtime \
  -f docker/Dockerfile.jetson \
  -t nvcr:orin-nano .

docker run --rm --runtime=nvidia --network=host \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NVCR_ENGINE_ROOT=/opt/nvcr/engines \
  -v "$PWD/build/engines:/opt/nvcr/engines:ro" \
  -v "/path/to/yuv-input:/input:ro" \
  -v "/path/to/nvcr-output:/output" \
  nvcr:orin-nano encode \
    -i /input/input.yuv -o /output/output.nvcr \
    -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32
```

Compose uses the same NVIDIA runtime and rolling-catalog installer:

```bash
export NVCR_INPUT_DIR="/path/to/yuv-input"
export NVCR_OUTPUT_DIR="/path/to/nvcr-output"

docker compose -f docker/compose.jetson.yaml run --rm --build engine-install
docker compose -f docker/compose.jetson.yaml run --rm --build nvcr --help
```

The public `l4t-jetpack:r36.4.0` base supplies the JetPack 6.1 CUDA 12.6 and
TensorRT 10.3 userspace. The reference target remains the newer recorded
`orin-nano-l4t3647` host. A successful image build is therefore development
convenience, not target-support evidence; run the complete registered GPU suite
and the roadmap protocol on that host. The NVIDIA runtime must expose `/dev/nvmap`
and the required `/dev/nvhost-*` devices or CUDA tests cannot pass.

## Dev Containers

The repository exposes two named configurations:

- `.devcontainer/x86_64/devcontainer.json` for an RTX/discrete x86_64 host;
- `.devcontainer/jetson/devcontainer.json` for a Jetson aarch64 host.

Choose the configuration matching the machine where Docker runs. Both use a
non-root `nvcr` user, keep the source bind-mounted by the editor, store the CMake
tree in a named volume, and configure a Debug GPU build after creation. The runtime images do not contain checkpoints, ONNX graphs, or TensorRT plans.
The Compose `engine-install` service downloads only exact-compatible published
engine bundles through the rolling catalog; it never downloads checkpoints or
creates engines. Dev Containers do not run that networked installation
automatically. Use the source-tree `nvcr-artifacts install` command with a
mounted engine volume, or mount a locally prepared target collection.

For CPU-only parser/rANS development, the same source tree can still use the
native CPU build documented in [Getting started](getting-started.md); these GPU
devcontainers intentionally model the two deployment architectures.

## Compatibility boundary

Image architecture and CUDA code generation are only the first compatibility
layer. Before execution, NVCR still checks the engine bundle's model/profile
hashes, GPU identity, compute capability, CUDA runtime compatibility, exact
TensorRT version, and plan checksums. Never copy an Orin engine bundle into the x86_64 image or an RTX
bundle into the Jetson image. A container does not make a TensorRT plan portable.
