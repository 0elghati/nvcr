# NVCR

NVCR is a native C++ runtime architecture for neural video codecs.

It provides the systems layer needed to operate learned codecs through
stateful encoder and decoder sessions, codec-adapter and execution-provider
contracts, target-aware artifact selection, and bounded, versioned neural-codec
access units. NVCR is neither a new compression model nor merely a C++ port of
DCVC-RT.

The first complete production vertical is DCVC-RT through TensorRT FP16 on
Linux/NVIDIA targets. That vertical validates the architecture end to end; it
does not define the long-term boundary of NVCR. The architecture is
codec-extensible, but DCVC-RT is currently the only production codec and
TensorRT is currently the only production provider. The deterministic test
codec and CPU provider are conformance fixtures, not additional products.

## What NVCR provides

- Native C++20 runtime and stateful encoder/decoder session contracts.
- `send`/`receive`, flush, drain, reset, statistics, and structured errors.
- Codec adapters separated from execution-provider mechanics.
- Static codec/provider discovery and provider-mediated runtime services.
- Artifact descriptors, catalogs, compatibility ranking, hashes, and license
  restrictions.
- Bounded `NVAU` v1 and generalized sectioned `NVAU` v2 access units.
- A DCVC-RT adapter with native entropy coding and TensorRT FP16 execution.
- CLI, CMake package, target-local artifact tooling, Docker images, and
  reproducible evaluation tooling.

## Install and run the current production vertical

The binary installer selects compatible engine profiles from the rolling
catalog. Generic packages do not contain checkpoints, exported model assets,
TensorRT plans, or datasets.

```bash
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash
export PATH="$HOME/.local/nvcr/bin:$PATH"
nvcr-artifacts install --profile qcif
nvcr --help
```

NVCR reads and writes headerless planar 8-bit YUV420, so dimensions and frame
rate must be supplied for raw input:

```bash
nvcr encode -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32
nvcr decode -i output.nvcr -o reconstructed.yuv
```

If no compatible engine is published, follow [model and engine preparation](docs/dcvcrt-artifacts.md).
TensorRT plans are target- and runtime-specific; a matching configuration file
is not evidence that a target is supported.

## Build from source

CPU-only contract build:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

Native production build, on a machine with the required CUDA/TensorRT stack:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

## Supported and validated targets

Current target profiles cover RTX 3050 Laptop, RTX 4070, RTX 5060 Laptop, and
Jetson Orin Nano. A target profile describes an artifact request; it is not a
support claim. Support requires a clean Release build, validated target-local
artifacts, runtime round trips, registered tests, and the applicable evidence
gates. See [scope and support](docs/scope-and-support.md) and
[compatibility](docs/compatibility.md).

## Documentation

- [Identity and scope](docs/identity-and-scope.md)
- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Extending NVCR](docs/extending-nvcr.md)
- [C++ API reference](docs/reference.md)
- [Streams and access units](docs/bitstream.md)
- [DCVC-RT production integration](docs/dcvcrt-integration.md)
- [Documentation map](docs/README.md)
- [Reproducibility and review path](docs/reproducibility/README.md)
- [Roadmap](ROADMAP.md)

## Reproducing the evaluation

The executable protocol is in [docs/experiments](docs/experiments/README.md),
and the reviewer-facing claim matrix is in
[docs/reproducibility/claims-and-evidence.md](docs/reproducibility/claims-and-evidence.md).
Only a clean, complete result package with current artifact identities,
registered gates, profiling metrics, and required pinned-reference rows is
publication evidence. Retained partial or diagnostic results are not a release
baseline.

## Current limitations

NVCR is not yet a supported v1 release. The current production boundary is
Linux/NVIDIA with DCVC-RT and TensorRT FP16. The public API is not ABI-frozen;
additional production codecs/providers, INT8 release support, FFmpeg,
standard containers, and universal TensorRT-plan portability remain outside
the current release boundary. `.nvcr`, `NVCR`, and `NVCS` are development or
application wrappers, not standard multimedia containers.

## Citation, license, and support

NVCR source is MIT-licensed. Model checkpoints, exported model assets, engine
plans, and datasets have separate terms and are excluded from generic packages;
see [the asset policy](ASSET_DISTRIBUTION_POLICY.md),
[model licensing](MODEL_LICENSES.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

Use [CITATION.cff](CITATION.cff) for the current software citation and
[SUPPORT.md](SUPPORT.md) for usage, reproducibility, bug, and security routes.
