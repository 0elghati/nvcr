# NVCR — Neural Video Codec Runtime

NVCR is a native C++20 runtime architecture for neural video codecs. It turns
learned compression models into deployable codec implementations with stateful
encoder and decoder sessions, explicit codec and execution-provider boundaries,
target-aware artifact selection, and bounded, versioned access units.

NVCR does not introduce a new compression model. Its first supported
end-to-end integration operationalizes
[DCVC-RT](https://github.com/microsoft/DCVC) with native entropy coding and
TensorRT FP16 execution on NVIDIA Linux systems. The architecture is
codec-extensible; DCVC-RT and TensorRT are the first complete codec/provider
pair, not the intended limit of the project.

## Main contributions

- A session-oriented C++20 runtime with explicit encode/decode, flush, reset,
  statistics, ownership, and error contracts.
- Separation of codec semantics from execution-provider mechanics, with static
  codec/provider discovery and provider-mediated runtime construction.
- Target-aware model and engine management with catalog resolution,
  compatibility ranking, integrity checks, and recorded artifact identity.
- Bounded `NVAU` v1 and sectioned v2 access units carrying codec, model,
  ordering, dependency, and payload information.
- A complete DCVC-RT I/P-frame integration using native rANS entropy coding,
  target-local TensorRT FP16 engines, a CLI, installable CMake packages, native
  archives, and container images.
- Reproducible validation tooling and retained comparisons against the pinned
  Python DCVC-RT reference implementation.

NVCR supplies the systems layer around a neural codec: lifecycle, state,
artifacts, framing, deployment, and validation. DCVC-RT remains the authority
for the learned compression method.

## DCVC-RT reference comparison

The retained comparisons use the same six sequence names and dimensions from
QCIF through 1080p, QP 32, 100 frames, and Python DCVC-RT as the reference. The
table compares inner entropy/rANS bits; NVCR access-unit and application
framing bytes are excluded from both sides.

| Retained platform | Directly comparable cases | NVCR entropy BPP relative to Python DCVC-RT |
|---|---|---:|
| Jetson Orin, reported 64-frame feature reset | GOP 1, 30, and 100; 18 cases | −0.16% to +1.07%; median +0.21% |
| RTX 4070, reported 32-frame feature reset | GOP 1 only; 6 all-intra cases | −0.12% to +0.30%; median +0.23% |

At this matched byte boundary, the retained results show that the native
integration closely tracks DCVC-RT's entropy rate while adding the runtime and
deployment facilities above. RTX 4070 inter-frame rows are excluded because
the retained NVCR and Python runs used different feature-reset intervals.

These are historical diagnostic results, not a current-release performance
claim. Their exact runtime and artifact identities are incomplete, and the
recorded clean-state metadata is inconsistent. The retained FPS and PSNR fields
also use different measurement definitions, so NVCR does not claim a speedup or
quality delta from these rows. See the full [Jetson Orin
comparison](results/jetson-orin/summary.md), [RTX 4070
comparison](results/rtx4070/summary.md), [result inventory](results/README.md),
and [measurement contract](docs/performance.md).

## Run NVCR on Linux

This path installs the latest stable native package on Ubuntu 24.04 x86_64,
the current native-package baseline. It requires an NVIDIA GPU supported by
the public engine catalog, a compatible NVIDIA driver, CUDA 12.8 and TensorRT
10.9 runtimes, Python 3, `curl`, `tar`, `sha256sum`, and network access.

Resolve the latest release once, install its verified package, and request the
QCIF engine for the detected GPU:

```bash
export NVCR_RELEASE="$(
  curl -fsSL https://api.github.com/repos/0elghati/nvcr/releases/latest |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])'
)"

curl -fsSLo /tmp/nvcr-install.sh \
  "https://raw.githubusercontent.com/0elghati/nvcr/${NVCR_RELEASE}/scripts/install.sh"
bash /tmp/nvcr-install.sh --tag "$NVCR_RELEASE" --profile qcif
export PATH="$HOME/.local/nvcr/bin:$PATH"
```

Create a deterministic four-frame QCIF YUV420P8 input, then encode and decode
it:

```bash
export NVCR_WORK_DIR="$PWD/nvcr-example"
mkdir -p "$NVCR_WORK_DIR"

python3 - "$NVCR_WORK_DIR/input.yuv" <<'PY'
from pathlib import Path
import sys

width, height, frames = 176, 144, 4
frame = bytes([16]) * (width * height) + bytes([128]) * (width * height // 2)
Path(sys.argv[1]).write_bytes(frame * frames)
PY

test "$(stat -c %s "$NVCR_WORK_DIR/input.yuv")" -eq 152064

nvcr codec list
nvcr provider list
nvcr encode \
  -i "$NVCR_WORK_DIR/input.yuv" \
  -o "$NVCR_WORK_DIR/output.nvcr" \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32
nvcr decode \
  -i "$NVCR_WORK_DIR/output.nvcr" \
  -o "$NVCR_WORK_DIR/reconstructed.yuv" \
  --quality-metrics "$NVCR_WORK_DIR/input.yuv"

test -s "$NVCR_WORK_DIR/output.nvcr"
test "$(stat -c %s "$NVCR_WORK_DIR/reconstructed.yuv")" -eq 152064
```

A successful run lists `dcvc-rt` and `tensorrt`, reports four encoded and four
decoded frames, creates a non-empty `.nvcr` stream, and writes 152064 bytes of
reconstructed YUV. NVCR is lossy, so the reconstruction is not expected to be
byte-identical to the input.

The installer resolves a moving release name only once, verifies the package
checksum, and records the selected release in `NVCR_RELEASE`. Engine selection
is automatic: exact device first, then a published same-compute-capability or
Ampere-and-newer bundle when compatible. TensorRT must still match the engine
manifest exactly, and Jetson accepts exact-target engines only.

## Other installation paths

| Environment or goal | Procedure |
|---|---|
| Linux x86_64 with Docker | [Docker encode/decode procedure](docs/first-run.md#linux-x86_64-with-an-nvidia-gpu-and-docker) |
| Windows 11 with Docker Desktop and WSL 2 | [Windows container procedure](docs/first-run.md#windows-11-with-docker-desktop-and-an-nvidia-gpu) |
| Jetson Orin | [Native Jetson procedure](docs/first-run.md#jetson-orin-native-installation) |
| Linux without a supported NVIDIA GPU | [CPU contract validation](docs/first-run.md#cpu-only-contract-validation) |
| Build or modify NVCR | [Build and contribution guide](CONTRIBUTING.md) |

See [Installation](docs/installation.md) for delivery boundaries and immutable
identity recording, [Docker on Linux](docs/docker-x86_64.md) for host setup,
and [Troubleshooting](docs/troubleshooting.md) when GPU or engine detection
fails.

## Architecture

```text
Application / CLI
        |
        v
NVCR runtime and stateful session contracts
        |
        +-- codec adapter: DCVC-RT semantics, I/P state, native entropy coding
        |
        +-- execution provider: TensorRT plans, CUDA execution, synchronization
        |
        +-- artifact resolver: model, profile, target, runtime, and digest identity
        |
        +-- access units: bounded, versioned NVAU framing
```

The current TensorRT provider owns construction of the complete DCVC-RT
backend. See [Architecture](docs/architecture.md), [DCVC-RT
integration](docs/dcvcrt-integration.md), and [Bitstreams and access
units](docs/bitstream.md) for the detailed contracts.

## Current scope and limitations

The supported end-to-end path is DCVC-RT with TensorRT FP16 on NVIDIA Linux.
NVCR provides native x86_64 packages and containers, native AArch64 packages
for the documented Jetson Orin stack, and Windows-hosted execution through the
Linux container in Docker Desktop/WSL 2. CPU builds validate portable runtime
and format contracts but do not run neural inference.

NVCR is pre-v1. The C++ API/ABI and application wrappers are not frozen. Native
Windows and macOS execution, CPU neural inference, INT8 release support, FFmpeg
integration, standard multimedia containers, arbitrary TensorRT-plan
portability, and additional supported codec/provider integrations are outside
the current release boundary. `.nvcr`, `NVCR`, and `NVCS` are development or
application wrappers, not standard multimedia containers.

See [Scope and support](docs/scope-and-support.md),
[Compatibility](docs/compatibility.md), and the [Roadmap](ROADMAP.md).

## Documentation

- [First run](docs/first-run.md)
- [Documentation map](docs/README.md)
- [Installation](docs/installation.md)
- [Command line](docs/cli.md)
- [C++ API](docs/reference.md)
- [Architecture](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Performance and benchmarking](docs/performance.md)
- [Retained results](results/README.md)

## Acknowledgements

NVCR's first supported codec integration builds on DCVC-RT, introduced by
Zhaoyang Jia, Bin Li, Jiahao Li, Wenxuan Xie, Linfeng Qi, Houqiang Li, and Yan
Lu in [*Towards Practical Real-Time Neural Video
Compression*](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html),
and on the reference implementation and model lineage released through the
[Microsoft/DCVC repository](https://github.com/microsoft/DCVC). DCVC-RT remains
the learned compression model; NVCR contributes the native runtime,
integration, artifact, framing, deployment, and validation layers.

CUDA and TensorRT are external NVIDIA components. The native rANS subset
retains its upstream DCVC and incorporated-source attributions. Exact source,
license, and redistribution boundaries are recorded in [Third-party
notices](THIRD_PARTY_NOTICES.md), [Model and checkpoint
terms](MODEL_LICENSES.md), and the [Asset distribution
policy](ASSET_DISTRIBUTION_POLICY.md).

## License and citation

NVCR source is distributed under the [MIT License](LICENSE). Generic packages
exclude checkpoints, exported model assets, TensorRT plans, and datasets.

Use [CITATION.cff](CITATION.cff) to cite NVCR. For assistance or vulnerability
reporting, see [Support](SUPPORT.md) and [Security](SECURITY.md).
