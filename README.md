# NVCR — Neural Video Codec Runtime

NVCR is a deployment-oriented runtime architecture for neural video codecs.
Its first and currently only supported backend is the pinned DCVC-RT I/P model
pair, executed through a Linux C++20/CUDA/TensorRT runtime and reproducible
artifact toolchain.

> Repository status: development snapshot `0.5.1` <!-- x-release-please-version -->.
> `v0.3` is the active scope-and-foundation target; `v1.0` is the first scoped
> product release. This repository is not currently a validated v1 release.

The authoritative promises, non-goals, target matrix, and evidence rules are in
[Scope and support](docs/scope-and-support.md). Work status and test evidence are
in [ROADMAP.md](ROADMAP.md).

## Declared v1 scope

NVCR v1 is intentionally narrow. It validates the runtime architecture through
one production-quality codec backend rather than claiming broad codec coverage:

- pinned `dcvcrt-cvpr2025` image/video checkpoints and source identity;
- Linux on an RTX 4070 reference desktop and Jetson Orin Nano reference edge target;
- CUDA, TensorRT, FP16, fixed-shape engine profiles, and raw YUV420P8 input/output;
- offline checkpoint-to-ONNX-to-target-engine preparation;
- native C++ I/P encode/decode, CPU rANS, explicit sequence state, and bounded access units.

NVCR v1 is not a training framework, arbitrary PyTorch/ONNX converter, universal
GPU runtime, multi-codec release, stable upstream bitstream implementation, C
ABI, FFmpeg codec, or container integration. INT8 is experimental and is not a
supported/default v1 profile. Checkpoints, ONNX graphs, runtime model assets,
and TensorRT plans are not distributed from this repository; users generate
derived bundles locally.

## Current implementation

The development runtime currently includes:

- seven I-frame and seven P-frame TensorRT stages;
- native C++ rANS encode/decode and I/P temporal state;
- a versioned `NVAU` codec access unit carrying model identity, dimensions,
  frame type, effective QP, reset state, and bounded payload length;
- bounds-checked development `NVCR` packet and `NVCS` sequence framing for the CLI;
- v2 model and engine manifests with checkpoint, graph, asset, engine, tool,
  CUDA, TensorRT, GPU, precision, workspace, and profile identities;
- runtime rejection of modified, wrong-model, wrong-GPU, wrong-CUDA, and
  wrong-TensorRT engine bundles before engine deserialization;
- registered CPU, rANS, CUDA, artifact/profile, engine-contract, and native I/P
  roundtrip tests when a declared engine bundle is configured.

Remaining v1 gates include the reusable device arena, GPU-resident I-frame and
complete P-frame state paths, stable plane/stride session API, pinned Python
reference-quality gates, performance gates, and a current clean-room Orin
validation.

## Measured RTX 4070 snapshot

The following numbers are practical development evidence, not portable
performance guarantees or passed v1 release gates. They use the Release CLI on
an NVIDIA GeForce RTX 4070 at QP 32 over 97 frames. Each FPS value is the mean
of three measured runs after one warm-up run.

| Resolution | Sequence | GOP | Encode | Decode |
|---|---|---:|---:|---:|
| 720p | FourPeople | 1 (all-I) | 30.525 fps | 35.340 fps |
| 720p | FourPeople | 97 (I/P) | 94.773 fps | 44.292 fps |
| 1080p | BQTerrace | 1 (all-I) | 13.160 fps | 14.967 fps |
| 1080p | BQTerrace | 97 (I/P) | 44.182 fps | 19.921 fps |

GOP-97 is the representative inter-frame throughput point; GOP-1 measures the
more expensive all-intra path. Full protocol, payload, timing, software, and
hardware details are in
[Performance](docs/performance.md#2026-07-30-rtx-4070-warmed-encode-and-decode-diagnostic).

The pinned 720p GOP-97 quality comparison shows that native NVCR closely tracks
the Python reference when each runtime decodes its own stream:

| Comparison | Average PSNR-YUV | Minimum / note |
|---|---:|---|
| Source to pinned Python reconstruction | 40.817787 dB | 97 frames |
| Source to native reconstruction | 40.726269 dB | 97 frames |
| Pinned Python to native reconstruction | 45.241436 dB | 44.137497 dB minimum |

That result does **not** mean the raw entropy payloads are interchangeable.
Framing-only two-frame probes completed decoding in both directions but failed
quality from the first I frame:

| Entropy payload direction | PSNR-YUV | Status |
|---|---:|---|
| Python payload decoded by native NVCR | 24.864091 dB | incompatible |
| Native NVCR payload decoded by Python | 25.152377 dB | incompatible |

The outer framing and model CDF assets match their expected contracts, while the
remaining incompatibility is localized to the entropy symbol/index-map
boundary. NVCR v1 therefore makes no upstream byte or payload-interchangeability
promise. See the
[GOP-97 conformance evidence](docs/evidence/2026-07-30-rtx4070-dcvcrt-gop97-conformance.json)
and
[cross-runtime probe evidence](docs/evidence/2026-07-30-rtx4070-dcvcrt-cross-runtime-probe.json).


## Workflow

```text
Pinned DCVC-RT source + checkpoint hashes
                  |
                  v
       nvcr-artifacts prepare
                  |
       ONNX + entropy/quant assets
                  |
                  v
        nvcr-artifacts build
                  |
      target-local TensorRT engines
                  |
                  v
        nvcr-artifacts validate
                  |
                  v
       native NVCR encode/decode
```

### 1. Build NVCR

Requirements for the supported runtime are Linux, CMake 3.24+, a C++20 compiler,
CUDA, and TensorRT. A CPU-only development build remains available for parsers,
rANS, manifests, and API tests.

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

For published releases, the convenience installer fetches the latest matching
Linux package and every available engine profile for the selected backend. If
more than one profile is available, it asks which backend/profile should become
the active CLI default:

```bash
curl -fsSL https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash
```

Development checkouts can still use the source-build helper, while forks and
private staging repositories can pass `--repo OWNER/REPO` to `scripts/install.sh`:

```bash
./scripts/install_from_source.sh --run-tests
```

Other NVIDIA Linux configurations are experimental until they have equivalent
recorded evidence. A multi-architecture CUDA binary does not make TensorRT plans
portable; every engine bundle is tied to its recorded target. Discrete GPUs use
persistent TensorRT execution contexts by default; set
`NVCR_TENSORRT_LOW_MEMORY_MODE=1` only when memory pressure requires it.

### 2. Prepare model artifacts and engines

Place the two documented checkpoints under `<dcvcrt-root>/checkpoints/`, create a
Python export environment containing PyTorch, ONNX, and ONNXScript, then use the
single supported front end:

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

The operations are:

- `prepare`: verify the pinned checkout/checkpoints, export portable model assets,
  and, unless `--skip-engine` is used, build the selected target engines;
- `build`: build engines from an already validated model directory;
- `inspect`: report bundle identity without executing it;
- `validate`: hash and validate every required model or engine artifact.

The backend-local shell/export scripts under `scripts/backends/dcvcrt/` are
implementation helpers. They remain callable for development, but are not
separate supported user workflows.

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt --json
```

See [Model and engine preparation](docs/dcvcrt-artifacts.md) for the exact source
commit, checkpoint hashes, target profiles, and clean-room procedure.

### 3. Register and run the complete local test suite

Configure with the generated engine directory so the GPU integration tests are
registered:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$PWD/build/engines/dcvcrt"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Without `NVCR_TENSORRT_ENGINE_DIR`, CTest cannot claim engine or I/P roundtrip
coverage; it runs only the tests available to that configuration.

### 4. Run the development CLI

```bash
./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

By default the CLI uses the active installed engine profile. Switch profiles
without passing an engine path:

```bash
./build-release/cli/nvcr encode ... --engine-profile 720p-fp16
NVCR_ENGINE_PROFILE=1080p-fp16 ./build-release/cli/nvcr decode ...
```

Use `--backend NAME` when multiple backends are installed; `dcvcrt`, `dcvc_rt`,
and `dcvc-rt` currently select the same backend. `--engine-dir` remains
available only for custom bundles and local development tests. Input and output
frames are planar YUV420P8. `NVCR`/`NVCS` are development
framing formats and have no stable pre-v1 compatibility promise. Container
timestamps and FFmpeg metadata are intentionally outside the codec access unit.

## Public C++ boundary

Applications create an `nvcr::Runtime` with an explicit per-session
`RuntimeConfiguration` and an injected `nvcr::codec::CodecBackend`. The DCVC-RT
TensorRT backend is the only backend implemented in this release. Errors cross
the boundary as `nvcr::Result<T>`; CUDA/TensorRT types and exceptions do not.
Encoder and decoder state are independent and `reset()`/`flush()` are explicit.

The v1 API still needs its final plane/stride/device-view stabilization. Do not
treat the current headers as a frozen ABI. A public C ABI and FFmpeg wrapper are
post-v1 work.

## Documentation

- [Documentation index](docs/README.md)
- [Scope and support](docs/scope-and-support.md)
- [Getting started and builds](docs/getting-started.md)
- [Compatibility matrix](docs/compatibility.md)
- [Architecture](docs/architecture.md)
- [Model and engine preparation](docs/dcvcrt-artifacts.md)
- [Bitstream and access units](docs/bitstream.md)
- [CLI](docs/cli.md)
- [Performance protocol and historical results](docs/performance.md)
- [Release gates](docs/releasing.md)
- [Roadmap and evidence](ROADMAP.md)

## License

NVCR is licensed under the [MIT License](LICENSE). Vendored third-party code
retains its own license. Model/checkpoint and derived-asset rights are separate;
NVCR releases do not ship those artifacts.
