# Getting started and builds

NVCR is currently a source-built development project. The runtime architecture is
intended for neural video codec backends, but the current release path supports
DCVC-RT only. Published `0.2.x` artifacts are historical snapshots, not
v1-compatible product packages. The supported release procedure is gated in
[Releasing](releasing.md).

## Choose a build

### CPU development build

This build validates public types, bounded packet/access-unit parsing, rANS,
configuration, and artifact profiles. It cannot encode or decode DCVC-RT frames.

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

### Target-profile GPU runtime build

Requirements:

- Linux and a C++20 compiler;
- CMake 3.24 or newer;
- the CUDA and TensorRT versions selected by a reference target profile;
- direct access to the selected NVIDIA device;
- locally generated compatible engines.

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON
cmake --build build-release --parallel
```

CMake detects the attached GPU architecture, CUDA compiler, and TensorRT install
on a normal native build. This produces a development build for the selected
profile; it is not support evidence by itself. The source-build helper performs
the same detection and installs to a local platform-specific prefix by default:

```bash
./scripts/install_from_source.sh --run-tests
```

Auto-detection is convenience, not support evidence. Use a matching
[reference target profile](compatibility.md) and target-local engine set for
support evidence. Explicit overrides are available for target bring-up:

```bash
cmake -S . -B build-target-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="${NVCR_CUDA_ARCHITECTURES:?set the target compute capability}" \
  -DTensorRT_ROOT=/usr
```

A CUDA fat binary (`NVCR_CUDA_ARCH_SET=portable`) is an experimental packaging
option. It does not make a binary universal and never makes TensorRT plans
portable across GPU models or TensorRT runtimes.

## Prepare the pinned model

NVCR ships no checkpoints, ONNX/model assets, or TensorRT plans. Obtain the two
checkpoint files described in [Model and engine preparation](dcvcrt-artifacts.md)
and place them under:

```text
/path/to/DCVC-RT/checkpoints/cvpr2025_image.pth.tar
/path/to/DCVC-RT/checkpoints/cvpr2025_video.pth.tar
```

The Python interpreter used offline must provide `torch`, `onnx`, and
`onnxscript`. Python is not a runtime dependency of `libnvcr` or the CLI.

Use the profile-aware artifact command. This generic front end dispatches to the
current DCVC-RT backend preparation helpers:

```bash
target_profile="${NVCR_TARGET_PROFILE:?set NVCR_TARGET_PROFILE to a target JSON profile}"

./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 1080p \
  --target-profile "$target_profile" \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
```

Run engine building on the final target described by `target_profile`.
Portable ONNX/assets may be transferred only after `nvcr-artifacts validate`
succeeds; plans must be rebuilt on the final target.

Inspect and validate without executing a bundle:

```bash
./scripts/nvcr_artifacts.py inspect build/models/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json
./scripts/nvcr_artifacts.py inspect build/engines/dcvcrt-1080p --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt-1080p --json
```

## Run the complete configured suite

The engine and native I/P tests are registered only when CMake receives an engine
bundle. Use a TensorRT engine set built for the active target and installed
runtime. Keep the generated engine directories outside the build directory;
CMake and the runtime consume them in place.

Use the 720p bundle as the primary bundle because the pinned Python I-frame
golden test uses a 720p input. Register the other five profiles through
`NVCR_TENSORRT_ENGINE_DIRS`:

```bash
target_profile="${NVCR_TARGET_PROFILE:?set NVCR_TARGET_PROFILE to a target JSON profile}"
engine_root="${NVCR_ENGINE_ROOT:?set NVCR_ENGINE_ROOT to the target-local engine set}"
golden_input="${NVCR_DCVCRT_720P_GOLDEN_INPUT:?set the 720p golden input path}"

cmake -S . -B build-release-clean \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$engine_root/dcvcrt-720p" \
  -DNVCR_TENSORRT_ENGINE_DIRS="$engine_root/dcvcrt-qcif;$engine_root/dcvcrt-cif;$engine_root/dcvcrt-360p;$engine_root/dcvcrt-540p;$engine_root/dcvcrt-1080p" \
  -DNVCR_DCVCRT_ROOT="${NVCR_DCVCRT_ROOT:?set the pinned DCVC-RT checkout path}" \
  -DNVCR_DCVCRT_720P_GOLDEN_INPUT="$golden_input"
cmake --build build-release-clean --parallel
ctest --test-dir build-release-clean --output-on-failure
```

Check `ctest --test-dir build-release-clean -N`: a full GPU configuration includes
artifact/profile, parser/rANS, CUDA ops, engine contracts, and I/P roundtrip
tests. The roundtrip test covers high effective QP, two GOPs, reset/reuse, and
encoder/decoder reconstruction equality. The engine-contract test covers
wrong-model, corrupt-manifest, and corrupt-checksum rejection.

Validate all six engine bundles before configuring CMake:

```bash
for profile in qcif cif 360p 540p 720p 1080p; do
  ./scripts/nvcr_artifacts.py validate \
    "$engine_root/dcvcrt-$profile" --json
done
```

Do not use an engine bundle with a different TensorRT runtime or target than the
one recorded in its manifest. Engine bundles are target-local and are
intentionally not copied into the installed Release prefix.

## Run the real resolution matrix

After the complete CTest suite passes, run warmed Release measurements at QP 32
for QCIF, CIF, 360p, 540p, 720p, and 1080p. This command runs all-intra and
GOP-97 cases, three measured repetitions, and ten warm-up frames per case. It
writes compact JSONL evidence and leaves raw `.nvcr` streams under the separate
output directory:

```bash
engine_root="${NVCR_ENGINE_ROOT:?set NVCR_ENGINE_ROOT to the target-local engine set}"
golden_input="${NVCR_DCVCRT_720P_GOLDEN_INPUT:?set the 720p golden input path}"
run_id="live-release-$(date +%Y%m%d)"
evidence_dir="$PWD/evidence/$run_id"
mkdir -p "$evidence_dir/streams"
scripts/benchmark_resolution_matrix.sh \
  --nvcr "$PWD/build-release-clean/cli/nvcr" \
  --frames 97 \
  --qp 32 \
  --gops "1 97" \
  --resolutions "qcif cif 360p 540p 720p 1080p" \
  --repetitions 3 \
  --warmup-frames 10 \
  --qcif-engine-dir "$engine_root/dcvcrt-qcif" \
  --cif-engine-dir "$engine_root/dcvcrt-cif" \
  --360p-engine-dir "$engine_root/dcvcrt-360p" \
  --540p-engine-dir "$engine_root/dcvcrt-540p" \
  --720p-engine-dir "$engine_root/dcvcrt-720p" \
  --1080p-engine-dir "$engine_root/dcvcrt-1080p" \
  --qcif-input "${NVCR_BENCH_QCIF_INPUT:?set the QCIF input path}" \
  --cif-input "${NVCR_BENCH_CIF_INPUT:?set the CIF input path}" \
  --360p-input "${NVCR_BENCH_360P_INPUT:?set the 360p input path}" \
  --540p-input "${NVCR_BENCH_540P_INPUT:?set the 540p input path}" \
  --720p-input "$golden_input" \
  --1080p-input "${NVCR_BENCH_1080P_INPUT:?set the 1080p input path}" \
  --output-dir "$evidence_dir/streams" \
  --jsonl "$evidence_dir/resolution-matrix.jsonl"
```

For a quicker single profile smoke test, replace the matrix options with
`--resolutions "720p" --gops "97" --repetitions 1 --warmup-frames 2`.

On Jetson, tests require access to `/dev/nvmap` and `/dev/nvhost-*`. A container
that hides those devices can compile successfully but cannot provide GPU test
evidence.

Architecture-specific Docker execution images and named RTX/Jetson Dev
Container configurations are documented in [Docker execution and
development](docker.md). They preserve the same target-local engine rule as a
native build; containerization does not make TensorRT plans portable.

## Encode and decode raw YUV420P8

```bash
export NVCR_ENGINE_ROOT="$PWD/build/engines"

./build-release/cli/nvcr encode \
  -i input.yuv -o output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32

./build-release/cli/nvcr decode \
  -i output.nvcr -o reconstructed.yuv
```

The CLI format is a development sequence format, not MP4/Matroska, FFmpeg, or a
stable upstream DCVC-RT stream. See [CLI](cli.md) and [Bitstream](bitstream.md).

## Install for local development

```bash
cmake --install build-release --prefix "$PWD/install-nvcr"
export PATH="$PWD/install-nvcr/bin:$PATH"
nvcr --help
```

The install contains NVCR code, headers, CMake metadata, configuration examples,
documentation, and notices. It does not contain checkpoints or derived model and
engine assets. Pass the local engine directory explicitly.

## Clean-room target evidence

A release validation starts without pre-existing build/model/engine directories
and records:

1. OS, compiler, CUDA, TensorRT, driver/JetPack, GPU, clocks/power mode;
2. pinned source commit and checkpoint hashes;
3. model preparation and model-bundle validation;
4. target-local engine build and engine-bundle validation;
5. the complete registered CTest suite;
6. native multi-frame encode/decode and reference comparison;
7. release-build latency, memory, throughput, rate/distortion, and applicable
   energy evidence using [Performance](performance.md).

v0.3 requires one complete reference-target foundation workflow. v1.0 requires
separate successful matrices for each declared reference target and all
applicable M1–M4 roadmap gates.
