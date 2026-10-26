# Phase C6.5 generic CLI session-driver evidence — 2026-09-15

## Scope and starting point

This stabilization makes the shipped CLI drive the codec-owned session
lifecycle and narrows the C5/C6 compatibility conclusion. It does not add a
codec, provider, capability taxonomy, NVAU change, or performance campaign.

- Branch: `codex/pre-1.1-cli-session-driver`
- Baseline and fetched `origin/main`:
  `5d05bd815dd56d2d7e873c505f620605edd409ac`
- Starting tree: clean
- GPU: NVIDIA GeForce RTX 4070, compute capability 8.9
- CUDA compiler/runtime: 12.8.93 / 12080
- TensorRT: 10.9.0

RTK could not run in this execution context because its sandbox failed before
the command with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
Validation used the commands below directly; `rtk proxy` was not used.

## Driver result

The CLI uses one private driver for both normal and final drains. Encode sends
each input frame, receives until `try_again`, flushes only the encoder, and
receives through `end_of_stream`. Decode sends each access unit, receives every
available frame, flushes only the decoder after input exhaustion, and receives
through `end_of_stream`. Decode `--frames N` counts emitted output frames and
stops at the limit without reading additional access units.

The grouped application-path contract uses this same driver with the registered
test-only codec. It covers seven inputs without output, a full eight-frame
group, a short flush-only group, one access unit producing eight frames,
non-uniform timestamps, order, directional drain, an output-frame limit, and
reset/reuse. The test codec remains linked only into the contract-test binary.

Codec time sums session send, receive, directional flush, and final-drain calls.
Packet serialization and raw file I/O remain outside the timing boundary.

## Validation

The benchmark-output parser unit test passed 16 tests:

```text
python3 -m unittest tests/softwarex_driver_tests.py
```

A clean CPU Release tree was configured and built:

```text
cmake -S . -B /tmp/nvcr-c6-5-release.crmuN2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=ON \
  -DNVCR_BUILD_EXAMPLES=ON \
  -DNVCR_BUILD_BENCHMARKS=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build /tmp/nvcr-c6-5-release.crmuN2 -j2
ctest --test-dir /tmp/nvcr-c6-5-release.crmuN2 --output-on-failure
cmake --install /tmp/nvcr-c6-5-release.crmuN2 \
  --prefix /tmp/nvcr-c6-5-install.OYwlDh
```

Result: build and install passed; 18/18 tests passed.

The normal Clang sanitizer/libFuzzer gate used:

```text
cmake -S . -B /tmp/nvcr-c6-5-sanitized.bs3rwX \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_ENABLE_SANITIZERS=ON \
  -DNVCR_ENABLE_FUZZING=ON \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build /tmp/nvcr-c6-5-sanitized.bs3rwX -j2
ctest --test-dir /tmp/nvcr-c6-5-sanitized.bs3rwX --output-on-failure
/tmp/nvcr-c6-5-sanitized.bs3rwX/tests/nvcr_access_unit_fuzz \
  -runs=1000 -timeout=5 -max_total_time=10
```

Result: 15/15 sanitizer tests passed; libFuzzer completed 1,000 runs without a
sanitizer failure.

A clean Release tree used CUDA 12.8, TensorRT 10.9, SM 8.9, the retained exact
B7 QCIF/CIF/360p/540p/720p/1080p catalog bundles, and the pinned DCVC-RT golden
input. The exact configure, build, and test commands were:

```text
cmake -S . -B /tmp/nvcr-c6-5-tensorrt.iBe8bH \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=ON \
  -DNVCR_BUILD_EXAMPLES=ON \
  -DNVCR_BUILD_BENCHMARKS=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF \
  -DNVCR_DCVCRT_ROOT=/home/oelghati/DCVC-RT \
  -DNVCR_DCVCRT_720P_GOLDEN_INPUT=/home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -DNVCR_TENSORRT_ENGINE_DIR=/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/1080p/9b452c220ac7354f6687b41c51dfe8c040c63b52a4775402f6d9f887239718a4 \
  '-DNVCR_TENSORRT_ENGINE_DIRS=/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/cif/45dbbbf89154a2707572f98b9bd075ac9f87484d08c0a35cf74041ff23bf0a2a;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/360p/f3ef0872e3f8b0964601886ef0c5714354104f9ec6a64527897e81169f366d1f;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/540p/47a971716fc2f2a39a21e9b3d210bb68fa4f20e5d14d4f6ae7df631cde80ccaa;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/720p/7c5cd5865de1d609d12fa9b35cc4669ea49cbd1bd21a794ac1e9aececac4372a'
cmake --build /tmp/nvcr-c6-5-tensorrt.iBe8bH -j2
ctest --test-dir /tmp/nvcr-c6-5-tensorrt.iBe8bH --output-on-failure
```

Result: the build passed and 32/32 tests passed. The suite includes all six
engine contracts and registered I/P round trips,
CUDA operations, the grouped CLI-driver contract, and the pinned Python/native
I-frame golden.

## DCVC-RT byte and reconstruction parity

The Release CLI encoded the first 65 frames of
`/home/oelghati/datasets/qcif/akiyo_qcif.yuv` at 176x144, 30 fps, QP 32, and
GOP 8 with the exact retained B7 QCIF catalog bundle. Decode used `--frames 0`
to exercise input exhaustion and final decoder drain.

```text
/tmp/nvcr-c6-5-tensorrt.iBe8bH/cli/nvcr encode \
  -i /home/oelghati/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/nvcr-c6-5-b7-parity.nvcr \
  -s 176x144 -r 30 --frames 65 --gop-size 8 --qp 32 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
/tmp/nvcr-c6-5-tensorrt.iBe8bH/cli/nvcr decode \
  -i /tmp/nvcr-c6-5-b7-parity.nvcr \
  -o /tmp/nvcr-c6-5-b7-reconstructed.yuv --frames 0 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
sha256sum /tmp/nvcr-c6-5-b7-parity.nvcr \
  /tmp/nvcr-c6-5-b7-reconstructed.yuv
```

Result: encode produced 65 frames and 14,669 payload bytes; decode produced 65
frames; both commands and the hash check passed.

- Complete `.nvcr` SHA-256:
  `9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a`
- Decoded YUV SHA-256:
  `10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008`
- NVAU payload bytes: 14,669

All three values match the retained B7 and C1-C3 evidence. A discarded
diagnostic using the later `build/engines-c1-trt109/dcvcrt-qcif` bundle produced
different bytes, confirming that parity depends on the recorded artifact
identity.

## Compatibility conclusion and handoff

C5 still records the unchanged PSNR/BPP limits and measurements. C6 still
records the raw-symbol divergence before entropy coding. The conclusion is now
limited to the tested pinned MLVC-S/default generic FP16 ONNX/ORT-CUDA pair: it
failed strict reference consistency and is not byte/payload interchangeable
with the PyTorch CPU reference path. MLVC self-conformance, MLVC's general
suitability, and ONNX Runtime CUDA's suitability as an NVCR provider remain
unresolved.

The next bounded experiment should hold DCVC-RT codec semantics constant and
test a second provider, with ONNX Runtime CUDA as the current candidate. The
DCVC-UF C7 codec-axis experiment remains pending.
