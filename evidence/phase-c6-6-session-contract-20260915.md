# Phase C6.6 session-contract evidence — 2026-09-15

## Scope

- Branch: `codex/c6-6-session-contract`
- Baseline and fetched `origin/main`:
  `f727491f9335c5add423058ce56c39bc0cb0b630`
- Starting tree: clean
- Release Please PR #146: open, mergeable, and not merged
- No production codec, provider, execution, NVAU, or CLI-driver code changed.

RTK could not run in this execution context because its sandbox failed before
the command with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`.
Validation used the commands below directly; `rtk proxy` was not used.

## Source audit

- No production `send_frame()` or `send_access_unit()` implementation returns
  `try_again`. DCVC-RT sends either consume the input or return a real error.
- No current test send implementation returns `try_again`.
- Runtime and CLI callers do not branch on send-side `try_again`. The CLI treats
  every failed send as an error.
- DCVC-RT queues output after a successful send. Its receive method reports
  `try_again` only when the output queue is empty before flush.
- The grouped encoder already expressed delay through successful sends followed
  by receive-side `try_again`. Its decoder produced output immediately, so it
  did not prove decoder lookahead before C6.6.
- The ambiguous wording appeared in `include/nvcr/codec/session.hpp` and was
  repeated in `docs/cross-codec-requirements.md`. The lifecycle text in
  `docs/extending-nvcr.md`, `docs/reference.md`, and `docs/cli.md` needed the
  same consumption/readiness distinction.

## Contract result

- Successful `send_frame()` and `send_access_unit()` calls accept and consume
  their input. The caller must not resubmit it.
- Successful receive calls return ready output.
- Receive-side `try_again` means no output is ready and more input may be
  required.
- Directional `flush()` signals that no more input will arrive. The caller
  drains the matching receive method through `end_of_stream`.
- Send-side `try_again` and receive-side `try_again` after successful flush are
  outside the v1 contract.

## Test result

The registered grouped test decoder now retains one accepted access unit. The
first send succeeds and the first receive returns `try_again`. A second,
distinct access unit is sent once and releases the first unit's eight frames in
order with their non-uniform timestamps. The second unit remains buffered until
decoder flush, then its final frame is emitted before `end_of_stream`.

The same fixture runs through the C6.5 CLI session driver. It covers two
single-submission inputs, output-frame limiting, final drain, reset, and reuse.

## Validation

CPU Release configure, build, 18/18 CTest, focused contract test, and install
passed:

```text
cmake -S . -B /tmp/nvcr-c6-6-release.jYmbEw \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=ON \
  -DNVCR_BUILD_EXAMPLES=ON \
  -DNVCR_BUILD_BENCHMARKS=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build /tmp/nvcr-c6-6-release.jYmbEw -j2
ctest --test-dir /tmp/nvcr-c6-6-release.jYmbEw --output-on-failure
/tmp/nvcr-c6-6-release.jYmbEw/tests/nvcr_contract_tests
cmake --install /tmp/nvcr-c6-6-release.jYmbEw \
  --prefix /tmp/nvcr-c6-6-install.DCNvWu
```

The Clang ASan/UBSan suite passed 15/15 tests. The bounded access-unit
libFuzzer target completed 1,000 runs without a sanitizer failure:

```text
cmake -S . -B /tmp/nvcr-c6-6-sanitized.z1lEv4 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_ENABLE_SANITIZERS=ON \
  -DNVCR_ENABLE_FUZZING=ON \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF
cmake --build /tmp/nvcr-c6-6-sanitized.z1lEv4 -j2
ctest --test-dir /tmp/nvcr-c6-6-sanitized.z1lEv4 --output-on-failure
/tmp/nvcr-c6-6-sanitized.z1lEv4/tests/nvcr_access_unit_fuzz \
  -runs=1000 -timeout=5 -max_total_time=10
```

The exact-artifact TensorRT Release suite used the RTX 4070, CUDA 12.8.93,
TensorRT 10.9.0, and the retained QCIF/CIF/360p/540p/720p/1080p B7 bundles.
The build and 32/32 tests passed:

```text
cmake -S . -B /tmp/nvcr-c6-6-tensorrt.ukO7L9 \
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
cmake --build /tmp/nvcr-c6-6-tensorrt.ukO7L9 -j2
ctest --test-dir /tmp/nvcr-c6-6-tensorrt.ukO7L9 --output-on-failure
```

## DCVC-RT regression

The Release CLI encoded and decoded the first 65 frames of
`/home/oelghati/datasets/qcif/akiyo_qcif.yuv` at 176x144, 30 fps, QP 32, and
GOP 8 with the retained B7 QCIF bundle. Both commands passed and produced 65
frames.

```text
/tmp/nvcr-c6-6-tensorrt.ukO7L9/cli/nvcr encode \
  -i /home/oelghati/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/nvcr-c6-6-b7-parity.nvcr \
  -s 176x144 -r 30 --frames 65 --gop-size 8 --qp 32 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
/tmp/nvcr-c6-6-tensorrt.ukO7L9/cli/nvcr decode \
  -i /tmp/nvcr-c6-6-b7-parity.nvcr \
  -o /tmp/nvcr-c6-6-b7-reconstructed.yuv --frames 0 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
sha256sum /tmp/nvcr-c6-6-b7-parity.nvcr \
  /tmp/nvcr-c6-6-b7-reconstructed.yuv
```

- Complete `.nvcr` SHA-256:
  `9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a`
- Decoded YUV SHA-256:
  `10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008`
- NVAU payload bytes: 14,669

All three values match the retained B7 and C6.5 evidence.

## Decision

The pre-1.1 session ambiguity is closed. All available release gates pass, and
production DCVC-RT semantics remain unchanged. The tree is ready for user
review and merge.
