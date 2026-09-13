# Vision B4 correctness and lifecycle closure evidence

Recorded on 2026-09-13 for the TensorRT execution-session extraction.

This record closes the technical B4 gates on the available RTX 4070. The
recorded performance comparison was explicitly accepted on 2026-09-13,
completing B4. Full measurements remain in
[`vision-b4-rtx4070-20260911.md`](vision-b4-rtx4070-20260911.md).

## Identity

- Current source: `65e52f3063f19c56bdcca3be349f77c0c113fe1d`
- Branch at validation start: `main`, clean and equal to `origin/main`
- Validation branch: `codex/close-b4`
- B3 source: `11208d1e23824f69045027af8aa0f17eeae5d6c4`
- Extracted B4 source: `10f4975e9e9499232cb57d7ec6994168b7e0bfb2`
- `rtk git diff --quiet 10f4975e9e9499232cb57d7ec6994168b7e0bfb2 HEAD -- src include tests CMakeLists.txt cmake cli`: pass
- GPU: NVIDIA GeForce RTX 4070, compute capability 8.9
- Driver: 580.173.02
- CUDA compiler: 12.8.93
- TensorRT: 10.9.0
- Exact engine profiles: QCIF, CIF, 360p, 540p, 720p, and 1080p from
  `build/engines-b3-rtx4070-catalog`

## Clean CPU Release

Configuration:

```bash
rtk cmake -S . -B /tmp/nvcr-b4-close-65e52f3-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/tmp/nvcr-b4-close-65e52f3-install \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_FETCH_DEPENDENCIES=OFF \
  -DNVCR_BUILD_TESTS=ON
```

Results:

- `rtk cmake --build /tmp/nvcr-b4-close-65e52f3-cpu --parallel`: pass
- `rtk ctest --test-dir /tmp/nvcr-b4-close-65e52f3-cpu --output-on-failure`: 17/17 pass
- `rtk cmake --install /tmp/nvcr-b4-close-65e52f3-cpu`: pass

GoogleTest was not installed, so CMake selected the repository's
dependency-free smoke-test fallback. The dedicated contract, format, parser,
artifact, payload, rANS, CLI, installer, onboarding, version, and experiment
driver tests were still registered and passed.

## Sanitizers and bounded fuzzing

Configuration:

```bash
rtk cmake -S . -B /tmp/nvcr-b4-close-65e52f3-sanitized \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DNVCR_ENABLE_TENSORRT=OFF \
  -DNVCR_ENABLE_SANITIZERS=ON \
  -DNVCR_ENABLE_FUZZING=ON \
  -DNVCR_FETCH_DEPENDENCIES=OFF \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_BUILD_CLI=OFF \
  -DNVCR_BUILD_EXAMPLES=OFF
```

Results:

- `rtk cmake --build /tmp/nvcr-b4-close-65e52f3-sanitized --parallel`: pass
- `rtk ctest --test-dir /tmp/nvcr-b4-close-65e52f3-sanitized --output-on-failure`: 14/14 pass
- `rtk /tmp/nvcr-b4-close-65e52f3-sanitized/tests/nvcr_access_unit_fuzz -runs=1000 -timeout=5 -max_total_time=10`: 1,000 runs pass under ASan/UBSan

## TensorRT Release and lifecycle gates

Configuration:

```bash
rtk cmake -S . -B /tmp/nvcr-b4-close-65e52f3-trt \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_FETCH_DEPENDENCIES=OFF \
  -DNVCR_BUILD_TESTS=ON \
  -DNVCR_DCVCRT_ROOT=/home/oelghati/DCVC-RT \
  -DNVCR_DCVCRT_720P_GOLDEN_INPUT=/home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -DNVCR_TENSORRT_ENGINE_DIR=/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/720p/7c5cd5865de1d609d12fa9b35cc4669ea49cbd1bd21a794ac1e9aececac4372a \
  '-DNVCR_TENSORRT_ENGINE_DIRS=/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/cif/45dbbbf89154a2707572f98b9bd075ac9f87484d08c0a35cf74041ff23bf0a2a;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/360p/f3ef0872e3f8b0964601886ef0c5714354104f9ec6a64527897e81169f366d1f;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/540p/47a971716fc2f2a39a21e9b3d210bb68fa4f20e5d14d4f6ae7df631cde80ccaa;/home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/1080p/9b452c220ac7354f6687b41c51dfe8c040c63b52a4775402f6d9f887239718a4'
```

`rtk cmake --build /tmp/nvcr-b4-close-65e52f3-trt --parallel`: pass.

The clean TensorRT Release build registered 31 tests. The non-GPU selection
passed 18/18:

```bash
rtk ctest --test-dir /tmp/nvcr-b4-close-65e52f3-trt \
  --output-on-failure -LE gpu
```

The `gpu` selection passed 13/13:

```bash
rtk ctest --test-dir /tmp/nvcr-b4-close-65e52f3-trt \
  --output-on-failure -L gpu
```

That selection covered engine contracts and I/P round trips for all six exact
profiles plus `nvcr_dcvcrt_i_frame_golden`. The round-trip test covers two
GOPs, I/P frame classification, dimensions, timestamps, reset, reuse, and
decoded output. Provider-session reset, dependency, ownership, asynchronous
completion, and failure paths passed in `nvcr_contract_tests`. Malformed and
bounded input checks passed in the parser, format, payload, rANS, artifact, and
engine-contract tests.

The pinned one-frame Python/native golden passed with:

- Python reference commit: `1aa39a0ec756669af2ca3aedf9cf934aef55b1a4`
- Native source PSNR-YUV: `38.63167056464599` dB
- Python source PSNR-YUV: `38.67122266855402` dB
- Python/native PSNR-YUV: `44.211195291562504` dB
- Native sequence SHA-256: `3c1f5792e961db76aea3ca2ac3148ff5e660921e654fb76a5a92a104fb2c4717`
- Native reconstruction SHA-256: `15638687cb0eb0f53219a3692bcc5b1839765d6c8996a5fe6b8663c0529c76a1`

The test checked the pinned checkpoint, source-frame, Python bitstream, and
Python reconstruction hashes before comparing native output.

## B3/B4 byte and reconstruction parity

The retained B3 Release binary came from the B3 worktree at
`11208d1e23824f69045027af8aa0f17eeae5d6c4`. The B4 binary was rebuilt from
the current source. Binary hashes differ as expected:

- B3 binary: `59e17ee0e079f183c4cda6365e66ba1c648df63176275bd3753e586557b0b2d3`
- B4 binary: `7486507f0ae98c99348b1c0918820074d664b8362da4eba2300620602c993fba`
- Input: `e1efee0e95c6d27aefe2294a727768334db8466e67bf4a49cf8f5b2fb8b49108`
- QCIF engine manifest: `11e53891e186ec2662a8c2022fdd50600e06ef342eb393c2ee3b11b8d0cd3ed4`

Both binaries encoded the same first 65 frames of
`/home/oelghati/datasets/qcif/akiyo_qcif.yuv` at 176x144, QP 32, GOP 8, and 30
fps through the same exact QCIF bundle. This covers nine I frames, 56 P frames,
and multiple GOP resets.

The B3 command was:

```bash
rtk /tmp/nvcr-b3-perf-build/cli/nvcr encode \
  -i /home/oelghati/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/nvcr-b4-byte-parity-65e52f3-b3.nvcr \
  -s 176x144 -r 30 --frames 65 --gop-size 8 --qp 32 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
```

The B4 command changed only the binary and output paths:

```bash
rtk /tmp/nvcr-b4-close-65e52f3-trt/cli/nvcr encode \
  -i /home/oelghati/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/nvcr-b4-byte-parity-65e52f3-b4.nvcr \
  -s 176x144 -r 30 --frames 65 --gop-size 8 --qp 32 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
```

Each encoder reported 14,669 NVAU payload bytes.

The decoder commands were:

```bash
rtk /tmp/nvcr-b3-perf-build/cli/nvcr decode \
  -i /tmp/nvcr-b4-byte-parity-65e52f3-b3.nvcr \
  -o /tmp/nvcr-b4-byte-parity-65e52f3-b3.yuv --frames 65 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61

rtk /tmp/nvcr-b4-close-65e52f3-trt/cli/nvcr decode \
  -i /tmp/nvcr-b4-byte-parity-65e52f3-b4.nvcr \
  -o /tmp/nvcr-b4-byte-parity-65e52f3-b4.yuv --frames 65 \
  --engine-dir /home/oelghati/NVCR/build/engines-b3-rtx4070-catalog/bundles/dcvcrt/rtx4070-ubuntu2404/qcif/ad3215592c2079b8719c7ac1deb19c99e5bd039d94d24a331eb364c300efed61
```

Comparison commands:

```bash
rtk cmp -s /tmp/nvcr-b4-byte-parity-65e52f3-b3.nvcr /tmp/nvcr-b4-byte-parity-65e52f3-b4.nvcr
rtk cmp -s /tmp/nvcr-b4-byte-parity-65e52f3-b3.yuv /tmp/nvcr-b4-byte-parity-65e52f3-b4.yuv
rtk sha256sum /tmp/nvcr-b4-byte-parity-65e52f3-b3.nvcr /tmp/nvcr-b4-byte-parity-65e52f3-b4.nvcr /tmp/nvcr-b4-byte-parity-65e52f3-b3.yuv /tmp/nvcr-b4-byte-parity-65e52f3-b4.yuv
```

Both `cmp -s` commands passed. The complete `.nvcr` streams, including outer
record framing and every serialized NVAU byte, have SHA-256:

`9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a`

The reconstructed YUV files have SHA-256:

`10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008`

Raw streams, reconstructed YUV, TensorRT plans, checkpoints, and source video
remain outside Git.

## Acceptance

All B4 technical gates pass on the available RTX 4070. The recorded pooled
throughput changes are -0.294% I/P encode, +1.206% I/P decode, -0.640%
all-intra encode, and +0.426% all-intra decode. The worst per-profile decrease
is -0.824%. The recorded result was explicitly accepted on 2026-09-13. With
the technical gates above passing, B4 is complete.
