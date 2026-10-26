# Phase C1-C3 runtime-ownership evidence — 2026-09-14

## Scope

This evidence closes the C1 codec-runtime ownership cleanup, C2
buffered/delayed-output lifecycle fixture, and C3 configuration ownership split
selected by [the C0 audit](../docs/cross-codec-requirements.md).

- Branch: `codex/phase-c1-codec-sessions`
- Baseline revision: `fe6cb8e9db0d90cf2ac33a6f1030a2600bd413e2`
- GPU: NVIDIA GeForce RTX 4070, compute capability 8.9
- Build runtime: CUDA 12.8.93 and TensorRT 10.9.0

## Ownership result

`ICodecAdapter::create_sessions` now returns paired codec-owned encoder and
decoder sessions. The generic `Runtime` is a serialized facade over those
sessions and retains shared `flush()`/`reset()` only for compatibility.
Direction-specific, mutex-protected methods expose independent drain/reset.

DCVC-RT owns its fixed-GOP I/P decisions, encoder and decoder `SequenceState`,
NVAU construction/validation, state commits, and output queues in
`src/dcvcrt/session.cpp`. The generic runtime no longer owns codec state,
one-input/one-output queues, or access-unit semantics.

The registered test codec has an eight-input/one-packet encoder and a
one-packet/eight-frame decoder. Its contract covers pre-output `try_again`, a
full group, ordered frame recovery, a short group emitted only on encoder
flush, full drain, independent encoder/decoder flush and reset, reuse, and a
fresh repeated session.

`RuntimeConfiguration` now has five explicit scopes: `runtime`, `codec`,
`provider`, `artifacts`, and `stream`. Existing CLI flags and flat config-file
keys map into those scopes, so user-facing configuration behavior is unchanged.

## Release verification

A clean Release tree was configured with an explicit matched toolchain and the
new target-local 1080p bundle:

```text
cmake -S . -B build-c1-release-cuda128 \
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
  -DNVCR_TENSORRT_ENGINE_DIR=/home/oelghati/NVCR/build/engines-c1-trt109/dcvcrt-1080p
cmake --build build-c1-release-cuda128 -j2
ctest --test-dir build-c1-release-cuda128 --output-on-failure
```

Result: 22/22 tests passed. This includes the registered lifecycle contracts,
parser and format contracts, CUDA operations, the 1080p TensorRT engine
contract, the registered I/P round trip, and the pinned Python/native I-frame
golden. A clean `cmake --install` to `/tmp/nvcr-c1-c3-install` also passes and
installs the new session and configuration headers with the Release library and
package metadata.

`readelf` records a runpath of `/usr/local/cuda-12.8/lib64:/usr/local/lib` for
the TensorRT test binary, and `ldd` resolves `libcudart.so.12` from the CUDA
12.8 directory. This matters because an older cached build tree remained linked
to CUDA 12.6 and correctly rejected the CUDA 12.8 engine plan; no compatibility
check was relaxed.

## Byte and reconstruction parity

The C1-C3 CLI encoded and decoded the first 65 frames of
`/home/oelghati/datasets/qcif/akiyo_qcif.yuv` at 176x144, QP 32, GOP 8, and
30 fps with the retained exact B7 QCIF catalog bundle. The values match
[the B7 baseline](vision-b7-rtx4070-20260914.md#byte-and-reconstruction-parity):

- complete `.nvcr` SHA-256:
  `9896684474818032506d32ac7d065a7710d9c26197a9d374f275d2b65f83f81a`;
- decoded YUV SHA-256:
  `10e02e460ec4477e9a7e41f72bbb44c5222e0f6773bf581cf874f74b97363008`;
- NVAU payload bytes: 14,669.

The exact stream and reconstruction prove that moving policy/state/AU handling
into codec sessions did not change current DCVC-RT behavior.

## Decision boundary

C1-C3 do not add a chunk API, second codec, second provider, or capability
taxonomy. C4 subsequently resolved the grouped-access-unit timing question at
the serialized `PacketIO` boundary; see
[the C4 evidence](phase-c4-grouped-au-timing-20260914.md). C5 subsequently
rejected the tested public MLVC default FP16 export under its predeclared
tolerance; see [the C5 evidence](phase-c5-mlvc-reference-export-20260914.md).
C6 traced the first stream-relevant divergence to encoder raw symbols before
entropy coding and rejected that pair as the next production axis; see
[the C6 evidence](phase-c6-mlvc-cuda-equivalence-20260914.md). The remaining
work is the DCVC-UF fused-operator export and the minimum
application-admission capability set. Their experiment and decision rules
remain in
[the open-question handoff](../docs/cross-codec-requirements.md#open-questions-requiring-prototypes).
