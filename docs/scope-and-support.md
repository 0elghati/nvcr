# NVCR scope and support contract

This document is the authoritative product boundary for NVCR. When another
document, release note, or example conflicts with this page, this page wins and
the conflict is a documentation bug.

## Product definition

NVCR is an open-source deployment-oriented runtime architecture for neural video
codecs. The current release validates that architecture through one implemented
codec backend: the pinned DCVC-RT CVPR 2025 image/video model pair on
Linux C++20/CUDA/TensorRT.

For the currently supported DCVC-RT backend, the deployed path is:

```text
pinned checkpoints
  -> offline ONNX, entropy, and quantization export
  -> target-local TensorRT FP16 engines
  -> native C++ I/P encode and decode
  -> versioned codec access units
  -> reconstruction, correctness, performance, and energy evidence
```

Python and PyTorch are offline export and reference-conformance dependencies for
the DCVC-RT backend. They are not runtime dependencies of `libnvcr` or the
`nvcr` executable.

## Release tracks

| Track | Meaning | Required evidence |
|---|---|---|
| Current `0.2.x` | Historical development snapshots | No current product-readiness claim |
| Target `0.3.0` | Scope, safety, artifact-workflow, and desktop foundation release | Clean RTX 4070 checkpoint-to-native I/P round trip |
| Target `1.0.0` | First scoped product release | Complete RTX 4070 and Jetson Orin Nano validation matrices |
| Post-v1 | C ABI, FFmpeg, containers, and additional optimization | Separate roadmap gates |

The repository may contain implemented behavior before its release gate passes.
That behavior is **implemented**, not **supported**, until the required evidence
is recorded in `ROADMAP.md`.

## Declared v1 profile

The profile below is the intended v1 support contract. It is a declaration of
scope, not evidence that the current snapshot has passed every release gate.
The profile becomes supported only when the target-specific evidence in the
final section is recorded for the selected model, engine profile, and target.

### Codec and model

- Codec: DCVC-RT only.
- Model profile: `dcvcrt-cvpr2025`.
- Source repository: `https://github.com/0elghati/DCVC-RT.git`.
- Upstream reference commit:
  `48ab0ac5e5199d78fffb944bfbafafb2b6142f7b`.
- Image checkpoint: `cvpr2025_image.pth.tar`, SHA-256
  `555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9`.
- Video checkpoint: `cvpr2025_video.pth.tar`, SHA-256
  `b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6`.
- Normal operation: stateful I/P GOP encode and decode.
- Pixel boundary: planar 8-bit 4:2:0 (`YUV420P8`).
- Supported precision: TensorRT FP16.

### Validation targets

| Target profile | Host | GPU | Required runtime record |
|---|---|---|---|
| `rtx4070-ubuntu2404` | Linux x86_64 | NVIDIA GeForce RTX 4070, SM 8.9 | Validation in progress; record OS, driver, CUDA, TensorRT, compiler, and engine profile |
| `orin-nano-l4t3647` | Linux aarch64 | Jetson Orin Nano, SM 8.7 | Validation incomplete; record JetPack/L4T, power mode, clocks, CUDA, TensorRT, compiler, and engine profile |

These are validation targets, not a claim that every NVIDIA GPU, CUDA release,
TensorRT release, Linux distribution, or model shape is supported. Other
machines may build and run, but remain unvalidated until a target profile and
evidence matrix are added.

### Engine and frame profile

- Engines are fixed-profile, target-local executable artifacts.
- The tracked v1 profiles are QCIF and 1080p, with the exact min/opt/max shapes
  in `configs/engine-profiles/`.
- Visible dimensions are distinct from padded tensor dimensions.
- The current implementation accepts even dimensions from 64x64 through
  1920x1080 when covered by the selected engine profile.
- A model bundle and an engine bundle are accepted only when their identities,
  hashes, precision, target, CUDA/TensorRT versions, and optimization profile
  pass validation.

## Explicit non-goals for v1

- Training, fine-tuning, dataset management, or checkpoint production.
- Arbitrary ONNX/model conversion or a dynamically loaded codec plugin system.
- Other DCVC releases, learned codecs, precisions, or checkpoint pairs.
- INT8 as a default or supported release profile.
- Windows, macOS, CPU inference, or non-NVIDIA accelerators.
- A stable public C ABI.
- FFmpeg encoder/decoder integration or a standard container mapping.
- Universal GPU or TensorRT-plan portability.
- Redistribution of checkpoints, exported model assets, or TensorRT engines.

Additional codec backends, the C ABI, FFmpeg wrapper, and container milestones
remain useful future work, but they cannot be used as v1 acceptance evidence.

## Artifact and redistribution policy

NVCR source releases contain source code, versioned profiles, documentation,
tests, and reproducible tooling. They do not contain Microsoft checkpoints,
exported ONNX files, derived entropy/quantization assets, or TensorRT engines.

Users obtain the two checkpoints under their applicable terms and build model
and engine bundles locally. This policy remains in force until redistribution
rights are reviewed and recorded in the roadmap.

## Compatibility policy

- The `NVCR` packet envelope and `NVCS` sequence file are development
  application framing, not a stable codec elementary-stream contract.
- New codec payloads use the versioned access-unit contract documented in
  `docs/bitstream.md`.
- Reading older development payloads may remain temporarily when inexpensive,
  but no pre-v1 bitstream compatibility promise is made.
- NVCR v1 does not promise upstream DCVC-RT byte or payload interchangeability.
  Independent reconstruction quality remains pinned reference evidence; any
  future interoperability claim requires passing Python-to-native and
  native-to-Python golden tests.
- TensorRT engines are never portable across a different GPU model or TensorRT
  runtime merely because deserialization succeeds.

## Evidence required for “supported”

A supported target/profile combination must have all of the following recorded
in `ROADMAP.md` or a linked machine-readable result:

1. Clean source configure, Release build, test, and install.
2. Checkpoint hash and upstream commit verification.
3. Model export, bundle validation, target-local engine build, and engine hash
   validation.
4. Native I/P round trip across at least two complete GOPs, including reset and
   reuse.
5. Malformed/truncated/oversized access-unit and wrong-bundle rejection.
6. Cross-runtime payload/reconstruction comparison against the pinned Python
   implementation with recorded tolerances.
7. Release-build latency, memory, throughput, and rate/distortion under
   `docs/performance.md` without discarding failed/superseded history.
8. For Orin, recorded JetPack/L4T, compiler, power mode, clocks, and energy
   protocol/results; for RTX, the recorded OS/compiler/driver profile.
9. Exact-tag packaging validation proving required docs/licenses/manifests are
   present and checkpoints/derived assets are absent.
