# Compatibility and support matrix

This page summarizes the authoritative policy in
[Scope and support](scope-and-support.md). “Implemented” means code exists;
“supported” additionally requires current evidence on the stated target.

## Release status

| Track | Status | Meaning |
|---|---|---|
| 0.2.x | Historical development snapshots | No stable model/engine/stream compatibility promise |
| 0.3 | Active foundation target | Scope, correctness, artifact, release, and RTX clean-room gate |
| 1.0 | Pending | First supported product release, gated on RTX 4070 and Orin Nano |

## Model identity

The only v1 codec backend is DCVC-RT, and the only v1 model profile is
`dcvcrt-cvpr2025`, defined by
[configs/models/dcvcrt-cvpr2025.json](../configs/models/dcvcrt-cvpr2025.json).
It binds the pinned DCVC-RT source commit, exact image/video checkpoint names and
SHA-256 digests, FP16 deployment precision, YUV420P8, and QP bounds.

A checkpoint with a different hash, another DCVC model family, a modified export,
or a bundle whose model-manifest digest does not match is incompatible. Supporting
another checkpoint requires a new versioned model profile and its own reference,
correctness, and target evidence; changing a filename is not sufficient.

## Target matrix

| Target profile | OS / platform | CUDA | TensorRT | GPU | Status |
|---|---|---|---|---|---|
| `rtx4070-ubuntu2404` | Ubuntu 24.04 x86_64 | 12.6 | 10.7.0 | RTX 4070, SM 8.9 | v0.3 validation in progress |
| `orin-nano-l4t3647` | L4T 36.4.7 aarch64 | 12.6.68 | 10.3.0.30 | Jetson Orin Nano, SM 8.7 | current v1 validation incomplete |

Exact recorded compiler, driver/JetPack, memory, power, and clock details belong
in the target profile and roadmap evidence. Other NVIDIA Linux hosts may build or
run but are experimental/community-supported until their own evidence is added.
Windows, macOS, non-NVIDIA GPUs, and non-Linux deployment are unsupported in v1.

## Public package families

Published binary archives use generic family names such as
`linux-x86_64-nvidia` and `linux-aarch64-jetson-l4t36`. These identifiers are
package labels, not support claims. The current validated reference targets and
evidence requirements remain the RTX 4070 and Jetson Orin Nano profiles above.
The two packages are required because native x86_64 and AArch64 ELF binaries are
not CPU-architecture portable. CUDA fat binaries only broaden GPU-code coverage
within one host architecture.

## Engine compatibility

TensorRT plans are target-local. An engine bundle is compatible only when all of
these match its v2 manifest and checksum set:
- exact SHA-256 digests of the model, target, and engine-profile JSON files;

- model profile and I/P model-manifest digests;
- FP16 engine profile and optimization dimensions;
- CUDA runtime and exact TensorRT version;
- GPU model, compute capability, and multiprocessor count;
- every required plan and runtime asset SHA-256 digest.

NVCR rejects a stale, edited, wrong-model, cross-GPU, cross-CUDA, or
cross-TensorRT bundle during initialization. A multi-architecture CUDA library
does not relax TensorRT plan compatibility. Rebuild engines on the final target.

Published engines live outside semver application releases in the rolling
`engine-assets` release. `nvcr-engine-catalog.json` first narrows by Linux host
architecture, then requires exact device name, compute capability, SM count,
CUDA runtime, and TensorRT version before downloading an archive.

TensorRT does provide optional discrete-GPU hardware-compatibility modes for some
deployment environments, but NVCR does not currently ship public compatibility-
mode engines. Jetson/L4T remains a target-local engine-build path.

If reviewer-convenience compatibility-mode bundles are added later, prefer
`SAME_COMPUTE_CAPABILITY` for discrete x86_64 NVIDIA targets. Treat broader
`AMPERE_PLUS` bundles as a separate non-default class with explicit correctness
and performance evidence, and not as a Jetson portability path.

## Feature matrix

The final column is a policy/status label, not a substitute for target evidence.
“Declared, target-gated” means the capability is in the v1 contract but cannot
be called supported until the evidence rules below pass on the relevant target.

| Capability | Development state | v1 policy/status |
|---|---|---|
| Native I/P FP16 encode/decode | Implemented; RTX integration tests pass | Declared, target-gated on both reference targets |
| YUV420P8 raw frame I/O | Implemented | Declared, target-gated |
| Fixed 64×64–1920×1080 engine profiles | Implemented profiles | Declared within the selected profile, target-gated |
| High P effective QP through 71 | Implemented and tested | Declared, target-gated with boundary evidence |
| Reset/reuse and multiple GOPs | Implemented and tested on RTX | Declared, target-gated on both targets |
| `NVAU` access unit | Implemented, versioned, bounded | Candidate v1 codec contract; cross-runtime evidence pending |
| `NVCR` packet / `NVCS` sequence files | Implemented CLI development framing | No stable pre-v1 compatibility promise |
| Upstream Python payload/frame compatibility | Bidirectional framing-only probe fails at the I-frame entropy boundary | Unsupported in v1; no byte-compatibility promise |
| Reusable CUDA arena / fully device-resident state | In progress | Required performance/runtime gate |
| INT8 | Experimental builder flag | Unsupported/default-disabled in v1 |
| Additional codec backends | Architecture boundary introduced; none implemented | Post-v1 |
| Public C ABI | Not implemented | Post-v1 |
| FFmpeg / standard container integration | Not implemented | Post-v1 |
| Prebuilt checkpoints, model assets, or engines | Deliberately not shipped | Users build derived bundles locally |

## Evidence required for “supported”

A feature is supported only after its relevant CPU/parser tests, GPU I/P tests,
reference comparison, clean-room artifact build, release-build performance
protocol, and target-specific evidence pass and are recorded in
[ROADMAP.md](../ROADMAP.md). Documentation or successful compilation alone is not
support evidence.
