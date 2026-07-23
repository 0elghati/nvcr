# DCVC-RT integration contract

NVCR has one codec backend: the pinned `dcvcrt-cvpr2025` I/P model profile. The
backend owns all TensorRT, CUDA, native rANS, and temporal-state ordering; these
cannot be separated into generic pre/post processors because entropy coding and
spatial priors alternate.

## Initialization

A session receives an explicit model ID, engine directory, device ID, QP, GOP,
packet/memory limits, TensorRT execution mode, and legacy-compatibility policy.
Before TensorRT plan deserialization, initialization validates the v2 bundle
schema, exact file/checksum set, model-manifest identities, FP16, CUDA/TensorRT,
and selected GPU identity. It then loads 14 plans and six runtime assets, creates
the stream/contexts, uploads all 72 P-frame quantization entries, and warms the
engines.

Artifact preparation is described in [Model and engine preparation](dcvcrt-artifacts.md).
`scripts/nvcr_artifacts.py` is the supported front end; individual exporters and
builders are internal helpers.

## Encode/decode

The I path owns:

- `i_analysis`, `i_hyper_analysis`, `i_hyper_synthesis`;
- `i_spatial_prior_1`, `_2`, and `_3` over four checkerboard passes;
- `i_synthesis`, image entropy tables/quantization, and native rANS.

The P path owns:

- frame and learned-feature reference adaptation;
- `p_analysis`, `p_hyper_analysis`, `p_prior`, `p_spatial_prior`, `p_synthesis`;
- adaptive effective QP (`0..71`), video entropy tables/quantization, native rANS;
- encoder and decoder device DPBs with frame/feature state.

The encoder returns the actual effective QP so the outer `NVAU` access unit binds
the quantization entry that was used. Invalid base/derived QP indexes, dimensions,
frame order, or missing references fail before unsafe access.

## State and execution policy

Encoder and decoder state are independent. An I-frame starts/reset a GOP; P-frames
require a reference. `reset()` clears state immediately and `flush()` completes
backend work then resets the session. Two-GOP and reset/reuse behavior is covered
by the registered TensorRT roundtrip test.

`automatic`, `low_memory`, and `performance` TensorRT execution modes are selected
per runtime configuration. Performance mode reuses execution contexts; low-memory
mode recreates them to lower residency. The reusable device arena and removal of
remaining avoidable per-frame allocations are active v1 work.

## Entropy and payloads

The vendored native rANS implementation is the runtime entropy coder. Python is
used only as an offline model/reference tool. `NVI1`/`NVP1` remain isolated inner
payload formats; the versioned `NVAU` codec contract carries model identity,
dimensions, type, effective QP, reset state, and bounded payload length.

No upstream Python payload or reconstruction compatibility claim is made until
cross-runtime golden tests pass. The CLI's `NVCR`/`NVCS` wrappers are also
pre-v1 development formats, not containers.

## Current gates

Implemented tests cover parser/allocation bounds, rANS vectors, CUDA operators,
model/profile and bundle tampering, wrong-model/corrupt bundle rejection, I/P
reconstruction equality, maximum effective P QP, two GOPs, and reset/reuse on the
configured RTX engine bundle.

Still required for v1 are reference Python golden evidence, a stable plane/stride
session API, the reusable device arena and complete GPU state path, release-build
performance gates, and a fresh Orin clean-room workflow. See [ROADMAP.md](../ROADMAP.md).
