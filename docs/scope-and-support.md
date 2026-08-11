# Scope and support

This page separates four concepts that are easy to conflate: what NVCR is,
what the architecture implements, what production support has validated, and
what remains on the roadmap.

## 1. NVCR identity

NVCR is a native C++ runtime architecture for neural video codecs. It is a
codec-extensible systems layer for stateful sessions, execution-provider
mediation, target-aware artifacts, and bounded access units. Linux/NVIDIA is
the current production deployment boundary, not the definition of NVCR.

## 2. Implemented architecture

The repository implements:

- encoder and decoder session contracts with send/receive, `try_again`, flush,
  drain, reset, state, statistics, and structured errors;
- codec descriptors, capabilities, option schemas, `ICodecAdapter`, and a
  static codec/provider registry;
- provider descriptors, capabilities, tensor bindings, `IExecutable`,
  `IExecutionProvider`, and the optional offline `IArtifactCompiler` contract;
- provider-mediated `RuntimeServices` and artifact descriptors, catalogs,
  resolution, compatibility ranking, hashes, and license restrictions;
- bounded `NVAU` v1 and implemented generalized sectioned `NVAU` v2 parsing and
  serialization;
- a deterministic test codec and CPU provider for conformance tests; and
- the DCVC-RT adapter, native entropy coding, TensorRT provider registration,
  CLI, packaging, and evaluation tooling.

These are implementation facts. They do not all represent production support.

## 3. Validated production support

The current production vertical is:

```text
DCVC-RT codec adapter -> TensorRT FP16 provider -> Linux/NVIDIA target
```

The checked-in target profiles are RTX 3050 Laptop, RTX 4070, RTX 5060 Laptop,
and Jetson Orin Nano. They describe requested identities and artifact classes;
they are not automatic support claims. The test codec and CPU provider are
conformance fixtures, not CPU neural-codec products or performance baselines.

Support evidence levels are cumulative:

| Evidence level | What it establishes |
|---|---|
| Core contract validation | CPU tests cover public values, session lifecycle, registry/services, format bounds, resolver behavior, and malformed input |
| Production codec/provider validation | DCVC-RT/TensorRT builds, artifact checks, I/P round trips, native entropy, and provider-mediated construction pass |
| Exact-target validation | The named target's current artifact bundle and runtime round trips pass with recorded identity and hashes |
| Compatibility-mode validation | A same-compute or Ampere-plus bundle passes its explicit correctness and comparison gates; Jetson does not use these classes |
| Execution-portability validation | The specific host binary, CUDA code, container userspace, or engine class is tested for the stated portability claim |
| Controlled performance evidence | Release measurements use the documented input, timing boundary, repetitions, instrumentation separation, and environment record |

A target becomes supportable only when the applicable levels pass. A matching
JSON profile, successful compilation, TensorRT deserialization, or CPU test run
alone is insufficient.

## 4. Current non-support claims

The current release track does not claim:

- a second production codec or execution provider;
- CPU neural-codec inference, Windows, macOS, or non-NVIDIA production support;
- universal TensorRT-plan portability or provider-independent execution;
- a stable C ABI or frozen C++ ABI;
- upstream Python DCVC-RT byte/payload interchangeability;
- `NVAU` as an industry standard, MP4/Matroska container, FFmpeg integration,
  or finalized NVIF specification; or
- redistribution permission for checkpoints, exported models, engine plans, or
  datasets without a separate asset review.

## Roadmap boundary

The current priorities are exact-target artifact gates, complete evaluation
matrices, pinned-reference comparisons, package/license review, and external
consumer verification. Later codec/provider integrations and media-framework
or standard-container integration remain future work. See [ROADMAP.md](../ROADMAP.md).
