# NVCR identity and scope

NVCR is a native C++ runtime architecture for neural video codecs.

## Purpose

Learned compression models are not usable codec implementations by themselves.
Applications also need stateful encoder and decoder sessions, execution and
memory ownership, compatible model artifacts, bounded stream framing, error
handling, tests, and reproducible deployment. NVCR provides that systems layer.

NVCR is neither a new compression model nor simply a C++ reimplementation of
DCVC-RT. Its architecture separates codec semantics from execution-provider
mechanics:

```text
NVCR runtime architecture
    |-- stateful encoder and decoder sessions
    |-- codec adapter and execution-provider contracts
    |-- registry and runtime services
    |-- target-aware artifact resolution
    |-- bounded NVAU access units
    |-- public C++ API and CLI
    `-- validation and reproducibility tools
          |
          `-- current supported integration
                `-- DCVC-RT codec adapter
                      `-- TensorRT FP16 provider
                            `-- Linux/NVIDIA targets
```

## Main capabilities

NVCR provides:

- stateful send/receive, flush, receive-until-end-of-stream, and reset
  semantics for encoders and decoders;
- explicit codec-adapter and execution-provider interfaces;
- codec and provider discovery through a static registry;
- model, target, and engine profiles with catalog selection, hashes, and
  compatibility checks;
- bounded, versioned `NVAU` access units with codec-owned payload sections;
- a C++20 library, command-line interface, release packages, and containers;
- CPU conformance fixtures for testing the runtime without neural-model
  artifacts; and
- benchmark and result-recording infrastructure.

## Current supported integration

The first supported codec/provider pair is DCVC-RT with TensorRT FP16 on
NVIDIA Linux systems. This path includes I/P-frame reference state, native
entropy coding, artifact validation, encoding, decoding, and target-aware
deployment.

The DCVC-RT adapter owns frame semantics, reference state, entropy behavior,
and codec-private payloads. The TensorRT provider owns engine loading,
TensorRT/CUDA execution, synchronization, and provider failures. NVCR core owns
the public session lifecycle, registration, artifact selection, access-unit
framing, and application-facing values.

DCVC-RT and TensorRT are the first supported integration, not the identity or
long-term limit of NVCR.

## Implementation status

| Area | Status | Boundary |
|---|---|---|
| Runtime values, sessions, registry, and errors | Implemented core contract | Public headers and contract tests |
| Codec adapter and provider interfaces | Implemented core contract | Static registration and `RuntimeServices` |
| Artifact catalog, resolver, hashes, and compatibility ranking | Implemented core contract | Artifact tests and validation tools |
| `NVAU` v1 and sectioned v2 | Implemented core contract | Format tests and parser bounds |
| Deterministic codec and CPU provider | Conformance fixtures | Tests only; not CPU neural inference |
| DCVC-RT adapter and native entropy coding | Supported integration | Current production codec path |
| TensorRT FP16 provider-owned backend | Supported integration | Current production provider path |
| Component-level TensorRT executable loading | Transitional | Interface exists; current provider does not implement independent stage loading |
| Additional codecs/providers, INT8, stable C ABI, FFmpeg, standard containers | Planned | Not current functionality |

## Stream boundary

`NVAU` is the codec access-unit contract. Version 1 is the current
narrow/default representation; version 2 is the implemented generalized
sectioned representation. Codec payload syntax remains owned by the adapter.

`NVCR`, `NVCS`, and `.nvcr` are application and development wrappers, not
standard multimedia containers. NVCR does not claim byte-level compatibility
with the upstream Python DCVC-RT payload.

## Current limitations

The public C++ API and ABI are not frozen. The supported neural-codec path is
currently Linux/NVIDIA and FP16 only. NVCR does not currently provide CPU
neural inference, a native Windows interface, multiple supported production
codecs or providers, universal TensorRT-plan portability, FFmpeg integration,
or a standard-container mapping.

See [Scope and support](scope-and-support.md) for validation boundaries and
[Roadmap](../ROADMAP.md) for planned work.
