# NVCR identity and scope

## One-line identity

NVCR is a native C++ runtime architecture for neural video codecs.

## The software problem

Learned compression models are not usable codec products by themselves. A
deployment needs stateful encoder and decoder sessions, lifecycle semantics,
execution and memory ownership, artifact selection, compatibility checks,
bounded stream framing, errors, tests, and reproducible release evidence.
NVCR supplies that systems layer without making the runtime core responsible
for one model architecture or one execution technology.

## What NVCR contributes

NVCR provides:

- stateful encoder and decoder session contracts, including delayed output,
  flush, drain, reset, and structured status/error behavior;
- a codec adapter contract for codec semantics and an execution-provider
  contract for executable artifacts and device execution;
- a static codec/provider registry and `RuntimeServices` bridge;
- artifact descriptors, catalogs, resolver ranking, hashes, provenance, and
  license-aware selection;
- bounded, versioned `NVAU` elementary access units with codec-private payload
  sections; and
- public C++/CLI interfaces, conformance tests, target-aware packaging, and
  reproducible evaluation infrastructure.

NVCR is therefore neither a new neural compression model nor simply a native
C++ reimplementation of DCVC-RT.

## The current production vertical

The first complete production integration is:

```text
NVCR runtime architecture
    |
    +-- DCVC-RT codec adapter
          |
          +-- TensorRT FP16 provider
                |
                +-- validated Linux/NVIDIA targets
```

The DCVC-RT adapter owns I/P semantics, reference state, native entropy coding,
and the codec-private payload. TensorRT owns target-local engine loading,
TensorRT/CUDA execution, synchronization, and provider-side failures. NVCR
core owns sessions, registry/services, artifact identity and selection, access
unit framing, and application-facing values.

DCVC-RT and TensorRT are the first supported codec/provider pair, not the
identity or long-term boundary of NVCR. The architecture is codec-extensible,
while DCVC-RT is currently the only production codec integration. The provider
contract is implemented, while TensorRT is currently the only production
provider.

## Capability classification

| Capability | Classification | Boundary and evidence |
|---|---|---|
| C++20 runtime values and session lifecycle | Implemented core contract | Public headers and CPU contract/unit tests |
| Codec adapter and provider interfaces | Implemented core contract | `ICodecAdapter`, provider API, registry/services tests |
| Static registration and runtime services | Implemented core contract | Registry implementation and contract tests |
| Artifact descriptors, catalog, resolver, ranking, hashes | Implemented core contract | Artifact and catalog tests plus validator scripts |
| Bounded `NVAU` v1 and sectioned v2 | Implemented core contract | Access-unit implementation, format tests, fuzz/boundary corpus |
| Deterministic test codec | Implemented conformance fixture | Test-only delayed-output and lifecycle coverage |
| CPU test provider | Implemented conformance fixture | Test-only tensor binding and provider error coverage |
| DCVC-RT adapter and native rANS | Implemented production path | Production codec sources and DCVC-RT tests |
| TensorRT FP16 provider-owned backend | Implemented production path | Target-local engine contracts and GPU round-trip gates |
| Independent TensorRT component loading via `IExecutable` | Implemented but transitional | Provider contract exists; current provider returns `not_implemented` for component-level loading |
| Additional codecs/providers, INT8 release, C ABI | Planned future work | Not current support claims |
| FFmpeg and standard containers | Planned future work | Explicitly outside the current release scope |

## Conformance fixtures are not products

The deterministic test codec and CPU provider make it possible to test the
runtime contracts without TensorRT or restricted model assets. They exercise
the architecture; they do not establish CPU neural inference, a second
production codec/provider, or a performance baseline.

## Stream and wrapper position

`NVAU` is the bounded codec access-unit contract. Version 1 is the narrow
DCVC-RT-shaped production/default representation; version 2 is the implemented
generalized sectioned representation. Learned payload syntax stays owned by
the codec adapter. `NVCR`, `NVCS`, and `.nvcr` are development/application
wrappers and must not be presented as standard multimedia containers. NVIF is
a future research direction, not a finalized standard.

NVCR does not claim byte or payload interchangeability with upstream Python
DCVC-RT. Reference consistency and payload compatibility are separate claims;
the former requires controlled comparison, while the latter requires clean
bidirectional golden tests.

## Current limitations and non-goals

The public C++ API/ABI is not frozen. Current production deployment is
Linux/NVIDIA and FP16-only. NVCR is not yet a supported v1 release. It is not
currently a multi-codec production runtime, multi-provider production runtime,
universal TensorRT-plan portability layer, CPU neural-codec implementation,
FFmpeg integration, or standard container implementation.

Check [scope and support](scope-and-support.md) for evidence requirements and
[ROADMAP.md](../ROADMAP.md) for planned work.
