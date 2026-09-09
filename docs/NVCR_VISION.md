# NVCR Vision

> **NVCR — Neural Video Codec Runtime**
>
> A native, codec-extensible, provider-extensible systems layer for turning neural video compression models into usable, portable, reproducible video codec implementations.

Status: Long-term direction

Last reviewed: 2026-09-09

Execution status: [NVCR roadmap](../ROADMAP.md)

Current boundary design: [NVCR v2 codec/provider RFC](provider-boundary-v2.md)

---

## 1. Purpose of This Document

This document defines the long-term vision for NVCR.

It is intentionally broader than the current implementation and broader than any single roadmap sprint. Its purpose is to answer:

- What is NVCR trying to become?
- What architectural boundaries must remain stable as the project grows?
- How should new codecs, execution providers, formats, platforms, and integrations fit together?
- What evidence is required before NVCR can claim to be generic?
- What work belongs inside NVCR, and what should remain outside it?
- How should the project evolve from a successful DCVC-RT/TensorRT implementation into a reusable neural-video-codec systems platform?

This is a **vision document**, not a statement that every capability described here already exists.

The current repository should be treated as the working foundation from which this vision is realized incrementally.

---

# 2. The Core Problem

Neural video codecs are usually delivered as research implementations.

A typical codec repository provides some combination of:

- Python model code;
- research checkpoints;
- entropy coding scripts;
- experimental evaluation tools;
- fixed assumptions about hardware;
- framework-specific execution;
- project-specific bitstream handling.

That is enough to reproduce a paper.

It is not enough to make a codec behave like deployable systems software.

A practical codec needs much more:

- persistent encode/decode sessions;
- temporal state management;
- execution on real hardware;
- model and engine discovery;
- versioning;
- artifact compatibility;
- deterministic errors;
- stream framing;
- bounded parsing;
- packaging;
- reproducible builds;
- deployment tooling;
- platform validation;
- application integration.

NVCR exists to provide this missing systems layer.

---

# 3. North Star

The long-term objective is:

> **Any supported neural video codec should be able to plug into NVCR, run through one or more execution providers, produce codec-neutral access units, and be consumed by normal applications without the application needing to understand the neural network implementation.**

Conceptually:

```text
                        Applications
                             │
                             ▼
                     ┌───────────────┐
                     │     NVCR      │
                     │    Runtime    │
                     └───────┬───────┘
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
        Codec Layer     Provider Layer    Stream Layer
             │               │               │
      ┌──────┴──────┐   ┌────┴─────┐    ┌────┴─────┐
      │             │   │          │    │          │
   DCVC-RT        MLVC TensorRT  ORT-CUDA NVAU v2  Future
      │             │   │          │              container
      └──────┬──────┘   └────┬─────┘
             │               │
             └───────┬───────┘
                     ▼
                 Hardware
```

The specific codecs and providers in the diagram are examples. **Applications
depend on NVCR contracts, not on DCVC-RT, TensorRT, PyTorch, ONNX Runtime, CUDA,
or any future codec implementation.**

---

# 4. NVCR Is a Runtime, Not a Codec

NVCR must never become confused with the neural codec itself.

A codec integration owns compression semantics.

NVCR owns the reusable runtime environment around those semantics.

## The codec owns

- model-specific coding decisions;
- GOP semantics;
- I/P/B or other frame semantics;
- reference-state transitions;
- latent/reference state;
- rate/quality controls;
- entropy meaning;
- codec-private payload syntax;
- model-component meaning;
- codec-specific compatibility rules.

## NVCR owns

- session lifecycle;
- runtime discovery;
- codec registration;
- provider registration;
- configuration;
- common errors;
- artifact lookup;
- provider negotiation;
- access-unit framing;
- bounded parsing;
- application-facing APIs;
- packaging;
- reproducibility;
- deployment contracts.

NVCR should make neural codecs usable without redefining their compression algorithm.

---

# 5. NVCR Must Also Be Provider-Independent

A neural codec should not be architecturally tied to one inference technology.

The provider layer exists so that the same codec semantics can be implemented through different hardware and runtime stacks.

Potential provider families include:

- TensorRT;
- ONNX Runtime CUDA;
- ONNX Runtime DirectML;
- OpenVINO;
- Windows ML;
- Core ML;
- QNN;
- other future accelerator runtimes.

The provider owns:

- executable artifact loading;
- device selection;
- device memory;
- tensor memory;
- tensor bindings;
- streams;
- synchronization;
- execution contexts;
- graph capture;
- provider-specific optimization;
- provider-specific errors.

The codec adapter must not know:

- TensorRT engine filenames;
- CUDA device pointers;
- TensorRT binding indexes;
- provider-specific graph objects;
- provider-specific serialized plans;
- local artifact paths.

The runtime should mediate this boundary.

---

# 6. The Replacement Invariant

The central long-term design constraint is:

> **Codec semantics and execution technology must be independently replaceable.**

That implies a matrix, not a chain.

The project should eventually be able to support combinations such as:

| Codec | TensorRT | ONNX Runtime CUDA | Other provider |
|---|---:|---:|---:|
| DCVC-RT | Yes | Yes | Future |
| MLVC or second codec | Yes/Optional | Yes | Future |
| Future codec | Optional | Yes | Future |

This matrix is the real proof of extensibility.

One codec with multiple providers proves provider independence.

Multiple codecs with one provider prove codec independence.

Multiple codecs with multiple providers prove the architecture.

---

# 7. The Current Foundation

The current NVCR repository already establishes important systems foundations:

- native C++20 runtime;
- stateful encoder and decoder sessions;
- runtime configuration;
- codec descriptors;
- provider descriptors;
- artifact descriptors;
- artifact resolution;
- model/engine compatibility metadata;
- TensorRT execution;
- native rANS integration;
- bounded access-unit parsing;
- NVAU v1;
- sectioned NVAU v2;
- CLI;
- CMake packaging;
- native installation;
- Docker workflows;
- x86-64 support;
- Jetson support;
- tests and fuzzing;
- benchmark tooling;
- reproducibility infrastructure;
- release automation;
- controlled result retention.

The first complete vertical is:

```text
DCVC-RT
   │
   ▼
NVCR runtime
   │
   ▼
TensorRT FP16
   │
   ▼
NVIDIA GPU
```

This vertical proves that NVCR can run a real neural video codec efficiently.

It does **not yet**, by itself, prove generality.

The next era of the project is about proving that the abstractions survive beyond this first vertical.

---

# 8. NVCR v2 Vision

The major architectural evolution should be treated as the foundation of an NVCR v2 generation.

The objective is not to rewrite the project.

The objective is to remove the remaining assumptions that are specific to the first implementation.

The v2 architecture should make these relationships explicit:

```text
Application
    │
    ▼
Runtime
    │
    ├── Codec Adapter
    │      │
    │      ├── codec semantics
    │      ├── reference state
    │      ├── rate/quality controls
    │      └── codec payload
    │
    ├── Runtime Services
    │      │
    │      ├── provider resolution
    │      ├── artifact resolution
    │      └── execution/session creation
    │
    ├── Execution Provider
    │      │
    │      ├── executable models
    │      ├── device buffers
    │      ├── scheduling
    │      ├── synchronization
    │      └── device execution
    │
    └── Stream Layer
           │
           ├── NVAU
           ├── dependencies
           ├── ordering
           ├── metadata
           └── codec-private sections
```

The production TensorRT path should eventually use the same generic execution contract that future providers use.

No hidden DCVC-RT/TensorRT-specific construction path should remain necessary for production execution.

---

# 9. Performance Is Part of the Architecture

A clean abstraction that destroys performance is not a successful abstraction.

NVCR is systems software.

Its execution-provider API must therefore preserve the performance characteristics that matter for neural video coding:

- GPU-resident intermediate tensors;
- persistent execution contexts;
- persistent memory where useful;
- bounded low-memory modes;
- shared workspaces where appropriate;
- CUDA graphs or equivalent provider optimizations;
- asynchronous execution where appropriate;
- controlled synchronization;
- reusable provider sessions;
- minimal host/device copies;
- predictable allocations;
- stable latency.

The generic provider boundary should not assume that every model invocation is:

```text
CPU tensor -> provider -> CPU tensor
```

Instead, provider-owned buffers and execution lifetimes should be first-class concepts.

The provider must be free to keep state and tensors resident on the accelerator when codec semantics permit it.

---

# 10. The Second Provider Is an Architectural Proof

The next provider integration should be used to test the architecture rather than simply to add another backend.

A strong first target is ONNX Runtime with CUDA.

Why:

- it is meaningfully different from direct TensorRT;
- it can consume ONNX assets already natural to the existing model-export flow;
- it has C/C++ APIs;
- it supports GPU execution;
- it can support device-side I/O;
- it makes cross-provider comparisons possible.

The key experiment is:

```text
                DCVC-RT
                   │
           ┌───────┴────────┐
           │                │
       TensorRT         ORT-CUDA
```

The same codec should run through both providers without duplicating codec semantics.

Both providers encoding video is insufficient.

Success means:

- the same runtime API;
- the same codec adapter;
- the same session semantics;
- equivalent frame/reference behavior;
- equivalent rate/quality configuration;
- controlled bitstream compatibility position;
- comparable correctness evidence;
- measured performance;
- measured memory use.

---

# 11. The Second Codec Is the Other Architectural Proof

The second codec should be selected because it stresses the abstraction.

It should not be added merely to increase the codec count.

A useful second codec should exercise differences in:

- model structure;
- state representation;
- frame dependencies;
- GOP semantics;
- artifact layout;
- rate control;
- entropy coding;
- delayed output;
- resolution profiles;
- provider requirements.

MLVC is a practical candidate because it already targets multiple consumer execution backends and uses exportable model formats.

Other codecs may be better choices if they provide stronger architectural diversity.

The second codec should force NVCR to prove that no DCVC-RT-specific assumptions have leaked into:

- runtime configuration;
- codec descriptors;
- provider APIs;
- artifact resolution;
- NVAU metadata;
- lifecycle;
- reference handling;
- error handling.

---

# 12. NVAU: Codec-Neutral Access Units

NVAU is a central long-term part of NVCR.

Its purpose is not to standardize neural compression algorithms.

Its purpose is to provide a runtime-facing envelope in which codec-specific data can be transported safely and consistently.

A mature NVAU contract should support:

- codec identity;
- codec profile;
- model-set identity where required;
- frame type;
- visible dimensions;
- quality/rate information;
- decode order;
- presentation order;
- dependencies;
- random access;
- reset/discontinuity;
- typed sections;
- codec-private payloads;
- forward-compatible optional extensions;
- bounded parsing.

The stream layer must remain ignorant of:

- TensorRT;
- CUDA;
- ONNX Runtime;
- engine files;
- graph internals;
- local device state.

Provider details are never part of normative stream identity.

---

# 13. NVAU Is Not the Final Container

NVAU should remain an elementary codec access-unit format.

It should not grow into a complete multimedia container.

A complete media container needs concepts such as:

- tracks;
- timestamps;
- time bases;
- duration;
- file indexes;
- random-access indexes;
- seeking;
- edit lists;
- audio;
- subtitles;
- metadata tracks;
- muxing;
- network-friendly segmentation.

Those concerns belong at a higher layer.

The long-term stack should look like:

```text
Neural Codec
    │
    ▼
NVCR Codec Adapter
    │
    ▼
NVAU
    │
    ▼
NVIF / Standard Container Mapping
    │
    ▼
FFmpeg / GStreamer / Applications
    │
    ▼
Streaming / Storage / Transport
```

The exact name and specification of the higher-level format may evolve.

The architectural boundary should not.

---

# 14. NVCR and the Unified Neural Video Format Direction

NVCR creates the execution-side foundation for a future codec-neutral video ecosystem.

The broader progression is:

```text
Neural codec
    ↓
Common runtime
    ↓
Common access-unit contract
    ↓
Common file/container representation
    ↓
Media-framework integration
    ↓
Streaming and adaptive delivery
```

NVCR is primarily responsible for the first three layers.

Higher-level file, container, and streaming specifications should remain sufficiently independent that they can evolve without coupling the runtime to a single media workflow.

---

# 15. Artifact Architecture

Neural codecs are unusually dependent on deployment artifacts.

A complete runtime cannot assume that “the model” is one portable file.

A deployed codec may require:

- checkpoints;
- ONNX models;
- quantization tables;
- entropy assets;
- provider-specific compiled graphs;
- TensorRT engines;
- Core ML packages;
- QNN binaries;
- OpenVINO IRs;
- shape/profile-specific assets.

NVCR should continue to treat artifacts as explicit, versioned, target-aware objects.

An artifact request should be able to identify:

- codec;
- codec version;
- model set;
- component;
- provider;
- precision;
- target;
- input/profile;
- runtime compatibility;
- schema version;
- digest;
- provenance;
- license status.

The resolver chooses candidates.

The provider validates final executability.

Packages should remain separate from model/engine bundles where licensing and target specificity require that separation.

---

# 16. Reproducibility Is a Product Feature

NVCR should remain unusually strict about reproducibility.

Performance and correctness claims are meaningful only when tied to:

- exact source revision;
- exact release;
- exact codec model;
- exact checkpoint;
- exact artifact;
- exact provider version;
- exact CUDA/runtime version;
- exact GPU/device;
- exact input sequence;
- exact codec configuration;
- exact benchmark procedure.

The repository should continue to retain machine-readable evidence rather than plots alone.

Reproducibility is not documentation overhead.

It is part of the runtime's scientific credibility.

---

# 17. Correctness Before Performance Claims

NVCR should distinguish multiple levels of compatibility.

For example:

## Self-conformant

NVCR encode can be decoded by NVCR decode using the same codec/provider family.

## Cross-provider conformant

A bitstream encoded through provider A can be decoded through provider B for the same codec.

## Reference-consistent

NVCR reconstruction and rate behavior agree with a pinned upstream implementation within declared tolerances.

## Byte-interchangeable

NVCR produces the exact same codec payload syntax as another implementation.

These are different claims.

They must never be conflated.

A new codec/provider integration should explicitly state which level is supported.

---

# 18. Cross-Provider Bitstream Compatibility as a Key Experiment

One of the strongest future validation experiments is:

```text
TensorRT encode
      │
      ▼
    NVAU
      │
      ▼
ORT-CUDA decode
```

and:

```text
ORT-CUDA encode
      │
      ▼
    NVAU
      │
      ▼
TensorRT decode
```

If this works with controlled quality/rate results, NVCR has demonstrated that:

- provider identity is not encoded into the stream;
- provider-specific execution is correctly abstracted;
- codec semantics live above the provider;
- the stream is provider-neutral.

This is more important than simply reporting two throughput numbers.

---

# 19. Public API Philosophy

The public API should be small, stable, and lifecycle-oriented.

The core abstraction should continue to resemble a streaming codec API:

## Encoder

```text
create session
send frame
receive access unit
send frame
receive access unit
...
flush
drain
reset/reuse
```

## Decoder

```text
create session
send access unit
receive frame
send access unit
receive frame
...
flush
drain
reset/reuse
```

The API must not assume:

- one frame in means one packet out;
- output is immediate;
- there is no reordering;
- every codec uses the same frame types;
- every provider executes synchronously.

This makes room for future codecs without redesigning the runtime.

---

# 20. Configuration Philosophy

Configuration should be divided into three domains.

## Runtime configuration

Examples:

- codec ID;
- provider preference;
- device;
- threading;
- profiling;
- logging;
- memory policy.

## Codec configuration

Examples:

- QP;
- quality level;
- GOP;
- reset interval;
- bitrate/rate mode;
- codec-specific tools.

## Provider configuration

Examples:

- precision;
- execution mode;
- device ID;
- graph capture;
- memory policy;
- provider-specific tuning.

These should remain separated even if the CLI exposes them through one command.

---

# 21. Hardware Scope

NVCR should begin from platforms where neural codecs can realistically run.

The current NVIDIA path is therefore strategically correct.

Long-term support can expand toward:

- desktop NVIDIA GPUs;
- Jetson;
- AMD/DirectML-class GPU execution;
- Intel CPU/GPU/NPU;
- Apple GPU/Neural Engine;
- Qualcomm NPU;
- embedded accelerators.

NVCR should not promise a platform merely because the provider SDK theoretically exists.

A platform is supported only when:

- the runtime builds;
- artifacts are available;
- sessions execute;
- correctness is validated;
- representative hardware is tested.

---

# 22. CLI Role

The CLI should remain a first-class user of the public runtime.

It should not become a privileged internal path.

Anything the CLI can do should conceptually be achievable through the public runtime/API.

The CLI should remain useful for:

- encode;
- decode;
- codec discovery;
- provider discovery;
- compatibility inspection;
- artifact inspection;
- stream inspection;
- validation;
- version reporting;
- diagnostics.

The CLI is both a user tool and an integration test for the runtime.

---

# 23. FFmpeg and Media-Framework Integration

FFmpeg integration is important, but it should come after the core runtime proves generality.

FFmpeg should not become the architecture.

NVCR should remain independently usable.

A future FFmpeg integration should mainly provide:

- AVCodec wrappers;
- pixel-format conversion;
- timestamps;
- packet mapping;
- mux/demux integration;
- standard application interoperability.

NVCR should remain the codec runtime underneath.

Conceptually:

```text
FFmpeg
   │
   ▼
NVCR adapter layer
   │
   ▼
NVCR Runtime
   │
   ├── Codec
   ├── Provider
   └── NVAU
```

This keeps NVCR useful outside FFmpeg as well.

---

# 24. Streaming Direction

Once runtime and format contracts are stable, NVCR becomes a foundation for neural-video streaming research and systems work.

Potential future work includes:

- segmentation;
- random access;
- adaptive quality;
- provider-aware adaptation;
- device-aware adaptation;
- compute-aware representation selection;
- latency-aware coding;
- edge/cloud split execution;
- neural codec negotiation;
- multi-codec adaptation;
- runtime resource sensing.

These belong downstream of a stable runtime and access-unit contract.

Streaming should not drive premature changes to the core codec/provider boundary.

---

# 25. Resource Awareness

The runtime is a natural place to expose resource information without embedding adaptation policy into every codec.

Potential runtime-observable signals include:

- memory pressure;
- provider availability;
- accelerator utilization;
- execution latency;
- queue depth;
- energy measurements where reliable;
- thermal state;
- hardware capability;
- artifact availability.

A future controller could use these signals to select:

- codec;
- provider;
- profile;
- precision;
- quality level;
- GOP;
- model variant;
- compute mode.

NVCR should expose the mechanisms needed for this.

It should avoid hardcoding one control strategy into the runtime itself.

---

# 26. What NVCR Should Not Become

NVCR should not become:

- a research training framework;
- a neural-network architecture library;
- a replacement for PyTorch;
- a replacement for ONNX;
- a replacement for TensorRT;
- a complete multimedia framework;
- a monolithic container specification;
- a DCVC-RT optimization fork;
- a benchmark-only repository;
- a collection of provider-specific special cases.

The project remains strongest when it focuses on the execution and systems gap between research codec implementations and deployable applications.

---

# 27. Architectural Invariants

The following invariants should be treated as non-negotiable.

## Invariant 1 — Codec/provider separation

Codec code must not directly depend on concrete execution-provider implementations.

## Invariant 2 — Provider-neutral streams

Normative stream data must not contain provider-local details.

## Invariant 3 — Bounded parsing

All externally supplied binary structures must remain bounded before allocation.

## Invariant 4 — Stateful lifecycle

The runtime must support delayed output, flushing, draining, reset, and reuse.

## Invariant 5 — Explicit compatibility

No portability claim without evidence.

## Invariant 6 — Artifact provenance

Runtime artifacts must remain versioned, target-aware, and attributable.

## Invariant 7 — Performance-aware abstraction

Generic interfaces must not force unnecessary device/host transfer or synchronization.

## Invariant 8 — Tests before claims

A new codec/provider/platform is not production-supported until its contract and target gates pass.

---

# 28. Development Strategy

The project should evolve through architectural proofs.

Not through feature accumulation.

The recommended sequence is:

```text
Phase A: baseline coherence [complete]
        │
        ▼
Phase B0: provider-boundary RFC [complete]
        │
        ▼
Phase B2: contracts and deterministic fixtures [next]
        │
        ▼
Execution-provider boundary implementation
        │
        ▼
Second provider
        │
        ▼
Cross-provider conformance
        │
        ▼
Second codec
        │
        ▼
Multi-codec / multi-provider matrix
        │
        ▼
NVAU v2 normative usage
        │
        ▼
API/ABI stabilization
        │
        ▼
Container / FFmpeg integration
        │
        ▼
Streaming and resource-aware systems
```

Each step should validate the previous abstraction.

---

# 29. Phase A — Baseline Coherence

Status: Complete in [PR #145](https://github.com/0elghati/nvcr/pull/145).

Before major architecture work:

- unify version identity;
- add reliable `--version` surfaces;
- remove stale pre-v1 wording where no longer accurate;
- separate software-release version from API/schema versions;
- update roadmap terminology;
- make release metadata internally consistent;
- preserve the current v1.x baseline.

Goal:

> A clean, internally coherent starting point for v2 work.

---

# 30. Phase B — Execution Boundary

Status: Design accepted in
[PR #145](https://github.com/0elghati/nvcr/pull/145); implementation is staged
as B2 through B7 in the [roadmap](../ROADMAP.md#active-v2-work).

Complete the generic provider model.

Key goals:

- no production-only `component_factory` escape hatch;
- provider-owned execution sessions;
- provider-owned accelerator memory;
- reusable executable components;
- explicit lifetime rules;
- explicit synchronization rules;
- tensor/buffer abstractions that do not force CPU staging;
- TensorRT migrated through the generic contract;
- no meaningful performance regression.

Goal:

> The direct TensorRT implementation becomes proof that the generic provider architecture can still be fast.

---

# 31. Phase C — Second Provider

Add ONNX Runtime CUDA or another suitably independent provider.

Validate:

- codec behavior;
- artifact resolution;
- tensor contracts;
- device memory;
- encode;
- decode;
- quality;
- rate;
- memory;
- throughput;
- error handling.

Then test cross-provider streams.

Goal:

> The same codec runs through multiple providers without changing codec semantics.

---

# 32. Phase D — Second Codec

Integrate a second real neural video codec.

Requirements:

- independent codec adapter;
- independent codec-private payload;
- lifecycle tests;
- artifact definitions;
- reference behavior;
- at least one production provider;
- NVAU mapping;
- reproducible evaluation.

Goal:

> The same runtime supports multiple neural video codec families.

---

# 33. Phase E — NVAU v2 Conformance

Once two codecs exist:

- make NVAU v2 the preferred writer;
- formalize codec/profile identity;
- formalize dependency semantics;
- formalize typed sections;
- create conformance vectors;
- build stream-inspection tools;
- support v1-to-v2 migration where reasonable.

Goal:

> NVAU is proven codec-neutral in practice.

---

# 34. Phase F — Stable Runtime API

After multiple codecs/providers exercise the contracts:

- freeze public C++ ownership rules;
- define semantic versioning for API/ABI;
- define compatibility guarantees;
- formalize plugin/registration strategy;
- strengthen external CMake consumer tests;
- define long-term deprecation rules.

Goal:

> Application developers can depend on NVCR as systems software.

---

# 35. Phase G — Media Ecosystem

Then add:

- FFmpeg integration;
- standard container mapping;
- NVIF or equivalent higher-level format work;
- indexing;
- seeking;
- muxing;
- media application examples.

Goal:

> Neural video codecs become usable through normal multimedia workflows.

---

# 36. Phase H — Adaptive and Resource-Aware Runtime

On top of the stable runtime:

- runtime resource telemetry;
- provider switching where valid;
- profile switching;
- resource-aware codec selection;
- energy-aware experiments;
- compute-quality tradeoff control;
- streaming adaptation.

Goal:

> NVCR becomes a platform for systems research beyond static neural video decoding.

---

# 37. Evidence Matrix

The project's claims should be tied to an explicit evidence matrix.

Example future state:

| Capability | Evidence |
|---|---|
| Stateful runtime | lifecycle contract tests |
| Safe stream parsing | malformed-input tests + fuzzing |
| TensorRT provider | target-local encode/decode validation |
| ORT-CUDA provider | target-local encode/decode validation |
| Multi-provider | same codec through two providers |
| Cross-provider stream | A→B and B→A decode tests |
| Multi-codec | two production codec adapters |
| NVAU codec neutrality | same NVAU contract used by multiple codecs |
| Platform portability | independent hardware validation |
| Performance | controlled benchmark matrices |
| Memory behavior | measured process/device memory |
| Energy claims | controlled energy methodology |
| Packaging | clean install from published artifacts |
| API stability | external consumer builds across releases |

---

# 38. Success Criteria for the Big Vision

NVCR has reached its big vision when an external developer can:

1. install NVCR;
2. discover available codecs;
3. discover available providers;
4. install compatible artifacts;
5. encode video with codec A/provider X;
6. decode it with codec A/provider Y when cross-provider compatibility is supported;
7. switch to codec B through the same runtime API;
8. inspect the resulting codec-neutral access units;
9. integrate NVCR into an application without linking against codec research code;
10. reproduce the claimed behavior from published artifacts and evidence.

At that point, NVCR is no longer only a successful native port.

It is a real runtime platform.

---

# 39. Research Position

NVCR's scientific value should come from systems questions rather than claims of inventing a new compression model.

Relevant research questions include:

- How should neural video codecs be abstracted for native execution?
- Which boundaries remain stable across codec families?
- How much runtime overhead does abstraction add?
- Can providers remain interchangeable without losing performance?
- Can neural codec access units remain provider-neutral?
- How portable are compiled neural codec artifacts?
- How should compatibility be specified for learned codecs?
- How should model, runtime, and bitstream versions interact?
- What resource signals should a neural codec runtime expose?
- How can neural codecs integrate into conventional media pipelines?

This creates a research direction distinct from neural network architecture design itself.

---

# 40. Product Position

NVCR should also remain useful as software independent of publications.

The runtime should aim to be:

- understandable;
- buildable;
- testable;
- installable;
- diagnosable;
- scriptable;
- embeddable;
- reproducible.

Research code often ends when a paper ends.

NVCR should be designed to continue.

---

# 41. Repository Structure Direction

As the project evolves, repository organization should reinforce ownership boundaries.

A conceptual organization is:

```text
include/nvcr/
    runtime/
    codec/
    provider/
    artifacts/
    stream/
    common/

src/
    runtime/
    stream/
    artifacts/

codecs/
    dcvcrt/
    mlvc/
    ...

providers/
    tensorrt/
    onnxruntime/
    ...

cli/
tests/
benchmarks/
docs/
scripts/
```

The exact paths may differ.

The principle matters more:

> Codec-specific and provider-specific implementation code should be visibly separate from the reusable runtime core.

---

# 42. Release Strategy

The current v1.x line should be treated as the stable published baseline.

Large architectural work should happen as deliberate v2 development rather than silently redefining v1 behavior.

A reasonable strategy is:

```text
v1.x
  stable DCVC-RT/TensorRT product line
  correctness fixes
  packaging fixes
  documentation fixes
  benchmark maintenance

v2.x
  completed provider abstraction
  multi-provider
  multi-codec
  NVAU v2 normative path
  stable next-generation API
```

This protects existing results while allowing meaningful architectural change.

---

# 43. Non-Goals for Early v2

Early v2 should not be blocked on:

- macOS;
- native Windows;
- every execution provider;
- INT8;
- FFmpeg;
- audio;
- subtitles;
- live streaming;
- standardized neural bitstreams;
- universal engine portability;
- arbitrary dynamic shapes.

The priority is architectural proof.

First prove:

```text
2 codecs × 2 providers
```

Then broaden the ecosystem.

---

# 44. Decision Rule for New Features

Before adding a feature, ask:

> Does this strengthen NVCR as a reusable runtime, or does it merely add another special case?

Prefer work that:

- removes hidden coupling;
- improves contracts;
- improves portability;
- improves reproducibility;
- improves application integration;
- proves an abstraction;
- generalizes existing functionality.

Be cautious with work that:

- only improves one codec;
- only improves one GPU;
- adds provider-specific API leakage;
- mixes container concerns into access units;
- duplicates provider logic in codec code.

---

# 45. The Strategic Milestone

The medium-term milestone is this:

```text
                    NVCR Runtime

            Codec A             Codec B
               │                   │
        ┌──────┴──────┐     ┌──────┴──────┐
        │             │     │             │
   Provider X     Provider Y ...      Provider Y
        │             │                   │
        └─────────────┴─────────┬─────────┘
                                │
                             NVAU v2
```

When this is real, NVCR's central claim is experimentally demonstrated.

Everything after that becomes a platform-building problem rather than an architectural uncertainty.

---

# 46. Final Vision

NVCR should become the systems boundary between neural compression research and real video applications.

A neural codec researcher should be able to bring:

- codec semantics;
- models;
- entropy logic.

An execution backend should be able to bring:

- hardware execution;
- provider artifacts;
- memory and scheduling.

NVCR should provide:

- lifecycle;
- contracts;
- discovery;
- artifacts;
- streams;
- portability;
- packaging;
- validation;
- application integration.

The long-term ambition can be summarized as:

> **Neural codecs should become deployable components, not isolated research repositories. NVCR should be the runtime that makes that possible.**
