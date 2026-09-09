# RFC: NVCR v2 codec/provider execution boundary

Status: Proposed
Date: 2026-09-09
Scope: Architecture and migration design only

## Decision summary

Adopt a codec-owned orchestration layer over a provider-owned execution session.
The codec layer will own DCVC-RT sequencing, quantization decisions, entropy
coding, payload syntax, and reference semantics. The provider session will own
TensorRT engines and contexts, device allocation, streams and events, shape
binding, graph capture, and model-stage execution.

This is a boundary extraction, not a codec rewrite. The first implementation
must keep the DCVC-RT/TensorRT result, access-unit bytes, lifecycle behavior,
and measured performance within the gates below. No second provider or codec is
required to justify the boundary.

## Scope and invariants

This RFC covers the v2 boundary between a codec adapter and an execution
provider. It does not change NVAU, NVI1, NVP1, rANS syntax, model identity,
engine manifests, supported platforms, or the v1.x public release status.

The following invariants apply throughout migration:

- Existing Runtime encode, decode, flush, and reset behavior remains observable.
- TensorRT plans remain target-local and are validated before deserialization.
- Intermediate tensors and reference features remain device-resident.
- The hot path does not add a host copy or a device-wide synchronization.
- Provider failures remain structured NVCR errors with useful subsystem context.
- CPU fixtures remain contract tests, not a production neural provider.

## Current production path

The production CLI does not use the generic per-component load path. The
source-traced path is:

    cli/main.cpp:create_runtime
      -> nvcr::dcvcrt::make_adapter
      -> DcvcrtAdapter::create_components
      -> RuntimeServices::create_components
      -> ProviderEntry::component_factory
      -> make_tensorrt_components
      -> make_tensorrt_backend
      -> TensorRTBackend as codec::Components.codec

The relevant definitions are:

- cli/main.cpp create_runtime directly constructs the DCVC-RT adapter.
- src/dcvcrt/adapter.cpp DcvcrtAdapter::create_components delegates to
  RuntimeServices::create_components.
- src/runtime/registry.cpp RuntimeServices::create_components selects a provider
  by provider_id and calls its component_factory.
- src/dcvcrt/backend/tensorrt/backend.cpp make_tensorrt_components installs one
  TensorRTBackend object as the codec backend.
- src/dcvcrt/registration.cpp register_codec registers the descriptor without an
  adapter factory, so Registry::create_codec cannot create the production
  DCVC-RT adapter.
- src/runtime/runtime.cpp registers DCVC-RT directly from generic runtime code.

RuntimeServices::resolve, IExecutionProvider::load, IExecutable,
TensorBindings, and ExecutionOptions form a separate path. Live production code
does not use that path. TensorRTProvider::load returns not_implemented.
IArtifactCompiler is declared but has no implementation or caller. Current
tests exercise the generic path through deterministic fixtures, which proves
the test contract but not a production provider split.

The C++ artifact resolver is likewise not the production CLI's bundle-selection
path. The CLI resolves an engine directory, nvcr-artifacts manages the catalog,
and TensorRTBackend validates the selected bundle before loading plans.

## Ownership found in TensorRTBackend

TensorRTBackend currently combines two kinds of responsibility.

Codec-specific responsibility:

- I/P model-stage ordering and DCVC-RT tensor names.
- Effective P-frame QP selection and quantization semantics.
- GOP and reference-frame/reference-feature selection.
- rANS encode/decode and private payload construction/parsing.
- Device-DPB updates tied to DCVC-RT sequence state.

Provider-specific responsibility:

- TensorRT runtime, engine, execution-context, and profile ownership.
- CUDA stream, event, pinned-buffer, and asynchronous-copy management.
- Stream-ordered allocation, scratch arena, and device tensor lifetime.
- Dynamic shape and tensor-address binding.
- Low-memory, persistent, and shared-workspace context policies.
- CUDA graph capture and bounded signature caching.
- Engine-manifest and runtime/device compatibility validation.

The device DPB crosses both concerns. Its storage and synchronization are
provider concerns; the meaning of a reference frame, reference feature,
generation, and frame index is a codec concern.

## Gaps in the existing generic API

include/nvcr/provider/provider_api.hpp is provider-neutral at the type-name
level, but it cannot describe the current production path precisely:

- TensorBinding exposes a host byte span while its comment says the provider
  owns tensor memory. Memory domain, allocation owner, and lifetime are absent.
- ExecutionOptions has only synchronize_after. It cannot express stream/event
  dependencies or asynchronous completion.
- ArtifactDescriptor describes one component, while the production bundle
  contains fourteen cooperating engines plus entropy and quantization assets.
- RuntimeServices::create_components has no codec identifier. A provider
  component factory must already know which codec it is constructing.
- ProviderKind and ProviderCapabilities contain TensorRT/CUDA-shaped vocabulary.
- RuntimeConfiguration contains TensorRT execution policy, engine paths, and a
  device arena alongside codec QP/GOP fields.
- Registry::compatible checks registered identifiers, not artifact, runtime,
  device, or tensor-contract compatibility.
- Generic runtime and codec runtime sources directly name DCVC-RT.

Adding another enum value or implementing TensorRTProvider::load against the
current TensorBinding would preserve these ambiguities. It would not produce a
sound production boundary.

## Design A: codec orchestrator plus provider session

This design introduces a provider session with opaque device resources and
makes the DCVC-RT adapter the orchestration owner.

The codec side owns:

- model-stage graph and stage-specific tensor contracts;
- GOP, frame index, reset, flush, and reference selection;
- codec quantization and entropy parameters;
- rANS and private payload syntax;
- mapping between codec state and opaque provider resources.

The provider side owns:

- validated provider artifacts and loaded executable stages;
- device buffers, memory pools, execution contexts, streams, and events;
- shape validation, binding, enqueue, graph capture, and completion;
- provider-specific profiling and error translation.

The minimum contract concepts are:

- MemoryDomain: host, pinned_host, or provider_device.
- BufferHandle: opaque storage with explicit size, domain, owner, and lifetime.
- TensorView: name, data type, shape, access mode, and a buffer slice.
- ExecutionDependency: prior provider event(s), never a raw CUDA type.
- Completion: an opaque event that can be waited on or chained.
- ExecutableStage: a loaded model component with an immutable tensor contract.
- ProviderSession: loads stages, allocates resources, and submits execution.

A codec session may retain BufferHandle values for its reference state. The
handle keeps storage alive, while the codec assigns semantic meaning to it.
Reset releases those references after their last Completion. ProviderSession
destruction waits only for work owned by that session.

Artifact validation remains layered. The catalog chooses a candidate; the
provider validates provider/runtime/device compatibility and plan integrity;
the codec validates that the complete set of named stages and non-plan assets
matches its model contract.

Advantages:

- Matches the ownership already required by the production hot path.
- Allows asynchronous, device-resident execution without exposing CUDA types.
- Can be extracted behind the current TensorRTBackend facade in small PRs.
- Keeps payload and entropy behavior outside a generic execution provider.

Costs:

- Requires explicit buffer/event lifetime rules.
- Requires the DCVC-RT orchestration now embedded in TensorRTBackend to move.
- Leaves stage ordering codec-specific, so providers do not optimize a whole
  codec graph unless a later contract adds that capability.

## Design B: provider-executed codec graph

This design gives the provider a declarative graph containing model stages,
CUDA-like transforms, entropy steps, and persistent state slots. The codec
adapter builds the graph and submits frames; the provider schedules the graph.

Advantages:

- Gives a provider maximum scheduling and graph-capture visibility.
- Can express whole-frame fusion and provider-specific graph optimization.
- Reduces codec-side submission calls.

Costs:

- A generic graph language must model DCVC-RT transforms, entropy boundaries,
  mutable DPB state, and conditional I/P execution.
- The graph contract risks becoming a second runtime and a TensorRT-shaped API.
- Migration cannot be staged cleanly without duplicating the current monolith.
- Correctness and performance review surface is much larger.

This design is viable only if whole-codec graph optimization becomes a measured
requirement. No current source or retained result establishes that need.

## Recommendation

Choose Design A. It is the smallest boundary that represents the production
requirements without hiding codec behavior in a provider factory. Design B is
not justified for the first v2 extraction.

Do not expose the draft concepts as frozen public C++ API in the first PR.
Introduce them under an internal or experimental namespace, write contract
tests, and promote only the subset exercised by the TensorRT production path.

The existing codec::Components and TensorRTBackend facade should remain in
place until the new path passes the same correctness and performance gates.
That permits one call site to switch at a time and avoids a flag-day refactor.

## Migration plan

### PR 1: contract vocabulary and fixtures

- Add provider-neutral memory, tensor-view, dependency, completion, executable,
  and session contracts.
- Extend the deterministic provider to test ownership, bounds, dependency
  chaining, reset, and error behavior.
- Keep production construction and bitstreams unchanged.

### PR 2: TensorRT execution session behind the facade

- Extract engine/context loading, shape binding, allocation, stream/event,
  graph-cache, and enqueue code into a TensorRT provider session.
- Keep TensorRTBackend as the only caller.
- Preserve context-policy selection and bundle validation.

### PR 3: I-frame orchestration

- Move I-frame stage ordering, quantization semantics, rANS, and payload
  assembly into the DCVC-RT codec side.
- Compare deterministic access-unit bytes and reconstructed output with the
  pre-extraction baseline.

### PR 4: P-frame state and orchestration

- Move P-frame stage ordering and reference semantics to the codec side.
- Keep reference storage opaque and device-resident through BufferHandle.
- Exercise GOP, reset, flush, repeated-session, and malformed-input cases.

### PR 5: construction cleanup

- Register a real DCVC-RT adapter factory.
- Select both codec_id and provider_id when creating a session.
- Remove direct DCVC-RT registration from generic runtime code.
- Retire component_factory and the unused production stub only after all
  production construction uses the session contract.

Each PR must be independently buildable and revertible. Compatibility shims
should be deleted when their last production caller is gone.

## Correctness gates

Every migration PR must pass:

- clean CPU Release configure, build, install, and full CTest;
- TensorRT Release build and non-GPU tests;
- registered TensorRT engine-contract and I/P round-trip tests on each retained
  supported profile and target;
- byte-for-byte NVAU equality for fixed deterministic fixtures unless a stream
  change is separately approved and versioned;
- identical decoded frame dimensions, timestamps, frame types, and lifecycle
  behavior;
- matching native/Python evidence only for the compatibility claims currently
  documented;
- sanitizer tests for provider-neutral ownership and malformed inputs.

No unit-only or CPU-only result is release evidence for neural inference.

## Performance and memory gates

Use the controlled procedure in docs/performance.md: Release builds, identical
source/input/engine bundle, one warm-up, three measured runs, clean throughput
separate from profiling, and retained raw per-run values.

Before PR 2, capture and approve a pre-refactor baseline. Proposed acceptance
limits are:

- no supported profile loses more than 3% median encode or decode FPS;
- pooled equal-work FPS loses no more than 2%;
- no new per-stage device synchronization or host transfer appears;
- peak device memory increases by no more than the larger of 5% or 64 MiB;
- context-policy and CUDA-graph hit/miss behavior remain explainable.

These numeric limits are proposed policy, not an existing measured guarantee.
Ratify or replace them before production extraction begins. A result outside a
limit blocks that PR until explained and explicitly accepted.

## Risks and open decisions

- BufferHandle destruction must not free storage before dependent work finishes.
- Error paths must not turn asynchronous provider failures into silent success.
- Stage tensor schemas need a versioning rule before they become artifacts.
- Profiling ownership must separate codec-stage names from provider timings.
- Shared-workspace mode on integrated devices needs explicit aliasing rules.
- The boundary between provider bundle validation and codec bundle-completeness
  validation must be represented once, not duplicated.
- Public promotion of any v2 contract requires an API/ABI compatibility policy.

## Non-goals

This RFC does not add ONNX Runtime, a second codec, CPU neural inference, INT8,
FFmpeg, standard containers, a new stream format, a model export pipeline, new
TensorRT engines, or benchmark results. It does not promise cross-provider
numerical identity or upstream DCVC-RT payload interchangeability.
