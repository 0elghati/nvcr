# Extending NVCR

NVCR is designed as a codec-extensible runtime architecture. An extension must
respect the ownership boundaries in [Architecture](architecture.md) and must
not turn a test fixture or transitional path into a production claim.

## Adding a codec adapter

Implement a codec adapter that provides:

1. `CodecDescriptor` with a stable machine-readable ID and API version.
2. `CodecCapabilities` for frame types, delay, QP/rate controls, and supported
   ordering behavior.
3. Namespaced encoder and decoder `OptionSchema` declarations.
4. `ICodecAdapter::create_components`, using `RuntimeServices` to obtain
   provider-owned components rather than naming a concrete provider type.
5. A `CodecBackend` that translates codec semantics into `CodecEncodeResult`
   and `CodecDecodeResult`, owns codec reference/latent state, and converts
   backend exceptions into `Result` errors.

The adapter owns GOP and frame-type meaning, reference-state transitions,
entropy semantics, codec-private payload syntax, and the compatibility position
of that payload. The runtime owns session lifecycle and common error behavior.

### Session obligations

The session contract permits lookahead and delayed output even though the
current DCVC-RT path emits one access unit per input frame. A codec may return
`try_again` when output is not ready. `flush()` signals end of input; callers
must drain `receive_access_unit()` or `receive_frame()` until
`end_of_stream`. `reset()` must discard all reference state and make the next
sequence start from frame index zero.

Document whether the codec reorders frames, delays output, requires a reference
for predicted frames, or has codec-specific reset/discontinuity rules.

### Access units and payloads

The adapter supplies codec identity, frame type, dimensions, model/profile
identity, dependencies, and bounded codec-private sections to the `NVAU`
contract. It must not put TensorRT plan names, CUDA device IDs, binding indexes,
or local filesystem paths into the normative stream identity. State whether
the payload is self-conformant only, reference-consistent, cross-provider
conformant, or byte/payload interchangeable with another implementation.

### Codec registration and tests

Register the codec in the static registry with an adapter factory. Required
tests include:

- descriptor, capabilities, option-schema, and registration discovery;
- I-frame and predicted-frame round trips where supported;
- delayed output, `try_again`, flush, drain, reset, and reuse;
- invalid state, malformed payload, bounds, and dependency rejection;
- `NVAU` writer/parser round trips and unknown-required-section handling; and
- reference comparison or golden vectors when compatibility is claimed.

## Adding an execution provider

Implement `ProviderDescriptor` and `ProviderCapabilities`, then implement
`IExecutionProvider`:

- `supports()` must reject artifacts outside the provider's identity and
  compatibility rules;
- `load()` must validate or rely on the artifact boundary as documented and
  return an `IExecutable` with a complete `ArtifactDescriptor`; and
- `IExecutable::execute()` must define tensor names, shapes, dtypes, ownership,
  synchronization, and error translation.

Use `IArtifactCompiler` only for an explicitly offline source-model to
provider-artifact workflow. Deployment should be able to consume prebuilt
artifacts without carrying the compiler toolchain.

The provider owns executable loading, tensor/device memory, synchronization,
execution contexts, and provider-specific failures. It does not own codec GOP
semantics, entropy meaning, or access-unit framing.

### Provider registration and artifact compatibility

Register the provider with a factory and, if needed, a provider-mediated
component factory. Artifact requests must identify codec, model set, component,
engine profile, provider, precision, target, runtime, API/schema versions,
digest expectations, and license policy. The resolver selects a candidate;
the provider remains responsible for rejecting an artifact it cannot execute.

Do not describe a same-compute or Ampere-plus artifact as universally portable.
Name exactly what was validated: host binary, CUDA code, engine bundle, catalog
entry, access unit, or container userspace.

### Provider tests

Required tests include:

- descriptor/capability discovery and codec/provider compatibility;
- tensor binding names, shapes, dtypes, ownership, and synchronization;
- missing, unavailable, restricted, digest-mismatched, and incompatible
  artifacts;
- provider error translation and invalid lifecycle states; and
- target-local engine loading and round trips for every production claim.

## Current TensorRT boundary

TensorRT is the only production provider. It currently creates a provider-owned
monolithic DCVC-RT backend through `component_factory`; its
`IExecutionProvider::load` component-level path returns `not_implemented`.
Independent loading of each neural model stage is transitional work. New
documentation must preserve that distinction.

## Build and contribution checklist

Keep codec-specific code under its adapter and provider-specific code under its
provider/backend. Add contract tests to `tests/`, update the relevant public
documentation and capability matrix, and record target/artifact evidence before
calling an extension production-supported. Do not commit checkpoints, ONNX
exports, TensorRT plans, datasets, or private credentials.
