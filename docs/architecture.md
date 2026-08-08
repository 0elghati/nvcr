# Architecture

NVCR keeps codec meaning separate from execution technology. The current
production path is DCVC-RT executed by the TensorRT backend.

```text
application / CLI
       |
   nvcr::Runtime
       |
 stream and access-unit checks
       |
   codec adapter  <---->  RuntimeServices
       |                         |
 DCVC-RT state             artifact resolver
                                 |
                         execution provider
                                 |
                         TensorRT / CUDA
```

## Ownership

| Part | Owns |
|---|---|
| Runtime | Session lifecycle, configuration, errors, reset, flush, and serialized calls |
| Stream/format | Bounded `NVAU` access units and development `NVCR`/`NVCS` wrappers |
| Codec adapter | GOP decisions, codec payload syntax, entropy meaning, model components, and reference state |
| Artifact resolver | Model/provider/component/profile identity, compatibility ranking, digest and license policy |
| Execution provider | Artifact loading, tensor execution, synchronization, and provider errors |
| TensorRT provider | TensorRT plans, CUDA bindings, execution contexts, and device-local work |
| CLI | Raw YUV420 I/O and development file handling |

Public headers do not expose CUDA, TensorRT, or DCVC-RT implementation types.
The static registry is intentional; v1 does not introduce a dynamic plugin ABI.

The diagram describes the provider-mediated production flow. The public
provider, registry, resolver, and `RuntimeServices` contracts are implemented
and covered with a deterministic test provider. The DCVC-RT adapter requests
provider-owned codec components through `RuntimeServices`; TensorRT is the only
production provider registered today. The finer-grained model-component
`IExecutionProvider::load` path is still transitional: TensorRT creates the
current monolithic codec backend instead of exposing each plan as an independent
executable component.

## Provider-mediated flow

1. The application creates a runtime with a codec adapter and configuration.
2. The adapter requests provider-owned codec components through `RuntimeServices`.
3. The selected provider creates the runtime components. It does not silently hand the request to another provider.
4. Future component-level paths may use the resolver to select candidates by codec, model set, component, engine profile, provider, target, precision, version, digest, availability, and license status.
5. Encoding and decoding update independent sequence state. An I-frame starts a GOP; P-frames require a reference.
6. Reset clears state. Flush completes pending work and returns the session to its initial state.

## Artifact boundary

A TensorRT bundle is target-local and must pass manifest and checksum validation before plans are deserialized. The catalog selects metadata; the Python validator remains responsible for byte-level archive and bundle validation.

## Current limits

The runtime API is usable but not ABI-frozen. Plane/stride ownership, a stable public session API, a C ABI, additional codecs, FFmpeg, and standard container mapping remain future work. See [Scope and support](scope-and-support.md) and [the accepted ADRs](adr/ADR-001-codec-provider-separation.md).
