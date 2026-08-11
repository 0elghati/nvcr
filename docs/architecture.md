# Architecture

NVCR is a native runtime architecture for neural video codecs. Its purpose is
to make stateful learned codecs usable as systems software while keeping codec
semantics, execution technology, artifact provenance, and stream framing at
separate ownership boundaries.

```text
                         Runtime plane
 application / CLI -> Runtime -> session -> codec adapter
                                      |           |
                                      |     RuntimeServices
                                      |           |
                         registry <---+-----------+
                                      |
                         execution provider

                         Artifact plane
 model profile -> provider/engine profile -> target profile
                                      |
                          catalog -> resolver -> hashes/license

                         Stream plane
 application mapping -> packet -> bounded NVAU -> codec-private payload
```

## Runtime plane

The runtime plane contains the application-facing `Frame` and `Packet` values,
`Runtime`, encoder/decoder session contracts, codec adapters, the static
registry, and `RuntimeServices`.

| Boundary | Responsibility |
|---|---|
| `Runtime` and sessions | Stateful send/receive behavior, flush, drain, reset, statistics, and serialized access to mutable codec state |
| Codec adapter | Codec descriptor/capabilities, option schemas, frame-type/GOP semantics, codec state, entropy meaning, and codec-private payloads |
| Registry | Static discovery and registration of codecs and providers; no dynamic plugin ABI in the current release track |
| `RuntimeServices` | Provider-mediated component creation and resolver-selected executable lookup |
| Execution provider | Artifact loading, tensor bindings, synchronization, device execution, and provider-specific errors |

The public provider headers do not expose CUDA, TensorRT, or DCVC-RT types.
`ICodecAdapter` and `IExecutionProvider` are implemented contracts; production
component creation is currently provider-mediated through
`ProviderEntry::component_factory`.

## Artifact plane

The artifact plane identifies and qualifies executable assets:

```text
codec ID + model set + provider + component + engine profile + target
precision + API/manifest versions + runtime + digest + license status
                                      |
                             catalog candidates
                                      |
                             resolver ranking
```

`ArtifactDescriptor` describes a resolved executable. `Catalog` reads the
versioned engine catalog. `Resolver` rejects mismatched codec/model/provider,
precision, API or manifest versions, architecture, target, runtime, digest,
availability, and license policy, then ranks exact, same-compute, and explicit
Ampere-plus candidates. A target profile is an input to this process, not proof
of support.

## Stream plane

The stream plane separates codec access units from application or container
metadata:

```text
application/container mapping -> Packet -> NVAU elementary stream unit
                                               |
                                  codec-private payload sections
```

`NVAU` v1 is the narrow DCVC-RT-shaped production/default representation.
`NVAU` v2 is an implemented generalized sectioned representation. Both are
bounded and versioned; malformed input is rejected before backend execution.
The learned payload remains codec-private. `NVCR` and `NVCS` are development
or application wrappers, not standard multimedia containers. Container tracks,
timestamps, indexes, audio, and subtitles remain outside the current stream
contract.

## Implemented versus future boundaries

| Class | Current meaning |
|---|---|
| Architecture contract | Runtime sessions, codec/provider interfaces, registry/services, artifact identity/resolution, and bounded access units |
| Conformance implementation | Deterministic test codec and CPU provider used by tests for delayed output, provider calls, tensor binding, errors, reset, and malformed input |
| Production implementation | DCVC-RT adapter, native rANS, and TensorRT FP16 provider-owned backend on validated Linux/NVIDIA targets |
| Transitional/future | Independent component-level TensorRT `IExecutable` loading, additional production codecs/providers, INT8 release profile, FFmpeg, standard containers, stable C ABI, and ABI freeze |

## Current production flow

1. The CLI or C++ application selects DCVC-RT and a provider configuration.
2. The adapter requests provider-owned runtime components through
   `RuntimeServices`.
3. TensorRT validates and owns the target-local engine bundle, CUDA work, and
   codec backend state.
4. The runtime drives I/P sequence state and emits bounded `NVAU` access units.
5. The decoder validates stream identity and payload bounds before updating its
   independent reference state.

The current TensorRT production path creates one provider-owned monolithic
DCVC-RT backend. It does not independently load every neural model stage
through the lower-level `IExecutionProvider::load` interface; that split is
transitional work and is not a current production claim.

## Stability

The public API and ABI are not frozen. Pin the NVCR revision, codec/provider
identities, stream format, model set, and artifact manifest when integrating.
See [scope and support](scope-and-support.md) for the support evidence rule.
