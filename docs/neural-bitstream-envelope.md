# Unified neural codec bitstream envelope

NVCR is intended to become a unified runtime for neural video codecs. The
bitstream design follows that goal by standardizing the outer access-unit
contract while allowing each codec backend to keep the payload syntax required by
its model.

The design choice is deliberately close to the way mature media stacks separate
containers, samples, and codec elementary streams. A common file or stream format
can carry AVC, HEVC, VVC, AV1, or other codecs without making their internal
bitstreams identical. NVCR should provide the same kind of runtime-level
structure for neural codecs: consistent framing, validation, ordering,
identity, dependencies, and capability negotiation, with codec-specific learned
payloads preserved inside bounded sections.

This page documents the direction for the NVCR bitstream family. It is not a
claim that the current development `NVCR`/`NVCS` file framing is a stable
standard, and it is not a promise of byte-level compatibility with upstream
research implementations.

## Goals

- Give every integrated codec the same runtime-facing packet shape.
- Keep stream parsing, size validation, model identity, frame order, dependency
  tracking, and metadata handling in the NVCR core.
- Let codec backends describe their learned payloads without leaking model
  internals across the public runtime boundary.
- Make access units inspectable, bounded, and rejectable before backend
  execution.
- Provide a clean path to future FFmpeg, Matroska, MP4, and interchange-format
  mappings.
- Avoid designing a final universal neural video standard before multiple codec
  families have been integrated and tested.

## Non-goals

- NVCR does not require all neural codecs to use the same entropy syntax.
- NVCR does not require the same latent tensor layout, prior model, reference
  state, or scale-index representation across codecs.
- NVCR does not claim that a stream from a research repository can be decoded by
  NVCR unless that compatibility is explicitly implemented and tested.
- NVCR does not expose TensorRT, CUDA, PyTorch, ONNX graph details, or checkpoint
  internals as part of the stable bitstream contract.
- NVCR does not use application timestamps or container metadata as hidden codec
  state.

## Layering model

The intended layering is:

```text
Application, CLI, FFmpeg wrapper, or container
                  |
                  v
        NVCR packet/container mapping
                  |
                  v
      NVCR codec access unit envelope
                  |
                  v
        Codec-specific payload sections
                  |
                  v
      Codec backend and model artifacts
```

The access-unit envelope is the common NVCR contract. It should be stable enough
for parsers, muxers, conformance tools, and backend integration. The payload
sections inside it may remain codec-specific as long as they are versioned,
bounded, and identified.

This keeps the core runtime responsible for stream safety and cross-codec
ergonomics, while leaving compression efficiency and model architecture to the
codec backend.

## Design principle

NVCR standardizes what every decoder must know before it enters a codec backend:

- which codec family is being used;
- which model/profile identity is required;
- which frame is being decoded;
- which previous frames or states it depends on;
- whether the frame is independently decodable;
- how large the payload is;
- which optional capabilities or extensions are present; and
- whether the access unit is valid enough to hand to the backend.

NVCR does not standardize the private representation of learned latents unless a
future codec family proves that such a representation is useful across more than
one backend.

## Proposed envelope shape

A future multi-codec NVCR access unit should contain a fixed common header plus a
small table of typed sections. The exact binary syntax can evolve, but the
semantic contract should remain stable.

Required common fields:

| Field | Purpose |
|---|---|
| Magic and version | Identify the access-unit syntax and reject unknown major versions. |
| Header size and total size | Bound parsing before allocation. |
| Codec ID | Select the codec family, such as `dcvcrt` or a future backend. |
| Codec profile ID | Select the versioned codec/profile contract. |
| Model bundle ID | Bind the stream to the required model artifacts. |
| Frame type | Identify intra, predicted, refresh, or future frame classes. |
| Flags | Carry common decode semantics such as reset, random access, or discontinuity. |
| Visible dimensions | Preserve the displayed frame size independently from padded tensor shapes. |
| Format | Record bit depth, chroma format, and color representation required by the codec contract. |
| Order indexes | Represent decode order and presentation order without relying on container-only state. |
| Dependency references | Describe previous access units or state slots needed for decode. |
| Quantization or rate-control descriptor | Carry common QP/rate-control identity when applicable. |
| Section table | Locate bounded typed payload and side-data sections. |
| Integrity field | Detect malformed or corrupted access units before backend execution. |

The current `NVAU` version 1 already contains a narrow subset of this model for
the DCVC-RT backend: model identity, dimensions, frame type, effective QP, reset
state, and bounded payload length. The next design step is to generalize that
contract without breaking the current separation between access units and
application/container metadata.

## Sections

Each section should have a type, version, flags, byte length, and payload. The
section table lets NVCR add common metadata without forcing every codec payload
to be rewritten.

Recommended section classes:

| Section class | Purpose |
|---|---|
| Codec payload | Required backend-owned encoded data. |
| Codec configuration | Optional stream-local codec settings when they are not fully captured by the model/profile ID. |
| Dependency map | Optional detailed reference structure for codecs with more than one reference state. |
| Quality/rate side data | Optional QP, lambda, target-rate, or quality-layer data when meaningful. |
| Color and mastering metadata | Optional video metadata for container mapping and display correctness. |
| Encoder statistics | Optional non-normative diagnostics that decoders may ignore. |
| Private extension | Namespaced extension data for experimental backends. |

Unknown required sections must be rejected. Unknown optional sections may be
skipped after their bounds are validated. This is the main extensibility rule
that lets the format grow without making old decoders unsafe.

## Codec backend contract

Backends should not write arbitrary top-level byte streams directly. They should
return structured access-unit data to the NVCR core:

```text
Encode input frame
        |
        v
Codec backend produces:
  frame type
  dependency information
  codec/profile/model identity
  quantization descriptor
  bounded codec-private sections
        |
        v
NVCR core serializes the access unit
```

Decode follows the reverse path:

```text
Serialized access unit
        |
        v
NVCR core validates common syntax and dispatch identity
        |
        v
Codec backend validates and decodes its own sections
```

This makes the core runtime the enforcement point. A future backend can be
integrated only after it can express its packets through the common access-unit
contract and pass parser, round-trip, corruption, and conformance tests.

## Identity and capability policy

A neural codec stream is only meaningful when paired with the correct model
artifacts. The envelope should make this explicit.

Recommended identity levels:

| Identity | Meaning |
|---|---|
| Codec ID | The codec family implemented by a backend. |
| Codec profile ID | The codec syntax and behavior contract. |
| Model bundle ID | The exact model/artifact family required to decode the stream. |
| Tool/runtime metadata | Optional encoder information for diagnostics, not required decoder state. |

The decoder must treat codec/profile/model identity as normative. Tool names,
encoder build versions, benchmark settings, and local paths are not normative and
should not be required to decode a conformant stream.

## Random access and state

Neural codecs often carry learned reference state, feature memory, or recurrent
context. The envelope should describe the required decode dependencies without
exposing backend memory layouts.

At minimum, every access unit should state whether it:

- resets decode state;
- is independently decodable;
- depends on the immediately previous decoded frame;
- depends on named state slots or previous access-unit indexes; or
- starts a new closed group of pictures.

The exact reference tensor contents remain backend-private. The stream should
describe dependency semantics, not CUDA buffers or model-layer names.

## Container mapping

The access-unit envelope should be independent of any one container.

Application packets and containers may add:

- timestamps and durations;
- track IDs;
- edit lists;
- indexing and seeking structures;
- muxed audio/subtitle/data streams;
- file-level metadata; and
- container-level checksums.

Codec access units should not duplicate that container layer except where decode
order, dependencies, or random-access semantics are necessary for decoding. This
keeps the same NVCR access unit usable in raw streams, test fixtures, FFmpeg
packets, Matroska blocks, or a future MP4 sample-entry mapping.

## Versioning

The envelope should use explicit major/minor versioning:

- A major version change may alter required parsing semantics.
- A minor version change may add optional fields or sections.
- Reserved bits must be zero and rejected unless a later version defines them.
- Required extensions must be declared before their section payloads.
- Encoders should write canonical field order and minimal required metadata.

This policy is stricter than a research prototype and more conservative than a
general metadata container. The goal is predictable decoder behavior.

## Conformance expectations

A codec backend should not be considered integrated until the following pass:

- parser and writer round-trip tests for valid access units;
- malformed, truncated, oversized, unknown-required-section, and trailing-data
  rejection tests;
- encode/decode round trips through the runtime;
- reset, flush, drain, and multi-GOP tests;
- target model/artifact identity rejection tests;
- reference reconstruction tests for that codec/profile; and
- clear documentation of whether upstream byte/payload compatibility is
  supported, unsupported, or intentionally out of scope.

For the current DCVC-RT backend, NVCR validates reconstruction behavior against
the pinned Python implementation but does not claim upstream payload
interchangeability.

## Relationship to a future interchange format

The unified NVCR access-unit envelope is a practical runtime contract. A future
neural video interchange format can build on it after more codec families expose
which fields are truly common and which fields must remain codec-private.

That future format should not be assumed to be identical to NVCR's development
framing. NVCR's role is to make codec integration disciplined now: every backend
uses the same outer structure, every stream is bounded and inspectable, and every
compatibility claim is backed by evidence.

## Current documentation boundary

- [`docs/bitstream.md`](bitstream.md) documents the current `NVAU` version 1
  access unit and development `NVCR`/`NVCS` framing.
- [`docs/architecture.md`](architecture.md) documents the runtime/backend
  boundary.
- [`docs/scope-and-support.md`](scope-and-support.md) defines which behavior is
  supported by the current release track.
- [`docs/releasing.md`](releasing.md) describes the release and evidence gates.
