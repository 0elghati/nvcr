# Bitstream and access-unit contracts

The current NVCR stream contract separates the codec access unit from
application/container metadata.
All integers below are little-endian.

`NVAU` is NVCR's bounded codec access-unit contract. Version 1 is the narrow
DCVC-RT-shaped production/default representation; version 2 is the implemented
generalized sectioned representation. The learned payload remains codec-private
and is not an upstream DCVC-RT interchange format. The design direction is documented in
[Neural codec access-unit envelope direction](neural-bitstream-envelope.md).
This page documents the implemented `NVAU` v1 syntax and the generalized
sectioned `NVAU` v2 envelope.

## Codec access unit: `NVAU` version 1

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 | ASCII `NVAU` |
| Version | `u16` | `1` |
| Frame type | `u8` | `0` intra, `1` predicted |
| Flags | `u8` | bit 0 = reset state; other bits rejected |
| Model ID size | `u16` | 1–128 bytes |
| Reserved | `u16` | zero |
| Width | `u32` | even visible width |
| Height | `u32` | even visible height |
| Effective QP | `u32` | I: 0–63; P: 0–71 |
| Payload size | `u64` | bounded codec payload length |
| Model ID | variable | portable ASCII profile ID |
| Payload | variable | DCVC-RT codec payload |

The parser bounds the complete access unit before allocation, limits dimensions
to 16,384 per axis and 256 Mi pixels, rejects odd YUV420 dimensions, validates
model-ID characters, and rejects unknown flags, invalid QP/frame/reset
combinations, zero/truncated/oversized payloads, and trailing data.

Every encoded access unit carries the configured public bitstream model ID. The
DCVC-RT integration defaults this value to `dcvcrt`; internal research,
checkpoint, and engine-profile identifiers are not written into `NVAU`. Runtime
decode compares the access-unit model ID with the configured bitstream model ID
before backend execution. An I-frame marks a state reset. Timestamps, time bases,
packet side data, and container metadata are not part of `NVAU`.

The TensorRT backend currently encapsulates entropy data in isolated internal
`NVI1` and `NVP1` payload layouts. They are NVCR implementation formats, not
upstream DCVC-RT frame records. A pinned framing-only bidirectional probe decoded
structurally but failed reconstruction quality from the first I frame, so v1
makes no upstream byte or payload-interchangeability promise. `NVAU` versioning
permits replacing these inner layouts without pretending they are standardized
DCVC-RT streams.

## Sectioned codec access unit: `NVAU` version 2

`NVAU` version 2 is an implemented NVCR-side sectioned envelope that can be
mapped to future NVIF or standard-container work. It keeps codec samples independent of file/container concerns while
adding codec identity, ordering, dependency, and typed-section metadata. The
current runtime can parse v1 and v2 access units. `AccessUnitIO::serialize` still
writes v1 for compatibility; `AccessUnitIO::serialize_sectioned` writes v2.

Fixed header, 64 bytes:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 | ASCII `NVAU` |
| Version | `u16` | `2` |
| Header size | `u16` | `64` |
| Flags | `u32` | bit 0 = reset state, bit 1 = random access, bit 2 = discontinuity |
| Frame type | `u8` | `0` intra, `1` predicted |
| Pixel format | `u8` | `1` YUV420P8 |
| Bit depth | `u8` | `8` |
| Reserved | `u8` | zero |
| Width | `u32` | even visible width |
| Height | `u32` | even visible height |
| Effective QP | `u32` | I: 0-63; P: 0-71 |
| Decode order index | `u64` | access-unit decode order within the elementary stream |
| Presentation order index | `u64` | presentation order within the elementary stream |
| Dependency count | `u16` | number of following `u64` dependency indexes, max 32 |
| Section count | `u16` | number of section table entries, max 64 |
| Codec ID size | `u16` | 1-128 bytes |
| Codec profile ID size | `u16` | 1-128 bytes |
| Model ID size | `u16` | 1-128 bytes |
| Reserved | `u16` | zero |
| Total size | `u64` | complete `NVAU` byte size |

The fixed header is followed by codec ID, codec profile ID, model ID, dependency
indexes, section table entries, and then section payloads in table order. IDs use
the same portable ASCII character set as v1 model IDs. These are stream-facing
identities: they describe the codec family and public decoder contract, not the
private checkpoint or internal build label used to create TensorRT artifacts.

Each section table entry is 16 bytes:

| Field | Size | Meaning |
|---|---:|---|
| Section type | `u16` | typed payload class |
| Section version | `u16` | section-local syntax version |
| Section flags | `u32` | bit 0 = required section |
| Section size | `u64` | following section payload bytes |

Known section types:

| Type | Meaning |
|---:|---|
| `1` | codec payload, exactly one required section |
| `2` | codec configuration |
| `3` | dependency map |
| `4` | quality/rate side data |
| `5` | color metadata |
| `6` | encoder statistics |
| `0x8000` | private extension |

Unknown required sections are rejected. Unknown optional sections are bounded and
preserved by the parser so tools can inspect or forward them without executing a
codec backend. v2 still does not define tracks, time bases, file indexes, edit
lists, audio/subtitle streams, or mux metadata; those belong to a future NVIF or
standard container mapping.

## Development packet envelope: `NVCR` version 1

The current CLI still wraps access units in a generic packet envelope:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 | ASCII `NVCR` |
| Version | `u16` | `1` |
| Frame type | `u8` | intra or predicted |
| Flags | `u8` | zero |
| Timestamp | `i64` bits | application timestamp |
| Metadata count | `u16` | bounded key/value count |
| Metadata entries | variable | `u16` key/value lengths and bytes |
| Payload size | `u64` | bounded `NVAU` size |
| Payload | variable | one complete codec access unit |

Metadata is limited to 1,024 entries and 1 MiB aggregate encoded metadata. The
parser rejects invalid magic/version/type/flags, duplicate keys, aggregate or
individual bounds, truncation, trailing bytes, and total/payload length overflow
before large allocation.

## Development sequence file: `NVCS` version 1

The CLI writes a streaming sequence header followed by repeated bounded `NVCR`
records:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 | ASCII `NVCS` |
| Version | `u16` | `1` |
| Flags | `u16` | zero |
| Packet size | `u64` | following packet length |
| Packet | variable | one `NVCR` packet |

`NVCR`/`NVCS` are application/development formats. They are not the v1 codec
access-unit promise, an upstream DCVC-RT container, MP4, Matroska, or FFmpeg
integration, and no pre-v1 backward-compatibility promise is made. Temporary
legacy raw `NVI1`/`NVP1` decode can be enabled per session only for isolated
development migration.

## Compatibility position

NVCR does not claim upstream DCVC-RT byte or payload interchangeability.
That claim requires clean bidirectional Python/native golden vectors. Additional
codec adapters will need their own access-unit compatibility position and tests.
See [Compatibility](compatibility.md).
