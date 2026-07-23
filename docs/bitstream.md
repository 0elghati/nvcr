# Bitstream and access-unit contracts

NVCR v1 separates the codec access unit from application/container metadata.
All integers below are little-endian.

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

Every encoded access unit carries `dcvcrt-cvpr2025`; runtime decode compares it
with the configured session model before backend execution. An I-frame marks a
state reset. Timestamps, time bases, packet side data, and container metadata are
not part of `NVAU`.

The TensorRT backend currently encapsulates entropy data in isolated internal
`NVI1` and `NVP1` payload layouts. Those are implementation formats until
cross-runtime Python golden tests define any upstream compatibility position.
`NVAU` versioning permits replacing the inner development layouts without
pretending they are standardized DCVC-RT streams.

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

`NVCR`/`NVCS` are application development formats. They are not the v1 codec
access-unit promise, an upstream DCVC-RT container, MP4, Matroska, or FFmpeg
integration, and no pre-v1 backward-compatibility promise is made. Temporary
legacy raw `NVI1`/`NVP1` decode can be enabled per session only for isolated
development migration.

## Compatibility evidence

No claim of upstream DCVC-RT byte or reconstruction compatibility is made until
native payload decoding and reconstructed frames pass golden tests against the
pinned Python implementation. See [Compatibility](compatibility.md) and the
M2 gates in [ROADMAP.md](../ROADMAP.md).
