# NVCR packet envelope

Return to the [docs index](README.md) or the [project overview](../README.md).

NVCR owns framing and metadata. The native DCVC-RT entropy coder owns only the
payload passed to and from its adapter.

All integers use little-endian byte order. Version 1 has this layout:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 bytes | ASCII `NVCR` |
| Version | `u16` | `1` |
| Frame type | `u8` | `0` intra, `1` predicted |
| Flags | `u8` | Must be zero |
| Timestamp | `i64` bits | Microseconds |
| Metadata count | `u16` | Number of key/value entries |
| Metadata entries | variable | `u16` key size, `u16` value size, then bytes |
| Payload size | `u64` | Entropy-coded payload length |
| Payload | variable | Native entropy coder output |

Keys are unique. The parser rejects invalid magic/version/type/flags, duplicates,
truncation, trailing bytes, integer overflow, and payloads above the configured
limit. Metadata ordering is deterministic because `PacketMetadata` is ordered.

For DCVC-RT I-frames the payload currently begins with an internal `NVI1` header
containing width, height, QP, and the two-coder flag, followed by the native rANS
stream. The outer `Packet` envelope remains separate from that codec-specific
payload.

The envelope is an initial NVCR contract, not a claim of compatibility with an
existing DCVC-RT research bitstream. Golden-vector tests must be added before the
format is declared stable.

## Sequence file

The native CLI stores zero or more serialized packet envelopes in a streaming
sequence file. All integers are little-endian:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 bytes | ASCII `NVCS` |
| Version | `u16` | `1` |
| Flags | `u16` | Must be zero |
| Packet size | `u64` | Size of the following serialized packet |
| Packet | variable | One complete `PacketIO` envelope |

The packet-size and packet fields repeat until EOF. This permits encode-only
streaming without knowing the final frame count and permits decoding later in a
separate process. Version 1 is an internal NVCR format, not an upstream DCVC-RT or
standard multimedia container.

## See also

- [Architecture](architecture.md)
- [Native command-line interface](cli.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
