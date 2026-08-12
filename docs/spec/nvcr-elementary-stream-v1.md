# NVCR Elementary Stream Specification (v1/v2)

Status: implemented specification; covered by format and parser tests
Scope: documents the existing `NVAU` v1 and `NVAU` v2 wire formats without changing bytes.

The filename is retained for link compatibility. It originated when v1 was the
only format; this specification now documents both implemented versions.

## 1. Byte order and primitive widths

- Endianness: little-endian for all multi-byte integer fields.
- Unsigned integer widths: u8, u16, u32, u64.
- Signed integers are not used in on-wire fixed headers.

## 2. Access-unit magic and version

- Magic bytes are ASCII NVAU: 0x4E 0x56 0x41 0x55.
- Immediately after magic, a u16 version selects parsing mode:
  - v1: 0x0001
  - v2: 0x0002

Unknown versions must be rejected with malformed-stream behavior.

## 3. NVAU v1 layout (current narrow/default)

Fixed header fields after magic/version:

1. frame_type (u8)
2. flags (u8), bit0 reset_state
3. model_id_len (u16)
4. reserved (u16) must be 0
5. width (u32)
6. height (u32)
7. qp (u32)
8. payload_len (u64)
9. model_id bytes (model_id_len)
10. payload bytes (payload_len)

Constraints:

- model_id_len and payload_len must fit in configured maximum access-unit size.
- reserved must be zero.
- Truncated buffers at any point are malformed.

## 4. NVAU v2 layout (sectioned)

Fixed header fields after magic/version:

1. fixed_header_bytes (u16)
2. flags (u32)
3. frame_type (u8)
4. pixel_format (u8)
5. bit_depth (u8)
6. reserved0 (u8)
7. width (u32)
8. height (u32)
9. qp (u32)
10. decode_order_index (u64)
11. presentation_order_index (u64)
12. dependency_count (u16)
13. section_count (u16)
14. codec_id_len (u16)
15. codec_profile_id_len (u16)
16. model_id_len (u16)
17. reserved1 (u16)
18. total_size (u64)
19. codec_id bytes
20. codec_profile_id bytes
21. model_id bytes
22. dependencies[dependency_count] as u64
23. section table entries:
   - type (u16)
   - version (u16)
   - flags (u32)
   - payload_size (u64)
24. section payload bytes concatenated in table order

Constraints:

- total_size must equal complete serialized byte count.
- reserved fields must be zero.
- Required section semantics are defined by section flags and must be enforced by parser.
- Length sums must use overflow-safe arithmetic.

## 5. Unknown extensions and section handling

- Unknown AU version: reject.
- Unknown section type in v2:
  - if section flag indicates required, reject.
  - otherwise ignore payload for semantic interpretation but preserve parse integrity.

## 6. Checksum behavior

Current v1/v2 headers do not include a mandatory per-AU checksum field. Integrity checks are provided externally by artifact/manifest workflows and test harnesses in current NVCR.

## 7. Truncation and malformed rejection

Parser must reject:

- invalid magic
- unsupported version
- truncated fixed or variable regions
- overflow in size arithmetic
- total size mismatch (v2)
- non-zero reserved fields where required

## 8. Stream lifecycle notes

An elementary stream is a sequence of complete NVAU records. Reset/discontinuity semantics are signaled by flags and consumed by codec/runtime session logic; stream framing itself remains codec-neutral.

## 9. Hex examples

NVAU v1 prefix example (magic + version + first two fields):

- 4E 56 41 55 01 00 00 01 ...

NVAU v2 prefix example (magic + version):

- 4E 56 41 55 02 00 ...

These prefixes are validated by format contract tests in tests/format_contract_tests.cpp.
