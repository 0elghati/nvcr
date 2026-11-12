#!/usr/bin/env python3
"""Writes truncated and corrupted variants of a DCVC-UF .nvcr file (C13).

Variants keep NVCR's framing consistent where the point is to reach the UF
entropy decoder (record, packet and NVAU v1 lengths are rewritten), and break
it where the point is the container parser. Each variant is listed in
variants.json with what was changed; decode them with the sanitizer build of
nvcr_dcvcuf_entropy_trace or the CLI.
"""

import argparse
import json
import random
import struct
from pathlib import Path

HEADER = b"NVCS\x01\x00\x00\x00"


def split_records(data):
    if data[:8] != HEADER:
        raise ValueError("unsupported NVCR sequence header")
    position, records = 8, []
    while position < len(data):
        (length,) = struct.unpack_from("<Q", data, position)
        records.append(bytearray(data[position + 8:position + 8 + length]))
        position += 8 + length
    return records


def join(records):
    out = bytearray(HEADER)
    for record in records:
        out += struct.pack("<Q", len(record)) + record
    return bytes(out)


def locate(record):
    """Offsets inside a packet record: NVAU start, payload start/size, UF stream start."""
    if record[:4] != b"NVCR":
        raise ValueError("missing packet magic")
    count = struct.unpack_from("<H", record, 16)[0]
    position = 18
    for _ in range(count):
        key, value = struct.unpack_from("<HH", record, position)
        position += 4 + key + value
    au_size_at = position
    au = position + 8
    if record[au:au + 4] != b"NVAU" or struct.unpack_from("<H", record, au + 4)[0] != 1:
        raise ValueError("expected NVAU v1")
    ids = struct.unpack_from("<H", record, au + 8)[0]
    payload_size_at = au + 24
    payload = au + 32 + ids
    frames = record[payload + 4]
    uf = payload + 8 + 8 * frames
    return {"au_size_at": au_size_at, "payload_size_at": payload_size_at,
            "payload": payload, "uf": uf}


def shorten_uf(record, cut):
    """Removes `cut` bytes from the end of the UF stream, fixing every length."""
    where = locate(record)
    del record[len(record) - cut:]
    for at in (where["au_size_at"], where["payload_size_at"]):
        (value,) = struct.unpack_from("<Q", record, at)
        struct.pack_into("<Q", record, at, value - cut)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=216)
    args = parser.parse_args()
    data = args.stream.read_bytes()
    records = split_records(data)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = []

    def emit(name, blob, description):
        path = args.output_dir / f"{name}.nvcr"
        path.write_bytes(blob)
        variants.append({"name": name, "file": path.name, "description": description,
                         "bytes": len(blob)})

    units = {"intra": 0, "group1": 1, "group6": min(6, len(records) - 1)}
    # Container-level truncation.
    first_end = 8 + 8 + len(records[0])
    for name, size in (("file_header_only", 8), ("file_record_length_cut", 12),
                       ("file_unit0_half", 16 + len(records[0]) // 2),
                       ("file_unit1_cut", first_end + 8 + len(records[1]) // 3),
                       ("file_last_byte_removed", len(data) - 1)):
        emit(name, data[:size], f"file truncated to {size} of {len(data)} bytes")
    for label, unit in units.items():
        uf_size = len(records[unit]) - locate(records[unit])["uf"]
        # UF entropy stream shortened with consistent framing.
        for cut in (1, 4, uf_size // 2, uf_size - 4, uf_size - 1):
            if cut <= 0 or cut >= uf_size:
                continue
            copy = [bytearray(r) for r in records]
            shorten_uf(copy[unit], cut)
            emit(f"{label}_uf_minus_{cut}", join(copy),
                 f"unit {unit}: UF stream shortened by {cut} of {uf_size} bytes, lengths fixed")
        # Bit flips at the start (initial state), middle and end of the UF stream.
        start = locate(records[unit])["uf"]
        for where, offset in (("start", 0), ("state3", 3), ("middle", uf_size // 2),
                              ("end", uf_size - 1)):
            copy = [bytearray(r) for r in records]
            bit = rng.randrange(8)
            copy[unit][start + offset] ^= 1 << bit
            emit(f"{label}_flip_{where}", join(copy),
                 f"unit {unit}: bit {bit} of UF byte {offset}/{uf_size} flipped")
        # Random bytes in place of the UF stream.
        copy = [bytearray(r) for r in records]
        copy[unit][start:] = bytes(rng.randrange(256) for _ in range(uf_size))
        emit(f"{label}_random_uf", join(copy), f"unit {unit}: UF stream replaced by random bytes")
        # Zeroed UF stream.
        copy = [bytearray(r) for r in records]
        copy[unit][start:] = bytes(uf_size)
        emit(f"{label}_zero_uf", join(copy), f"unit {unit}: UF stream zeroed")
        # Header fields of the UF payload.
        payload = locate(records[unit])["payload"]
        for field, offset, values in (("parallel", 5, (0, 2, 3, 7, 8, 9, 255)),
                                      ("flags", 6, (1, 2, 0x80)),
                                      ("frames", 4, (0, 2, 9))):
            for value in values:
                copy = [bytearray(r) for r in records]
                if copy[unit][payload + offset] == value:
                    continue
                if field == "frames":
                    continue  # changes the payload layout; covered by the adapter parser tests
                copy[unit][payload + offset] = value
                emit(f"{label}_{field}_{value}", join(copy),
                     f"unit {unit}: UF payload {field} byte set to {value}")
    # Units out of order and a missing intra unit.
    emit("drop_intra", join(records[1:]), "intra unit removed")
    emit("drop_group1", join(records[:1] + records[2:]), "first HT-S group removed")
    swapped = records[:1] + [records[2], records[1]] + records[3:]
    emit("swap_groups_1_2", join(swapped), "HT-S groups 1 and 2 swapped")
    (args.output_dir / "variants.json").write_text(json.dumps(variants, indent=1) + "\n")
    print(f"wrote {len(variants)} variants to {args.output_dir}")


if __name__ == "__main__":
    main()
