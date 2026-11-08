#!/usr/bin/env python3
"""Lists DCVC-UF y CDF slots that a wrong entropy index cannot be detected on.

The UF decoder checks that each frame's rANS stream ends at the encoder's
initial state with every byte consumed. A decoder that computes a different
entropy index (CDF row) for a symbol takes the same rANS state transition when
the other row holds the coded slot with the same start and frequency; the stream
stays in sync and only the symbol's meaning can change. This script lists those
(coded row, decoder row, slot) cases for a TensorRT bundle's tables.bin, and
their symbol probabilities. Same-meaning cases (same slot, or both escape slots
of rows with the same escape offset) are harmless and only counted.

Layout of tables.bin: see add_tensorrt_tables.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from add_tensorrt_tables import TABLES


def y_tables(data):
    offset, found = 0, {}
    for name, shape, dtype in TABLES:
        count = int(np.prod(shape))
        array = np.frombuffer(data, np.dtype(dtype), count, offset).reshape(shape)
        offset += array.nbytes
        found[name] = array
    if offset != len(data):
        raise SystemExit(f"tables.bin has {len(data)} bytes, layout needs {offset}")
    return {model: (found[f"{model}.gaussian_encoder.quantized_cdf"],
                    found[f"{model}.gaussian_encoder.cdf_length"]) for model in ("image", "video")}


def aliasing(cdf, lengths, max_distance):
    silent, harmless = [], 0
    rows = len(lengths)
    for coded in range(rows):
        coded_max = int(lengths[coded]) - 2
        for other in range(rows):
            if other == coded or abs(other - coded) > max_distance:
                continue
            other_max = int(lengths[other]) - 2
            for slot in range(coded_max + 1):
                start, end = int(cdf[coded][slot]), int(cdf[coded][slot + 1])
                for alt in range(other_max + 1):
                    if int(cdf[other][alt]) != start or int(cdf[other][alt + 1]) != end:
                        continue
                    coded_escape, other_escape = slot == coded_max, alt == other_max
                    if coded_escape != other_escape:
                        continue  # bypass bits read on one side only: desynchronizes
                    if (coded_escape and coded_max == other_max) or \
                            (not coded_escape and slot == alt):
                        harmless += 1
                        continue
                    silent.append({"coded_row": coded, "decoder_row": other, "slot": slot,
                                   "decoder_slot": alt, "escape": coded_escape,
                                   "probability": (end - start) / 65536})
    return silent, harmless


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--max-distance", type=int, default=127,
                        help="largest |coded row - decoder row| to consider")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {}
    for model, (cdf, lengths) in y_tables(args.tables.read_bytes()).items():
        silent, harmless = aliasing(cdf, lengths, args.max_distance)
        near = sorted({(c["coded_row"], c["decoder_row"]) for c in silent
                       if abs(c["coded_row"] - c["decoder_row"]) <= 4})
        report[model] = {
            "silent_cases": len(silent), "harmless_cases": harmless,
            "silent_escape_cases": sum(1 for c in silent if c["escape"]),
            "max_symbol_probability": max((c["probability"] for c in silent), default=0.0),
            "row_pairs_within_4": near, "cases": silent}
        print(f"{model}: {len(silent)} silent cases (max symbol probability "
              f"{report[model]['max_symbol_probability']:.6f}), {harmless} harmless; "
              f"row pairs within 4: {near}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
