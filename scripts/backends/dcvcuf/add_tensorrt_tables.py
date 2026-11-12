#!/usr/bin/env python3
"""Add DCVC-UF quantization and entropy tables to a TensorRT stage bundle.

The plans hold the networks; the codec also needs the per-QP scale vectors and
the z/y CDF tables. They are copied from NVCR's converted native weight bundle
(the same values the fused path uses) into tables.bin, and the bundle's
engine.sha256 and engine_manifest.json are refreshed so the TensorRT session
verifies the new file like every plan.

tables.bin is the concatenation, in this order, of little-endian arrays:
  image q_scale_enc, q_scale_dec (64x384 f16), q_scale_y_enc, q_scale_y_dec (64x256 f16),
        z cdf (8192x19 i32), z cdf_length (8192 i32), y cdf (128x19 i32), y cdf_length (128 i32)
  video q_encoder, q_decoder, q_feature (64x512 f16), then the same four CDF arrays.
"""

import argparse
import hashlib
import json
from pathlib import Path

TABLES = [
    ("image.q_scale_enc", (64, 384), "float16"),
    ("image.q_scale_dec", (64, 384), "float16"),
    ("image.q_scale_y_enc", (64, 256), "float16"),
    ("image.q_scale_y_dec", (64, 256), "float16"),
    ("image.bit_estimator_z.quantized_cdf", (8192, 19), "int32"),
    ("image.bit_estimator_z.cdf_length", (8192,), "int32"),
    ("image.gaussian_encoder.quantized_cdf", (128, 19), "int32"),
    ("image.gaussian_encoder.cdf_length", (128,), "int32"),
    ("video.q_encoder", (64, 512), "float16"),
    ("video.q_decoder", (64, 512), "float16"),
    ("video.q_feature", (64, 512), "float16"),
    ("video.bit_estimator_z.quantized_cdf", (8192, 19), "int32"),
    ("video.bit_estimator_z.cdf_length", (8192,), "int32"),
    ("video.gaussian_encoder.quantized_cdf", (128, 19), "int32"),
    ("video.gaussian_encoder.cdf_length", (128,), "int32"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(shape):
    total = 1
    for size in shape:
        total *= size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="dcvcuf-cvpr2026-hts.pt")
    source.add_argument("--tables", type=Path,
                        help="an existing tables.bin (for example another target's bundle)")
    parser.add_argument("--tables-sha256", help="required digest of --tables")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    if args.tables is not None:
        data = args.tables.read_bytes()
        expected = sum(count(shape) * (2 if dtype == "float16" else 4)
                       for _name, shape, dtype in TABLES)
        if len(data) != expected:
            raise SystemExit(f"{args.tables}: {len(data)} bytes, expected {expected}")
        if args.tables_sha256 and hashlib.sha256(data).hexdigest() != args.tables_sha256:
            raise SystemExit(f"{args.tables}: SHA-256 does not match --tables-sha256")
    else:
        import torch

        module = torch.jit.load(str(args.weights), map_location="cpu")
        buffers = {name.replace("__nvcr_dot__", "."): value
                   for name, value in module.named_buffers()}
        data = bytearray()
        for name, shape, dtype in TABLES:
            value = buffers[name].contiguous()
            if tuple(value.shape) != shape or str(value.dtype) != f"torch.{dtype}":
                raise SystemExit(
                    f"{name}: expected {shape} {dtype}, found {tuple(value.shape)} {value.dtype}")
            data += value.numpy().tobytes()
    (args.bundle / "tables.bin").write_bytes(bytes(data))

    manifest_path = args.bundle / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    files["tables.bin"] = sha256(args.bundle / "tables.bin")
    checksums = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    (args.bundle / "engine.sha256").write_text(checksums, encoding="utf-8")
    manifest["checksum_manifest_sha256"] = sha256(args.bundle / "engine.sha256")
    manifest_path.write_text(json.dumps(manifest, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(data)} bytes to {args.bundle / 'tables.bin'}")


if __name__ == "__main__":
    main()
