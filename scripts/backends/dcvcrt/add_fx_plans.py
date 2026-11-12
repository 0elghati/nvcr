#!/usr/bin/env python3
"""Adds DCVC-RT's nvcr.fx.v1 stages to a TensorRT engine bundle.

Copies the fx programs and fx_entropy.json into <bundle>/normative, compiles each program into
<bundle>/<stage>.fx.plan with nvcr-fx-plans over the bundle's frame sizes (its visible minimum to maximum, optimized
for its optimum, each padded to 16), and lists the new files in engine.sha256 and engine_manifest.json.

usage: add_fx_plans.py --bundle DIR --programs DIR --nvcr-fx-plans PATH
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shapes(width: int, height: int) -> dict:
    """Input shapes [channels, height, width] of every DCVC-RT fx stage for a frame size."""
    h, w = -(-height // 16) * 16, -(-width // 16) * 16
    yh, yw = h // 16, w // 16
    zh, zw = -(-yh // 4), -(-yw // 4)
    feature, latent, z = [256, h // 8, w // 8], [1, yh, yw], [128, zh, zw]
    qp, parity = [1, 1, 1], [4, yh, yw]
    step = {"symbols": [64, yh, yw], "symbol_means": [64, yh, yw], "parity": parity}
    return {
        "i_prior": {"z_fixed": z, "latent_ref": latent, "parity": parity},
        "i_spatial_prior_1": {**step, "common": [256, yh, yw]},
        **{f"i_spatial_prior_{k}": {**step, "y_hat": [256, yh, yw], "common": [256, yh, yw]} for k in (2, 3)},
        "i_synthesis": {**step, "y_hat": [256, yh, yw], "params": [514, yh, yw], "qp": qp},
        "p_reference_frame": {"frame": [3, h, w], "qp": qp},
        "p_reference_feature": {"feature": feature, "qp": qp},
        "p_prior": {"z_fixed": z, "temporal_context": feature, "latent_ref": latent, "parity": parity},
        "p_spatial_prior": {**step, "params": [384, yh, yw]},
        "p_synthesis": {**step, "y_hat": [128, yh, yw], "params": [384, yh, yw], "context": feature, "qp": qp},
        "p_reset_frame": {"feature": feature, "qp": qp},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--programs", type=Path, required=True)
    ap.add_argument("--nvcr-fx-plans", type=Path, required=True)
    args = ap.parse_args()
    manifest_path = args.bundle / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    visible = manifest["visible_dimensions"]
    low, opt, high = (shapes(*visible[key]) for key in ("minimum", "optimum", "maximum"))
    ranged = {stage: {name: [low[stage][name], opt[stage][name], high[stage][name]] for name in opt[stage]}
              for stage in opt}

    normative = args.bundle / "normative"
    normative.mkdir(exist_ok=True)
    for path in [*args.programs.glob("*.fxprog.json"), *args.programs.glob("*.fxw"), args.programs / "fx_entropy.json"]:
        shutil.copy2(path, normative / path.name)
    with tempfile.TemporaryDirectory() as tmp:
        shape_file = Path(tmp) / "shapes.json"
        shape_file.write_text(json.dumps(ranged))
        subprocess.run([str(args.nvcr_fx_plans), str(normative), str(shape_file), str(args.bundle)], check=True)

    files = manifest["files"]
    for path in sorted(args.bundle.glob("*.fx.plan*")):
        files[path.name] = sha256(path)
    checksum = args.bundle / manifest["checksum_manifest"]
    checksum.write_text("".join(f"{files[name]}  {name}\n" for name in sorted(files)))
    manifest["checksum_manifest_sha256"] = sha256(checksum)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.bundle}: fx plans for {visible['minimum']} to {visible['maximum']}")


if __name__ == "__main__":
    main()
