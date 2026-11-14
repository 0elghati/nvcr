"""Adds a codec's nvcr.fx.v1 stages to a TensorRT engine bundle (shared by the codecs' add_fx_plans.py).

Copies the fx programs and fx_entropy.json into <bundle>/normative, compiles each program into
<bundle>/<stage>.fx.plan with nvcr-fx-plans over the given input shapes, and lists the new files in engine.sha256
and engine_manifest.json.
"""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_fx_plans(bundle: Path, programs: Path, nvcr_fx_plans: Path, ranged: dict) -> None:
    """ranged: {stage: {input: [[c, h, w] minimum, optimum, maximum]}}."""
    manifest_path = bundle / "engine_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    normative = bundle / "normative"
    normative.mkdir(exist_ok=True)
    for path in [*programs.glob("*.fxprog.json"), *programs.glob("*.fxw"), programs / "fx_entropy.json"]:
        shutil.copy2(path, normative / path.name)
    with tempfile.TemporaryDirectory() as tmp:
        shape_file = Path(tmp) / "shapes.json"
        shape_file.write_text(json.dumps(ranged))
        subprocess.run([str(nvcr_fx_plans), str(normative), str(shape_file), str(bundle)], check=True)

    files = manifest["files"]
    for path in sorted(bundle.glob("*.fx.plan*")):
        files[path.name] = sha256(path)
    checksum = bundle / manifest["checksum_manifest"]
    checksum.write_text("".join(f"{files[name]}  {name}\n" for name in sorted(files)))
    manifest["checksum_manifest_sha256"] = sha256(checksum)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
