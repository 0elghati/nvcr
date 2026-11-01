#!/usr/bin/env python3
"""Replay verified native tensors through the built UF stages without native entropy."""
import argparse
import hashlib, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path, help="Passing probe_nvcr_session.py report")
parser.add_argument("output", type=Path, help="New replay result JSON")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("replay output must be new")
report_path = args.report.resolve()
r = json.loads(report_path.read_text())
root = Path(__file__).resolve().parents[3]
if not r["passed"]:
    raise ValueError("reference stage report did not pass")
binary = report_path.parent / "binding/nvcr_dcvcuf_session_probe.so"
vectors = report_path.parent / "vectors.npz"
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
for path, expected in r["source_sha256"].items():
    verify(root / path, expected)
for value in r["inputs"].values():
    verify(value["path"], value["sha256"])
verify(binary, r["session_binary_sha256"])
verify(vectors, r["vectors_sha256"])
sys.path.insert(0, str(root / "scripts/backends/dcvcuf"))
from probe_hts_export import verify_reference
reference = root / "assets/dcvc-uf-reference"
if verify_reference(reference) != r["reference_commit"]:
    raise ValueError("reference revision mismatch")
sys.path[:0] = [str(reference), str(Path(r["inputs"]["extension"]["path"]).parent), str(binary.parent)]
import numpy as np
import torch
import inference_extensions_cuda
import nvcr_dcvcuf_session_probe as binding
if Path(inference_extensions_cuda.__file__).resolve() != Path(r["inputs"]["extension"]["path"]).resolve():
    raise ValueError("imported upstream binary differs from verified extension")
if Path(binding.__file__).resolve() != binary:
    raise ValueError("imported provider binary differs from verified binding")
from src.models.video_model_ht import DMC
from src.utils.common import get_state_dict, set_torch_env
set_torch_env()
torch.cuda.set_stream(torch.cuda.Stream())
m = DMC().eval()
m.load_state_dict(get_state_dict(r["inputs"]["checkpoint"]["path"]))
m = m.half().cuda().to(memory_format=torch.channels_last)
checks = []
with np.load(vectors) as data, torch.inference_mode():
    a = b = None
    for i, chunk in enumerate(r["chunks"]):
        _, h, w, _ = chunk["feature_shape_nhwc"]
        if a is None:
            a = binding.SessionProbe(m.recon_head.state_dict(), m.decoder.state_dict(), h, w, 0)
            b = binding.SessionProbe(m.recon_head.state_dict(), m.decoder.state_dict(), h, w, 0)
        inputs = [torch.from_numpy(data[f"chunk{i}_{name}"]) for name in ("latent", "context", "quant")]
        expected = inputs + [torch.from_numpy(data[f"chunk{i}_{name}"]) for name in ("feature", "frames", "reset")]
        for session in (a, b, a):
            session.reset()
            actual = session.run_pipeline(*inputs)
            checks.append(all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True)))
    contracts = a.rejection_checks(b)
result = {"scope": "provider_synthesis_replay_without_native_entropy", "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
          "replay_source_sha256": digest(__file__), "reference_report_sha256": digest(report_path), "vectors_sha256": digest(vectors), "session_binary_sha256": digest(binary),
          "chunks": len(r["chunks"]), "pipeline_runs": len(checks), "passed": all(checks), "contract_checks": len(contracts)}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["passed"] else 1)
