#!/usr/bin/env python3
"""Replay verified native tensors through the built UF stages without native entropy."""
import argparse
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
process_start = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path, help="Passing probe_nvcr_session.py report")
parser.add_argument("output", type=Path, help="New replay result JSON")
parser.add_argument("--video-output", type=Path,
                    help="New MP4 containing the I-frame and NVCR provider output")
parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate")
parser.add_argument("--benchmark-runs", type=int, default=0,
                    help="Timed provider-pipeline repetitions after correctness checks")
parser.add_argument("--warmup-runs", type=int, default=2,
                    help="Untimed provider-pipeline repetitions before benchmarking")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("replay output must be new")
if args.video_output is not None and args.video_output.exists():
    raise ValueError("video output must be new")
if args.fps <= 0 or args.benchmark_runs < 0 or args.warmup_runs < 0:
    raise ValueError("fps must be positive and run counts must be nonnegative")
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
from src.utils.transforms import yuv_444_to_420
set_torch_env()
torch.cuda.set_stream(torch.cuda.Stream())
m = DMC().eval()
m.load_state_dict(get_state_dict(r["inputs"]["checkpoint"]["path"]))
m = m.half().cuda().to(memory_format=torch.channels_last)
checks = []
benchmark_inputs = []
rendered_chunks = []
with np.load(vectors) as data, torch.inference_mode():
    a = b = None
    for i, chunk in enumerate(r["chunks"]):
        _, h, w, _ = chunk["feature_shape_nhwc"]
        if a is None:
            a = binding.SessionProbe(m.recon_head.state_dict(), m.decoder.state_dict(), h, w, 0)
            b = binding.SessionProbe(m.recon_head.state_dict(), m.decoder.state_dict(), h, w, 0)
        inputs = [torch.from_numpy(data[f"chunk{i}_{name}"]) for name in ("latent", "context", "quant")]
        benchmark_inputs.append(inputs)
        expected = inputs + [torch.from_numpy(data[f"chunk{i}_{name}"]) for name in ("feature", "frames", "reset")]
        for session in (a, b, a):
            session.reset()
            actual = session.run_pipeline(*inputs)
            checks.append(all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True)))
        rendered_chunks.append(actual[4])
    contracts = a.rejection_checks(b)
    for _ in range(args.warmup_runs if args.benchmark_runs else 0):
        for inputs in benchmark_inputs:
            a.run_pipeline(*inputs)
    torch.cuda.synchronize()
    benchmark_start = time.perf_counter()
    for _ in range(args.benchmark_runs):
        for inputs in benchmark_inputs:
            a.run_pipeline(*inputs)
    torch.cuda.synchronize()
    benchmark_duration = time.perf_counter() - benchmark_start
    rendered = torch.cat([
        torch.from_numpy(data["intra0_frame"]).permute(0, 2, 3, 1),
        *rendered_chunks,
    ])

video = None
if args.video_output is not None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --video-output")
    frames = rendered.permute(0, 3, 1, 2).float() + 0.5
    y, uv = yuv_444_to_420(frames)
    y = torch.clamp(y * 255, 0, 255).round().byte().numpy()
    uv = torch.clamp(uv * 255, 0, 255).byte().numpy()
    raw = b"".join(y[i].tobytes() + uv[i].tobytes() for i in range(len(y)))
    height, width = rendered.shape[1:3]
    completed = subprocess.run([
        ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s:v", f"{width}x{height}", "-r", str(args.fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.video_output.resolve()),
    ], input=raw, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    video = {"path": str(args.video_output.resolve()), "sha256": digest(args.video_output),
             "frames": len(rendered), "width": width, "height": height, "fps": args.fps}

provider_frames = args.benchmark_runs * len(benchmark_inputs) * 8
timing = None if not args.benchmark_runs else {
    "scope": "NVCR provider synthesis and reconstruction from frozen native tensors, including host transfers",
    "warmup_runs": args.warmup_runs, "repetitions": args.benchmark_runs,
    "chunks_per_repetition": len(benchmark_inputs), "frames_per_chunk": 8,
    "decoded_frames": provider_frames, "duration_s": benchmark_duration,
    "fps": provider_frames / benchmark_duration,
    "milliseconds_per_chunk": 1000 * benchmark_duration / args.benchmark_runs / len(benchmark_inputs),
}
result = {"scope": "provider_synthesis_replay_without_native_entropy", "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
          "replay_source_sha256": digest(__file__), "reference_report_sha256": digest(report_path), "vectors_sha256": digest(vectors), "session_binary_sha256": digest(binary),
          "chunks": len(r["chunks"]), "pipeline_runs": len(checks), "passed": all(checks),
          "contract_checks": len(contracts), "timing": timing, "video": video,
          "gpu": torch.cuda.get_device_name(),
          "output_composition": {"native_intra_frames": 1, "nvcr_provider_predicted_frames": 16},
          "limitations": ["I-frame and entropy-decoded tensors come from the pinned native reference",
                          "timing covers NVCR synthesis and reconstruction stages, not full bitstream decode"],
          "process_duration_s": time.perf_counter() - process_start,
          "process_timing_scope": "verification, model load, correctness, benchmark, and video generation"}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
raise SystemExit(0 if result["passed"] else 1)
