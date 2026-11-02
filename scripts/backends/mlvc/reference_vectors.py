#!/usr/bin/env python3
"""Record an upstream MLVC reference vector for NVCR's MLVC parity gate.

Runs Microsoft MLVC's own conversion runtime (load_split_model + FrameLoop) on
an ONNX model bundle and writes what that runtime produces: the `.mlvc` record
stream, the decoded 8-bit YUV, and a JSON summary of the per-frame decisions.
Nothing in the upstream path is reimplemented here.

Run with the Python of a pinned MLVC checkout's environment, isolated so that
only the checkout passed with --mlvc-video-dir is importable:

  <mlvc>/.venv/bin/python -I reference_vectors.py \
      --mlvc-video-dir <mlvc>/video --bundle <bundle-dir> --model-id 320x192 \
      --input akiyo_qcif.yuv --width 176 --height 144 --frames 66 \
      --q-index 32 --output <dir>

--bitrate selects MLVC's rate controller instead of a fixed Q index; its
streams can contain dropped frames (Q index -1, empty payload).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlvc-video-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="bundle canvas, e.g. 320x192")
    parser.add_argument("--input", type=Path, required=True, help="8-bit I420 input")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    rate = parser.add_mutually_exclusive_group(required=True)
    rate.add_argument("--q-index", type=int)
    rate.add_argument("--bitrate", type=float, help="bits per second; MLVC rate control")
    parser.add_argument("--iframe-period", type=int, default=None)
    parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda", "qnn"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video_dir = args.mlvc_video_dir.resolve()
    sys.path.insert(0, str(video_dir))
    import numpy as np
    import onnxruntime as ort
    from conversion import FrameLoop, load_split_model
    from conversion.types import OnnxExecutionProvider, RuntimeParams

    frame_bytes = args.width * args.height * 3 // 2
    if args.input.stat().st_size < frame_bytes * args.frames:
        # FrameLoop ping-pongs short clips; the gate needs distinct frames.
        parser.error("input has fewer frames than requested")

    provider = {
        "cpu": OnnxExecutionProvider.CPU,
        "cuda": OnnxExecutionProvider.CUDA,
        "qnn": OnnxExecutionProvider.QNN,
    }[args.execution_provider]
    split_model = load_split_model(
        model_path=args.bundle,
        model_id=args.model_id,
        runtime_params=RuntimeParams(onnx_execution_provider=provider),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    loop = FrameLoop(
        split_model,
        video_path=args.input.resolve(),
        image_width=args.width,
        image_height=args.height,
        frame_count=args.frames,
        output_data_dir=args.output,
    )
    results = loop.run(
        q_index=args.q_index,
        bitrate=args.bitrate,
        iframe_period=args.iframe_period,
        progress_bar=False,
        save_yuv=True,
    )

    # FrameLoop names its output directory after the rate setting.
    run_dir = args.output / (f"q_index={args.q_index}" if args.q_index is not None
                             else f"bitrate_kbps={int(args.bitrate / 1000)}")
    commit = subprocess.run(
        ["git", "-C", str(video_dir), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(video_dir), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.strip()
    summary = {
        "upstream": {"repository": "https://github.com/microsoft/MLVC.git",
                     "commit": commit, "clean": not dirty},
        "runtime": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                    "numpy": np.__version__, "execution_provider": args.execution_provider,
                    "machine": platform.machine(), "system": platform.system()},
        "bundle": {"path": str(args.bundle.resolve()), "model_id": args.model_id,
                   "manifest_sha256": sha256(args.bundle / "bundle_manifest.json")},
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input),
                  "width": args.width, "height": args.height, "frames": args.frames},
        "q_index": args.q_index,
        "bitrate": args.bitrate,
        "iframe_period": args.iframe_period,
        "outputs": {"mlvc": "output.mlvc", "mlvc_sha256": sha256(run_dir / "output.mlvc"),
                    "yuv": "output.yuv", "yuv_sha256": sha256(run_dir / "output.yuv")},
        "frames": [
            {"frame_type": frame.frame_type.name, "frame_idx": frame.frame_idx,
             "ref_frame_idx": frame.ref_frame_idx, "q_index": frame.q_index,
             "marked_as_ltr": frame.marked_as_ltr, "feature_reset": frame.feature_reset,
             "bytes": len(frame.encoder.bitstream) if frame.encoder is not None else 0}
            for frame in results.frames
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("upstream", "runtime", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
