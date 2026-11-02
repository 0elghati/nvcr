#!/usr/bin/env python3
"""Capture the C8 sequence-level DCVC-UF reference with pinned upstream test_video.py.

Runs the unmodified upstream script on hash-verified sequences and checkpoints
and keeps a compact report: per-frame bpp and PSNR, frame types, and stream
digests for each sequence and QP. This is the reference that a TensorRT port is
compared against; it does not execute NVCR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
DEFAULT_QPS = (0, 21, 42, 63)  # upstream --rate_num 4: linspace(0, 63, 4) rounded
FRAME_FIELDS = ("frame_type", "frame_bpp", "frame_psnr", "frame_psnr_y",
                "frame_psnr_u", "frame_psnr_v")
AVERAGE_FIELDS = ("ave_all_frame_bpp", "ave_all_frame_psnr", "ave_all_frame_psnr_y",
                  "ave_all_frame_psnr_u", "ave_all_frame_psnr_v",
                  "i_frame_num", "p_frame_num")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_reference(root, expected=PINNED_REFERENCE_COMMIT):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if commit != expected:
        raise ValueError(f"expected upstream {expected}, found {commit}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("reference checkout must be clean, including untracked source files")
    return commit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--image-checkpoint-sha256", required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    parser.add_argument("--video-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("c8_reference_sequences.json"))
    parser.add_argument("--sequence", action="append",
                        help="Restrict to these manifest sequence names (repeatable)")
    parser.add_argument("--qp", type=int, nargs="+", default=list(DEFAULT_QPS))
    parser.add_argument("--intra-period", type=int, default=-1,
                        help="Upstream intra_period; -1 means intra only at frame 0")
    parser.add_argument("--reset-interval", type=int, default=32)
    parser.add_argument("--skip-thres", type=float, default=0.0)
    parser.add_argument("--frames", type=int, help="Override every sequence's frame count")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # Upstream runs from its own checkout. Keep the venv interpreter path
    # (absolute, not symlink-resolved) so its site-packages still apply.
    args.python = args.python.absolute()
    if any(qp < 0 or qp > 63 for qp in args.qp):
        parser.error("QPs must be in [0, 63]")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    return args


def select_sequences(manifest, names):
    sequences = manifest["sequences"]
    if not names:
        return sequences
    known = {sequence["name"]: sequence for sequence in sequences}
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown manifest sequences: {', '.join(unknown)}")
    return [known[name] for name in names]


def verify_inputs(args, sequences):
    checks = [
        (args.image_checkpoint, args.image_checkpoint_sha256, "image checkpoint"),
        (args.video_checkpoint, args.video_checkpoint_sha256, "video checkpoint"),
    ]
    checks += [(args.dataset_root / sequence["path"], sequence["sha256"], sequence["name"])
               for sequence in sequences]
    for path, expected, label in checks:
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")


def write_config(path, sequence, dataset_root, intra_period, frames):
    source = dataset_root / sequence["path"]
    config = {
        "root_path": str(source.parent),
        "test_classes": {
            sequence["class"]: {
                "test": 1,
                "base_path": ".",
                "src_type": "yuv420",
                "sequences": {
                    source.name: {
                        "width": sequence["width"],
                        "height": sequence["height"],
                        "frames": frames or sequence["frames"],
                        "intra_period": intra_period,
                    }
                },
            }
        },
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return source.name


def summarize(raw, sequence, source_name, qps, stream_dir):
    results = raw[sequence["class"]][source_name]
    points = []
    for rate_key in sorted(results):
        result = results[rate_key]
        qp = qps[result["rate_idx"]]
        stream = stream_dir / sequence["class"] / f"{source_name}_q{qp}.bin"
        point = {"qp": qp, "stream_bytes": stream.stat().st_size,
                 "stream_sha256": sha256(stream)}
        point.update({field: result[field] for field in AVERAGE_FIELDS})
        point.update({field: result[field] for field in FRAME_FIELDS})
        point["i_frames"] = [index for index, kind in enumerate(result["frame_type"])
                             if kind == 0]
        points.append(point)
    return points


def child_environment():
    """Returns the environment with PYTHONPATH made absolute for the upstream cwd."""
    env = os.environ.copy()
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(entry).absolute()) for entry in env["PYTHONPATH"].split(os.pathsep) if entry)
    return env


def environment(python):
    probe = ("import json, torch; print(json.dumps({'torch': torch.__version__, "
             "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}))")
    versions = json.loads(subprocess.check_output([str(python), "-c", probe], text=True))
    versions["python"] = subprocess.check_output(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        text=True).strip()
    versions["platform"] = platform.platform()
    return versions


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory must be new: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sequences = select_sequences(manifest, args.sequence)
    commit = verify_reference(args.reference_root)
    verify_inputs(args, sequences)

    args.output_dir.mkdir(parents=True)
    config_dir = args.output_dir / "configs"
    raw_dir = args.output_dir / "raw"
    stream_dir = args.output_dir / "streams"
    config_dir.mkdir()
    raw_dir.mkdir()

    report = {
        "schema": "nvcr.dcvcuf.rd_reference.v1",
        "upstream": {"repository": "https://github.com/microsoft/DCVC", "commit": commit,
                     "script": "test_video.py", "model_structure": "hts"},
        "checkpoints": {
            "image": {"path": str(args.image_checkpoint), "sha256": args.image_checkpoint_sha256},
            "video": {"path": str(args.video_checkpoint), "sha256": args.video_checkpoint_sha256},
        },
        "recipe": {"qps": args.qp, "intra_period": args.intra_period,
                   "reset_interval": args.reset_interval, "skip_thres": args.skip_thres,
                   "frames_override": args.frames},
        "environment": environment(args.python),
        "manifest_sha256": sha256(args.manifest),
        "sequences": [],
    }
    for sequence in sequences:
        config = config_dir / f"{sequence['name']}.json"
        source_name = write_config(
            config, sequence, args.dataset_root, args.intra_period, args.frames)
        raw = raw_dir / f"{sequence['name']}.json"
        command = [
            str(args.python), "test_video.py",
            "--model_path_i", str(args.image_checkpoint.resolve()),
            "--model_path_p", str(args.video_checkpoint.resolve()),
            "--model_structure", "hts", "--test_config", str(config.resolve()),
            "--rate_num", str(len(args.qp)),
            "--qp_i", *map(str, args.qp), "--qp_p", *map(str, args.qp),
            "--reset_interval", str(args.reset_interval),
            "--skip_thres", str(args.skip_thres),
            "--worker", "1", "--cuda_idx", "0", "--verbose_json", "True",
            "--stream_path", str(stream_dir.resolve()), "--output_path", str(raw.resolve()),
        ]
        print(f"[capture] {sequence['name']}", flush=True)
        subprocess.run(command, cwd=args.reference_root, check=True, env=child_environment())
        report["sequences"].append({
            "name": sequence["name"],
            "class": sequence["class"],
            "sha256": sequence["sha256"],
            "width": sequence["width"],
            "height": sequence["height"],
            "frames": args.frames or sequence["frames"],
            "points": summarize(json.loads(raw.read_text(encoding="utf-8")),
                                sequence, source_name, args.qp, stream_dir),
        })
    verify_reference(args.reference_root)  # upstream must stay untouched by the run
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[capture] wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
