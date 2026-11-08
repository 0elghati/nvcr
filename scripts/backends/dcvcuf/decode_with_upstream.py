#!/usr/bin/env python3
"""C11 G4: pinned upstream DCVC-UF decoding NVCR .nvcr streams.

Each NVCR DCVC-UF access unit carries upstream's entropy unit unchanged, with
its QP, entropy-parallel count and feature-memory-reset bit (C7). This script
extracts them and runs upstream's own decoder on them exactly as pinned
test_video.py's decode loop does (DMCI.decompress, then clear_dpb and
add_ref_feature_from_frame for an intra unit; DMC.decompress for an HT-S
group), and writes the reconstruction with test_video.py's YUV420 conversion.
Only the real frames of a padded tail are written, as test_video.py does.

Upstream has no entropy desynchronization check, so a stream it cannot decode
may complete with wrong frames, crash or hang. Every stream therefore runs in a
child process with a timeout, frames are flushed as they are decoded, and the
parent records the exit status, signal and frames written.

Run with the interpreter that has upstream's dependencies (PyTorch, the built
inference_extensions_cuda and MLCodec_extensions_cpp modules on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
INTRA_MAGIC = b"UFI1"
PREDICTED_MAGIC = b"UFP1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, count):
        if count < 0 or count > len(self.data) - self.pos:
            raise ValueError("truncated")
        out = self.data[self.pos:self.pos + count]
        self.pos += count
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))


def access_unit(data):
    """NVAU v1/v2 -> (codec id, frame kind, width, height, qp, codec payload)."""
    r = Reader(data)
    if r.take(4) != b"NVAU":
        raise ValueError("missing NVAU")
    (version,) = r.unpack("<H")
    if version == 1:
        kind, _flags, ids, _reserved, width, height, qp, size = r.unpack("<BBHHIIIQ")
        identifier = r.take(ids).decode("ascii")
        payload = r.take(size)
    elif version == 2:
        (_header, _flags, kind, _pixel, _depth, _reserved, width, height, qp, _doi, _poi, deps,
         sections, cid, pid, mid, _reserved2, _total) = r.unpack("<HIBBBBIIIQQHHHHHHQ")
        identifier = r.take(cid).decode("ascii")
        r.take(pid + mid + deps * 8)
        entries = [r.unpack("<HHIQ") for _ in range(sections)]
        bodies = [(entry[0], r.take(entry[3])) for entry in entries]
        payload = next(body for kind_id, body in bodies if kind_id == 1)
    else:
        raise ValueError(f"unsupported NVAU version {version}")
    return identifier, kind, width, height, qp, payload


def read_units(path):
    """Yields upstream entropy units from a CLI .nvcr file (NVCS v1)."""
    data = Path(path).read_bytes()
    r = Reader(data)
    if r.take(8) != b"NVCS\x01\x00\x00\x00":
        raise ValueError("unsupported NVCR sequence header")
    units = []
    while r.pos < len(data):
        (length,) = r.unpack("<Q")
        packet = Reader(r.take(length))
        if packet.take(4) != b"NVCR":
            raise ValueError("missing packet magic")
        _version, _kind, _flags, _timestamp, count = packet.unpack("<HBBQH")
        for _ in range(count):
            key, value = packet.unpack("<HH")
            packet.take(key + value)
        (size,) = packet.unpack("<Q")
        identifier, kind, width, height, qp, payload = access_unit(packet.take(size))
        if identifier != "dcvcuf-cvpr2026-hts":
            raise ValueError(f"not a DCVC-UF HT-S stream: {identifier}")
        p = Reader(payload)
        magic = p.take(4)
        frames, ec_part, flags, _reserved = p.unpack("<BBBB")
        p.take(8 * frames)
        intra = magic == INTRA_MAGIC
        if magic not in (INTRA_MAGIC, PREDICTED_MAGIC) or intra != (kind == 0):
            raise ValueError("unexpected DCVC-UF payload")
        units.append({"intra": intra, "width": width, "height": height, "qp": qp,
                      "frames": frames, "ec_part": ec_part, "reset": flags & 1,
                      "bit_stream": payload[p.pos:]})
    return units


def worker(args):
    """Decodes one stream in this process (child of `main`)."""
    sys.path.insert(0, str(args.reference_root.resolve()))
    os.chdir(args.reference_root)
    import torch
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC
    from src.utils.common import ModelStructure, get_state_dict, set_torch_env
    from src.utils.transforms import yuv_444_to_420

    set_torch_env()
    device = "cuda:0"
    torch.cuda.set_stream(torch.cuda.Stream(device, 0))  # as test_video.py init_func

    def finalize(net):
        net = net.half().to(device)
        return net.to(memory_format=torch.channels_last)

    i_net = DMCI().eval()
    i_net.load_state_dict(get_state_dict(args.image_checkpoint))
    i_net.update(0.0)
    i_net = finalize(i_net)
    # Upstream creates the fused image proxy lazily in DMCI.compress (test_video.py
    # always encodes first); these are its lines. The HT-S proxy is created by
    # add_ref_feature_from_frame after the first intra unit, as upstream does.
    from inference_extensions_cuda import DMCIProxy
    state_dict = i_net.add_cdf_to_state_dict(i_net.state_dict())
    i_net.proxy = DMCIProxy()
    i_net.proxy.set_param(state_dict, i_net.gaussian_encoder.skip_thres)
    p_net = DMC(model_structure=ModelStructure.HTS).eval()
    p_net.load_state_dict(get_state_dict(args.video_checkpoint))
    p_net.update(0.0)
    p_net = finalize(p_net)

    units = read_units(args.stream)
    width, height = units[0]["width"], units[0]["height"]
    sps = {"sps_id": 0, "height": height, "width": width}
    progress = Path(args.progress)
    with torch.no_grad(), Path(args.output).open("wb") as output:
        for index, unit in enumerate(units):
            if unit["intra"]:
                decoded = i_net.decompress(unit["bit_stream"], sps, unit["qp"], unit["ec_part"])
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(decoded["x_hat"], apply_feature_adaptor=False)
                frames = [decoded["x_hat"]]
            else:
                decoded = p_net.decompress(unit["bit_stream"], sps, unit["qp"], unit["ec_part"],
                                           unit["reset"])
                frames = decoded["x_hat"][:unit["frames"]]
            for x_hat in frames:
                x_hat = x_hat[:, :, :height, :width]
                y_rec, uv_rec = yuv_444_to_420(x_hat + 0.5)
                y_rec = torch.clamp(y_rec * 255, 0, 255).round().byte().squeeze(0).cpu().numpy()
                uv_rec = torch.clamp(uv_rec * 255, 0, 255).byte().squeeze(0).cpu().numpy()
                output.write(y_rec.tobytes())
                output.write(uv_rec.tobytes())
            output.flush()
            progress.write_text(json.dumps({"units_done": index + 1}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    parser.add_argument("--streams", type=Path, help="directory with streams.json (and SHA256SUMS)")
    parser.add_argument("--stream", type=Path, help="one .nvcr file (worker mode with --worker)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)

    def git(*command):
        return subprocess.check_output(["git", "-C", str(args.reference_root), *command],
                                       text=True).strip()
    if git("rev-parse", "HEAD") != PINNED_REFERENCE_COMMIT:
        raise SystemExit("reference checkout is not the pinned commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("reference checkout must be clean")
    manifest = json.loads((args.streams / "streams.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in manifest["streams"]:
        stream = args.streams / entry["stream"]
        if sha256(stream) != entry["sha256"]:
            raise SystemExit(f"{stream}: digest mismatch")
        case = args.output_dir / Path(entry["stream"]).with_suffix("")
        case.mkdir(parents=True, exist_ok=True)
        output, progress = case / "upstream.yuv", case / "progress.json"
        progress.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--reference-root", str(args.reference_root.resolve()),
                   "--image-checkpoint", str(args.image_checkpoint.resolve()),
                   "--video-checkpoint", str(args.video_checkpoint.resolve()),
                   "--stream", str(stream.resolve()), "--output", str(output.resolve()),
                   "--progress", str(progress.resolve())]
        print(f"[upstream] {entry['stream']}", flush=True)
        started = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=args.timeout)
            status, stderr = completed.returncode, completed.stderr
            timed_out = False
        except subprocess.TimeoutExpired as expired:
            status, stderr, timed_out = None, (expired.stderr or b"").decode(errors="replace"), True
        frame_bytes = entry["width"] * entry["height"] * 3 // 2
        written = output.stat().st_size // frame_bytes if output.exists() else 0
        row = {"stream": entry["stream"], "stream_sha256": entry["sha256"],
               "encoder_path": entry["encoder_path"], "encoder_gpu": manifest.get("gpu_label"),
               "sequence": entry["sequence"], "qp": entry["qp"], "frames": entry["frames"],
               "width": entry["width"], "height": entry["height"],
               "exit": status, "signal": (signal.Signals(-status).name
                                          if isinstance(status, int) and status < 0 else None),
               "timed_out": timed_out, "seconds": round(time.monotonic() - started, 2),
               "units_done": (json.loads(progress.read_text())["units_done"]
                              if progress.exists() else 0),
               "frames_written": written,
               "yuv": str(output.relative_to(args.output_dir)) if output.exists() else None,
               "yuv_sha256": sha256(output) if output.exists() else None,
               "stderr_tail": stderr.strip()[-1500:]}
        rows.append(row)
        print(f"  exit={status} signal={row['signal']} timed_out={timed_out} "
              f"units={row['units_done']} frames={written}", flush=True)
    (args.output_dir / "upstream.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
