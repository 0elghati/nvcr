#!/usr/bin/env python3
"""Capture C8 per-stage DCVC-UF reference tensors from pinned upstream modules.

For one intra frame and the following three eight-frame HT-S groups of each
manifest stage vector (the second resets feature memory), this runs:

- the upstream PyTorch `forward_one_frame` path in FP32 and in upstream's FP16
  channels-last configuration, recording every sub-network's inputs and
  outputs through forward hooks plus the quantized latents; and
- the upstream fused CUDA path (`compress`/`decompress`, as test_video.py does)
  for the deployed reconstruction.

The report gives each stage's FP16-vs-FP32 error, the quantized-symbol
agreement, and how far each PyTorch reconstruction is from the fused one. These
define what a TensorRT export can be compared against; this tool does not run
NVCR or TensorRT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
GROUP_FRAMES = 8
# Upstream's reset schedule with --reset_interval 16 over three groups (steps 1,
# 9, 17): the second group resets feature memory, so the vectors exercise the
# memory update (feature_adaptor_m), the reset head, and a post-reset group.
GROUP_RESETS = (False, True, False)
VECTOR_FRAMES = 1 + GROUP_FRAMES * len(GROUP_RESETS)
IMAGE_STAGES = ("enc", "hyper_enc", "hyper_dec", "y_prior_fusion",
                "y_spatial_prior_reduction", "y_spatial_prior_adaptor_1",
                "y_spatial_prior_adaptor_2", "y_spatial_prior_adaptor_3",
                "y_spatial_prior", "dec")
VIDEO_STAGES = ("feature_adaptor_i", "feature_adaptor_m", "feature_extractor", "encoder",
                "hyper_encoder", "hyper_decoder", "temporal_prior_encoder",
                "y_prior_fusion", "y_spatial_prior_reduction",
                "y_spatial_prior_adaptor_1", "y_spatial_prior_adaptor_2",
                "y_spatial_prior_adaptor_3", "y_spatial_prior", "decoder", "recon_head")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_reference(root, expected=PINNED_REFERENCE_COMMIT):
    import subprocess

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
    parser.add_argument("--vector", action="append",
                        help="Restrict to these manifest stage-vector names (repeatable)")
    parser.add_argument("--qp", type=int, default=32)
    parser.add_argument("--max-tensor-pixels", type=int, default=256 * 256,
                        help="Store raw tensors only for vectors up to this many pixels")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.qp <= 63:
        parser.error("--qp must be in [0, 63]")
    return args


def verify_inputs(args, vectors):
    checks = [
        (args.image_checkpoint, args.image_checkpoint_sha256, "image checkpoint"),
        (args.video_checkpoint, args.video_checkpoint_sha256, "video checkpoint"),
    ]
    checks += [(args.dataset_root / vector["path"], vector["sha256"], vector["name"])
               for vector in vectors]
    for path, expected, label in checks:
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")


def read_frames(path, width, height, count):
    import numpy as np

    frame_bytes = width * height * 3 // 2
    data = np.fromfile(path, dtype=np.uint8, count=frame_bytes * count)
    if data.size != frame_bytes * count:
        raise ValueError(f"{path} has fewer than {count} frames")
    return data.reshape(count, frame_bytes)


def crop_frames(frames, width, height, crop):
    """Crops planar 4:2:0 frames at an even offset and size."""
    import numpy as np

    x, y, w, h = crop["x"], crop["y"], crop["width"], crop["height"]
    if min(x, y, w, h) < 0 or (x | y | w | h) & 1 or x + w > width or y + h > height:
        raise ValueError("crop must be even and inside the frame")
    output = []
    for frame in frames:
        luma = frame[:width * height].reshape(height, width)[y:y + h, x:x + w]
        chroma = frame[width * height:].reshape(2, height // 2, width // 2)[
            :, y // 2:(y + h) // 2, x // 2:(x + w) // 2]
        output.append(np.concatenate((luma.ravel(), chroma.ravel())))
    return np.stack(output)


def check_latent_alignment(width, height):
    """Upstream forward_one_frame matches the fused stage structure only without pad_for_y."""
    padded = [(size + 15) // 16 for size in (width, height)]
    if any(size % 4 for size in padded):
        raise ValueError("upstream forward_one_frame needs a 1/16 latent divisible by 4; "
                         "the fused path would pad it (pad_for_y), so stages would differ")


def to_444(frame, width, height):
    """Upstream ycbcr420_to_444_np: nearest-neighbour chroma replication."""
    import numpy as np

    y = frame[:width * height].reshape(1, height, width)
    uv = frame[width * height:].reshape(2, height // 2, width // 2)
    uv = uv.repeat(2, axis=1).repeat(2, axis=2)
    return np.concatenate((y, uv), axis=0)


def model_input(frames_444, dtype, device, padding_b, padding_r):
    """Upstream get_src_frame ops, then the fused path's replicate padding."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.concatenate(frames_444, axis=0)[None]).to(device)
    x = x.half() / 255.0 - 0.5 if dtype == torch.float16 else x.float() / 255.0 - 0.5
    return F.pad(x, (0, padding_r, 0, padding_b), mode="replicate")


def to_8bit(x_hat, width, height):
    """Upstream save path: clamp, round luma, 2x2-average and truncate chroma."""
    import torch
    import torch.nn.functional as F

    x = (x_hat[:, :, :height, :width].float() + 0.5).clamp(0, 1)
    y = torch.clamp(x[:, :1] * 255, 0, 255).round().byte()
    uv = torch.clamp(F.avg_pool2d(x[:, 1:], 2, 2) * 255, 0, 255).byte()
    return torch.cat((y.flatten(1), uv.flatten(1)), 1)[0].cpu().numpy()


def psnr(reference, actual, width, height):
    import numpy as np

    plane = width * height
    mse = np.mean((reference[:plane].astype(np.float64) - actual[:plane]) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


class StageRecorder:
    """Forward hooks recording each named sub-network call's tensors."""

    def __init__(self, model, names, prefix):
        self.calls = []
        self.handles = []
        for name in names:
            module = getattr(model, name)
            self.handles.append(module.register_forward_hook(self._hook(f"{prefix}.{name}")))

    def _hook(self, name):
        def record(_module, inputs, output):
            self.calls.append((name, [t.detach() for t in flatten(inputs)],
                               [t.detach() for t in flatten(output)]))
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()


def flatten(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for element in value for item in flatten(element)]
    return []


def wrap_prior(model, store):
    """Records upstream forward_prior_4x outputs (y_res, y_q, y_hat, scales)."""
    original = model.forward_prior_4x

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        store.append([t.detach() for t in result if t is not None])
        return result

    model.forward_prior_4x = recorded


def load_models(args, dtype, device):
    import torch
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC
    from src.utils.common import ModelStructure, get_state_dict

    image = DMCI().eval()
    image.load_state_dict(get_state_dict(args.image_checkpoint))
    image.update(0.0)
    video = DMC(model_structure=ModelStructure.HTS).eval()
    video.load_state_dict(get_state_dict(args.video_checkpoint))
    video.update(0.0)
    image, video = image.to(device), video.to(device)
    if dtype == torch.float16:  # upstream test_video.py finalize_model
        image = image.half().to(memory_format=torch.channels_last)
        video = video.half().to(memory_format=torch.channels_last)
    return image, video


def run_pytorch(args, frames_444, dtype, padding_b, padding_r):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda:0")
    image, video = load_models(args, dtype, device)
    records = {"calls": [], "latents": [], "reconstructions": []}
    with torch.no_grad():
        recorder = StageRecorder(image, IMAGE_STAGES, "image")
        wrap_prior(image, records["latents"])
        x = model_input(frames_444[:1], dtype, device, padding_b, padding_r)
        qp = torch.tensor([args.qp], device=device)  # upstream training-path QP index
        intra = image.forward_one_frame(x, qp)
        recorder.close()
        records["calls"] += recorder.calls
        records["reconstructions"].append(intra["x_hat"])

        video.clear_dpb()
        video.ref_feature = F.pixel_unshuffle(intra["x_hat"], 8)
        wrap_prior(video, records["latents"])
        for group, reset in enumerate(GROUP_RESETS):
            recorder = StageRecorder(video, VIDEO_STAGES, f"video.g{group}")
            first = 1 + group * GROUP_FRAMES
            x = model_input(frames_444[first:first + GROUP_FRAMES], dtype, device,
                            padding_b, padding_r)
            predicted = video.forward_one_frame(x, qp, reset_feature_memory=reset)
            recorder.close()
            records["calls"] += recorder.calls
            records["reconstructions"] += list(predicted["x_hat"])
    torch.cuda.synchronize()
    return records


def run_fused(args, frames_444, padding_b, padding_r):
    """test_video.py's compress-then-decompress sequence for the vector's frames."""
    import torch

    device = torch.device("cuda:0")
    # test_video.py encodes the sequence, then decodes it with the same models.
    image, video = load_models(args, torch.float16, device)
    height, width = frames_444[0].shape[1:]
    sps = {"height": height, "width": width}

    def source(first, count):
        return model_input(frames_444[first:first + count], torch.float16, device,
                           0, 0).contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        encoded = image.compress(source(0, 1), args.qp, padding_b, padding_r)
        video.clear_dpb()
        video.add_ref_feature_from_frame(encoded["x_hat"])
        groups = [video.compress(source(1 + index * GROUP_FRAMES, GROUP_FRAMES), args.qp,
                                 reset, padding_b, padding_r)
                  for index, reset in enumerate(GROUP_RESETS)]

        # Fused outputs are borrowed proxy buffers that the next call overwrites;
        # upstream consumes them immediately, so snapshot each one.
        intra = image.decompress(encoded["bit_stream"], sps, args.qp,
                                 encoded["ec_parallel"])["x_hat"]
        reconstructions = [intra.clone()]
        video.clear_dpb()
        video.add_ref_feature_from_frame(intra, apply_feature_adaptor=False)
        for group, reset in zip(groups, GROUP_RESETS):
            reconstructions += [frame.clone() for frame in video.decompress(
                group["bit_stream"], sps, args.qp, group["ec_parallel"], reset)["x_hat"]]
    torch.cuda.synchronize()
    return {"reconstructions": reconstructions,
            "bytes": {"intra": len(encoded["bit_stream"]),
                      "predicted": [len(group["bit_stream"]) for group in groups]}}


def tensor_error(reference, actual):
    import torch

    reference = reference.float()
    actual = actual.float()
    if reference.shape != actual.shape:
        return {"shape_mismatch": [list(reference.shape), list(actual.shape)]}
    delta = (reference - actual).abs()
    scale = reference.abs().max().item()
    return {"shape": list(reference.shape), "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(), "reference_max_abs": scale,
            "finite": bool(torch.isfinite(actual).all().item())}


def symbol_agreement(reference, actual):
    reference = reference.float().round()
    actual = actual.float().round()
    differing = (reference != actual).sum().item()
    return {"symbols": reference.numel(), "differing": int(differing),
            "fraction_differing": differing / reference.numel()}


def capture_vector(args, vector, output_dir):
    import numpy as np
    import torch
    from src.models.common_model import CompressionModel

    width, height = vector["width"], vector["height"]
    raw = read_frames(args.dataset_root / vector["path"], width, height, VECTOR_FRAMES)
    if "crop" in vector:
        raw = crop_frames(raw, width, height, vector["crop"])
        width, height = vector["crop"]["width"], vector["crop"]["height"]
    padding_r, padding_b = CompressionModel.get_padding_size(height, width, 16)
    check_latent_alignment(width, height)
    frames_444 = [to_444(frame, width, height) for frame in raw]

    fp32 = run_pytorch(args, frames_444, torch.float32, padding_b, padding_r)
    fp16 = run_pytorch(args, frames_444, torch.float16, padding_b, padding_r)
    fused = run_fused(args, frames_444, padding_b, padding_r)

    stages = []
    for (name, in32, out32), (name16, _in16, out16) in zip(fp32["calls"], fp16["calls"]):
        if name != name16:
            raise RuntimeError(f"stage order differs between precisions: {name} vs {name16}")
        stages.append({"stage": name,
                       "inputs": [list(t.shape) for t in in32],
                       "outputs": [tensor_error(a, b) for a, b in zip(out32, out16)]})
    latents = []
    labels = ["intra"] + [f"group{index}" for index in range(len(GROUP_RESETS))]
    for label, latent32, latent16 in zip(labels, fp32["latents"], fp16["latents"]):
        latents.append({"frame": label,
                        "y_q_fp16_vs_fp32": symbol_agreement(latent32[1], latent16[1]),
                        "y_hat": tensor_error(latent32[2], latent16[2])})

    frames = []
    for index in range(VECTOR_FRAMES):
        source = raw[index]
        rec32 = to_8bit(fp32["reconstructions"][index], width, height)
        rec16 = to_8bit(fp16["reconstructions"][index], width, height)
        rec_fused = to_8bit(fused["reconstructions"][index], width, height)
        frames.append({
            "frame": index,
            "psnr_y_vs_source": {"pytorch_fp32": psnr(source, rec32, width, height),
                                 "pytorch_fp16": psnr(source, rec16, width, height),
                                 "fused": psnr(source, rec_fused, width, height)},
            "psnr_y_vs_fused": {"pytorch_fp32": psnr(rec_fused, rec32, width, height),
                                "pytorch_fp16": psnr(rec_fused, rec16, width, height)},
        })

    stored = None
    if width * height <= args.max_tensor_pixels:
        arrays = {}
        for precision, records in (("fp32", fp32), ("fp16", fp16)):
            for call, (name, inputs, outputs) in enumerate(records["calls"]):
                for kind, tensors in (("in", inputs), ("out", outputs)):
                    for position, tensor in enumerate(tensors):
                        arrays[f"{precision}/{call:03d}/{name}/{kind}{position}"] = (
                            tensor.float().cpu().numpy())
        for index, recon in enumerate(fused["reconstructions"]):
            arrays[f"fused/reconstruction/{index}"] = recon.float().cpu().numpy()
        path = output_dir / f"{vector['name']}_qp{args.qp}.npz"
        np.savez_compressed(path, **arrays)
        stored = {"path": path.name, "sha256": sha256(path), "arrays": len(arrays)}

    return {"name": vector["name"], "sha256": vector["sha256"], "crop": vector.get("crop"),
            "width": width,
            "height": height, "padding": {"bottom": padding_b, "right": padding_r},
            "qp": args.qp, "group_resets": list(GROUP_RESETS),
            "fused_bytes": fused["bytes"], "stages": stages,
            "latents": latents, "frames": frames, "tensors": stored}


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory must be new: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    vectors = manifest["stage_vectors"]
    if args.vector:
        vectors = [vector for vector in vectors if vector["name"] in set(args.vector)]
        if len(vectors) != len(set(args.vector)):
            raise SystemExit("unknown stage vector name")
    commit = verify_reference(args.reference_root)
    verify_inputs(args, vectors)

    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch
    from src.utils.common import set_torch_env

    set_torch_env()
    # Upstream init_func: fused CUDA-graph capture cannot use the legacy stream.
    torch.cuda.set_stream(torch.cuda.Stream(device=torch.device("cuda:0")))
    torch.utils.deterministic.fill_uninitialized_memory = True  # as the C7 probes
    args.output_dir.mkdir(parents=True)
    report = {
        "schema": "nvcr.dcvcuf.stage_vectors.v1",
        "upstream": {"commit": commit, "model_structure": "hts"},
        "checkpoints": {"image": args.image_checkpoint_sha256,
                        "video": args.video_checkpoint_sha256},
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
        "skip_thres": 0.0,
        "vectors": [capture_vector(args, vector, args.output_dir) for vector in vectors],
    }
    verify_reference(args.reference_root)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[stages] wrote {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
