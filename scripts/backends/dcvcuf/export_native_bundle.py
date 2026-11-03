#!/usr/bin/env python3
"""Convert pinned DCVC-UF checkpoints into one digest-bound native runtime bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from export_weights_artifact import export_state_dict


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.reference_root.resolve()
    sys.path.insert(0, str(root))
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC
    from src.utils.common import get_state_dict, set_torch_env

    import torch
    set_torch_env()
    image = DMCI().eval()
    image.load_state_dict(get_state_dict(args.image_checkpoint.resolve()))
    image.update(0)
    image = image.half().to(memory_format=torch.channels_last)

    video = DMC().eval()
    video.load_state_dict(get_state_dict(args.video_checkpoint.resolve()))
    video.update(0)
    video = video.half().to(memory_format=torch.channels_last)

    image_state = image.add_cdf_to_state_dict(image.state_dict())
    video_state = video.add_cdf_to_state_dict(video.state_dict())
    combined = {f"image.{name}": tensor for name, tensor in image_state.items()}
    combined.update({f"video.{name}": tensor for name, tensor in video_state.items()})
    artifact = export_state_dict(
        combined, args.output.resolve(), preserve_channels_last=True)
    manifest = {
        "schema": "nvcr.dcvcuf.bundle.v1",
        "codec": "dcvc-uf",
        "model_set": "dcvcuf-cvpr2026-hts",
        "upstream_commit": "cbdae87a5445114cdc7f48816da63ea80bdeac40",
        "image_checkpoint": {
            "path": str(args.image_checkpoint.resolve()),
            "sha256": sha256(args.image_checkpoint),
        },
        "video_checkpoint": {
            "path": str(args.video_checkpoint.resolve()),
            "sha256": sha256(args.video_checkpoint),
        },
        "artifact": artifact,
        "image_tensor_count": len(image_state),
        "video_tensor_count": len(video_state),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
