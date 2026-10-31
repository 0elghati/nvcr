#!/usr/bin/env python3
"""Compare the pinned HT-S reconstruction provider proxy with PyTorch.

This is a provider-stage vector only. It loads the real HT-S reconstruction
weights, feeds one deterministic feature tensor through the upstream Python
head and the upstream fused CUDA proxy, and records every branch plus the
unclamped reset feature. It does not register a codec or establish bitstream
interchangeability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
ATOL = 3e-3
RTOL = 3e-3


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_reference(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if commit != PINNED_REFERENCE_COMMIT:
        raise ValueError(f"expected upstream {PINNED_REFERENCE_COMMIT}, found {commit}")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("reference checkout must be clean, including untracked source files")
    return commit


def compare(expected, actual):
    import numpy as np

    expected = expected.astype(np.float32, copy=False)
    actual = actual.astype(np.float32, copy=False)
    if expected.shape != actual.shape or not np.isfinite(expected).all() \
            or not np.isfinite(actual).all():
        return {"passed": False, "reason": "shape mismatch or nonfinite output",
                "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    scale = np.abs(expected.astype(np.float64))
    return {
        "passed": bool(np.allclose(expected, actual, atol=ATOL, rtol=RTOL)),
        "exact": bool(np.array_equal(expected, actual)),
        "max_abs_error": float(delta.max()),
        "max_rel_error": float((delta / np.maximum(scale, 1e-12)).max()),
        "shape": list(actual.shape),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-height", type=int, default=3)
    parser.add_argument("--feature-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)
    if min(args.feature_height, args.feature_width) <= 0:
        parser.error("feature dimensions must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.reference_root.resolve()
    commit = verify_reference(root)
    checkpoint = args.checkpoint.resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError(f"SHA-256 mismatch: {checkpoint}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("provider parity requires CUDA")

    sys.path.insert(0, str(root))
    from src.models.video_model_ht import ReconHead
    from src.utils.common import get_state_dict

    import inference_extensions_cuda

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    state = get_state_dict(checkpoint)
    head = ReconHead(is_hts=True).eval()
    head.load_state_dict({key.removeprefix("recon_head."): value
                          for key, value in state.items()
                          if key.startswith("recon_head.")}, strict=True)
    head = head.half().cuda().to(memory_format=torch.channels_last)

    feature = torch.randn(
        1, 512, args.feature_height, args.feature_width,
        device="cuda", dtype=torch.float16,
    ).to(memory_format=torch.channels_last)
    feature_path = output / "feature.f16.npy"
    np.save(feature_path, feature.cpu().numpy())

    proxy = inference_extensions_cuda.DMCHTSReconHeadProxy()
    proxy.set_param({key: value for key, value in head.state_dict().items()})
    pre_shuffle_out = torch.empty(
        1, 192, args.feature_height, args.feature_width,
        device="cuda", dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    proxy.pre_allocate_tensors(feature, pre_shuffle_out)
    with torch.inference_mode():
        expected_frames = [frame.clamp(-0.5, 0.5) for frame in head(feature)]
        expected_reset = head(feature, for_reset=True)
        actual_frames, actual_reset = proxy.forward(feature)
        actual_reset_direct = proxy.forward_reset(feature)
        torch.cuda.synchronize()

    frame_checks = [compare(expected.cpu().numpy(), actual.cpu().numpy())
                    for expected, actual in zip(expected_frames, actual_frames, strict=True)]
    reset_check = compare(expected_reset.cpu().numpy(), actual_reset.cpu().numpy())
    reset_direct_check = compare(expected_reset.cpu().numpy(), actual_reset_direct.cpu().numpy())
    report = {
        "schema_version": 1,
        "scope": "dcvcuf_hts_provider_stage_parity",
        "reference_commit": commit,
        "probe_sha256": sha256(Path(__file__)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "feature_sha256": sha256(feature_path),
        "feature_dtype": "float16",
        "feature_shape": list(feature.shape),
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "acceptance": {"atol": ATOL, "rtol": RTOL,
                        "frame_outputs_clamped": True,
                        "reset_output_unclamped": True},
        "frames": frame_checks,
        "reset": reset_check,
        "reset_direct": reset_direct_check,
        "status": "provider_stage_passed"
        if all(item["passed"] for item in frame_checks)
        and reset_check["passed"] and reset_direct_check["passed"]
        else "provider_stage_failed",
        "c7_complete": False,
        "remaining_gates": ["NVCR provider implementation parity",
                            "pretrained upstream encode/decode vector and entropy parity",
                            "codec-owned I/chunk/tail/reset lifecycle"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}; C7 remains open. Evidence: {output / 'report.json'}")
    return 0 if report["status"] == "provider_stage_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
