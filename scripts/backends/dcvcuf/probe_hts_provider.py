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
import json
from pathlib import Path
import sys

from probe_hts_export import PINNED_REFERENCE_COMMIT, sha256, verify_reference
ATOL = 3e-3
RTOL = 3e-3


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
        "mean_abs_error": float(delta.mean()),
        "different_elements": int(np.count_nonzero(delta)),
        "outside_tolerance": int(np.count_nonzero(delta > ATOL + RTOL * scale)),
        "shape": list(actual.shape),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True,
                        help="Unmodified upstream inference_extensions_cuda shared library")
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--feature", type=Path, help="Optional float16 NCHW .npy vector")
    parser.add_argument("--feature-sha256")
    parser.add_argument("--feature-height", type=int, default=3)
    parser.add_argument("--feature-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args(argv)
    if min(args.feature_height, args.feature_width) <= 0:
        parser.error("feature dimensions must be positive")
    if bool(args.feature) != bool(args.feature_sha256):
        parser.error("--feature and --feature-sha256 must be supplied together")
    return args


def load_proxy(root, extension, output):
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME must identify the toolkit used by the upstream extension")
    binding = Path(__file__).with_name("provider_probe.cpp").resolve()
    build = output / "binding-build"
    build.mkdir()
    return load(
        name="nvcr_dcvcuf_provider_probe", sources=[str(binding)],
        build_directory=str(build),
        extra_include_paths=[str(root / "src/layers/extensions/inference"),
                             str(root / "src/cpp/py_rans"), str(Path(CUDA_HOME) / "include")],
        extra_ldflags=[str(extension)], extra_cflags=["-O2"], with_cuda=True,
    )


def diagnose_first_operation(binding, head, feature):
    """Hold FP16 inputs/weights fixed and isolate the first conv + activation."""
    import torch
    from torch.nn import functional as F

    def array(tensor):
        return tensor.detach().float().cpu().numpy()

    layer = head.conv1[0][0].dc[0]
    plain = layer(feature)
    python_activation = head.conv1[0][0].dc[1](plain)
    fused_conv = binding.conv1x1(feature, layer.weight, layer.bias, False)
    fused_activation = binding.conv1x1(feature, layer.weight, layer.bias, True)
    # Remove bias/activation entirely to isolate the matrix accumulation.
    python_no_bias = F.conv2d(feature, layer.weight)
    fused_no_bias = binding.conv1x1(feature, layer.weight, torch.zeros_like(layer.bias), False)
    # Float64 CPU oracle uses the SAME quantized FP16 weights/input. It isolates
    # evaluation/rounding order rather than changing the model checkpoint.
    oracle_conv = F.conv2d(feature.double().cpu(), layer.weight.double().cpu(),
                          layer.bias.double().cpu())
    oracle_activation = (oracle_conv * torch.sigmoid(4 * oracle_conv)).half()
    return {
        "location": "recon_head.conv1.0.0.dc: convolution and WSiLU",
        "no_bias_python_vs_fused": compare(array(python_no_bias), array(fused_no_bias)),
        "conv_python_vs_fused": compare(array(plain), array(fused_conv)),
        "activation_python_vs_fused": compare(array(python_activation), array(fused_activation)),
        "oracle_vs_python_conv": compare(array(oracle_conv.half()), array(plain)),
        "oracle_vs_fused_conv": compare(array(oracle_conv.half()), array(fused_conv)),
        "oracle_vs_python_activation": compare(array(oracle_activation), array(python_activation)),
        "oracle_vs_fused_activation": compare(array(oracle_activation), array(fused_activation)),
    }


def main(argv=None):
    args = parse_args(argv)
    root = args.reference_root.resolve()
    commit = verify_reference(root)
    checkpoint = args.checkpoint.resolve()
    extension = args.extension.resolve()
    for path, digest in ((checkpoint, args.checkpoint_sha256),
                         (extension, args.extension_sha256), (args.feature, args.feature_sha256)):
        if path and sha256(path) != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("provider parity requires CUDA")

    sys.path.insert(0, str(root))
    from src.models.video_model_ht import ReconHead
    from src.utils.common import get_state_dict

    binding = load_proxy(root, extension, output)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    state = get_state_dict(checkpoint)
    head = ReconHead(is_hts=True).eval()
    head.load_state_dict({key.removeprefix("recon_head."): value
                          for key, value in state.items()
                          if key.startswith("recon_head.")}, strict=True)
    head = head.half().cuda().to(memory_format=torch.channels_last)

    if args.feature:
        value = np.load(args.feature, allow_pickle=False)
        if (value.dtype != np.float16 or value.ndim != 4 or value.shape[:2] != (1, 512)
                or min(value.shape[2:]) <= 0 or not np.isfinite(value).all()):
            raise ValueError("feature must be finite float16 [1,512,H>0,W>0]")
        feature = torch.from_numpy(value.copy()).cuda()
    else:
        feature = torch.randn(1, 512, args.feature_height, args.feature_width,
                              device="cuda", dtype=torch.float16)
    feature = feature.to(memory_format=torch.channels_last)
    feature_path = output / "feature.f16.npy"
    np.save(feature_path, feature.cpu().numpy())

    proxy = binding.ReconHead()
    proxy.set_param({key: value for key, value in head.state_dict().items()})
    pre_shuffle_out = torch.empty(
        1, 192, *feature.shape[2:],
        device="cuda", dtype=torch.float16,
        memory_format=torch.channels_last,
    )
    proxy.pre_allocate_tensors(feature, pre_shuffle_out)
    with torch.inference_mode():
        original_feature = feature.clone()
        expected_frames = [frame.clamp(-0.5, 0.5) for frame in head(feature)]
        expected_reset = head(feature, for_reset=True)
        actual_frames, actual_reset = proxy.forward(feature)
        # Proxy outputs are borrowed buffers; snapshot before any later call.
        actual_frames = [frame.clone() for frame in actual_frames]
        actual_reset = actual_reset.clone()
        actual_reset_direct = proxy.forward_reset(feature).clone()
        repeated_frames, repeated_reset = proxy.forward(feature)
        torch.cuda.synchronize()
        input_unchanged = torch.equal(feature, original_feature)
        repeat_exact = all(torch.equal(a, b) for a, b in
                           zip(actual_frames, repeated_frames, strict=True))
        repeat_exact = repeat_exact and torch.equal(actual_reset, repeated_reset)
        reset_exact = torch.equal(actual_reset, actual_reset_direct)
        reset_display_exact = torch.equal(
            torch.nn.functional.pixel_shuffle(actual_reset, 8).clamp(-0.5, 0.5),
            actual_frames[-1])
        diagnosis = diagnose_first_operation(binding, head, feature)

    frame_checks = [compare(expected.cpu().numpy(), actual.cpu().numpy())
                    for expected, actual in zip(expected_frames, actual_frames, strict=True)]
    reset_check = compare(expected_reset.cpu().numpy(), actual_reset.cpu().numpy())
    reset_direct_check = compare(expected_reset.cpu().numpy(), actual_reset_direct.cpu().numpy())
    tensors = {"expected_reset": expected_reset.cpu().numpy(),
               "fused_reset": actual_reset.cpu().numpy(),
               "fused_reset_direct": actual_reset_direct.cpu().numpy()}
    for index, (expected, actual) in enumerate(zip(expected_frames, actual_frames, strict=True)):
        tensors[f"expected_frame_{index}"] = expected.cpu().numpy()
        tensors[f"fused_frame_{index}"] = actual.cpu().numpy()
    tensor_path = output / "outputs.npz"
    np.savez(tensor_path, **tensors)
    report = {
        "schema_version": 1,
        "scope": "dcvcuf_hts_provider_stage_parity",
        "reference_commit": commit,
        "probe_sha256": sha256(Path(__file__)),
        "binding_sha256": sha256(Path(__file__).with_name("provider_probe.cpp")),
        "extension_sha256": args.extension_sha256,
        "outputs_sha256": sha256(tensor_path),
        "feature_source": "supplied" if args.feature else "synthetic",
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
        "lifecycle": {"input_unchanged": input_unchanged, "repeat_exact": repeat_exact,
                      "reset_exact": reset_exact, "reset_display_exact": reset_display_exact},
        "first_operation": diagnosis,
        "status": "provider_stage_passed"
        if all(item["passed"] for item in frame_checks)
        and reset_check["passed"] and reset_direct_check["passed"]
        and input_unchanged and repeat_exact and reset_exact and reset_display_exact
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
