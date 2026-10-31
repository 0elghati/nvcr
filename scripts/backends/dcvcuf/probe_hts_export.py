#!/usr/bin/env python3
"""Bounded DCVC-UF HT-S reconstruction export probe, not a codec adapter.

Exports the shuffle/clamp primitive and one frame branch. Checks ONNX CPU
execution against pinned PyTorch source; optionally checks TensorRT via trtexec.
Synthetic weights are opt-in and never establish codec/reference compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
PINNED_CUTLASS_COMMIT = "4370102f9dacab813282e1d67722fceb0b90a019"  # v4.4.1
ATOL = 1e-5
RTOL = 1e-4


def sha256(path):
    with path.open("rb") as stream:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--feature", type=Path, help="Captured float32 NCHW .npy feature")
    parser.add_argument("--feature-sha256")
    parser.add_argument("--branch", type=int, choices=range(8), default=7)
    parser.add_argument("--feature-height", type=int, default=2)
    parser.add_argument("--feature-width", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trtexec", type=Path, help="Also run a bounded FP32 TensorRT probe")
    parser.add_argument("--cutlass-root", type=Path,
                        help="Build and compare the actual upstream fused CUDA shuffle kernel")
    args = parser.parse_args(argv)
    if bool(args.checkpoint) != bool(args.checkpoint_sha256):
        parser.error("--checkpoint and --checkpoint-sha256 must be supplied together")
    if bool(args.feature) != bool(args.feature_sha256):
        parser.error("--feature and --feature-sha256 must be supplied together")
    if min(args.feature_height, args.feature_width) <= 0:
        parser.error("feature dimensions must be positive")
    return args


def compare(expected, actual, *, exact=False):
    import numpy as np

    if (expected.shape != actual.shape or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()):
        return {"passed": False, "reason": "shape mismatch or nonfinite output"}
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "passed": bool(np.array_equal(expected, actual) if exact else
                       np.allclose(expected, actual, atol=ATOL, rtol=RTOL)),
        "exact": bool(np.array_equal(expected, actual)),
        "max_abs_error": float(delta.max()),
        "shape": list(actual.shape),
    }


def compare_fused_shuffle(root, cutlass, output, value):
    import torch
    from torch.utils.cpp_extension import load

    commit = verify_reference(cutlass, PINNED_CUTLASS_COMMIT)
    if not torch.cuda.is_available():
        raise ValueError("--cutlass-root requires CUDA; no CPU fallback")
    build = output / "fused-build"
    build.mkdir()
    binding = build / "binding.cpp"
    # Compile the original translation unit, not a copied/reimplemented kernel.
    binding.write_text("""#include <torch/extension.h>
at::Tensor pixel_shuffle_8_cuda(const at::Tensor&, bool, at::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shuffle", &pixel_shuffle_8_cuda);
}
""")
    major, minor = torch.cuda.get_device_capability()
    module = load(
        name="nvcr_dcvcuf_shuffle_probe", build_directory=str(build),
        sources=[str(binding), str(root / "src/layers/extensions/inference/elementwise/shuffle.cu")],
        extra_include_paths=[str(cutlass / "include")],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-DCURRENT_DEVICE_SM={major * 10 + minor}"],
    )
    checks = {}
    for name, layout in (("contiguous", torch.contiguous_format),
                         ("channels_last", torch.channels_last)):
        x = value.to(device="cuda", dtype=torch.float16, memory_format=layout)
        expected = torch.nn.functional.pixel_shuffle(x, 8).clamp(-0.5, 0.5)
        actual = torch.empty_like(expected, memory_format=layout)
        module.shuffle(x, True, actual)
        torch.cuda.synchronize()
        checks[name] = compare(expected.cpu().numpy(), actual.cpu().numpy(), exact=True)
    return {"cutlass_commit": commit, "gpu": torch.cuda.get_device_name(),
            "torch_cuda": torch.version.cuda, "checks": checks}


def main(argv=None):
    args = parse_args(argv)
    root = args.reference_root.resolve()
    commit = verify_reference(root)
    for path, digest in ((args.checkpoint, args.checkpoint_sha256),
                         (args.feature, args.feature_sha256)):
        if path and sha256(path) != digest:
            raise ValueError(f"SHA-256 mismatch: {path}")
    # A fresh directory prevents failed reruns from leaving stale passing evidence.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn
    from torch.nn import functional as F

    sys.path.insert(0, str(root))
    from src.models.video_model_ht import ReconHead
    from src.utils.common import get_state_dict

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    head = ReconHead(is_hts=True).eval()
    if args.checkpoint:
        state = get_state_dict(args.checkpoint)
        head.load_state_dict({key.removeprefix("recon_head."): value
                              for key, value in state.items()
                              if key.startswith("recon_head.")}, strict=True)

    class ShuffleClamp(nn.Module):
        def forward(self, x):
            # Inference is centered at zero; the upstream CUDA kernel clamps here.
            return F.pixel_shuffle(x, 8).clamp(-0.5, 0.5)

    class FrameBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.common = head.conv1[args.branch // 2]
            self.frame = head.conv2[args.branch]
            self.shuffle = ShuffleClamp()

        def forward(self, feature):
            pre_shuffle = self.frame(self.common(feature))
            # Branch 7's unclamped pre-shuffle output is also the reset feature.
            return pre_shuffle, self.shuffle(pre_shuffle)

    if args.feature:
        feature_array = np.load(args.feature, allow_pickle=False)
        if (feature_array.dtype != np.float32 or feature_array.ndim != 4
                or feature_array.shape[:2] != (1, 512)
                or min(feature_array.shape[2:]) <= 0 or not np.isfinite(feature_array).all()):
            raise ValueError("feature must be finite float32 [1, 512, H>0, W>0]")
        feature = torch.from_numpy(np.ascontiguousarray(feature_array))
    else:
        feature = torch.randn(1, 512, args.feature_height, args.feature_width)
    height, width = feature.shape[2:]
    # Exact binary fractions exercise channel placement and both clamp boundaries.
    shuffle_input = ((torch.arange(192 * height * width) % 1024) - 512).float()
    shuffle_input = (shuffle_input / 256).reshape(1, 192, height, width)
    branch = FrameBranch().eval()

    report = {
        "schema_version": 1,
        "scope": "dcvcuf_hts_stage_export_probe",
        "reference_commit": commit,
        "probe_sha256": sha256(Path(__file__)),
        "weights": "checkpoint" if args.checkpoint else "synthetic",
        "checkpoint_sha256": args.checkpoint_sha256,
        "feature_source": "captured" if args.feature else "synthetic",
        "feature_sha256": args.feature_sha256,
        "seed": args.seed,
        "branch": args.branch,
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__},
        "acceptance": {"shuffle": "exact", "branch_atol": ATOL, "branch_rtol": RTOL},
        "c7_complete": False,
        "remaining_gates": ["upstream fused CUDA operator comparison",
                            "pretrained upstream encode/decode vector and entropy parity",
                            "provider-stage comparison on captured reconstruction/reset features"],
        "stages": {},
    }
    with torch.inference_mode():
        if args.cutlass_root:
            report["fused_shuffle"] = compare_fused_shuffle(
                root, args.cutlass_root.resolve(), output, shuffle_input)
            if all(item["passed"] for item in report["fused_shuffle"]["checks"].values()):
                report["remaining_gates"].remove("upstream fused CUDA operator comparison")
        # Independent full-head selection catches the wrong shared pair or frame branch.
        pre_shuffle, frame = branch(feature)
        torch.testing.assert_close(frame, head(feature)[args.branch].clamp(-0.5, 0.5),
                                   rtol=0, atol=0)
        if args.branch == 7:
            torch.testing.assert_close(pre_shuffle, head(feature, for_reset=True), rtol=0, atol=0)

        stages = (("shuffle_clamp", ShuffleClamp(), shuffle_input, ["frame"]),
                  ("hts_frame_branch", branch, feature, ["pre_shuffle", "frame"]))
        for name, module, value, names in stages:
            model_path = output / f"{name}.onnx"
            expected = module(value)
            if not isinstance(expected, tuple):
                expected = (expected,)
            expected = [item.numpy() for item in expected]
            torch.onnx.export(module, (value,), str(model_path), input_names=["input"],
                              output_names=names, opset_version=18, dynamo=False)
            onnx.checker.check_model(str(model_path))
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            actual = session.run(names, {"input": value.numpy()})
            entry = {
                "onnx_sha256": sha256(model_path),
                "input_shape": list(value.shape),
                "onnx_cpu": {key: compare(ref, out, exact=name == "shuffle_clamp")
                             for key, ref, out in zip(names, expected, actual, strict=True)},
            }
            input_path = output / f"{name}.input.f32"
            value.numpy().tofile(input_path)
            entry["input_sha256"] = sha256(input_path)
            for key, ref in zip(names, expected, strict=True):
                ref.tofile(output / f"{name}.{key}.expected.f32")
            if args.trtexec:
                trt_output = output / f"{name}.tensorrt"
                trt_output.mkdir()
                command = [str(args.trtexec.resolve()), f"--onnx={model_path}",
                           f"--loadInputs=input:{input_path}", "--dumpRawBindingsToFile",
                           "--noTF32", "--builderOptimizationLevel=0", "--memPoolSize=workspace:64",
                           "--warmUp=0", "--duration=0", "--iterations=1"]
                entry["trtexec_command"] = command
                with (output / f"{name}.trtexec.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            cwd=trt_output, timeout=180, check=False)
                if result.returncode != 0:
                    entry["tensorrt"] = {"execution": {"passed": False,
                                                        "returncode": result.returncode}}
                else:
                    entry["tensorrt"] = {}
                    for key, ref in zip(names, expected, strict=True):
                        dimensions = ".".join(str(size) for size in ref.shape)
                        raw_path = trt_output / f"{key}.output.{dimensions}.fp32.raw"
                        actual = np.fromfile(raw_path, dtype=np.float32).reshape(ref.shape)
                        entry["tensorrt"][key] = compare(ref, actual, exact=name == "shuffle_clamp")
                        entry["tensorrt"][key]["output_sha256"] = sha256(raw_path)
            report["stages"][name] = entry

    passed = all(check["passed"] for stage in report["stages"].values()
                 for backend in ("onnx_cpu", "tensorrt")
                 for check in stage.get(backend, {}).values())
    passed = passed and all(check["passed"] for check in
                            report.get("fused_shuffle", {}).get("checks", {}).values())
    report["status"] = "stage_probe_passed" if passed else "stage_probe_failed"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']}; C7 remains open. Evidence: {output / 'report.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
