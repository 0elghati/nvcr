#!/usr/bin/env python3
"""Check DCVC-UF TensorRT stages in isolation against upstream reference tensors (C8 phase 2).

Every stage call recorded by capture_stage_vectors.py (upstream PyTorch FP32)
becomes one case: its recorded FP32 inputs, rounded to FP16, go through the
TensorRT plan via NVCR's codec-agnostic TensorRT session
(nvcr_tensorrt_stage_runner) and through upstream's own PyTorch module in its
FP16 channels-last inference configuration. Both are compared with the FP32
outputs. Upstream FP16 is the band a faithful FP16 port is expected to fall in;
a stage is flagged when TensorRT's error exceeds twice that band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorrt_stages  # noqa: E402

PINNED_REFERENCE_COMMIT = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
BAND_FACTOR = 2.0


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--image-checkpoint-sha256", required=True)
    parser.add_argument("--video-checkpoint", type=Path, required=True)
    parser.add_argument("--video-checkpoint-sha256", required=True)
    parser.add_argument("--stage-vectors", type=Path, required=True,
                        help="NPZ written by capture_stage_vectors.py")
    parser.add_argument("--stage-vectors-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--stage", action="append",
                        help="Check only these stages (the bundle may hold only these)")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def recorded_calls(arrays):
    """Returns the FP32 calls in order as (module, inputs, outputs)."""
    calls = {}
    for key in arrays.files:
        if not key.startswith("fp32/"):
            continue
        _precision, index, name, slot = key.split("/")
        kind, position = ("in", int(slot[2:])) if slot.startswith("in") else ("out", int(slot[3:]))
        entry = calls.setdefault(int(index), {"name": name, "in": {}, "out": {}})
        entry[kind][position] = arrays[key]
    ordered = []
    for index in sorted(calls):
        entry = calls[index]
        module = entry["name"].split(".")[-1]
        family = entry["name"].split(".")[0]
        ordered.append((f"{family}.{module}",
                        [entry["in"][i] for i in sorted(entry["in"])],
                        [entry["out"][i] for i in sorted(entry["out"])]))
    return ordered


def pixel_unshuffle_8(x):
    """Inverse of the 8x pixel shuffle, in torch's channel order."""
    n, c, h, w = x.shape
    return (x.reshape(n, c, h // 8, 8, w // 8, 8).transpose(0, 1, 3, 5, 2, 4)
            .reshape(n, c * 64, h // 8, w // 8))


def stage_cases(calls):
    """Maps recorded module calls onto the exported stages."""
    import numpy as np

    cases = []
    index = 0
    while index < len(calls):
        name, inputs, outputs = calls[index]
        following = [call[0] for call in calls[index + 1:index + 4]]

        def case(stage, case_inputs, case_outputs, consumed=1):
            cases.append({"stage": stage, "inputs": case_inputs, "outputs": case_outputs,
                          "label": f"{len(cases):02d}_{stage}"})
            return consumed

        if name == "image.enc":
            used = case("i_encoder", {"frame": inputs[0], "q_enc": inputs[1]}, {"y": outputs[0]})
        elif name == "image.hyper_enc":
            used = case("i_hyper_encoder", {"y_padded": inputs[0]}, {"z": outputs[0]})
        elif name == "image.hyper_dec":
            assert following[:2] == ["image.y_prior_fusion", "image.y_spatial_prior_reduction"]
            used = case("i_hyper_decoder", {"z_hat": inputs[0]},
                        {"params": calls[index + 1][2][0], "common": calls[index + 2][2][0]}, 3)
        elif name.startswith("image.y_spatial_prior_adaptor_"):
            assert following[0] == "image.y_spatial_prior"
            joined = inputs[0]
            used = case(f"i_spatial_prior_{name[-1]}",
                        {"y_hat": joined[:, :256], "common": joined[:, 256:]},
                        {"scales_means": calls[index + 1][2][0]}, 2)
        elif name == "image.dec":
            used = case("i_decoder", {"y_hat": inputs[0], "q_dec": inputs[1]},
                        {"frame_pre_shuffle": pixel_unshuffle_8(outputs[0])})
        elif name == "video.feature_adaptor_i":
            used = case("p_feature_adaptor_i", {"feature": inputs[0]}, {"memory": outputs[0]})
        elif name == "video.feature_adaptor_m":
            used = case("p_feature_adaptor_m", {"memory": inputs[0], "feature": inputs[1]},
                        {"memory_out": outputs[0]})
        elif name == "video.feature_extractor":
            used = case("p_feature_extractor", {"memory": inputs[0]}, {"context": outputs[0]})
        elif name == "video.encoder":
            used = case("p_encoder",
                        {"frames": inputs[0], "context": inputs[1], "q_encoder": inputs[2]},
                        {"y": outputs[0]})
        elif name == "video.hyper_encoder":
            used = case("p_hyper_encoder", {"y_padded": inputs[0]}, {"z": outputs[0]})
        elif name == "video.temporal_prior_encoder":
            assert following == ["video.hyper_decoder", "video.y_prior_fusion",
                                 "video.y_spatial_prior_reduction"]
            hyper, fusion, reduction = calls[index + 1], calls[index + 2], calls[index + 3]
            case("p_hyper_decoder", {"z_hat": hyper[1][0]}, {"hyper_params": hyper[2][0]}, 0)
            used = case("p_prior",
                        {"hyper_params": fusion[1][0], "memory": inputs[0],
                         "q_feature": inputs[1]},
                        {"params": fusion[2][0], "common": reduction[2][0]}, 4)
        elif name.startswith("video.y_spatial_prior_adaptor_"):
            assert following[0] == "video.y_spatial_prior"
            used = case(f"p_spatial_prior_{name[-1]}", {"y_hat": inputs[0], "common": inputs[1]},
                        {"means": calls[index + 1][2][0]}, 2)
        elif name == "video.decoder":
            used = case("p_decoder",
                        {"y_hat": inputs[0], "context": inputs[1], "q_decoder": inputs[2]},
                        {"feature": outputs[0]})
        elif name == "video.recon_head" and len(outputs) == 8:
            used = case("p_reconstruction", {"feature": inputs[0]},
                        {f"frame_{index}_pre_shuffle": pixel_unshuffle_8(output)
                         for index, output in enumerate(outputs)})
        elif name == "video.recon_head":
            used = case("p_reset", {"feature": inputs[0]}, {"reset_feature": outputs[0]})
        else:
            raise ValueError(f"unmapped upstream call {name}")
        index += used
    return cases


def write_job(cases, bundle, directory, model):
    import numpy as np

    lines = [f"bundle {bundle.resolve()}", f"model {model}"]
    contracts = json.loads((bundle / "stages.json").read_text(encoding="utf-8"))["stages"]
    for stage in sorted({case["stage"] for case in cases}):
        tensors = contracts[stage]
        lines.append(f"stage {stage}.plan {len(tensors)}")
        for tensor in tensors:
            fixed = tensor["profile"]["min"] == tensor["profile"]["max"]
            dims = tensor["profile"]["min"] if fixed else [1, tensor["channels"], -1, -1]
            lines.append(f"tensor {tensor['name']} {tensor['access']} {' '.join(map(str, dims))}")
    for case in cases:
        lines.append(f"run {case['stage']}.plan {len(case['inputs'])} {len(case['outputs'])}")
        formats = {tensor["name"]: tensor.get("format", "chw")
                   for tensor in contracts[case["stage"]]}
        for name, value in case["inputs"].items():
            path = directory / f"{case['label']}.{name}.in.f16"
            stored = value.transpose(0, 2, 3, 1) if formats[name] == "hwc8" else value
            np.ascontiguousarray(stored, dtype=np.float16).tofile(path)
            lines.append(f"input {name} {' '.join(map(str, value.shape))} {path.resolve()}")
        for name in case["outputs"]:
            lines.append(f"output {name} {(directory / (case['label'] + '.' + name + '.out.f16')).resolve()}")
    job = directory / "job.txt"
    job.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return job


def error(reference, actual):
    import numpy as np

    if reference.shape != actual.shape:
        return {"shape_mismatch": [list(reference.shape), list(actual.shape)]}
    delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "reference_peak": float(np.abs(reference).max()),
            "reference_mean_abs": float(np.abs(reference).mean()),
            "finite": bool(np.isfinite(actual).all())}


def pytorch_fp16(modules, case):
    import torch

    module = modules[case["stage"]]
    inputs = [torch.from_numpy(value.astype("float16")).cuda() for value in case["inputs"].values()]
    with torch.no_grad():
        outputs = module(*inputs)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    return [output.float().cpu().numpy() for output in outputs]


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory must be new: {args.output_dir}")
    import export_tensorrt_stages as export

    export.verify_reference(args.reference_root)
    export.verify_inputs(args)
    if sha256(args.stage_vectors) != args.stage_vectors_sha256:
        raise ValueError("stage vector SHA-256 mismatch")
    import numpy as np
    import torch

    sys.path.insert(0, str(args.reference_root.resolve()))
    work = args.output_dir / "tensors"
    work.mkdir(parents=True)
    cases = stage_cases(recorded_calls(np.load(args.stage_vectors)))
    if args.stage:
        cases = [case for case in cases if case["stage"] in set(args.stage)]
    job = write_job(cases, args.bundle, work, export.MODEL_PROFILE_ID)
    completed = subprocess.run([str(args.runner), str(job)], capture_output=True, text=True)
    (args.output_dir / "runner.log").write_text(completed.stdout + completed.stderr,
                                                encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"stage runner failed: {completed.stderr[-2000:]}")
    shapes = [line.split()[1:] for line in completed.stdout.splitlines()
              if line.startswith("shape ")]
    contracts = json.loads((args.bundle / "stages.json").read_text(encoding="utf-8"))["stages"]

    image, video = export.load_models(args)
    modules = tensorrt_stages.modules(image.half().cuda().to(memory_format=torch.channels_last),
                                      video.half().cuda().to(memory_format=torch.channels_last))
    results = []
    shape_index = 0
    for case in cases:
        baseline = pytorch_fp16(modules, case)
        outputs = []
        for (name, reference), torch_output in zip(case["outputs"].items(), baseline):
            _, *dims = shapes[shape_index]
            shape_index += 1
            path = work / f"{case['label']}.{name}.out.f16"
            dims = [int(d) for d in dims]
            layout = next(t.get("format", "chw") for t in contracts[case["stage"]]
                          if t["name"] == name)
            if layout == "hwc8":  # NHWC bytes for NCHW logical dims
                actual = np.fromfile(path, dtype=np.float16).reshape(
                    [dims[0], dims[2], dims[3], dims[1]]).transpose(0, 3, 1, 2)
            else:
                actual = np.fromfile(path, dtype=np.float16).reshape(dims)
            trt = error(reference, actual.astype(np.float32))
            torch16 = error(reference, torch_output)
            within = (trt.get("finite", False) and "shape_mismatch" not in trt and
                      trt["mean_abs"] <= BAND_FACTOR * torch16["mean_abs"] + 1e-6 and
                      trt["max_abs"] <= BAND_FACTOR * torch16["max_abs"] + 1e-6)
            outputs.append({"output": name, "tensorrt_fp16": trt, "upstream_fp16": torch16,
                            "within_fp16_band": bool(within)})
        results.append({"case": case["label"], "stage": case["stage"],
                        "input_shapes": {k: list(v.shape) for k, v in case["inputs"].items()},
                        "outputs": outputs})

    report = {
        "schema": "nvcr.dcvcuf.tensorrt_stage_check.v1",
        "upstream": PINNED_REFERENCE_COMMIT,
        "stage_vectors_sha256": args.stage_vectors_sha256,
        "bundle_manifest_sha256": sha256(args.bundle / "engine_manifest.json"),
        "runner_sha256": sha256(args.runner),
        "band_factor": BAND_FACTOR,
        "cases": results,
        "all_within_fp16_band": all(o["within_fp16_band"] for r in results for o in r["outputs"]),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"[check] {len(results)} cases; all within FP16 band: {report['all_within_fp16_band']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
