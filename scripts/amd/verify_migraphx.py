#!/usr/bin/env python3
"""Verify that ONNX Runtime can actually place work on the MIGraphX execution
provider on this machine. Fails loudly if the EP is missing, if the session
silently falls back to CPU, or if the EP cannot run a trivial graph.

Issue #224 acceptance criterion: "Verify with a trivial ONNX model that
MIGraphXExecutionProvider is listed and actually used, not silently falling
back to CPU."

Usage:
    python3 scripts/amd/verify_migraphx.py
    python3 scripts/amd/verify_migraphx.py --json  # machine-readable output

Exit code 0 on success, 1 on any failure. Prints a report that can be pasted
into the issue or redirected into an evidence file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def build_trivial_model() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w_init = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4],
        [1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0],
    )
    b_init = helper.make_tensor("b", TensorProto.FLOAT, [4], [0.5] * 4)
    matmul = helper.make_node("MatMul", ["x", "w"], ["xw"])
    add = helper.make_node("Add", ["xw", "b"], ["y"])
    graph = helper.make_graph([matmul, add], "trivial", [x], [y], [w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def run(args: argparse.Namespace) -> int:
    report: dict[str, object] = {
        "ok": False,
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION", ""),
    }

    try:
        import onnxruntime as ort
    except ImportError as exc:
        report["error"] = f"onnxruntime not importable: {exc}"
        emit(report, args.json)
        return 1
    try:
        import numpy as np  # noqa: F401
    except ImportError as exc:
        report["error"] = f"numpy not importable: {exc}"
        emit(report, args.json)
        return 1
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        report["error"] = f"onnx not importable: {exc}"
        emit(report, args.json)
        return 1

    import numpy as np

    report["onnxruntime_version"] = ort.__version__
    report["available_providers"] = ort.get_available_providers()
    if "MIGraphXExecutionProvider" not in report["available_providers"]:
        report["error"] = (
            "MIGraphXExecutionProvider missing from ort.get_available_providers(). "
            "The installed onnxruntime wheel does not expose the MIGraphX EP."
        )
        emit(report, args.json)
        return 1

    model_bytes = build_trivial_model()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "trivial.onnx"
        model_path.write_bytes(model_bytes)

        options = ort.SessionOptions()
        options.log_severity_level = 1
        try:
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["MIGraphXExecutionProvider"],
            )
        except Exception as exc:
            report["error"] = f"InferenceSession creation raised: {exc!r}"
            emit(report, args.json)
            return 1

        used = session.get_providers()
        report["session_providers"] = used
        if used != ["MIGraphXExecutionProvider"]:
            report["error"] = (
                f"Session resolved to providers {used}, not MIGraphX alone. "
                "ORT silently fell back; the EP is not actually in use."
            )
            emit(report, args.json)
            return 1

        x = np.arange(4, dtype=np.float32).reshape(1, 4)
        try:
            (y,) = session.run(["y"], {"x": x})
        except Exception as exc:
            report["error"] = f"session.run raised on MIGraphX EP: {exc!r}"
            emit(report, args.json)
            return 1
        report["input"] = x.tolist()
        report["output"] = y.tolist()
        expected = (x + 0.5).tolist()
        if y.tolist() != expected:
            report["error"] = (
                f"MIGraphX EP produced {y.tolist()}, expected {expected}. "
                "Numerics are wrong on a trivial identity graph."
            )
            emit(report, args.json)
            return 1

    report["ok"] = True
    emit(report, args.json)
    return 0


def emit(report: dict, as_json: bool) -> None:
    if as_json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    width = max(len(k) for k in report) + 2
    for key in sorted(report):
        value = report[key]
        if isinstance(value, list):
            value = ", ".join(repr(v) for v in value)
        sys.stdout.write(f"{(key + ':').ljust(width)} {value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
