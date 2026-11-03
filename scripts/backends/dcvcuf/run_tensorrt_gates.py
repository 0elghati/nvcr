#!/usr/bin/env python3
"""C11 fidelity gates: DCVC-UF TensorRT against the native fused path.

Encodes and decodes every sequence and QP of the reference set with both NVCR
paths (upstream recipe: intra only at frame 0, reset interval 32, skip 0) and
reports, per point, payload bytes and mean per-frame PSNR. The native path is
byte-identical to upstream test_video.py, so its PSNR must equal the upstream
reference report; TensorRT is then held to the C11 G3 limits against it:
mean PSNR-Y within 0.05 dB, bits within 1%, and BD-rate per class within 0.5%.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

QPS = (0, 21, 42, 63)


def frame_psnr(reference, decoded, width, height, frames):
    size = width * height
    planes = ((0, size), (size, size // 4), (size + size // 4, size // 4))
    rows = []
    with open(reference, "rb") as a, open(decoded, "rb") as b:
        for _ in range(frames):
            x = np.frombuffer(a.read(size * 3 // 2), np.uint8).astype(np.float64)
            y = np.frombuffer(b.read(size * 3 // 2), np.uint8).astype(np.float64)
            psnr = []
            for start, count in planes:
                mse = np.mean((x[start:start + count] - y[start:start + count]) ** 2)
                psnr.append(100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse))
            rows.append(psnr)
    return np.array(rows)


def run_point(args, path, sequence, qp, work):
    binary, bundle = (args.native, args.native_bundle) if path == "native" else (args.tensorrt, args.tensorrt_bundle)
    source = args.dataset_root / sequence["path"]
    stream = work / f"{path}.nvcr"
    decoded = work / f"{path}.yuv"
    size = f"{sequence['width']}x{sequence['height']}"
    common = ["--codec", "dcvc-uf", "--engine-dir", str(bundle)]
    encoded = subprocess.run([str(binary), "encode", *common, "-i", str(source), "-o", str(stream), "-s", size,
                              "--frames", str(sequence["frames"]), "--qp", str(qp), "--gop-size", "65528"],
                             capture_output=True, text=True, check=True).stdout
    subprocess.run([str(binary), "decode", *common, "-i", str(stream), "-o", str(decoded)],
                   capture_output=True, text=True, check=True)
    payload = int(next(line for line in encoded.splitlines() if line.startswith("Encoded")).split()[3])
    psnr = frame_psnr(source, decoded, sequence["width"], sequence["height"], sequence["frames"])
    decoded.unlink()
    weighted = (6 * psnr[:, 0] + psnr[:, 1] + psnr[:, 2]) / 8
    return {"payload_bytes": payload, "psnr_y": float(psnr[:, 0].mean()),
            "psnr_yuv": float(weighted.mean()), "frames": sequence["frames"]}


def bd_rate(anchor, test):
    """Bjontegaard delta rate (%) of test against anchor; points are (bits, psnr)."""
    def fit(points):
        rate = np.log([p[0] for p in points])
        psnr = np.array([p[1] for p in points])
        return np.polyfit(psnr, rate, 3), psnr.min(), psnr.max()
    pa, lo_a, hi_a = fit(anchor)
    pt, lo_t, hi_t = fit(test)
    lo, hi = max(lo_a, lo_t), min(hi_a, hi_t)
    ia, it = np.polyint(pa), np.polyint(pt)
    diff = ((np.polyval(it, hi) - np.polyval(it, lo)) - (np.polyval(ia, hi) - np.polyval(ia, lo))) / (hi - lo)
    return float((math.exp(diff) - 1) * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--tensorrt", type=Path, required=True)
    parser.add_argument("--tensorrt-bundle", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, default=Path(__file__).with_name("c8_reference_sequences.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = {s["name"]: {p["qp"]: p for p in s["points"]}
                 for s in json.loads(args.reference_report.read_text())["sequences"]}
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for sequence in json.loads(args.sequences.read_text())["sequences"]:
        for qp in QPS:
            point = {}
            for path in ("native", "tensorrt"):
                point[path] = run_point(args, path, sequence, qp, args.output)
            upstream = reference[sequence["name"]][qp]
            point["upstream_psnr_y"] = upstream["ave_all_frame_psnr_y"]
            point["native_minus_upstream_psnr_y"] = point["native"]["psnr_y"] - upstream["ave_all_frame_psnr_y"]
            point["tensorrt_minus_native_psnr_y"] = point["tensorrt"]["psnr_y"] - point["native"]["psnr_y"]
            point["tensorrt_bits_ratio"] = point["tensorrt"]["payload_bytes"] / point["native"]["payload_bytes"]
            results.setdefault(sequence["name"], {"class": sequence["class"], "points": {}})["points"][qp] = point
            print(f"{sequence['name']} qp {qp}: native {point['native']['payload_bytes']} B "
                  f"{point['native']['psnr_y']:.4f} dB (upstream {upstream['ave_all_frame_psnr_y']:.4f}); "
                  f"tensorrt {point['tensorrt']['payload_bytes']} B {point['tensorrt']['psnr_y']:.4f} dB; "
                  f"dPSNR-Y {point['tensorrt_minus_native_psnr_y']:+.4f} dB, "
                  f"bits {100 * (point['tensorrt_bits_ratio'] - 1):+.2f}%", flush=True)
            (args.output / "gates.json").write_text(json.dumps(results, indent=2) + "\n")

    classes = {}
    for name, sequence in results.items():
        points = sequence["points"].values()
        rate = bd_rate([(p["native"]["payload_bytes"], p["native"]["psnr_yuv"]) for p in points],
                       [(p["tensorrt"]["payload_bytes"], p["tensorrt"]["psnr_yuv"]) for p in points])
        sequence["bd_rate_percent"] = rate
        classes.setdefault(sequence["class"], []).append(rate)
    summary = {"classes": {name: float(np.mean(rates)) for name, rates in classes.items()},
               "max_abs_dpsnr_y": max(abs(p["tensorrt_minus_native_psnr_y"])
                                      for s in results.values() for p in s["points"].values()),
               "max_abs_bits_percent": max(abs(100 * (p["tensorrt_bits_ratio"] - 1))
                                           for s in results.values() for p in s["points"].values()),
               "max_abs_native_minus_upstream_psnr_y": max(abs(p["native_minus_upstream_psnr_y"])
                                                           for s in results.values() for p in s["points"].values())}
    summary["g3_pass"] = (summary["max_abs_dpsnr_y"] <= 0.05 and summary["max_abs_bits_percent"] <= 1.0
                          and all(abs(v) <= 0.5 for v in summary["classes"].values()))
    results["summary"] = summary
    (args.output / "gates.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
