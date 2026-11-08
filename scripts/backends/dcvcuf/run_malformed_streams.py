#!/usr/bin/env python3
"""Decodes make_malformed_streams.py variants and classifies each outcome (C13).

Run with an AddressSanitizer/UBSan build of the CLI and of
nvcr_dcvcuf_entropy_trace. A sanitizer report makes the process exit with
--sanitizer-exit-code; that, a signal or a timeout is a failure of the
malformed-stream check. A defined NVCR error is a clean rejection. A complete
decode is compared with the decode of the unmodified stream: equal frames, or
different frames that the format cannot detect (no integrity field).
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def run(command, timeout, env):
    try:
        done = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                              timeout=timeout, env=env)
        return done.returncode, done.stderr, False
    except subprocess.TimeoutExpired as expired:
        return None, (expired.stderr or b"").decode(errors="replace"), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True, help="directory with variants.json")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frame-bytes", type=int, required=True)
    parser.add_argument("--sanitizer-exit-code", type=int, default=99)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    code = args.sanitizer_exit_code
    env["ASAN_OPTIONS"] = f"protect_shadow_gap=0:detect_leaks=0:exitcode={code}:abort_on_error=0"
    env["UBSAN_OPTIONS"] = f"halt_on_error=1:print_stacktrace=1:exitcode={code}"
    work = args.output.parent / (args.output.stem + "-work")
    work.mkdir(parents=True, exist_ok=True)
    reference = work / "original.yuv"
    status, stderr, _ = run([args.cli, "decode", "--codec", "dcvc-uf", "--engine-dir", args.bundle,
                             "-i", args.original, "-o", reference], args.timeout, env)
    if status != 0:
        raise SystemExit(f"original stream does not decode: {stderr[-500:]}")
    original = reference.read_bytes()
    rows = []
    for variant in json.loads((args.variants / "variants.json").read_text()):
        stream = args.variants / variant["file"]
        cli_yuv, trace_yuv, trace = work / "cli.yuv", work / "trace.yuv", work / "trace.bin"
        for path in (cli_yuv, trace_yuv, trace):
            path.unlink(missing_ok=True)
        row = dict(variant)
        for tool, command, out in (
                ("cli", [args.cli, "decode", "--codec", "dcvc-uf", "--engine-dir", args.bundle,
                         "-i", stream, "-o", cli_yuv], cli_yuv),
                ("harness", [args.harness, args.bundle, stream, trace_yuv, trace], trace_yuv)):
            status, stderr, timed_out = run(command, args.timeout, env)
            sanitizer = ("AddressSanitizer" in stderr or "runtime error:" in stderr or
                         status == code)
            written = out.read_bytes() if out.exists() else b""
            frames = len(written) // args.frame_bytes
            prefix_ok = written == original[:len(written)]
            if timed_out or sanitizer or status is None or status < 0:
                outcome = "FAIL sanitizer/crash/timeout"
            elif status == 0:
                outcome = ("complete, identical" if written == original
                           else "complete, different (undetectable corruption)")
            elif (tool == "cli" and status == 1 and "nvcr: " in stderr) or \
                    (tool == "harness" and status == 3):
                outcome = "rejected" if prefix_ok else "rejected after different frames"
            elif tool == "harness" and status == 2:
                outcome = "harness container error"
            else:
                outcome = f"FAIL unexpected exit {status}"
            row[tool] = {"exit": status, "timed_out": timed_out, "sanitizer": sanitizer,
                         "frames_written": frames, "partial_bytes": len(written) % args.frame_bytes,
                         "written_prefix_equals_original": prefix_ok, "outcome": outcome,
                         "error": next((line for line in stderr.splitlines()
                                        if line.startswith(("nvcr: ", "unit ", "error: "))), None),
                         "sanitizer_report": stderr[-3000:] if sanitizer else None}
        rows.append(row)
        print(f"{variant['name']}: cli={row['cli']['outcome']} harness={row['harness']['outcome']}",
              flush=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    failures = [r for r in rows if r["cli"]["outcome"].startswith("FAIL") or
                r["harness"]["outcome"].startswith("FAIL")]
    print(f"{len(rows)} variants, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
