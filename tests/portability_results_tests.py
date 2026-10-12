#!/usr/bin/env python3
"""Checks for compact portability statistics and binary/harness provenance."""
import copy
import json
import os
import shutil
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from consolidate_portability_results import GOPS, PROFILES, QPS, consolidate


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.image = {"Config": {"Labels": {"org.opencontainers.image.revision": "image-revision",
                       "org.opencontainers.image.version": "1.0.7"}}, "RepoDigests": ["image@sha256:test"]}
        self.rows = []
        for p, size in PROFILES.items():
            w, h = map(int, size.split("x"))
            for q in QPS:
                for g in GOPS:
                    for op in ("encode", "decode"):
                        for i in range(1, 11):
                            fps = 2 + i / 1000
                            self.rows.append(dict(schema="nvcr.benchmark.resolution-matrix.v1",
                                container_digest="image@sha256:test", execution_mode="docker",
                                run_index=str(i), size=size, resolution=p, frames=100,
                                warmup_frames=10, runs_planned=10, memory_sampler="none",
                                peak_memory_mb=None, peak_gpu_memory_mb=None, peak_system_memory_mb=None,
                                codec_time_seconds=round(100 / fps, 3), throughput_fps=fps,
                                payload_bytes=1000, payload_bpp=round(8000 / (w * h * 100), 9),
                                qp=q, gop_size=g, operation=op, psnr_yuv=35 if op=="decode" else None,
                                hardware="laptop", nvcr_commit="harness-revision", nvcr_dirty=True))

    def test_statistics_use_repetitions_and_image_identity(self):
        original = copy.deepcopy(self.rows)
        averaged = self.rows[0] | {"run_index": "average", "throughput_fps": 999}
        records = consolidate(self.rows + [averaged], self.image)
        self.assertEqual(len(records), 144)
        r = records[0]
        self.assertEqual(r["nvcr_commit"], "image-revision")
        self.assertNotIn("nvcr_dirty", r)
        self.assertEqual(r["codec_fps_statistics"]["n"], 10)
        self.assertAlmostEqual(r["codec_fps_statistics"]["mean"], 2.0055)
        self.assertEqual(self.rows, original)

    def test_missing_and_duplicate_runs_fail(self):
        for rows in (self.rows[1:], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                consolidate(rows, self.image)

    def test_incompatible_metric_and_image_fail(self):
        for field, value in (("container_digest", "another-image"), ("memory_sampler", "procfs"),
                             ("payload_bpp", 1), ("throughput_fps", float("nan"))):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                consolidate(rows, self.image)



    @unittest.skipUnless(shutil.which("bash"), "Bash required")
    def test_bash_launcher_separates_binary_from_dirty_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text("""#!/usr/bin/env bash
if [[ "$1" == inspect ]]; then exit 1; fi
if [[ "$1" == image ]]; then
  if [[ "$*" == *org.opencontainers.image.revision* ]]; then
    echo published-binary-revision
  else
    echo example/nvcr@sha256:test
  fi
elif [[ "$1" == run ]]; then
  printf '%s\\n' "$@"
else
  exit 2
fi
""", encoding="utf-8")
            git = bin_dir / "git"
            git.write_text("""#!/usr/bin/env bash
if [[ "$*" == *status* ]]; then echo '?? harness-change'; else echo harness-revision; fi
""", encoding="utf-8")
            for p in (docker, git):
                p.chmod(0o755)
            env = os.environ | {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
            (root / "version.txt").write_text("1.0.7", encoding="utf-8")
            launcher = root / "scripts/benchmark_docker.sh"
            launcher.parent.mkdir()
            launcher.write_text((Path(__file__).resolve().parents[1] /
                "scripts/benchmark_docker.sh").read_text(), encoding="utf-8")
            result = subprocess.run(["bash", str(launcher), "--image", "example/nvcr:1.0.7", "--skip-pull",
                "--input-dir", tmp, "--results-dir", str(root / "results"),
                "--output-dir", str(root / "streams")], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for value in ("NVCR_BENCH_COMMIT=published-binary-revision", "NVCR_BENCH_DIRTY=null",
                          "NVCR_BENCH_HARNESS_COMMIT=harness-revision", "NVCR_BENCH_HARNESS_DIRTY=true"):
                self.assertIn(value, result.stdout.splitlines())



    @unittest.skipUnless(shutil.which("bash"), "Bash required")
    def test_matrix_json_preserves_distinct_provenance(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/benchmark_resolution_matrix.sh").read_text()
        function = "append_row() {" + source.split("append_row() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = f"""jsonl={root / 'rows.jsonl'}
csv={root / 'rows.csv'}
report={root / 'report.md'}
hardware=laptop; execution_mode=docker; container_image=image; container_digest=digest
commit=published; dirty=null; harness_commit=harness; harness_dirty=true
frames=100; warmup_frames=10; repetitions=10; qp=0
csv_escape() {{ printf '%s' "$1"; }}
"""
            run = subprocess.run(["bash"], input=setup + function +
                "append_row encode qcif /input 176x144 30 1 qcif 1 1000 1 100 1 100 null 0.003 null null null null none 0\n",
                text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            row = json.loads((root / "rows.jsonl").read_text())
            self.assertEqual(row["nvcr_commit"], "published")
            self.assertIsNone(row["nvcr_dirty"])
            self.assertEqual(row["harness_commit"], "harness")
            self.assertIs(row["harness_dirty"], True)
            self.assertEqual(row["throughput_fps"], 100)


if __name__ == "__main__":
    unittest.main()
