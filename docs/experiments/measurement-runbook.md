# Measurement execution runbook

Run from `/home/oelghati/nvcr` on the intended GPU host. The default manifest is
local to this checkout and its known input locations. It retains the configured
four QPs and ten repetitions. For other hosts, copy it and change only paths,
reference interpreter, execution identity and the exact target/engine bundle.
Do not mix existing observations from different hosts or metric definitions.

The NVCR campaign and smoke passed; full comparison validation is **pending**. Read
[the validation record](measurement-readiness/validation.md) before launching.
Published bundles pass offline integrity/profile checks. The model-profile digest
difference was CRLF versus LF; the checker preserves an exact matching LF copy
without changing original files. The registered Orin marketing-name alias is
explicitly recorded while all device/runtime checks remain enforced. The
completed NVCR campaign has no allocation failures. The reference CUDA failure
and TensorRT device-model warnings remain unresolved.

## NVCR-only launch with downloaded bundles

For NVCR-only execution, run from the checkout root:

```bash
bash scripts/measurement_nvcr.sh
```

This uses the existing Release executable and installed bundles under
`build/engines-measurement-published`. It performs preflight and an eight-operation
smoke first, then the full NVCR matrix if smoke passes. It does not download,
rebuild or execute the Python codec. All six sequences, 100 frames, GOPs 1/30/100,
QPs 0/21/42/63 and ten repetitions per throughput/memory condition are retained.
There is one separate quality pass per condition, giving 3,024 full operations
(encode and decode counted separately). Python codec dependencies do not gate
an NVCR-only run.

Results are in `evidence/measurement-nvcr/`, including `observations.jsonl`,
`analysis.json` (mean/sample SD) and `rd-points.json`. Logs are saved in
`evidence/measurement-assets/logs/`. Use `bash scripts/measurement_nvcr.sh resume`
for compatible interrupted work, `check` for a dry-run, or `analyze` to regenerate
summaries. A failed smoke stops before the full campaign.

## Full NVCR-versus-Python path on this Jetson

All further builds, validation and measurements are launched by the operator.
The convenience entry point saves a separate log on every invocation and stops
on failure. It preserves existing model/engine bundles and datasets:

```bash
cd /home/oelghati/nvcr
bash scripts/measurement_jetson.sh prepare
bash scripts/measurement_jetson.sh smoke
bash scripts/measurement_jetson.sh run
```

Run each step only after the previous one succeeds. `prepare` builds the changed
Release CLI, runs focused tests and **downloads the six published Orin engine
bundles** using the existing catalog installer. It does not export models,
rebuild TensorRT engines or rebuild Python extensions. `smoke` runs only the
bounded validation. `run` performs the configured campaign and generates statistics.

The live [engine-assets release](https://github.com/0elghati/nvcr/releases/tag/engine-assets)
and [catalog](https://github.com/0elghati/nvcr/releases/download/engine-assets/nvcr-engine-catalog.json)
were checked on 2026-09-19. All six Orin entries specify exact matching for
AArch64 Linux, Orin SM 8.7 with 8 SMs, CUDA 12.6 and TensorRT 10.3.0.
The downloaded manifests still undergo provenance validation; catalog matching
alone is not a completed encode/decode smoke.

Use `bash scripts/measurement_jetson.sh check` for preflight/dry-run after
preparation, `resume` for compatible interrupted work, and `analyze` to regenerate
summaries. Logs live in `evidence/measurement-assets/logs/`; raw smoke directories
are unique and a successful path is saved automatically. No shell environment
setup is needed for the default launcher path. If selecting diagnostic rebuilt
Python extensions manually, keep that PYTHONPATH unchanged across runs. The
launcher is syntax checked; installation and end-to-end execution remain unverified.

If running remotely and the local graphical display is not needed, its login
service can be stopped explicitly before measurement:

```bash
sudo systemctl stop gdm3
# Restore the local login screen after measurements:
# sudo systemctl start gdm3
```

This stops the local graphical login service. Close unnecessary editor sessions
yourself before measuring; the launcher does not terminate connections, delete
files, clear caches, or change power/clock settings. Available system RAM does not establish CUDA allocation capacity.

## Build and focused verification

Use the current authorized source checkout; keep the Release CMake settings:

```bash
cd /home/oelghati/nvcr
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target nvcr_cli nvcr_measurement_wire_fixture -j 2
ctest --test-dir build-release --output-on-failure \
  -R 'nvcr_measurement_semantics|nvcr_legacy_consolidation|nvcr_softwarex_driver|nvcr_contract_tests|nvcr_format_contract_tests|nvcr_dcvcrt_payloads|nvcr_cli_accepts_inter_gop'
```

NumPy is required by the offline quality evaluator; SciPy is required only for
optional BD integration. The selected reference interpreter must contain a
CUDA-enabled PyTorch and native inference/rANS extensions for the pinned source.
No automatic package installation or host reconfiguration occurs.

## Capture and preflight

```bash
python3 scripts/measurement_campaign.py capture \
  --output evidence/measurement-environment/environment.json

python3 scripts/measurement_campaign.py preflight \
  --output evidence/measurement-preflight

python3 scripts/measurement_campaign.py run --dry-run \
  --output evidence/measurement-dry-run
```

Each output directory must be new or empty. Preflight exits nonzero with exact
failed prerequisites and preserves an environment/preflight JSON. Dry-run writes
`plan.json` and `commands.txt` without starting any measurement. It initializes
CUDA only for capability checks. It does not load models or encode video. The
configured full campaign has 6,048 operations: 5,760 throughput/memory operations
plus 288 quality operations, across both implementations. No full campaign was
started during implementation.

## Select published bundles without changing historical identities

The default registered target profile remains the build-provenance source. The
measurement checker records its known `Jetson Orin Nano` / CUDA `Orin` name alias
and still checks architecture, SM capability/count and exact runtime versions.
A CRLF-to-LF copy is accepted only when its SHA256 exactly matches the published
model-profile digest. Original profiles and bundle manifests remain untouched.

If installation is needed on another matching host:

```bash
python3 scripts/nvcr_artifacts.py install \
  --repo 0elghati/nvcr --asset-release engine-assets \
  --profile qcif cif 360p 540p 720p 1080p \
  --engine-root build/engines-measurement-published
```

All six downloaded bundles in this checkout have now passed offline integrity
and profile validation against the previously captured host identity. The user
launcher repeats live device checks before GPU execution.

## Optional Python extension diagnosis

This is separate from engine installation and is not part of `prepare`.
Only use it when investigating the previously observed reference CUDA error.

The current installed Python CUDA extension imports successfully but fails the
pinned QCIF operation with `CUDA error: misaligned address`. To distinguish an
extension-build mismatch from a pinned-source defect, build the pinned extensions
into an isolated local directory and select them explicitly (no installation over
the existing interpreter):

```bash
export NVCR_REFERENCE_EXTENSIONS="$PWD/evidence/measurement-assets/reference-extensions"
mkdir -p "$NVCR_REFERENCE_EXTENSIONS"
(
  cd assets/src/cpp
  /home/oelghati/DCVC-RT/.venv-jetson/bin/python setup.py build_ext \
    --build-lib "$NVCR_REFERENCE_EXTENSIONS" \
    --build-temp "$NVCR_REFERENCE_EXTENSIONS/build-rans"
)
(
  cd assets/src/layers/extensions/inference
  /home/oelghati/DCVC-RT/.venv-jetson/bin/python setup.py build_ext \
    --build-lib "$NVCR_REFERENCE_EXTENSIONS" \
    --build-temp "$NVCR_REFERENCE_EXTENSIONS/build-inference"
)
export PYTHONPATH="$NVCR_REFERENCE_EXTENSIONS${PYTHONPATH:+:$PYTHONPATH}"
```

A rebuild is a diagnostic prerequisite, not an established fix for the observed
misalignment. If it persists, the optimized reference implementation needs a
separately verified alignment repair with pinned-source patch provenance. Do not
silently fall back to CPU or disable its optimized CUDA path to obtain a result.

## Bounded smoke and full launch

After prerequisite repair, use the generated local manifest in every command:

```bash
python3 scripts/measurement_campaign.py smoke \
  --manifest evidence/measurement-assets/campaign.json \
  --output evidence/measurement-smoke
```

Smoke selects the first configured sequence and QP, the first inter GOP, three
frames, two warm-up frames, and one execution per mode. These are explicitly
smoke-only overrides; the full manifest is unchanged. Encode and decode are
separate for both implementations. Additional cold quality passes check stream
and reconstruction hashes against warmed/reset passes (16 total operations,
180-second timeout per operation). `analysis.json` must be complete and every
reset check must pass. A preflight failure or mock/unit-test pass is not this
hardware gate.

```bash
python3 scripts/measurement_campaign.py run \
  --manifest evidence/measurement-assets/campaign.json \
  --smoke-evidence evidence/measurement-smoke \
  --output evidence/measurement-campaign

# Resume only the SAME build, inputs, artifacts and environment:
python3 scripts/measurement_campaign.py run --resume \
  --manifest evidence/measurement-assets/campaign.json \
  --smoke-evidence evidence/measurement-smoke \
  --output evidence/measurement-campaign

python3 scripts/measurement_campaign.py analyze \
  --output evidence/measurement-campaign
```

The full launch checks prerequisites again and requires a complete compatible
smoke package, revalidated from its raw observations. Keep the same `PYTHONPATH` when
using isolated rebuilt extensions. `observations.jsonl` retains individual
operations and failures; `analysis.json` holds mean/sample-SD results and
completion accounting; `rd-points.json` holds common quality/rate points. A
failure, timeout, missing observation or incompatible resume returns nonzero.
The campaign never consumes legacy average rows as observations. Keep the local
raw evidence directory and failed products; do not commit large binaries or raw
runs. Commit only a current compact validation/publication summary as appropriate.

## Optional RD integration

Create a JSON with `quality_contract` equal to
`decoded-yuv420p8-pooled-plane-6-1-1-v1`, `rate_definition` equal to `file_bpp` or
`entropy_bpp`, and `reference`/`candidate` arrays of `[rate, quality]` points for
one matched source, GOP and frame range. Supply increasing rate/quality order
explicitly. Only finite complete observations from the same measurement contract
are eligible. Then run:

```bash
python3 scripts/measurement_campaign.py bd-rate \
  --curves /path/to/matched-curves.json \
  --output evidence/measurement-campaign/bd-rate.json
```

The result records the integration bounds and SciPy version. No BD result is
produced for a nonmonotonic, incomplete, infinite or non-overlapping curve.
