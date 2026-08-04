# Model and engine preparation

NVCR v1 prepares exactly one profile: `dcvcrt-cvpr2025`. It does not convert
arbitrary PyTorch models. Preparation is offline; deployed NVCR does not load
Python, PyTorch, or checkpoint files.

NVCR releases ship no checkpoints, ONNX graphs, entropy/quant assets, or TensorRT
plans. Users obtain the checkpoints under their applicable terms and build all
derived bundles locally.

## Bound source and checkpoints

The versioned model profile is
[configs/models/dcvcrt-cvpr2025.json](../configs/models/dcvcrt-cvpr2025.json):

```text
Source repository: https://github.com/0elghati/DCVC-RT.git
Source commit:     48ab0ac5e5199d78fffb944bfbafafb2b6142f7b
Image checkpoint:  cvpr2025_image.pth.tar
SHA-256:           555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9
Video checkpoint:  cvpr2025_video.pth.tar
SHA-256:           b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6
```

The checkpoint download locations are maintained by the upstream/model provider:

```text
https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8
https://1drv.ms/f/c/2866592d5c55df8c/EozfVVwtWWYggCitBAAAAAABbT4z2Z10fMXISnan72UtSA?e=BID7DA
```

Place the files at `<dcvcrt-root>/checkpoints/`. The exporter verifies the exact
Git commit and both checkpoint hashes; it does not trust filenames alone.

## Supported command

`scripts/nvcr_artifacts.py` is the public artifact front end. Its operations are:

- `prepare`: verify source/checkpoints, export the model bundle, and build engines;
- `build`: build engines from an existing model bundle;
- `inspect`: report identity and versions without execution;
- `validate`: validate schema, portable paths, required files, and SHA-256 digests.

The DCVC-RT preparation helper, the two exporters, the TensorRT builder, and the
manifest writer are retained under `scripts/backends/dcvcrt/` as backend-local
implementation helpers. Direct use is an expert/development path and does not
define another supported workflow.

## Artifact identity model

NVCR identifies prepared artifacts by the tuple
`(model_profile_id, target_profile_id, engine_profile_id)`.

- the model profile binds the pinned upstream commit, checkpoints, exporter
  inputs, and portable model-bundle identity;
- the target profile binds the validated deployment environment and runtime
  compatibility checks;
- the engine profile binds visible dimensions, optimization shapes, precision,
  workspace, and builder-level choices.

Resolution is therefore one field inside the engine profile, not the whole
portability contract. Do not manage or retain engine bundles by resolution name
alone.

## Prepare from checkpoints

Create a Python environment with PyTorch, ONNX, and ONNXScript, then select one
resolution profile. The registered target is auto-detected unless overridden:

```bash
/path/to/python -c 'import torch, onnx, onnxscript'

./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 1080p \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
```

The helper clones/checks out the pinned source by default when the target path is
absent. Use `--skip-clone` only for an already verified checkout. Use
`--skip-engine` to stop after portable model assets are ready.
Exactly one of `--profile NAME` or `--all` is required because TensorRT builds
are expensive. `--all --engines-root build/engines` creates one bundle per
registered resolution and reuses the model export.

The declared FP16 profiles are:

| Profile | Visible dimensions | Workspace | Builder level | Purpose |
|---|---|---:|---:|---|
| `qcif` | 64×64 to 176×144 | 512 MiB | 1 | Small correctness/development bundle |
| `cif` | 64×64 to 352×288 | 512 MiB | 1 | CIF correctness/development bundle |
| `360p` | fixed 640×360 | 1024 MiB | 4 | Edge latency specialization |
| `540p` | fixed 960×540 | 1024 MiB | 4 | Edge latency specialization |
| `720p` | 64×64 to 1280×720 | 1024 MiB | 2 | 720p target validation |
| `1080p` | 64×64 to 1920×1080 | 1024 MiB | 2 | Reference target validation |

Internal graph shapes may be padded (for example 1080 to 1088 lines); manifests
record user-visible dimensions and the engine builder records the actual TensorRT
profiles. FP16 is the only supported v1 precision. The lower-level INT8 flag is
experimental and must not be used for release evidence.

## Build engines from a model bundle

A validated model directory can be transferred to the final target. Build plans
there, selecting that target's profile:

```bash
./scripts/nvcr_artifacts.py validate build/models/dcvcrt --json

./scripts/nvcr_artifacts.py build \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --profile 1080p \
  --target-profile configs/targets/orin-nano-l4t3647.json \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt-1080p \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device-id 0
```

The TensorRT builder validates the complete model bundle before creating the
first plan. Legacy manifests, missing graphs, and graph/hash mismatches are
hard failures; do not bypass this check by mixing files from separate exports.

Never copy an RTX plan to Orin, or an engine across a different GPU model,
CUDA/TensorRT runtime, or model export. Rebuild on the final target.

If engine bundles are retained internally for CI or reviewer convenience, store
them under `dcvcrt-cvpr2025/<target-profile>/<engine-profile>/` rather than by
resolution name alone.

For the rolling GitHub engine release, package a validated bundle as:

```bash
./scripts/package_engine_bundle.sh \
  --engine-dir build/engines/dcvcrt-1080p \
  --output-dir dist
```

The stable archive name is derived from the target/model/profile identity:

```text
nvcr-engines-<target-profile>-dcvcrt-cvpr2025-<resolution>.tar.gz
```

The archive contains one `dcvcrt/` engine bundle plus
`ENGINE-ASSET-MANIFEST.sha256`. It remains a separate rolling GitHub Release asset,
not part of the generic `linux-x86_64-nvidia` or `linux-aarch64-jetson-l4t36`
binary packages.

If a future reviewer-convenience path adds compatibility-mode discrete-GPU
engines, treat them as a separate non-default engine class and record the chosen
compatibility mode in the engine manifest and support docs. Jetson/L4T is not
part of that portability path; build Jetson engines on the final target.

## Model bundle contract

`i_frame_manifest.json` and `p_frame_manifest.json` use
`nvcr.model-manifest.v2`. Together they bind:

- model profile, exporter version, source commit, checkpoint file/hash;
- PyTorch, ONNX, opset, precision, sample shapes, and QP;
- every ONNX graph and entropy/quant asset name, size, shape metadata, and hash;
- relative bundle filenames only—no host-specific absolute paths.

Validation requires all 14 graph files and four runtime assets and checks each
digest.

## Engine bundle contract

`nvcr.engine-bundle.v2` records:

- model, target, and engine-profile identities;
- SHA-256 digests of the exact model, target, and engine profile JSON files;
- optimization point, visible dimensions, FP16, workspace, and builder level;
- whether runtime-variable TensorRT axes use a `dynamic` or fully `fixed`
  shape profile;
- CUDA runtime, TensorRT version, GPU name, compute capability, and SM count;
- I/P model-manifest digests;
- every generated plan/runtime-asset hash;
- the digest of the relative `engine.sha256` checksum manifest.

A complete runtime bundle contains 14 plans and six copied runtime assets. Both
`nvcr-artifacts validate` and runtime initialization reject missing, extra,
modified, stale, cross-model, or incompatible bundles. Legacy v2 bundles are
treated as dynamic for compatibility. A fixed bundle must declare
`shape_profile: fixed`, and every runtime-variable axis in all 14 plans must be
concrete; partial fixed/dynamic hybrids are rejected.

```bash
./scripts/nvcr_artifacts.py inspect build/engines/dcvcrt --json
./scripts/nvcr_artifacts.py validate build/engines/dcvcrt --json
```

## Register integration tests

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=ON \
  -DNVCR_TENSORRT_ENGINE_DIR="$PWD/build/engines/dcvcrt"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

The engine and I/P roundtrip tests are absent from CTest when the engine-directory
option is omitted; an unconfigured CPU/CUDA-only suite is not end-to-end evidence.

## Clean-room rule

Release evidence uses empty model/engine/build directories and the selected
versioned profiles. Record all commands, hashes, tool versions, hardware, engine
build settings, test results, and reference/performance evidence in
[ROADMAP.md](../ROADMAP.md). Preserve failed or superseded results as labeled
history.

## Validated FP16 resolution profiles

The discrete-GPU release matrix contains four independently packaged TensorRT
profiles. Visible dimensions are the CLI/video dimensions; TensorRT profiles
also include the runtime padding required by the codec.

| Profile | Visible optimum | Release gate |
|---|---:|---|
| `qcif` | 176x144 | Contract plus native I/P roundtrip |
| `cif` | 352x288 | Contract plus native I/P roundtrip |
| `720p` | 1280x720 | Contract plus native I/P roundtrip |
| `1080p` | 1920x1080 | Contract plus native I/P roundtrip |

The fixed-shape `360p` and `540p` profiles are edge-performance
candidates. On Orin Nano, a conservative 13-fixed-engine candidate retained the
validated dynamic `p_synthesis.plan` while a stale source graph was
quarantined. Repeated BasketballDrive measurements improved 360p encode/decode
by 7.74%/3.65% and 540p encode/decode by 5.18%/2.35%. The 360p candidate and
540p encode clear the 3% candidate gate; 540p decode does not. See
`evidence/orin-fixed-edge-profiles-2026-08-02.json` for run order, hashes,
quality, and artifact provenance. This hybrid is retained as historical
performance evidence only: the hardened validator and runtime now reject it as
an incomplete fixed bundle. Fully fixed bundles remain pending a clean export
and locked-clock repetition.

The profile definitions and runtime support are architecture-neutral. Their
generated TensorRT plans remain target-specific, so Orin and desktop RTX use
separate builds of the same profile. Existing dynamic profiles remain valid and
retain their previous runtime behavior.

Configure `NVCR_TENSORRT_ENGINE_DIR` for one primary bundle and provide any
additional bundles through semicolon-separated `NVCR_TENSORRT_ENGINE_DIRS`.
CTest registers a profile-labeled contract and roundtrip gate for every
manifest. Engine plans remain specific to the target GPU, CUDA, and TensorRT
runtime recorded in `engine_manifest.json`.
