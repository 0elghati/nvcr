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
model, engine, and target profile explicitly:

```bash
/path/to/python -c 'import torch, onnx, onnxscript'

./scripts/nvcr_artifacts.py prepare \
  --model-profile configs/models/dcvcrt-cvpr2025.json \
  --engine-profile configs/engine-profiles/1080p-fp16.json \
  --target-profile configs/targets/rtx4070-ubuntu2404.json \
  --dcvcrt-root /path/to/DCVC-RT \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --python /path/to/python
```

The helper clones/checks out the pinned source by default when the target path is
absent. Use `--skip-clone` only for an already verified checkout. Use
`--skip-engine` to stop after portable model assets are ready.

The declared FP16 profiles are:

| Profile | Visible dimensions | Workspace | Builder level | Purpose |
|---|---|---:|---:|---|
| `qcif-fp16` | 64×64 to 176×144 | 512 MiB | 1 | Small correctness/development bundle |
| `1080p-fp16` | 64×64 to 1920×1080 | 1024 MiB | 2 | Reference target validation |

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
  --engine-profile configs/engine-profiles/1080p-fp16.json \
  --target-profile configs/targets/orin-nano-l4t3647.json \
  --models build/models/dcvcrt \
  --engines build/engines/dcvcrt \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device-id 0
```

Never copy an RTX plan to Orin, or an engine across a different GPU model,
CUDA/TensorRT runtime, or model export. Rebuild on the final target.

If engine bundles are retained internally for CI or reviewer convenience, store
them under `dcvcrt-cvpr2025/<target-profile>/<engine-profile>/` rather than by
resolution name alone.

For a reviewer-convenience GitHub Release asset, package a validated bundle as:

```bash
./scripts/package_engine_bundle.sh \
  --version 0.3.0 \
  --engine-dir build/engines/dcvcrt \
  --output-dir dist
```

The archive name is derived from the manifest and uses the public package family:

```text
nvcr-v0.3.0-<package-family>-dcvcrt-cvpr2025-<engine-profile>-engines.tar.gz
```

The archive contains one `dcvcrt/` engine bundle plus
`ENGINE-ASSET-MANIFEST.sha256`. It must remain a separate GitHub Release asset,
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
- CUDA runtime, TensorRT version, GPU name, compute capability, and SM count;
- I/P model-manifest digests;
- every generated plan/runtime-asset hash;
- the digest of the relative `engine.sha256` checksum manifest.

A complete runtime bundle contains 14 plans and six copied runtime assets. Both
`nvcr-artifacts validate` and runtime initialization reject missing, extra,
modified, stale, cross-model, or incompatible bundles.

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
