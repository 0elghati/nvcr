# Binary package status

The current repository is a development snapshot; it does not advertise a
supported v1 binary download or prebuilt engine bundle. Use
[Getting started](getting-started.md) to build from source and generate target-local
assets.

When a gated NVCR package is published, it contains the native library/CLI,
headers, CMake metadata, `nvcr-artifacts`, versioned profiles, configuration,
documentation, roadmap/changelog, licenses/notices, and a
`PACKAGE-MANIFEST.sha256`. It does **not** contain checkpoints, ONNX/model assets,
or TensorRT engines.

A future validated archive can be installed as follows:

```bash
export NVCR_ARCHIVE=nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz
export NVCR_PREFIX="$HOME/.local/nvcr"
mkdir -p "$NVCR_PREFIX"
sha256sum -c "$NVCR_ARCHIVE.sha256"
tar -xzf "$NVCR_ARCHIVE" -C "$NVCR_PREFIX" --strip-components=1
export PATH="$NVCR_PREFIX/bin:$PATH"
nvcr --help
nvcr-artifacts --help
```

The package must match the recorded target runtime. Build the pinned model and
engines locally with [Model and engine preparation](dcvcrt-artifacts.md), validate
them, and pass their directory to `nvcr --engine-dir`. Never substitute an engine
from another GPU, CUDA/TensorRT runtime, or model manifest.

If a release also provides a package-family reviewer-convenience engine asset,
download it from the same GitHub Release, verify it, and validate it before use:

```bash
export NVCR_TAG=vX.Y.Z
export NVCR_PACKAGE_FAMILY=linux-x86_64-nvidia
export NVCR_ENGINE_PROFILE=1080p-fp16
export NVCR_ENGINE_ASSET=nvcr-$NVCR_TAG-$NVCR_PACKAGE_FAMILY-dcvcrt-cvpr2025-$NVCR_ENGINE_PROFILE-engines.tar.gz

gh release download "$NVCR_TAG" \
  --pattern "$NVCR_ENGINE_ASSET" \
  --pattern "$NVCR_ENGINE_ASSET.sha256"

sha256sum -c "$NVCR_ENGINE_ASSET.sha256"
mkdir -p engines
tar -xzf "$NVCR_ENGINE_ASSET" -C engines
nvcr-artifacts validate "engines/${NVCR_ENGINE_ASSET%.tar.gz}/dcvcrt" --json
```

Use the validated bundle explicitly:

```bash
nvcr encode ... --engine-dir "engines/${NVCR_ENGINE_ASSET%.tar.gz}/dcvcrt"
nvcr decode ... --engine-dir "engines/${NVCR_ENGINE_ASSET%.tar.gz}/dcvcrt"
```

Engine assets are not generic TensorRT plans. Their filename records the package family,
model, and engine-profile identity, and the runtime still checks the
manifest and hashes before loading any TensorRT plan.

The public archive family name is generic. It does not broaden the current
support claim beyond the validated reference targets recorded in the roadmap and
compatibility matrix.

See [Compatibility](compatibility.md) for support status and
[Releasing](releasing.md) for the publication gates.
