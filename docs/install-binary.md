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

The public archive family name is generic. It does not broaden the current
support claim beyond the validated reference targets recorded in the roadmap and
compatibility matrix.

See [Compatibility](compatibility.md) for support status and
[Releasing](releasing.md) for the publication gates.
