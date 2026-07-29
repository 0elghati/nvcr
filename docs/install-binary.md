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

A future validated archive and matching default engine bundle can be installed
with the convenience installer:

```bash
curl -fsSL https://raw.githubusercontent.com/0elghati/NVCR/main/scripts/install.sh | bash
```

By default it installs NVCR under `$HOME/.local/nvcr`, validates the downloaded
checksums, extracts the matching backend engine bundle under
`$XDG_DATA_HOME/nvcr/engines/releases/`, points
`$XDG_DATA_HOME/nvcr/engines/<backend>` at that bundle, and points
`$XDG_DATA_HOME/nvcr/engines/default` at the active backend. The CLI uses that path
automatically, so normal encode/decode commands do not need `--engine-dir`.

Pin a release or install only the binary package when needed:

```bash
NVCR_TAG=vX.Y.Z ./scripts/install.sh --run-tests
./scripts/install.sh --no-engines
```

Manual installation remains supported for offline or audited environments:

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

The package must match the recorded target runtime. Build or download the pinned
model and engines with [Model and engine preparation](dcvcrt-artifacts.md),
validate them, and install them at the default engine path or pass their
directory to `nvcr --engine-dir`. Never substitute an engine from another GPU,
CUDA/TensorRT runtime, or model manifest.

Engine assets are not generic TensorRT plans. Their filename records the package family,
model, and engine-profile identity, and the runtime still checks the
manifest and hashes before loading any TensorRT plan.

The public archive family name is generic. It does not broaden the current
support claim beyond the validated reference targets recorded in the roadmap and
compatibility matrix.

See [Compatibility](compatibility.md) for support status and
[Releasing](releasing.md) for the publication gates.
