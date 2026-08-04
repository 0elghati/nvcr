# Binary and engine installation

NVCR publishes two semver'd Linux binary families because native x86_64 and
AArch64 executables are not interchangeable:

- `linux-x86_64-nvidia` for desktop/discrete NVIDIA hosts;
- `linux-aarch64-jetson-l4t36` for Jetson L4T 36.

The executable inside either package is named `nvcr`. Binary packages contain
the runtime, headers, CMake metadata, configuration, documentation, licenses,
and `nvcr-artifacts`. They remain free of checkpoints, ONNX graphs, runtime
model data, and TensorRT plans.

The convenience installer selects the binary family from `uname`, then delegates
engine installation to the GPU-aware rolling catalog:

```bash
curl -fsSL https://raw.githubusercontent.com/0elghati/nvcr/main/scripts/install.sh | bash
```

With no profile option, the installer selects the best published compatible
bundle for each canonical profile independently: exact device first, then
same-compute-capability desktop bundles, then Ampere-plus desktop bundles.
CUDA runtime and TensorRT version remain exact in every class. To install a
subset or no engines:

```bash
./scripts/install.sh --profile 720p
./scripts/install.sh --profile 360p --profile 540p
./scripts/install.sh --no-engines
```

The binary and engine lifecycles are independent. Pinning `--tag vX.Y.Z` pins
the binary package; engines still come from the rolling `engine-assets` release
unless `--asset-release` is overridden:

```bash
./scripts/install.sh --tag vX.Y.Z --run-tests
./scripts/install.sh --repo OWNER/REPO --asset-release engine-assets
```

After a binary-only install, manage engines directly:

```bash
nvcr-artifacts install
nvcr-artifacts install --profile 1080p --device-id 0
```

Validated bundles are stored in content-addressed directories under
`$XDG_DATA_HOME/nvcr/engines/bundles/`. Atomic aliases live under
`profiles/<backend>/<resolution>`, and `profiles/default` points to the default
backend collection. Encode selects a resolution from its raw input dimensions;
decode selects from the first access unit. Neither command performs downloads.

Force an installed resolution only when needed:

```bash
nvcr encode ... --engine-profile 720p
NVCR_ENGINE_PROFILE=1080p nvcr decode ...
```

For the transition release only, `720p-fp16` and the installer flag
`--engine-profile` remain accepted with a deprecation warning. FP16 is still the
internal v1 engine precision; it is no longer part of the user-facing resolution
name.

Manual binary extraction remains available for audited environments:

```bash
export NVCR_ARCHIVE=nvcr-vX.Y.Z-linux-x86_64-nvidia.tar.gz
export NVCR_PREFIX="$HOME/.local/nvcr"
mkdir -p "$NVCR_PREFIX"
sha256sum -c "$NVCR_ARCHIVE.sha256"
tar -xzf "$NVCR_ARCHIVE" -C "$NVCR_PREFIX" --strip-components=1
export PATH="$NVCR_PREFIX/bin:$PATH"
```

If no compatible catalog bundle matches, the installer fails rather than
starting a background TensorRT build or downloading an unsupported plan. Build
locally using [Model and engine preparation](dcvcrt-artifacts.md). See
[Compatibility](compatibility.md) for the target and compatibility contract and
[Releasing](releasing.md) for publication gates.
