# Jetson Orin

The recommended Jetson delivery is the latest stable native AArch64 package.
It is resolved from the GitHub release API and installed with an exact-target
engine from the public catalog.

The complete runtime inspection and encode/decode procedure is in
[First run](first-run.md#jetson-orin-native-installation).

## Target boundary

The documented reference family is Jetson Orin on JetPack 6.1 / L4T 36.4,
AArch64, CUDA 12.6, TensorRT 10.3, and compute capability 8.7. Record the full
detected module and software versions; the family name alone is insufficient
for engine selection.

Jetson engines are exact-target artifacts. Same-compute-capability and
Ampere-plus desktop bundles are rejected on AArch64/Jetson. A successful run
establishes functionality for the recorded target and does not generalize to
every Jetson module or JetPack release.

## Preflight

Run these commands on the Jetson:

```bash
uname -m
head -n 1 /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
dpkg-query -W -f='${Package}\t${Version}\n' \
  nvidia-jetpack libnvinfer10 2>/dev/null || true
```

Expected platform characteristics are `aarch64`, an L4T R36 release, CUDA
12.6, and TensorRT major version 10. Keep the exact output when reporting a
problem.

## Resolve and install the native release

```bash
export NVCR_RELEASE="$(
  curl -fsSL https://api.github.com/repos/0elghati/nvcr/releases/latest |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])'
)"
printf 'NVCR release: %s\n' "$NVCR_RELEASE"

curl -fsSLo /tmp/nvcr-install.sh \
  "https://raw.githubusercontent.com/0elghati/nvcr/${NVCR_RELEASE}/scripts/install.sh"
bash /tmp/nvcr-install.sh \
  --tag "$NVCR_RELEASE" --profile qcif --run-tests

export PATH="$HOME/.local/nvcr/bin:$PATH"
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
```

The installer downloads the AArch64 package and its checksum from the resolved
release, verifies the archive, installs the binaries, and requests only the
QCIF engine. `--run-tests` checks command availability; the first-run
encode/decode procedure is the functional validation.

Verify the installed registry and bundle before encoding:

```bash
nvcr codec list
nvcr provider list
nvcr-artifacts validate \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
```

The current public catalog uses anonymous access. A
`no compatible published dcvcrt engine` error means no accepted exact bundle
matches the detected Jetson/runtime identity. It is not an authentication
failure. Build and validate the engines on the target by following
[Model and engine preparation](dcvcrt-artifacts.md).

## Jetson container status

NVCR also publishes the architecture-qualified rolling alias:

```text
omarelghati/nvcr:latest-jetson-l4t36.4
```

This mutable alias can trail the latest native GitHub release. Do not infer its
application version from the word `latest`; pull and inspect it:

```bash
export NVCR_JETSON_IMAGE="omarelghati/nvcr:latest-jetson-l4t36.4"
docker pull "$NVCR_JETSON_IMAGE"
docker image inspect "$NVCR_JETSON_IMAGE" \
  --format 'version={{ index .Config.Labels "org.opencontainers.image.version" }} revision={{ index .Config.Labels "org.opencontainers.image.revision" }} digest={{ index .RepoDigests 0 }}'
```

Choose the rolling container only after its recorded version and runtime
contract satisfy the intended use. For general Jetson installation, use the
native release procedure above.

A locally built Jetson image is a source build, not a released image. Build it
natively on AArch64 and retain its source revision and local image identity.

## Operational considerations

For functional and performance reports, retain:

- module and GPU identity;
- JetPack/L4T, CUDA, and TensorRT versions;
- resolved NVCR release;
- exact engine manifest and hashes;
- `nvpmodel`, clocks, temperature, and throttling state; and
- input and timing identities.

Do not use desktop plans on Jetson or infer plan portability solely from
compute capability.

## Next step

Run the canonical
[Jetson functional validation](first-run.md#jetson-orin-native-installation).
Use [Troubleshooting](troubleshooting.md) for target or runtime failures.
