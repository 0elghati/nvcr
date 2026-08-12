# Installation

General users should install the latest stable release available for their
platform. Resolve that release once and retain its immutable tag, checksum, or
container digest with any bug report or result.

## Choose a delivery method

| Host | Recommended delivery | Boundary |
|---|---|---|
| Linux x86_64 with an NVIDIA GPU | [Architecture-qualified Docker image](docker-x86_64.md) | Linux/amd64 container and catalog-selected engine |
| Windows 11 with an NVIDIA GPU | [The same Linux image through Docker Desktop/WSL 2](docker-windows.md) | Container execution; no native Windows binary |
| Jetson Orin | [Native AArch64 package](docker-jetson.md) | JetPack/L4T package and exact-target engine |
| CPU-only Linux | [Source build](first-run.md#cpu-only-contract-validation) | Contract tests; no DCVC-RT inference |
| Contributor workstation | [Contributing](../CONTRIBUTING.md) | Current `main` revision |

Read [Compatibility](compatibility.md) before interpreting a successful build
or a registered target as supported execution.

## Native Linux package

The native installer supports Linux x86_64 and Linux AArch64. It requires
`curl`, `tar`, `sha256sum`, and Python 3. The host must already provide
the NVIDIA driver and CUDA/TensorRT runtime expected by the selected package.

Resolve GitHub's latest stable release, fetch the installer from that exact
tag, and pass the same tag to the installer:

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
```

The installer:

1. selects the package family from `uname -m`;
2. locates the matching archive and checksum in the resolved release;
3. verifies the archive with `sha256sum`;
4. installs the application under the selected prefix; and
5. asks the rolling public catalog for the best compatible QCIF engine.

The release assets follow these patterns:

```text
nvcr-<tag>-linux-x86_64-nvidia.tar.gz
nvcr-<tag>-linux-aarch64-jetson-l4t36.tar.gz
```

Each archive has a sibling `.sha256` asset. Generic packages contain no
checkpoints, model exports, TensorRT plans, or datasets.

### Installed locations

```bash
export PATH="$HOME/.local/nvcr/bin:$PATH"
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
```

| Content | Default |
|---|---|
| Binary prefix | `$HOME/.local/nvcr` |
| Commands | `$HOME/.local/nvcr/bin` |
| Engine collection | `${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines` |
| Active profile | `$NVCR_ENGINE_ROOT/profiles/dcvcrt/<profile>` |
| Immutable bundle | `$NVCR_ENGINE_ROOT/bundles/dcvcrt/<target>/<profile>/<sha256>` |

The CLI does not yet provide a `--version` option. Record `NVCR_RELEASE`
and retain the installed `PACKAGE-MANIFEST.sha256` when installation identity
matters.

Override installation paths only when required:

```bash
bash /tmp/nvcr-install.sh \
  --tag "$NVCR_RELEASE" \
  --profile qcif \
  --prefix "$HOME/opt/nvcr" \
  --engine-root "$HOME/.local/share/nvcr/engines"
```

Use `--no-engines` to install only the application. Install the engine later
with:

```bash
nvcr-artifacts install --profile qcif
```

Omitting `--profile` installs every compatible published profile. The
first-run procedure selects only QCIF.

## Engine selection and public access

The `engine-assets` GitHub release and
`nvcr-engine-catalog.json` are public. The current client uses anonymous
access and does not consume `GH_TOKEN` or `GITHUB_TOKEN`.

For each requested profile, the client validates catalog identity, detects the
GPU and runtime, and ranks compatible assets in this order:

1. exact device;
2. same compute capability; and
3. Ampere-plus, when a compatible broad desktop bundle is published.

A same-compute bundle must match the detected SM value. An SM 8.9 bundle does
not apply to SM 12.0, and the reverse is also invalid. Jetson accepts
exact-target bundles only. The client verifies archive size, SHA-256, safe
extraction, the bundle manifest, and required files before updating the active
profile link.

If no compatible published bundle exists, follow
[Model and engine preparation](dcvcrt-artifacts.md). Catalog installation does
not build engines or substitute another target's plan.

## Docker

Linux and Windows container users should start with:

- [Docker overview](docker.md)
- [Linux x86_64](docker-x86_64.md)
- [Windows and WSL 2](docker-windows.md)

Container examples use an architecture-qualified rolling alias for
installation, resolve it to an immutable OCI digest, and use that digest for
the actual run. This gives ordinary users the latest delivery while preserving
the exact runtime identity used in a session.

Jetson users should prefer the native package. The rolling Jetson container can
trail the latest GitHub native release; inspect its OCI version and revision
before choosing it.

## Source builds

Use [Building from source](building-from-source.md) for a tagged or development
build. A source result must record `git describe --tags --always --dirty`;
the CMake project version alone does not prove that a checkout is an immutable
release.
