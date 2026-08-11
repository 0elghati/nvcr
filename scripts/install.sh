#!/usr/bin/env bash
set -euo pipefail

repo="${NVCR_REPO:-0elghati/nvcr}"
tag="${NVCR_TAG:-latest}"
backend="${NVCR_BACKEND:-dcvcrt}"
prefix="${NVCR_PREFIX:-$HOME/.local/nvcr}"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
engine_root="${NVCR_ENGINE_ROOT:-$data_home/nvcr/engines}"
asset_release="${NVCR_ENGINE_ASSET_RELEASE:-engine-assets}"
device_id=0
run_tests=0
install_engines=1
profiles=()

usage() {
    cat <<EOF_USAGE
Usage: install.sh [options]

Installs the matching semver'd NVCR binary package, then delegates exact GPU,
CUDA, and TensorRT engine selection to 'nvcr-artifacts install'. If no profile
is selected, every compatible published profile is installed.

Options:
  --repo OWNER/REPO       GitHub repository (default: $repo)
  --tag TAG               Binary release tag, or "latest" (default: latest)
  --backend NAME          Backend to install (default: $backend)
  --profile NAME [...]    Install one or more profiles; may be repeated
  --device-id N           CUDA device used for engine matching (default: 0)
  --asset-release TAG     Rolling engine release tag (default: $asset_release)
  --prefix DIR            NVCR binary install prefix (default: $prefix)
  --engine-root DIR       Engine install root (default: $engine_root)
  --no-engines            Install only the NVCR binary package
  --run-tests             Run installed command help smokes
  -h, --help              Show this help

The legacy --engine-profile option and *-fp16 values are accepted for this
transition release with a warning.
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --profile)
        shift
        if (($# == 0)) || [[ "$1" == -* ]]; then
            echo "nvcr-install: --profile requires at least one profile" >&2
            exit 2
        fi
        while (($#)) && [[ "$1" != -* ]]; do
            profiles+=("$1")
            shift
        done
        ;;
    --engine-profile)
        echo "nvcr-install: warning: --engine-profile is deprecated; use --profile" >&2
        profiles+=("$2")
        shift 2
        ;;
    --device-id) device_id="$2"; shift 2 ;;
    --asset-release) asset_release="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --engine-root) engine_root="$2"; shift 2 ;;
    --no-engines) install_engines=0; shift ;;
    --non-interactive)
        echo "nvcr-install: warning: --non-interactive is no longer needed" >&2
        shift
        ;;
    --run-tests) run_tests=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "nvcr-install: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

for command in curl tar sha256sum python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "nvcr-install: required command not found: $command" >&2
        exit 1
    fi
done
if [[ ! "$device_id" =~ ^[0-9]+$ ]]; then
    echo "nvcr-install: --device-id must be a non-negative integer" >&2
    exit 2
fi
case "$(uname -s):$(uname -m)" in
Linux:x86_64|Linux:amd64) package_family="linux-x86_64-nvidia" ;;
Linux:aarch64|Linux:arm64) package_family="linux-aarch64-jetson-l4t36" ;;
*)
    echo "nvcr-install: unsupported platform: $(uname -s) $(uname -m)" >&2
    exit 2
    ;;
esac
if [[ ! "$repo" =~ ^[0-9A-Za-z._-]+/[0-9A-Za-z._-]+$ || "$repo" == *..* ]]; then
    echo "nvcr-install: invalid repo: $repo" >&2
    exit 2
fi

download_dir="$(mktemp -d "${TMPDIR:-/tmp}/nvcr-install.XXXXXX")"
trap 'rm -rf -- "$download_dir"' EXIT
api_url="https://api.github.com/repos/$repo/releases"

if [[ "$tag" == latest ]]; then
    release_json="$download_dir/latest.json"
    curl -fsSL "$api_url/latest" -o "$release_json"
    tag="$(python3 - "$release_json" <<'PY_RELEASE_TAG'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["tag_name"])
PY_RELEASE_TAG
)"
else
    release_json="$download_dir/release.json"
    curl -fsSL "$api_url/tags/$tag" -o "$release_json"
fi

if [[ ! "$tag" =~ ^[0-9A-Za-z._+-]+$ || "$tag" == *..* ]]; then
    echo "nvcr-install: invalid binary release tag: $tag" >&2
    exit 2
fi

package_asset="nvcr-$tag-$package_family.tar.gz"
asset_url() {
    python3 - "$release_json" "$1" <<'PY_ASSET_URL'
import json
import sys
for asset in json.load(open(sys.argv[1], encoding="utf-8")).get("assets", []):
    if asset.get("name") == sys.argv[2]:
        print(asset["browser_download_url"])
        raise SystemExit(0)
raise SystemExit(1)
PY_ASSET_URL
}
download_asset() {
    local name="$1"
    local url
    if ! url="$(asset_url "$name")"; then
        echo "nvcr-install: binary release $tag does not contain asset: $name" >&2
        exit 1
    fi
    curl -fL "$url" -o "$download_dir/$name"
}

echo "Installing NVCR $tag for $package_family"
download_asset "$package_asset"
download_asset "$package_asset.sha256"
(
    cd "$download_dir"
    sha256sum -c "$package_asset.sha256"
)
mkdir -p "$prefix"
tar -xzf "$download_dir/$package_asset" -C "$prefix" --strip-components=1

bin_dir="$prefix/bin"
if ((install_engines)); then
    artifact_args=(
        install
        --repo "$repo"
        --asset-release "$asset_release"
        --backend "$backend"
        --device-id "$device_id"
        --engine-root "$engine_root"
    )
    for profile in "${profiles[@]}"; do
        artifact_args+=(--profile "$profile")
    done
    "$bin_dir/nvcr-artifacts" "${artifact_args[@]}"
fi

if ((run_tests)); then
    "$bin_dir/nvcr" --help >/dev/null
    "$bin_dir/nvcr-artifacts" --help >/dev/null
fi

cat <<EOF_DONE
NVCR ${tag#v} installed.

Binary prefix:
  $prefix

Add this to PATH if needed:
  export PATH="$bin_dir:\$PATH"
EOF_DONE
if ((install_engines)); then
    cat <<EOF_ENGINES

Engine root:
  $engine_root

NVCR selects the installed resolution profile automatically during encode and decode.
EOF_ENGINES
fi
