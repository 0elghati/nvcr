#!/usr/bin/env bash
set -euo pipefail

repo="${NVCR_REPO:-0elghati/NVCR}"
tag="${NVCR_TAG:-latest}"
backend="${NVCR_BACKEND:-default}"
requested_backend="$backend"
engine_profile="${NVCR_ENGINE_PROFILE:-1080p-fp16}"
prefix="${NVCR_PREFIX:-$HOME/.local/nvcr}"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
engine_root="${NVCR_ENGINE_ROOT:-$data_home/nvcr/engines}"
bin_dir="$prefix/bin"
download_dir=""
run_tests=0
install_engines=1

usage() {
    cat <<EOF_USAGE
Usage: install.sh [options]

Installs the latest NVCR binary package and, by default, the matching backend
engine bundle. The CLI then finds engines without --engine-dir at:
  \$NVCR_ENGINE_DIR, or $data_home/nvcr/engines/default

Options:
  --repo OWNER/REPO       GitHub repository (default: $repo)
  --tag TAG               Release tag, or "latest" (default: latest)
  --backend NAME          Engine backend to install (default: default)
  --engine-profile NAME   Engine profile asset to install (default: 1080p-fp16)
  --prefix DIR            NVCR install prefix (default: $prefix)
  --engine-root DIR       Engine install root (default: $engine_root)
  --no-engines            Install only the NVCR binary package
  --run-tests             Run nvcr --help and nvcr-artifacts --help after install
  -h, --help              Show this help

Environment overrides: NVCR_REPO, NVCR_TAG, NVCR_BACKEND, NVCR_ENGINE_PROFILE,
NVCR_PREFIX, NVCR_ENGINE_ROOT, XDG_DATA_HOME.
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --engine-profile) engine_profile="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --engine-root) engine_root="$2"; shift 2 ;;
    --no-engines) install_engines=0; shift ;;
    --run-tests) run_tests=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "nvcr-install: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "nvcr-install: required command not found: $1" >&2
        exit 1
    fi
}

require_command curl
require_command tar
require_command sha256sum
require_command python3

case "$(uname -s):$(uname -m)" in
Linux:x86_64) package_family="linux-x86_64-nvidia" ;;
Linux:aarch64) package_family="linux-aarch64-jetson-l4t36" ;;
*)
    echo "nvcr-install: unsupported platform: $(uname -s) $(uname -m)" >&2
    exit 2
    ;;
esac

case "$backend" in
default)
    resolved_backend="dcvcrt"
    backend_asset_id="dcvcrt-cvpr2025"
    backend_archive_dir="dcvcrt"
    ;;
dcvcrt)
    resolved_backend="dcvcrt"
    backend_asset_id="dcvcrt-cvpr2025"
    backend_archive_dir="dcvcrt"
    ;;
*)
    echo "nvcr-install: unsupported backend: $backend" >&2
    exit 2
    ;;
esac
if [[ "$repo" != */* || "$repo" == *..* || "$repo" == /* ]]; then
    echo "nvcr-install: invalid repo: $repo" >&2
    exit 2
fi

download_dir="$(mktemp -d "${TMPDIR:-/tmp}/nvcr-install.XXXXXX")"
trap 'rm -rf -- "$download_dir"' EXIT

api_url="https://api.github.com/repos/$repo/releases"
if [[ "$tag" == "latest" ]]; then
    release_json="$download_dir/latest.json"
    curl -fsSL "$api_url/latest" -o "$release_json"
    tag="$(python3 - "$release_json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["tag_name"])
PY
)"
else
    release_json="$download_dir/release.json"
    curl -fsSL "$api_url/tags/$tag" -o "$release_json"
fi

version="${tag#v}"
package_asset="nvcr-$tag-$package_family.tar.gz"
engine_asset="nvcr-$tag-$package_family-$backend_asset_id-$engine_profile-engines.tar.gz"

asset_url() {
    python3 - "$release_json" "$1" <<'PY'
import json
import sys
release = json.loads(open(sys.argv[1], encoding="utf-8").read())
name = sys.argv[2]
for asset in release.get("assets", []):
    if asset.get("name") == name:
        print(asset["browser_download_url"])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

download_asset() {
    local name="$1"
    local url
    if ! url="$(asset_url "$name")"; then
        echo "nvcr-install: release $tag does not contain asset: $name" >&2
        exit 1
    fi
    curl -fL "$url" -o "$download_dir/$name"
}

verify_checksum() {
    local name="$1"
    download_asset "$name.sha256"
    (
        cd "$download_dir"
        sha256sum -c "$name.sha256"
    )
}

echo "Installing NVCR $tag for $package_family"
download_asset "$package_asset"
verify_checksum "$package_asset"
mkdir -p "$prefix"
tar -xzf "$download_dir/$package_asset" -C "$prefix" --strip-components=1

if ((install_engines)); then
    echo "Installing $requested_backend backend engines ($engine_profile)"
    download_asset "$engine_asset"
    verify_checksum "$engine_asset"
    engine_install="$engine_root/releases/${engine_asset%.tar.gz}"
    rm -rf -- "$engine_install"
    mkdir -p "$engine_install"
    tar -xzf "$download_dir/$engine_asset" -C "$engine_install" --strip-components=1
    mkdir -p "$engine_root"
    ln -sfn "$engine_install/$backend_archive_dir" "$engine_root/$resolved_backend"
    ln -sfn "$engine_install/$backend_archive_dir" "$engine_root/default"
    "$bin_dir/nvcr-artifacts" validate "$engine_root/default" --json >/dev/null
fi

if ((run_tests)); then
    "$bin_dir/nvcr" --help >/dev/null
    "$bin_dir/nvcr-artifacts" --help >/dev/null
fi

cat <<EOF_DONE
NVCR $version installed.

Binary prefix:
  $prefix

Add this to PATH if needed:
  export PATH="$bin_dir:\$PATH"
EOF_DONE

if ((install_engines)); then
    cat <<EOF_ENGINES

Default engine directory:
  $engine_root/default

Backend engine directory:
  $engine_root/$resolved_backend

The nvcr CLI will use that directory automatically unless --engine-dir or
NVCR_ENGINE_DIR overrides it.
EOF_ENGINES
fi
