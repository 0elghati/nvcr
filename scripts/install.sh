#!/usr/bin/env bash
set -euo pipefail

repo="${NVCR_REPO:-0elghati/nvcr}"
tag="${NVCR_TAG:-latest}"
backend="${NVCR_BACKEND:-default}"
requested_backend="$backend"
engine_profile="${NVCR_ENGINE_PROFILE:-}"
prefix="${NVCR_PREFIX:-$HOME/.local/nvcr}"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
engine_root="${NVCR_ENGINE_ROOT:-$data_home/nvcr/engines}"
bin_dir="$prefix/bin"
download_dir=""
run_tests=0
install_engines=1
non_interactive=0

usage() {
    cat <<EOF_USAGE
Usage: install.sh [options]

Installs the latest NVCR binary package and backend engine bundles. If no engine
profile is selected, all matching profiles are downloaded and the installer asks
which backend/profile should become the CLI default.

Options:
  --repo OWNER/REPO       GitHub repository (default: $repo)
  --tag TAG               Release tag, or "latest" (default: latest)
  --backend NAME          Backend to install/select (default: default)
  --engine-profile NAME   Download/select one engine profile, for example 720p-fp16
  --prefix DIR            NVCR install prefix (default: $prefix)
  --engine-root DIR       Engine install root (default: $engine_root)
  --no-engines            Install only the NVCR binary package
  --non-interactive       Do not prompt; choose a deterministic default profile
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
    --backend) backend="$2"; requested_backend="$backend"; shift 2 ;;
    --engine-profile) engine_profile="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --engine-root) engine_root="$2"; shift 2 ;;
    --no-engines) install_engines=0; shift ;;
    --non-interactive) non_interactive=1; shift ;;
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
default|dcvcrt|dcvc_rt|dcvc-rt)
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
    tag="$(python3 - "$release_json" <<'PY_RELEASE_TAG'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["tag_name"])
PY_RELEASE_TAG
)"
else
    release_json="$download_dir/release.json"
    curl -fsSL "$api_url/tags/$tag" -o "$release_json"
fi

version="${tag#v}"
package_asset="nvcr-$tag-$package_family.tar.gz"
engine_asset_prefix="nvcr-$tag-$package_family-$backend_asset_id-"
engine_asset_suffix="-engines.tar.gz"

asset_url() {
    python3 - "$release_json" "$1" <<'PY_ASSET_URL'
import json
import sys
release = json.loads(open(sys.argv[1], encoding="utf-8").read())
name = sys.argv[2]
for asset in release.get("assets", []):
    if asset.get("name") == name:
        print(asset["browser_download_url"])
        raise SystemExit(0)
raise SystemExit(1)
PY_ASSET_URL
}

list_engine_assets() {
    python3 - "$release_json" "$engine_asset_prefix" "$engine_asset_suffix" <<'PY_LIST_ASSETS'
import json
import sys
release = json.loads(open(sys.argv[1], encoding="utf-8").read())
prefix = sys.argv[2]
suffix = sys.argv[3]
names = sorted(
    asset["name"] for asset in release.get("assets", [])
    if asset.get("name", "").startswith(prefix)
    and asset.get("name", "").endswith(suffix)
    and not asset.get("name", "").endswith(".sha256")
)
print("\n".join(names))
PY_LIST_ASSETS
}

profile_from_asset() {
    local name="$1"
    local without_prefix="${name#"$engine_asset_prefix"}"
    printf '%s\n' "${without_prefix%"$engine_asset_suffix"}"
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

select_default_profile() {
    local -n profiles_ref=$1
    if [[ -n "$engine_profile" ]]; then
        printf '%s\n' "$engine_profile"
        return
    fi
    if ((non_interactive)) || [[ ! -t 0 ]] || [[ "${#profiles_ref[@]}" -eq 1 ]]; then
        for candidate in 1080p-fp16 720p-fp16; do
            for profile in "${profiles_ref[@]}"; do
                if [[ "$profile" == "$candidate" ]]; then
                    printf '%s\n' "$profile"
                    return
                fi
            done
        done
        printf '%s\n' "${profiles_ref[0]}"
        return
    fi

    echo "Available $resolved_backend engine profiles:"
    local index=1
    for profile in "${profiles_ref[@]}"; do
        printf '  %d) %s\n' "$index" "$profile"
        index=$((index + 1))
    done
    local choice=""
    while true; do
        printf 'Choose the default engine profile [1]: '
        read -r choice
        [[ -z "$choice" ]] && choice=1
        if [[ "$choice" =~ ^[0-9]+$ ]] && ((choice >= 1 && choice <= ${#profiles_ref[@]})); then
            printf '%s\n' "${profiles_ref[$((choice - 1))]}"
            return
        fi
        echo "nvcr-install: enter a number between 1 and ${#profiles_ref[@]}" >&2
    done
}

echo "Installing NVCR $tag for $package_family"
download_asset "$package_asset"
verify_checksum "$package_asset"
mkdir -p "$prefix"
tar -xzf "$download_dir/$package_asset" -C "$prefix" --strip-components=1

installed_profiles=()
selected_profile=""
if ((install_engines)); then
    mapfile -t engine_assets < <(list_engine_assets)
    if [[ "${#engine_assets[@]}" -eq 0 ]]; then
        echo "nvcr-install: release $tag has no $resolved_backend engine assets for $package_family" >&2
        exit 1
    fi
    if [[ -n "$engine_profile" ]]; then
        requested_asset="$engine_asset_prefix$engine_profile$engine_asset_suffix"
        found=0
        for asset in "${engine_assets[@]}"; do
            [[ "$asset" == "$requested_asset" ]] && found=1
        done
        if ((found == 0)); then
            echo "nvcr-install: release $tag does not contain engine profile: $engine_profile" >&2
            echo "Available profiles:" >&2
            for asset in "${engine_assets[@]}"; do
                echo "  $(profile_from_asset "$asset")" >&2
            done
            exit 1
        fi
        engine_assets=("$requested_asset")
    fi

    echo "Installing $requested_backend backend engines"
    for asset in "${engine_assets[@]}"; do
        profile="$(profile_from_asset "$asset")"
        echo "  - $profile"
        download_asset "$asset"
        verify_checksum "$asset"
        engine_install="$engine_root/releases/${asset%.tar.gz}"
        rm -rf -- "$engine_install"
        mkdir -p "$engine_install"
        tar -xzf "$download_dir/$asset" -C "$engine_install" --strip-components=1
        "$bin_dir/nvcr-artifacts" validate "$engine_install/$backend_archive_dir" --json >/dev/null
        mkdir -p "$engine_root/profiles/$resolved_backend"
        ln -sfn "$engine_install/$backend_archive_dir" "$engine_root/profiles/$resolved_backend/$profile"
        installed_profiles+=("$profile")
    done

    selected_profile="$(select_default_profile installed_profiles)"
    ln -sfn "$engine_root/profiles/$resolved_backend" "$engine_root/profiles/default"
    ln -sfn "$engine_root/profiles/$resolved_backend/$selected_profile" "$engine_root/$resolved_backend"
    ln -sfn "$engine_root/profiles/$resolved_backend/$selected_profile" "$engine_root/default"
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

Installed engine profiles:
EOF_ENGINES
    for profile in "${installed_profiles[@]}"; do
        echo "  $resolved_backend/$profile -> $engine_root/profiles/$resolved_backend/$profile"
    done
    cat <<EOF_DEFAULT

Default engine selection:
  backend: $resolved_backend
  profile: $selected_profile

Use a profile without passing an engine path:
  nvcr encode ... --backend $resolved_backend --engine-profile $selected_profile
  nvcr decode ... --backend $resolved_backend --engine-profile $selected_profile

Override only for custom or local development bundles:
  nvcr encode ... --engine-dir /path/to/custom/engines
EOF_DEFAULT
fi
