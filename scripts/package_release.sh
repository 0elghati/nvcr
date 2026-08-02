#!/usr/bin/env bash
set -euo pipefail

version=""
platform=""
install_prefix=""
output_dir="dist"

usage() {
    cat <<EOF_USAGE
Usage: $0 --version X.Y.Z --platform TAG --install-prefix DIR [options]

Packages an installed NVCR tree. Checkpoints, ONNX/model assets, TensorRT plans,
and engine bundles are deliberately rejected and never included.

Options:
  --version X.Y.Z        Release version without leading v
  --platform TAG         Public package family tag
  --install-prefix DIR   Installed NVCR prefix to package
  --output-dir DIR       Archive output directory (default: dist)
  -h, --help             Show this help
EOF_USAGE
}

while (($#)); do
    case "$1" in
    --version) version="$2"; shift 2 ;;
    --platform) platform="$2"; shift 2 ;;
    --install-prefix) install_prefix="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
        echo "unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

if [[ -z "$version" || -z "$platform" || -z "$install_prefix" ]]; then
    usage >&2
    exit 2
fi
if [[ "$version" == v* || "$version" =~ [^0-9A-Za-z.+-] ||
      "$platform" =~ [^0-9A-Za-z._-] ]]; then
    echo "invalid version or platform label" >&2
    exit 2
fi
case "$platform" in
linux-x86_64-nvidia)
    target_profile="rtx4070-ubuntu2404.json"
    ;;
linux-aarch64-jetson-l4t36)
    target_profile="orin-nano-l4t3647.json"
    ;;
*)
    echo "unsupported release platform: $platform (expected linux-x86_64-nvidia or linux-aarch64-jetson-l4t36)" >&2
    exit 2
    ;;
esac
if [[ ! -d "$install_prefix" ]]; then
    echo "install prefix does not exist: $install_prefix" >&2
    exit 1
fi

install_prefix="$(cd "$install_prefix" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
staging_root="$(mktemp -d "${TMPDIR:-/tmp}/nvcr-package.XXXXXX")"
trap 'rm -rf -- "$staging_root"' EXIT
package_name="nvcr-v$version-$platform"
package_root="$staging_root/$package_name"
mkdir -p "$package_root"
cp -a "$install_prefix/." "$package_root/"

required_files=(
    bin/nvcr
    bin/nvcr-artifacts
    share/nvcr/scripts/backends/dcvcrt/prepare_artifacts.sh
    share/nvcr/scripts/backends/dcvcrt/build_tensorrt.sh
    share/nvcr/configs/models/dcvcrt-cvpr2025.json
    share/nvcr/configs/engine-profiles/qcif-fp16.json
    share/nvcr/configs/engine-profiles/cif-fp16.json
    share/nvcr/configs/engine-profiles/360p-fp16.json
    share/nvcr/configs/engine-profiles/540p-fp16.json
    share/nvcr/configs/engine-profiles/720p-fp16.json
    share/nvcr/configs/engine-profiles/1080p-fp16.json
    "share/nvcr/configs/targets/$target_profile"
    share/doc/nvcr/README.md
    share/doc/nvcr/ROADMAP.md
    share/doc/nvcr/LICENSE
    share/doc/nvcr/third_party/dcvc_rans/LICENSE.MIT
    share/doc/nvcr/third_party/dcvc_rans/NOTICE.txt
)
for required in "${required_files[@]}"; do
    if [[ ! -f "$package_root/$required" ]]; then
        echo "release install is missing required file: $required" >&2
        exit 1
    fi
done

forbidden="$(find "$package_root" -type f \( \
    -name '*.plan' -o -name '*.engine' -o -name '*.onnx' -o \
    -name '*.pth' -o -name '*.pth.tar' -o -name 'i_entropy.bin' -o \
    -name 'i_quant.bin' -o -name 'p_entropy.bin' -o -name 'p_quant.bin' -o \
    -name 'engine_manifest.json' -o -name 'engine.sha256' \
    \) -print -quit)"
if [[ -n "$forbidden" ]]; then
    echo "release package contains forbidden model/engine asset: $forbidden" >&2
    exit 1
fi

manifest_tmp="$staging_root/PACKAGE-MANIFEST.sha256.tmp"
(
    cd "$package_root"
    find . -type f ! -name PACKAGE-MANIFEST.sha256 -print0 |
        LC_ALL=C sort -z |
        xargs -0 sha256sum >"$manifest_tmp"
)
mv "$manifest_tmp" "$package_root/PACKAGE-MANIFEST.sha256"
archive="$output_dir/$package_name.tar.gz"
tar -C "$staging_root" -czf "$archive" "$package_name"
(
    cd "$output_dir"
    sha256sum "$(basename "$archive")" >"$(basename "$archive").sha256"
)

echo "Wrote $archive and $archive.sha256"
