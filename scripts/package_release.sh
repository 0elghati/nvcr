#!/usr/bin/env bash
set -euo pipefail

version=""
platform=""
install_prefix=""
engine_dir=""
output_dir="dist"

usage() {
    cat <<EOF
Usage: $0 --version X.Y.Z --platform TAG --install-prefix DIR [options]

Options:
  --version X.Y.Z        Release version without leading v
  --platform TAG         Release platform tag, for example linux-x86_64-discrete
  --install-prefix DIR   Installed NVCR prefix to package
  --engine-dir DIR       Optional TensorRT engine bundle to package
  --output-dir DIR       Archive output directory (default: dist)
  -h, --help             Show this help
EOF
}

while (($#)); do
    case "$1" in
    --version)
        version="$2"
        shift 2
        ;;
    --platform)
        platform="$2"
        shift 2
        ;;
    --install-prefix)
        install_prefix="$2"
        shift 2
        ;;
    --engine-dir)
        engine_dir="$2"
        shift 2
        ;;
    --output-dir)
        output_dir="$2"
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
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

if [[ ! -d "$install_prefix" ]]; then
    echo "install prefix does not exist: $install_prefix" >&2
    exit 1
fi

mkdir -p "$output_dir"

binary_archive="$output_dir/nvcr-v$version-$platform.tar.gz"
tar -C "$install_prefix" -czf "$binary_archive" .

if [[ -n "$engine_dir" ]]; then
    if [[ ! -d "$engine_dir" ]]; then
        echo "engine directory does not exist: $engine_dir" >&2
        exit 1
    fi
    engine_archive="$output_dir/dcvcrt-engines-v$version-$platform.tar.gz"
    tar -C "$engine_dir/.." -czf "$engine_archive" "$(basename "$engine_dir")"
fi

sha256sum "$output_dir"/*.tar.gz >"$output_dir/SHA256SUMS"

echo "Wrote release archives to $output_dir"