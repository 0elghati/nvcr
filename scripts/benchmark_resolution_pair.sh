#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/benchmark_resolution_matrix.sh" --resolutions "720p 1080p" "$@"
