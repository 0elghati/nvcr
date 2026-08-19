# Command line

Use the `nvcr` command to discover installed codecs and providers, then select a
codec explicitly with `-c CODEC` or `--codec CODEC` before encoding or decoding
planar 8-bit YUV420 video. Install a compatible artifact bundle first:
commands that run the codec use that local engine and never download or build
one automatically.

## Help and build identity

```bash
nvcr --help
```

There is no `nvcr --version` or `nvcr-artifacts --version` yet. Identify a
packaged binary by its archive manifest, a source build with
`git describe --tags --always --dirty`, and a container with its immutable
tag, digest, and OCI labels.

A build with TensorRT disabled still builds `nvcr` for registry discovery and
for codecs/providers that do not require TensorRT.

## Discover codecs and providers

```bash
nvcr codec list
nvcr codec describe dcvc-rt
nvcr provider list
nvcr provider describe tensorrt
nvcr compatibility check --codec dcvc-rt --provider tensorrt
```

The lists include the `dcvc-rt` codec and `tensorrt` provider. The provider
description reports the TensorRT version compiled into that build; do not infer
it from an example or image name.

The compatibility command checks only that the registered IDs can be paired.
It does not query a GPU, select or validate a bundle, compare CUDA/TensorRT
versions, or deserialize an engine. Catalog compatibility is evaluated by
`nvcr-artifacts install`. The CLI resolves the installed bundle and enforces
runtime compatibility when the encode backend is initialized.

Provider descriptors report general capabilities. The supported DCVC-RT
execution mode is FP16 even if a descriptor exposes a broader vocabulary.

## Install and inspect engines

The public catalog uses anonymous access:

```bash
nvcr-artifacts install --profile qcif
nvcr-artifacts install --profile 360p 720p 1080p
export NVCR_ENGINE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/nvcr/engines"
nvcr-artifacts inspect \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
```

`--profile` accepts one or more values and may be repeated. Omitting it
installs every compatible catalog profile. For each profile, the installer
prefers `exact`, then `same_compute_capability`, then `ampere_plus`.
Same-compute matching compares the numeric major and minor (for example,
SM 8.9 matches only SM 8.9), and Jetson is exact-only. Every class still
requires the TensorRT `major.minor.patch` recorded in the bundle.

`inspect` reports the installed manifest without running a TensorRT plan.
`validate` checks the bundle schema, file inventory, and digests:

```bash
nvcr-artifacts validate \
  "$NVCR_ENGINE_ROOT/profiles/dcvcrt/qcif" --json
```

The final compatibility check occurs during backend initialization and frame
processing. A successful inspection or registry check is not execution
evidence.

Engine selection precedence is:

1. `--engine-dir`, a direct bundle;
2. `NVCR_ENGINE_DIR`, a direct bundle;
3. `NVCR_ENGINE_ROOT`, a catalog-style collection;
4. `$XDG_DATA_HOME/nvcr/engines`;
5. `$HOME/.local/share/nvcr/engines`; then
6. `./engines`.

Within a collection, installed profiles live under
`profiles/<backend>/<profile>`; DCVC-RT currently uses
`profiles/dcvcrt/<profile>`. Dimensions map to `qcif`, `cif`, exact
`360p`, exact `540p`, `720p`, or `1080p`; larger input requires an
explicit compatible bundle.

## Encode

Source users can generate deterministic raw input with the repository helper.
Packaged users may supply any headerless planar YUV420P8 input with matching
dimensions and frame count:

```bash
python3 scripts/generate_sample_yuv.py \
  --output nvcr-example/input.yuv \
  --width 176 --height 144 --frames 4
```

Encode:

```bash
nvcr encode \
  -i nvcr-example/input.yuv \
  -o nvcr-example/output.nvcr \
  -s 176x144 -r 30 \
  -c dcvc-rt --backend dcvcrt \
  --frames 4 --gop-size 2 --qp 32 --verbose
```

`-c` and `--codec` are equivalent. A codec is required for both encode and
decode because existing NVAU v1 access units identify a model, not a codec
implementation. `--backend` is an optional artifact namespace; DCVC-RT uses
`dcvcrt` for its catalog layout.

`--frames 0` processes complete frames to end of input. `--gop-size 1`
explicitly selects all-intra development mode. Normal operation uses I/P
coding. `--provider`, `--device-id`, `--engine-profile`, and
`--engine-dir` override their defaults.

The final payload-byte total is the complete packet payload passed by the
runtime—currently a bounded `NVAU` access unit. It excludes the outer
`NVCS` record bytes but is not an entropy-only byte count.

## Decode and quality

```bash
nvcr decode \
  -i nvcr-example/output.nvcr \
  -o nvcr-example/reconstructed.yuv \
  -c dcvc-rt --backend dcvcrt

nvcr decode \
  -i nvcr-example/output.nvcr \
  -o nvcr-example/reconstructed.yuv \
  -c dcvc-rt --backend dcvcrt \
  --quality-metrics nvcr-example/input.yuv
```

Decode reads dimensions from the first access unit and rejects resolution
changes in raw output. The quality option reports per-plane and weighted YUV
PSNR against a raw reference with matching frames and dimensions.

## Diagnostics and timing boundary

`--verbose` prints the selected artifact bundle before backend initialization,
then reports per-frame progress. Treat successful frame processing, not the
selection line alone, as execution evidence. `--profile` may add
adapter-specific diagnostics and synchronization, so do not use profiling
repetitions for throughput.

CLI FPS uses time around the runtime encode/decode call. It does not include
process startup, file/container orchestration, or the full subprocess boundary.
Use [Performance and benchmarking](performance.md) when comparing end-to-end
workflows.

The `.nvcr` file is an application sequence wrapper (`NVCS` records
containing `NVCR` packets and `NVAU` access units), not a standard media
container or an upstream Python DCVC-RT file.
