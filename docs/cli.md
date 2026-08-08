# Command line

The `nvcr` CLI reads and writes planar 8-bit YUV420. Encode and decode are separate commands; runtime commands do not download or build engines.

## Install engines

Use the artifact client before running the CLI:

```bash
nvcr-artifacts install
nvcr-artifacts install --profile 720p
```

The installer selects a compatible target-local bundle from the rolling catalog. A missing or incompatible bundle is an error.

## Encode

```bash
nvcr encode \
  -i /path/to/input.yuv \
  -o /tmp/output.nvcr \
  -s 176x144 -r 30 --frames 4 --gop-size 2 --qp 32
```

The CLI chooses an engine profile from the visible dimensions. Use `--engine-profile 720p` when an explicit profile is needed. `--provider ID` selects the execution provider; `tensorrt` is the only production provider in this build. Normal operation uses an I/P GOP; `--gop-size 1` is an all-intra development mode.

## Decode

```bash
nvcr decode \
  -i /tmp/output.nvcr \
  -o /tmp/reconstructed.yuv
```

For quality metrics, provide the matching raw source:

```bash
nvcr decode \
  -i /tmp/output.nvcr \
  -o /tmp/reconstructed.yuv \
  --quality-metrics /path/to/input.yuv
```

## Diagnostics

`--verbose` prints progress. `--profile` prints TensorRT/CUDA stage counters and adds synchronization, so it is for diagnosis rather than timing.

The `.nvcr` file is an application wrapper around `NVAU` access units. It is not a standard container or an upstream DCVC-RT file.
