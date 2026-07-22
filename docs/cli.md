# Native command-line interface

Return to the [docs index](README.md) or the [project overview](../README.md).

The release install places the CLI binary at `bin/nvcr` under the chosen install
prefix. On Linux, add that `bin` directory to `PATH` and run `nvcr` directly.
Encoding and decoding are separate commands and processes. Both commands
currently use planar 8-bit YUV 4:2:0 at the CLI boundary.

The CLI expects a generated engine directory. By default the helper script writes
one to `build/engines/dcvcrt`, and release-style installs can point to a copied
or symlinked `/opt/nvcr/engines/dcvcrt` directory instead. Build TensorRT plans
on the final target runtime and selected CUDA device. The directory must contain
`engine_manifest.json`; NVCR validates it before loading plans and rejects
bundles built for a different GPU model, compute capability, multiprocessor
count, or TensorRT version.

The native backend supports configured I/P GOPs through fourteen TensorRT plans.
It remains correctness-first and performance work is still active, so use Release
builds for timing.

## Encode

Encode one frame:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

The default GOP size is 32. For explicit all-intra development or benchmarking
only, pass `--gop-size 1`:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.all-intra.nvcr \
  -s 176x144 -r 30 --frames 4 --qp 32 --gop-size 1 \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

That mode is not equivalent to normal DCVC-RT I/P encoding and is not a completed
video-codec milestone.

Encode a normal 1080p I/P GOP sequence:

```bash
nvcr encode \
  -i /home/oelghati/datasets/hd/BasketballDrive_1920x1080_50.yuv \
  -o /tmp/basketball_hd.nvcr \
  -s 1920x1080 -r 50 --frames 97 --qp 32 \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

## Decode

```bash
nvcr decode \
  -i /tmp/akiyo_qcif.nvcr \
  -o /tmp/akiyo_qcif.decoded.yuv \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

The decoder reads dimensions, QP, frame type, and timestamp information from the
stored packets. Raw YUV output cannot represent mid-stream resolution changes, so
the CLI rejects those sequences.

Preview the result:

```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 176x144 \
  /tmp/akiyo_qcif.decoded.yuv
```

## HD I-frame benchmark

A single frame can be used to measure the currently supported HD I-frame path:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/720p/FourPeople_1280x720_60.yuv \
  -o /tmp/fourpeople-i.nvcr -s 1280x720 -r 60 --frames 1 \
  --engine-dir build/engines/dcvcrt-1080p-orin
```

The `.nvcr` sequence wrapper is versioned and bounds-checked, but it is an
internal development format. It is not compatible with the upstream research
container, MP4, Matroska, or an FFmpeg codec registration.

## See also

- [Architecture](architecture.md)
- [NVCR packet envelope](bitstream.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
