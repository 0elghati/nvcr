# Native command-line interface

Return to the [docs index](README.md) or the [project overview](../README.md).

The release install places the CLI binary at `bin/nvcr` under the chosen install
prefix. On Linux, add that `bin` directory to `PATH` and run `nvcr` directly.
Encoding and decoding are separate commands and processes. Both commands
currently use planar 8-bit YUV 4:2:0 at the CLI boundary.

The CLI expects a generated engine directory. By default the helper script writes
one to `build/engines/dcvcrt`, and release-style installs can point to a copied
or symlinked `/opt/nvcr/engines/dcvcrt` directory instead.

The native backend currently implements I-frames only. NVCR does not treat that as
complete video encoding and does not silently force normal multi-frame input into
an all-I stream.

## Encode

Encode one supported I-frame:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir build/engines/dcvcrt
```

The default GOP size is 32. Until the P-frame backend lands, requests for more
than one frame with a normal GOP fail before creating an output stream. For
explicit all-intra development or benchmarking only, pass `--gop-size 1`:

```bash
nvcr encode \
  -i /home/oelghati/DCVC/datasets/qcif/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.all-intra.nvcr \
  -s 176x144 -r 30 --frames 4 --qp 32 --gop-size 1 \
  --engine-dir build/engines/dcvcrt
```

That mode is not equivalent to normal DCVC-RT I/P encoding and is not a completed
video-codec milestone. `--frames 0` reads until EOF and therefore also requires
`--gop-size 1` while predicted frames remain unavailable.

## Decode

```bash
nvcr decode \
  -i /tmp/akiyo_qcif.nvcr \
  -o /tmp/akiyo_qcif.decoded.yuv \
  --engine-dir build/engines/dcvcrt
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
  --engine-dir build/engines/dcvcrt
```

The `.nvcr` sequence wrapper is versioned and bounds-checked, but it is an
internal development format. It is not compatible with the upstream research
container, MP4, Matroska, or an FFmpeg codec registration.

## See also

- [Architecture](architecture.md)
- [NVCR packet envelope](bitstream.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
