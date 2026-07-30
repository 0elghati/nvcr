# Native command-line interface

Return to the [docs index](README.md) or the [project overview](../README.md).

The local install places the CLI binary at `bin/nvcr` under the chosen install
prefix. On Linux, add that `bin` directory to `PATH` and run `nvcr` directly.
Encoding and decoding are separate commands and processes. Both commands
currently use planar 8-bit YUV 4:2:0 at the CLI boundary.

The CLI expects a generated engine directory. By default the helper script writes
one to `build/engines/dcvcrt`, and local installs can point to a copied
or symlinked `/opt/nvcr/engines/dcvcrt` directory instead. Build TensorRT plans
on the final target runtime and selected CUDA device. The directory must contain
`engine_manifest.json` plus `engine.sha256`; NVCR hashes the complete bundle before loading plans and rejects
bundles built for a different GPU model, compute capability, multiprocessor
count, or CUDA/TensorRT version or model identity.

The CLI currently selects the DCVC-RT TensorRT backend. That backend supports
configured I/P GOPs through fourteen TensorRT plans. It remains
correctness-first and performance work is still active, so use Release builds
for timing.

By default, encode/decode print only a final summary. Pass `--verbose` (`-v`) to
print per-frame CLI progress lines. Pass `--profile` for detailed TensorRT/CUDA
stage counters; profile output is diagnostic and may add synchronization overhead.

## Encode

Encode one frame:

```bash
nvcr encode \
  -i /path/to/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.nvcr \
  -s 176x144 -r 30 --frames 1 --qp 32 \
  --engine-dir build/engines/dcvcrt
```

The default GOP size is 32. For explicit all-intra development or benchmarking
only, pass `--gop-size 1`:

```bash
nvcr encode \
  -i /path/to/akiyo_qcif.yuv \
  -o /tmp/akiyo_qcif.all-intra.nvcr \
  -s 176x144 -r 30 --frames 4 --qp 32 --gop-size 1 \
  --engine-dir build/engines/dcvcrt
```

That mode is not equivalent to normal DCVC-RT I/P encoding and is not a completed
video-codec milestone.

Encode a normal 1080p I/P GOP sequence:

```bash
nvcr encode \
  -i /path/to/BasketballDrive_1920x1080_50.yuv \
  -o /tmp/basketball_hd.nvcr \
  -s 1920x1080 -r 50 --frames 97 --qp 32 \
  --engine-dir build/engines/dcvcrt
```

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

Report objective reconstruction quality against the matching raw source:

```bash
nvcr decode \
  -i /tmp/akiyo_qcif.nvcr \
  -o /tmp/akiyo_qcif.decoded.yuv \
  --quality-metrics /path/to/akiyo_qcif.yuv \
  --engine-dir build/engines/dcvcrt
```

The summary reports aggregate PSNR for Y, U, and V plus the DCVC 6:1:1 weighted
YUV value. Metric calculation is outside the reported codec time. The reference
must be planar 8-bit YUV 4:2:0 with the decoded dimensions and frame order.

Preview the result:

```bash
ffplay -f rawvideo -pixel_format yuv420p -video_size 176x144 \
  /tmp/akiyo_qcif.decoded.yuv
```

## HD I-frame benchmark

A single frame can be used to measure the implemented HD I-frame development path:

```bash
nvcr encode \
  -i /path/to/FourPeople_1280x720_60.yuv \
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
