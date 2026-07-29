# Experimental `mlvc-fast` backend

Return to the [docs index](README.md), [performance evidence](performance.md), or
[project roadmap](../ROADMAP.md).

## Status and purpose

`mlvc-fast-v1` is an experimental, self-contained CPU video codec behind the
existing NVCR backend interface. It replaces the original `mlvc-fast-v0` raw
frame transport with actual lossy intra/inter compression and deterministic
closed-loop reconstruction.

The implementation takes one architectural idea from learned inter-frame codecs
such as MLVC: separate motion-compensated prediction from residual coding and
make the decoder reproduce the encoder reference exactly. It is not an MLVC
paper reproduction, does not run a trained neural model, and is not compatible
with DCVC-RT bitstreams. Its purpose is to provide a fast, inspectable comparison
backend while a genuinely learned fast backend is researched.

The format is experimental. `mlvc-fast-v0` streams are intentionally rejected by
`mlvc-fast-v1` because the payload changed from raw YUV to compressed data.

## Codec pipeline

Input and reconstructed output are planar 8-bit YUV 4:2:0.

For intra frames:

1. Each Y, U, and V sample uses a median-edge spatial prediction from already
   reconstructed left, top, and top-left samples.
2. The prediction residual is scalar-quantized.
3. The quantized residual is immediately reconstructed, keeping the encoder in
   the same closed loop as the decoder.

For predicted frames:

1. Luma is divided into 32x32 blocks.
2. A sampled-SAD search checks a +/-8-pixel window at four-pixel spacing, then
   refines the best vector at two-pixel and one-pixel spacing.
3. Motion rows are searched in parallel. Chroma reuses the luma motion field at
   half resolution.
4. Every plane is predicted from the previously reconstructed frame, and the
   residual is quantized and reconstructed in parallel.

Quantized signed values are ZigZag mapped. Every 256-sample tile records the
minimum bit width needed by that tile and packs values densely. At QP 32 the
quantization step is 8, so the per-sample reconstruction error is bounded to four
levels before edge clipping. The current QP mapping starts at step 2 for QP 20
and doubles the step for every six QP values.

## Payload format

The backend payload begins with `NVM1` and contains:

| Field | Size | Meaning |
|---|---:|---|
| Magic | 4 bytes | `NVM1` |
| Version | 2 bytes | Payload syntax version 1 |
| Flags | 2 bytes | Bit 0 marks a predicted frame |
| Width, height | 8 bytes | Visible YUV dimensions |
| Quantization step | 2 bytes | Decoder reconstruction step |
| Motion block size | 2 bytes | Currently 32 |
| Motion-vector count | 4 bytes | Zero for intra frames |
| Y/U/V section sizes | 12 bytes | Bounds each coefficient plane |
| Motion field | 2 bytes/block | Signed horizontal and vertical displacement |
| Coefficient planes | variable | 256-value adaptive bit-packed tiles |

The decoder validates magic, version, flags, dimensions, quantizer, block size,
motion count, every section size, tile bit widths, truncation, and trailing data
before committing a reference frame.

## CLI usage

No engine directory or engine profile is needed because this experimental
backend is self-contained.

```bash
nvcr encode \
  -i input_1280x720.yuv -o output.nvcr -s 1280x720 -r 30 \
  --frames 97 --gop-size 97 --qp 32 --backend mlvc-fast

nvcr decode \
  -i output.nvcr -o decoded_1280x720.yuv --backend mlvc-fast

ffplay \
  -f rawvideo -pixel_format yuv420p -video_size 1280x720 -framerate 30 \
  decoded_1280x720.yuv
```

The decoded file is ordinary planar YUV420P8 and can also be opened in tools such
as FFmpeg, mpv, or YUView when the dimensions and frame rate are supplied.

## Measured implementation result

The following single-run development measurements were captured on 2026-07-29
from the release build, QP 32, 97 frames. They include compression and closed-loop
reconstruction. They are not warmed repeated publication results.

| Sequence | GOP | Encode FPS | Payload | Raw reduction |
|---|---:|---:|---:|---:|
| FourPeople 1280x720 | 1 | 73.833 | 42,320,776 bytes | 3.17x |
| FourPeople 1280x720 | 97 | 201.101 | 38,264,072 bytes | 3.50x |
| BQTerrace 1920x1080 | 1 | 29.674 | 127,451,430 bytes | 2.37x |
| BQTerrace 1920x1080 | 97 | 97.808 | 126,328,902 bytes | 2.39x |

GOP-97 round-trip evidence:

| Sequence | Decode FPS | Y PSNR | U PSNR | V PSNR | Average PSNR |
|---|---:|---:|---:|---:|---:|
| FourPeople 1280x720 | 337.361 | 40.919 dB | 41.298 dB | 41.452 dB | 41.065 dB |
| BQTerrace 1920x1080 | 189.587 | 40.695 dB | 41.469 dB | 41.599 dB | 40.957 dB |

The earlier raw scaffold reported 781-1,962 FPS because it only copied bytes and
wrote raw-sized payloads. Those values remain historical upper-bound data and
must not be compared with the compressed v1 result as codec throughput.

DCVC-RT still produces dramatically smaller payloads at the tested QP. Therefore
the FPS rows above are useful implementation data, not a rate-distortion win or
an MLVC/DCVC quality-equivalent comparison.

## Verification and limitations

The release smoke test uses separate encoder and decoder backend instances. It
checks compressed intra and predicted payloads, exact equality between encoder
and decoder reconstructions, the QP-32 sample error bound, reference commits,
and truncated-payload rejection. Full CLI streams were additionally decoded at
720p and 1080p and compared with FFmpeg PSNR.

Known limitations:

- CPU-only scalar prediction and bit packing; no CUDA path yet.
- Scalar quantization without transform coding or a learned entropy model.
- Fixed 32x32 translational luma motion and derived chroma motion.
- Experimental payload syntax with no compatibility guarantee.
- No claim of MLVC paper accuracy, bitrate, or model behavior.
- No claim of rate-distortion parity with DCVC-RT, x265, AV1, or other codecs.
