# MLVC integration

[MLVC](https://github.com/microsoft/MLVC) (Multi-platform Learned Video Codec)
is Microsoft's learned video codec designed for deployment across hardware. It
is NVCR's second codec integration and the first that runs through the ONNX
Runtime providers instead of TensorRT. Status: experimental (roadmap C10).

The integration follows two rules. NVCR stays a common runtime: MLVC uses the
same registry, provider sessions, `NVAU` access units, CLI and session contract
as DCVC-RT, and the only provider change is codec-neutral. NVCR also stays
faithful to the codec: it runs Microsoft's published model bundles unchanged
and reproduces MLVC's own conversion runtime byte for byte.

## What NVCR runs

MLVC ships a codec as an ONNX *model bundle* produced by its `convert.py bundle`
command. Microsoft's native codec repository
[libmlvc](https://github.com/microsoft/libmlvc) publishes ready bundles for
MLVC-S under `data/model_bundles/`:

| Bundle | Use |
|---|---|
| `dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1.tar` | CPU and other generic ONNX Runtime targets |
| `dmc61sbr_mini_reglu-perceptual_v1-onnx-qualcomm-v0.1.tar` | Snapdragon (Qualcomm QNN HTP); graph passes for the Hexagon NPU |

Extract a bundle and pass its directory to NVCR. NVCR reads
`bundle_manifest.json`, `metadata_<W>x<H>.json`, both PMF tables, the
`MLVCEncoder_<W>x<H>.onnx` and `MLVCDecoder_<W>x<H>.onnx` parts and the shared
`weights.bin`. It checks each part against the bundle's registry digest, which
is SHA-256 over the model bytes followed by the weights bytes, before the first
use.

A bundle holds several fixed canvases, for example 960x544, 640x368, 432x240
and 320x192. MLVC never scales a frame. NVCR encodes on the smallest canvas the
frame fits, first as-is and then with width and height swapped, just as MLVC
places a frame. The frame is edge-padded to the bottom and right. The decoder
selects the canvas named in each intra access unit.

## Command line

```bash
nvcr encode -i input.yuv -o output.nvcr -s 176x144 -r 30 --qp 32 \
  --codec mlvc --engine-dir dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1
nvcr decode -i output.nvcr -o decoded.yuv \
  --codec mlvc --engine-dir dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1
```

- `--qp` is MLVC's Q index, 0 to 63; a higher value gives higher quality.
- Without `--gop-size`, MLVC keeps the I period published in the bundle: 1024
  coded frames for MLVC-S and 64 for MLVC. `--gop-size N` sets it explicitly.
- Without `--provider`, MLVC uses `onnxruntime-cpu`. `onnxruntime-qnn` selects
  the Snapdragon NPU, and `onnxruntime-cuda` selects ONNX Runtime CUDA when
  that build is linked.

Library users set `codec.id = "mlvc"`, `artifacts.intra_engine_path` to the
bundle directory, `codec.intra_qp`, and `codec.gop_size` (or
`nvcr::mlvc::use_model_iframe_period`). See `include/nvcr/mlvc/codec.hpp`.

## How frames are handled

The MLVC sessions follow the upstream frame loop
(`video/conversion/_frame_loop.py`):

- An I frame runs the same encoder graph with a zero feature and clears both
  directions' reference state. MLVC has no separate intra model.
- A P frame references the previous coded frame. On MLVC-S, frame 8 is marked
  as a long-term reference (LTR), and every 64th frame after it becomes an
  LTR-recovery frame that references the latest LTR.
- Each direction keeps one short-term and up to three long-term features. The
  encoder references its own feature output, not a decoded one.
- The Q index sent to the graph is shifted by the position in an eight-frame
  cycle (+8, 0, +4, 0, +4, 0, +4, 0). The unshifted Q index selects the
  hyperprior distributions.
- Entropy coding uses Microsoft's `msrtc_rans` coder, vendored unmodified in
  `third_party/msrtc_rans`, with MLVC's settings: RansByte, 16 symbol bits and
  2 bypass bits. The coder pushes `y_raw_1`, then `y_raw_0`, then `z_raw`, and
  the decoder requires the stream to be fully consumed.
- Host conversions use upstream's rounding points: 8-bit to float as `u / 255`
  in float32, nearest-neighbour chroma upsampling, a cast to the model
  precision, a 2x2 float32 chroma mean cast back to that precision, and
  round-half-even to 8 bits.

The encoder uses a fixed Q index. MLVC's rate controller, and with it encoder
frame dropping, is not implemented. The decoder does accept dropped frames
from upstream streams (Q index -1 with an empty payload). It repeats the last
frame without changing reference state.

## Stream compatibility

Each `NVAU` access unit carries one upstream MLVC frame record as its codec
payload: a little-endian int32 Q index, a uint32 length and the rANS bytes.
The payloads of an NVCR MLVC stream therefore concatenate to the `.mlvc` file
written by MLVC's conversion tooling. `NVAU` adds what that file leaves out of
band: the frame dimensions, the intra/reset flag, and a model identity made of
the bundle name and canvas, such as
`dmc61sbr_mini_reglu-perceptual_v1-320x192`. libmlvc's own NALU-based `.mlvc`
format is a different syntax and is not produced.

Compatibility position, as validated (see the
[C10 evidence](../evidence/phase-c10-mlvc-codec-20261102.md)):

- **Byte/payload interchangeable with MLVC's conversion runtime** on the same
  bundle and execution provider. On ONNX Runtime 1.26.0 CPU, every record
  NVCR encodes equals the upstream record. Decoding either stream reproduces
  upstream's 8-bit output byte for byte, including LTR recovery, an I-period
  override, a transposed canvas and a rate-controlled stream with 16 dropped
  frames.
- **Across targets or providers** (for example, encoded on CPU and decoded on
  the Snapdragon NPU), MLVC is designed so that entropy decoding still works:
  scales are sent explicitly through `z`. Reconstructions are not expected to
  be identical and can drift until the next I or LTR-recovery frame. On the
  S25, decodes of one 640x360 stream on two backends fall to 34–43 dB Y PSNR
  between each other by frame 63, then recover at the frame-64 LTR recovery.
- **Same provider, different CPU architecture**: not identical either. ONNX
  Runtime's arm64 CPU kernels produce different FP16 results from x86, so
  upstream-runtime parity is established on x86 only.
- **Not** consistent with MLVC's PyTorch reference beyond what the exported
  graph itself provides. That is an export property; see C5/C6 in the
  [roadmap](../ROADMAP.md).

Per access unit, `NVAU` v1 adds a fixed header and the model identity string,
about 81 bytes per frame for MLVC-S at 320x192, on top of MLVC's 8-byte
record header. At low rates this overhead is larger than the payload. Reducing
it is a stream-format change outside this integration.

## Providers

| Provider | Status for MLVC |
|---|---|
| `onnxruntime-cpu` | Validated: upstream-runtime parity on the generic and qualcomm bundles |
| `onnxruntime-qnn` (Snapdragon NPU) | Runs on the S25 with every node on the HTP; no upstream reference on the device, so parity is not established |
| `onnxruntime-cuda` | Not validated for MLVC |
| `tensorrt` | Rejected with `incompatible_target`; the TensorRT session loads DCVC-RT engine bundles only |

The ONNX Runtime provider now binds float16, float32 and int32 stage tensors.
It checks each binding against the model's declared type. A stage's floating
tensors must match its declared precision (`fp16` or `fp32`), so DCVC-RT's
fp16-only rule is unchanged. Part names may use upper-case letters
(`MLVCEncoder_320x192`).

### Snapdragon

MLVC bundles already have fixed shapes per canvas, so the QNN HTP needs no
`fix_onnx_shapes.py` step. Use the qualcomm bundle: it carries upstream's HTP
graph workarounds (Floor after Round, a split gated convolution, 2-D Gather,
and Clip as Max). Build NVCR for arm64 Android as in the
[C9 evidence](../evidence/phase-c9-snapdragon-qnn-20261101.md), push the
extracted bundle, and run `nvcr encode/decode --codec mlvc --provider
onnxruntime-qnn --engine-dir <bundle>`. The provider disables CPU fallback, so a
part the NPU cannot run fails to load instead of silently running on the CPU.

On the S25 (2026-11-03), both parts load and run with CPU fallback disabled.
Streams decode across the NPU, the phone CPU and x86 CPU without an entropy
error, with the drift described above.

Upstream's QNN session uses `ORT_DISABLE_ALL` graph optimization,
`htp_performance_mode=burst` and `htp_graph_finalization_optimization_mode=3`.
NVCR's QNN provider uses `ORT_ENABLE_ALL` and sets only the backend path. On
the S25, upstream's settings produced byte-identical streams and
reconstructions while cutting the mean model time at 640x360 from 13.8 ms to
2.4 ms (encode) and from 15.3 ms to 3.2 ms (decode). NVCR's defaults are
unchanged. Until a default is chosen, these environment variables select the
options for diagnosis:

| Variable | Effect |
|---|---|
| `NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION` | `disable`, `basic`, `extended` or `all` (default) |
| `NVCR_QNN_HTP_PERFORMANCE_MODE` | Passed as QNN `htp_performance_mode`, for example `burst` |
| `NVCR_QNN_HTP_FINALIZATION_MODE` | Passed as QNN `htp_graph_finalization_optimization_mode`, for example `3` |
| `NVCR_ONNXRUNTIME_DISABLE_CPU_FALLBACK` | `1` disables CPU fallback for the CPU and CUDA providers too |
| `NVCR_ONNXRUNTIME_PROFILE` | Writes an ONNX Runtime profile per part with this file prefix |

With `--verbose`, the MLVC sessions log mean host, model and entropy time per
frame when they flush. Short CLI runs on the device are dominated by session
creation, about 4.8 s at 640x360, which is outside these phases. See the
[C10 evidence](../evidence/phase-c10-mlvc-codec-20261102.md#snapdragon-device-run--2026-11-03).

## Reproducing the parity gate

`scripts/backends/mlvc/reference_vectors.py` drives MLVC's own
`load_split_model` and `FrameLoop` to record `output.mlvc`, `output.yuv` and a
summary. Run it with the Python environment of a pinned MLVC checkout and
ONNX Runtime matching NVCR's build. `tests/mlvc_parity_tests.cpp` compares NVCR
with that output. Configure
`-DNVCR_MLVC_PARITY_CASES="BUNDLE|REFERENCE|INPUT|W|H|FRAMES|PERIOD"` to register
cases with CTest (`PERIOD` is a number or `model`). Append `|decode-only` for an
upstream stream NVCR cannot encode, such as a `--bitrate` reference. Bundles, checkpoints, references and inputs are not committed.

## Current limits

- One split, `dmc61sbr_e1d1` (both public models), with the upsample scale
  decoder and unit pixel range. Other splits fail closed.
- NCHW bundles only. MLVC's current exporter writes NHWC image and feature
  tensors for the qualcomm target (upstream `64b10d4`). Such bundles are
  rejected with `not_implemented` until a real one has been validated.
- No encoder rate control, no Q-index overrides, no LTR or feature-reset
  overrides beyond the bundle's own values.
- Only MLVC-S perceptual bundles have been run. The standard MLVC model needs a
  bundle exported from its public checkpoint.
- Codec/provider admission is still coarse: `nvcr compatibility check` reports
  `mlvc` with `tensorrt` as compatible because both are registered; the adapter
  rejects the pair when sessions are created.
