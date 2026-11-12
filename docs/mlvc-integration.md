# MLVC integration

[MLVC](https://github.com/microsoft/MLVC) (Multi-platform Learned Video Codec)
is Microsoft's learned video codec designed for deployment across hardware. It
is NVCR's second codec integration and the first that runs through the ONNX
Runtime providers; it also runs on TensorRT. Status: experimental (roadmap C10,
TensorRT C12).

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
| `tensorrt` | Experimental (C12): runs from an MLVC engine bundle; streams decode across TensorRT and ONNX Runtime CPU with bounded drift, see [TensorRT](#tensorrt) |

The ONNX Runtime provider now binds float16, float32 and int32 stage tensors.
It checks each binding against the model's declared type. A stage's floating
tensors must match its declared precision (`fp16` or `fp32`), so DCVC-RT's
fp16-only rule is unchanged. Part names may use upper-case letters
(`MLVCEncoder_320x192`).

### TensorRT

TensorRT runs MLVC from an *engine bundle*: Microsoft's ONNX bundle, unchanged,
plus one FP16 plan per canvas and part (`mlvc_encoder_<W>x<H>.plan`,
`mlvc_decoder_<W>x<H>.plan`) and the `engine_manifest.json` and
`engine.sha256` that NVCR validates for every TensorRT bundle. Plans are
target-local. Install a published bundle instead of building one:

```bash
nvcr-artifacts install --backend mlvc --profile perceptual
nvcr encode -i input.yuv -o output.nvcr -s 640x360 -r 30 --qp 32 \
  --codec mlvc --provider tensorrt \
  --engine-dir "$NVCR_ENGINE_ROOT/profiles/mlvc/perceptual"
```

A profile is one MLVC model and its weights, each a generic ONNX export with
all four canvases:

| Profile | Model | Weights | Source of the ONNX bundle |
|---|---|---|---|
| `s-perceptual` | MLVC-S (5.4M parameters) | perceptual | libmlvc's published `onnx-generic-v0.1` bundle |
| `perceptual` | MLVC (18.3M parameters) | perceptual | exported from `mlvc-perceptual-v1.ckpt` |
| `psnr` | MLVC (18.3M parameters) | psnr | exported from `mlvc-psnr-v1.ckpt` |

Without `--profile`, the installer installs all three. A stream records its
model in every access unit, so it decodes only with the profile that encoded
it. MLVC bundles live on their own rolling release, `engine-assets-mlvc`, so
installers released before MLVC never see them; installation never changes
`profiles/default`, which stays DCVC-RT's. The first published profile was
named `s-generic`; it is now `s-perceptual`, and installers skip catalog
profiles they do not know.

Only MLVC-S perceptual has a published bundle.
`scripts/backends/mlvc/export_onnx_bundle.sh` exports the others with MLVC's own
`convert.py` from the pinned checkout (`1425880`) and a published checkpoint,
using an interpreter with MLVC's `uv.lock` exporter versions, CPU PyTorch 2.10.0
and `packages/msrtc_rans` installed. Upstream validates each canvas on VCD clips;
the script uses NVCR's datasets instead, so the metrics recorded in the bundle's
metadata describe those clips.

To build a bundle for another GPU, extract a generic bundle and run, inside
the TensorRT toolchain container on that GPU:

```bash
python3 scripts/backends/mlvc/build_tensorrt_bundle.py \
  --bundle <extracted-onnx-bundle> --output-dir build/mlvc-trt \
  --target-profile configs/targets/<target>.json \
  --hardware-compatibility same_compute_capability
```

The script refuses a target profile that does not describe the build GPU.
Package and publish `build/mlvc-trt/bundle` like a DCVC-RT bundle
(`scripts/release_engine_assets.sh --asset-release engine-assets-mlvc`).

Behaviour on TensorRT:

- Every canvas's plans load and warm up when the session is created. The
  TensorRT session cannot load stages after it starts running, so MLVC's lazy
  per-canvas loading does not apply. A bundle missing any plan fails at start.
- MLVC's tensors are pinned host memory that the engines read and write
  directly. This is simple but not the fastest path; see C12 in the roadmap.
- The engine manifest names model `mlvc`. The CLI sets it; library users set
  `artifacts.model_id = "mlvc"`, or the TensorRT session rejects the bundle as
  another model.
- Decoded output is not byte-identical to ONNX Runtime CPU. Streams decode in
  both directions between the providers. On 65 frames the two decoders drift
  apart until the frame-64 LTR recovery: to 57 dB Y PSNR of each other at
  176x144, 44 dB at 640x360 and 39 dB at 960x540, then back to about 58 dB.
  Quality against the source and stream size match within 0.2 dB and 0.2%.

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

Upstream's QNN session (`_model_wrapper.py`) uses `ORT_DISABLE_ALL` graph
optimization, `htp_performance_mode=burst` and
`htp_graph_finalization_optimization_mode=3`. MLVC requests the same settings
for its stages through codec provider options (`qnn.graph_optimization`,
`qnn.htp_performance_mode`, `qnn.htp_graph_finalization_optimization_mode`).
On the S25 they produced byte-identical streams and reconstructions to
ONNX Runtime's defaults while cutting the mean model time at 640x360 from
13.8 ms to 2.4 ms (encode) and from 15.3 ms to 3.2 ms (decode). They are not
provider defaults: as defaults, QNN failed to finalize DCVC-RT's `i_analysis`
graph. Other codecs keep ONNX Runtime's defaults (`ORT_ENABLE_ALL`, backend
path only).

`onnxruntime-qnn` also caches compiled HTP graphs as ONNX Runtime EPContext
models, by default in `<bundle>/.nvcr-cache`. The cache file name is a digest
of the ONNX Runtime version, the session and QNN options, the SoC and build
fingerprint, the name, size and modification time of each QNN library next to
`libQnnHtp.so`, and the bytes of the source model and its external data files.
A changed input therefore never reuses an old graph. An entry that fails to
load is deleted and rebuilt, and a model whose compiled graph cannot be saved
loads uncached. Each MLVC part is compiled only when a session
first needs it, so an encode-only process never compiles the decoder.

| Variable | Effect |
|---|---|
| `NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION` | `disable`, `basic`, `extended` or `all`; overrides the codec request and the default |
| `NVCR_QNN_HTP_PERFORMANCE_MODE` | QNN `htp_performance_mode`, for example `burst`, `balanced` or `default`; overrides the codec request |
| `NVCR_QNN_HTP_FINALIZATION_MODE` | QNN `htp_graph_finalization_optimization_mode` (`0` is ONNX Runtime's default); overrides the codec request |
| `NVCR_ONNXRUNTIME_CACHE_DIR` | Cache directory for compiled QNN graphs; `off` disables the cache |

The CLI prints the codec time to the first output, which includes lazy model
loading, separately from the steady rate of the remaining frames.

With `--verbose`, the MLVC sessions log mean host, model and entropy time per
frame when they flush. Before the cache, short CLI runs on the device were
dominated by session creation, about 4.8 s at 640x360, which is outside these
phases. See the
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
- Upstream-runtime parity (C10) covers MLVC-S perceptual only. The other three
  bundles are NVCR's own exports from the public checkpoints; they run on
  ONNX Runtime CPU and TensorRT, but no upstream reference vectors exist for
  them.
- TensorRT engine bundles exist only for the RTX 5060 Laptop GPU class
  (SM 12.0, TensorRT 10.9).
