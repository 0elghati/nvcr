# Phase C5 MLVC reference/export evidence — 2026-09-14

## Scope

This evidence records the bounded upstream experiment selected by
[the C0 audit](../docs/cross-codec-requirements.md). It compares one public MLVC
model through the pinned in-memory PyTorch CPU reference and an exported ONNX
model running through ONNX Runtime CUDA. It does not add an NVCR provider, codec
adapter, artifact format, or stream change.

## Pinned inputs

- Source: `https://github.com/microsoft/mlvc.git`, `main` at
  `b881d799af62b8640c14c9ee6df3fb7fea679156`
- Model: `dmc61sbr_mini_reglu`, public `mlvc-s-psnr-v1.ckpt`
- Checkpoint SHA-256:
  `1b86b757ddb115342293efb57719d6216c6ee2e459ae796ec41723b5c05ca896`
- Input: `akiyo_qcif.yuv`, 176x144, first two frames
- Input SHA-256:
  `e1efee0e95c6d27aefe2294a727768334db8466e67bf4a49cf8f5b2fb8b49108`
- Rate points: Q indices 21 and 49
- Export: ONNX, generic target, upstream-default FP16 precision
- Reference runtime: Python 3.12.3, PyTorch 2.10.0 on CPU
- Candidate runtime: ONNX Runtime GPU 1.26.0, CUDA execution provider
- Entropy package: `msrtc-rans` 0.1.0
- Hardware: NVIDIA GeForce RTX 4070, compute capability 8.9, 12,282 MiB
- Driver: 580.173.02

The first environment resolution selected unpinned `onnxruntime-gpu` 1.29.0.
Its CUDA provider required `libcublasLt.so.13` and fell back to CPU. That run is
rejected. [ONNX Runtime's CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
state that PyPI GPU packages from 1.27 use CUDA 13 by default, while releases
1.21 through 1.26 use CUDA 12.8 and cuDNN 9. The audit therefore pinned 1.26.0.
A direct `InferenceSession` over the exported encoder reported
`CUDAExecutionProvider` followed by `CPUExecutionProvider`, and
`onnxruntime.get_device()` reported `GPU`.

A C6 source trace found that the accepted `--torch-device cuda` setting is not
applied to the in-memory model used for conversion results. A direct check
reported `requested=cuda` and `parameter_device=cpu`. C5 is therefore a
PyTorch CPU reference versus ONNX Runtime CUDA comparison.

## Acceptance rule

The limits were declared before the comparison:

- absolute drift no greater than 0.05 dB for every reported PSNR aggregate;
- absolute mean-BPP drift no greater than 0.0001 at every rate point; and
- an actual CUDA execution provider must load rather than silently falling back.

Every condition must pass. The limits were not changed after observing the
result.

## Default FP16 result

The upstream exporter completed, loaded both ONNX parts with its CUDA provider,
ran the identical two-frame input through both paths, and wrote
`validate_conversion.json`.

| Q index | Mean PSNR delta (dB) | Largest reported PSNR delta (dB) | Mean BPP delta | Result |
|---:|---:|---:|---:|---|
| 21 | -0.0326 | -0.1608 (`V` mean) | -0.001736 | Fail |
| 49 | +0.0137 | -0.0285 (`V` mean) | -0.000631 | Fail |

Q=21 exceeds both limits. Q=49 stays inside the PSNR limit but exceeds the BPP
limit. The default FP16 artifact/provider pair is rejected.

## FP32 diagnostic

FP32 was tested with the same unchanged limits to separate default-FP16 drift
from general ORT-CUDA feasibility. The pinned exporter cannot run its normal
FP32 optimization path: `_onnx_exporter.py` imports `onnx` only inside the FP16
branch and then references that local name in the shared optimization branch,
raising `UnboundLocalError` at line 95.

No upstream source was patched. The exporter-supported
`optimization_passes=[]` parameter allowed a clearly labelled unoptimized FP32
diagnostic:

| Q index | Mean PSNR delta (dB) | Largest reported PSNR delta (dB) | Mean BPP delta | Result |
|---:|---:|---:|---:|---|
| 21 | +0.0193 | +0.1186 (`V` mean) | -0.000631 | Fail |
| 49 | +0.0429 | +0.0572 (`PSNR` max) | -0.000947 | Fail |

This diagnostic also fails, but its missing graph passes prevent treating it as
a production FP32 candidate.

## Commands

The accepted CUDA 12-compatible run used the upstream entry point:

```text
rtk uv run --no-sync python convert.py \
  --job-outputs-dir /tmp/nvcr-c5-mlvc-job \
  --test-data-dir /home/oelghati/datasets/qcif \
  export --model-version dmc61sbr_mini_reglu \
  --model-type onnx --target-device generic --precision fp16 \
  --model-width 176 --model-height 144 --frame-count 2 \
  --output-path /tmp/nvcr-c5-mlvc-output-ort126 \
  --torch-device cuda --onnx-execution-provider cuda \
  --exporter-params-json '{"test_video_path":"/home/oelghati/datasets/qcif/akiyo_qcif.yuv","image_width":176,"image_height":144,"q_index_list":[21,49]}'
```

The result JSON was queried at full precision rather than copied from the
rounded console table.

## Decision

C5 is complete as a rejected feasibility result. The public checkpoint and the
upstream default generic FP16 export are reproducible, and ONNX Runtime CUDA is
available with an explicit CUDA 12-compatible pin, but the pair does not meet
the predeclared numerical contract. NVCR must not add an ORT-CUDA provider or
MLVC adapter on this evidence.

C6 subsequently located the first stream-relevant divergence in encoder raw
symbols before entropy coding and rejected this artifact/provider pair as the
next production axis. See
[the C6 evidence](phase-c6-mlvc-cuda-equivalence-20260914.md).
