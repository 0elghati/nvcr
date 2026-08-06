# Result schema

Each experiment row is one JSON object in a JSONL file. Required rows use schema `nvcr.softwarex.result.v1`.

The existing `benchmark_resolution_matrix.sh` writes `nvcr.benchmark.resolution-matrix.v1` rows with useful timing and PSNR fields, but it does not yet satisfy this complete SoftwareX schema. Do not silently treat those rows as publication-ready; wrap or extend the benchmark output before the paper run.

## Required fields

| Group | Fields |
|---|---|
| Run | `schema`, `run_id`, `timestamp_utc`, `nvcr_commit`, `git_dirty`, `commands`, `status`, `error_message` |
| Environment | `container_image`, `container_digest`, `os`, `architecture`, `driver_version`, `cuda_runtime_version`, `tensorrt_version` |
| Target | `target_id`, `gpu_name`, `compute_capability`, `compatibility_class` |
| Identity | `codec_id`, `model_set_id`, `provider_id`, `engine_profile_id`, `engine_manifest_sha256`, `engine_bundle_sha256`, `model_profile_sha256` |
| Input | `sequence_id`, `input_sha256`, `pixel_format`, `width`, `height`, `frames`, `qp`, `gop_size`, `mode`, `warmup_runs`, `measured_runs` |
| Runtime | `encode_fps_mean`, `encode_fps_stddev`, `decode_fps_mean`, `decode_fps_stddev`, `encode_latency_ms_median`, `encode_latency_ms_p95`, `decode_latency_ms_median`, `decode_latency_ms_p95` |
| Quality/size | `payload_bytes`, `bits_per_pixel`, `psnr_y`, `psnr_u`, `psnr_v`, `psnr_yuv` |
| Memory/reference | `peak_gpu_memory_mb`, `peak_host_memory_mb`, `python_reference_available`, `python_psnr_yuv`, `python_payload_bytes`, `python_vs_nvcr_psnr_yuv` |

Additional fields are welcome, but required fields must not disappear. Use JSON `null` for a metric that was not measured and explain why in `error_message` or the run summary.

## Example

```json
{
  "schema": "nvcr.softwarex.result.v1",
  "run_id": "20260806-rtx4070-exact-720p-gop97-qp32",
  "timestamp_utc": "2026-08-06T00:00:00Z",
  "nvcr_commit": "",
  "git_dirty": false,
  "container_image": "",
  "container_digest": "",
  "target_id": "rtx4070-ubuntu2404",
  "gpu_name": "NVIDIA GeForce RTX 4070",
  "compute_capability": "8.9",
  "architecture": "x86_64",
  "os": "",
  "driver_version": "",
  "cuda_runtime_version": "12.6",
  "tensorrt_version": "10.7.0",
  "codec_id": "dcvc-rt",
  "model_set_id": "dcvcrt-cvpr2025",
  "provider_id": "tensorrt",
  "compatibility_class": "exact",
  "engine_profile_id": "720p",
  "engine_manifest_sha256": "",
  "engine_bundle_sha256": "",
  "model_profile_sha256": "",
  "sequence_id": "",
  "input_sha256": "",
  "pixel_format": "yuv420p8",
  "width": 1280,
  "height": 720,
  "frames": 97,
  "qp": 32,
  "gop_size": 97,
  "mode": "ip",
  "warmup_runs": 1,
  "measured_runs": 3,
  "encode_fps_mean": null,
  "encode_fps_stddev": null,
  "decode_fps_mean": null,
  "decode_fps_stddev": null,
  "encode_latency_ms_median": null,
  "encode_latency_ms_p95": null,
  "decode_latency_ms_median": null,
  "decode_latency_ms_p95": null,
  "payload_bytes": null,
  "bits_per_pixel": null,
  "psnr_y": null,
  "psnr_u": null,
  "psnr_v": null,
  "psnr_yuv": null,
  "peak_gpu_memory_mb": null,
  "peak_host_memory_mb": null,
  "python_reference_available": false,
  "python_psnr_yuv": null,
  "python_payload_bytes": null,
  "python_vs_nvcr_psnr_yuv": null,
  "status": "pass",
  "error_message": "",
  "commands": []
}
```

## Derivation notes

`bits_per_pixel = payload_bytes * 8 / (width * height * frames)` when the payload covers the recorded frame range. State any other convention. Keep encode and decode rows distinct when their timings differ.
