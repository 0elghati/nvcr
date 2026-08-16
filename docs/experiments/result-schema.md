# Result schema

Each experiment row is one JSON object in a JSONL file. Required rows use the
legacy schema identifier `nvcr.softwarex.result.v1`, which remains unchanged
for tooling compatibility.

The existing `benchmark_resolution_matrix.sh` writes
`nvcr.benchmark.resolution-matrix.v1` rows with useful timing and PSNR fields,
but it does not satisfy the complete evaluation schema. Do not silently treat
those rows as complete; use `benchmark_softwarex_matrix.py`.

The diagnostic rows additionally record `execution_mode`, `container_image`,
and `container_digest`. They do not replace the strict evaluation driver's
`total_wall_time_ms`.

`scripts/benchmark_softwarex_matrix.py` is the schema-producing driver. It
writes one aggregate encode/decode-roundtrip row per sequence/profile/QP/GOP
case after the configured warm-up and measured repetitions. Optional profile
metrics come from additional repetitions requested with `--profile`.

## Required fields

| Group | Fields |
|---|---|
| Run | `schema`, `run_id`, `timestamp_utc`, `nvcr_commit`, `git_dirty`, `operation`, `command`, `commands`, `status`, `error_message` |
| Environment | `container_image`, `container_digest`, `native_build_id`, `os`, `architecture`, `driver_version`, `cuda_runtime_version`, `tensorrt_version` |
| Target | `target_id`, `gpu_name`, `compute_capability`, `multiprocessor_count`, `compatibility_class` |
| Identity | `codec_id`, `model_set_id`, `provider_id`, `build_target_id`, `engine_profile_id`, `engine_manifest_sha256`, `engine_bundle_sha256`, `engine_bundle_digest_kind`, `model_profile_sha256`, `target_profile_sha256`, `engine_profile_sha256` |
| Input | `sequence_id`, `input_sha256`, `input_bytes`, `input_redistribution`, `pixel_format`, `resolution`, `width`, `height`, `source_fps`, `frames`, `qp`, `gop_size`, `mode` |
| Measurement | `warmup_runs`, `measured_runs`, `profiling_enabled`, `profile_runs`, `performance_instrumentation` |
| Runtime | `encode_fps_mean`, `encode_fps_stddev`, `decode_fps_mean`, `decode_fps_stddev`, `total_wall_time_ms`, `encode_latency_ms_median`, `encode_latency_ms_p95`, `decode_latency_ms_median`, `decode_latency_ms_p95`, `first_frame_latency_ms` |
| Quality/size | `payload_bytes`, `bits_per_pixel`, `psnr_y`, `psnr_u`, `psnr_v`, `psnr_yuv` |
| Memory/reference | `peak_gpu_memory_mb`, `peak_host_memory_mb`, `python_reference_available`, `python_psnr_yuv`, `python_payload_bytes`, `python_vs_nvcr_psnr_yuv` |
| Compatibility comparison | `compatibility_baseline_available`, `exact_encode_fps_mean`, `exact_decode_fps_mean`, `encode_fps_ratio_vs_exact`, `decode_fps_ratio_vs_exact` |

Additional fields are welcome, but required fields must not disappear. Every
passing case requires finite clean-pass FPS, FPS standard deviation, total wall
time, payload bytes, and BPP. When `profiling_enabled` is `false`,
`profile_runs` is zero and the latency, first-frame, PSNR, and memory fields are
`null`; the case may pass as a performance-only diagnostic, but the package
cannot become `complete`. When `profiling_enabled` is `true`, those fields must
be finite for the case to pass. A complete package requires profiling on
every row.

A complete run records either `native_build_id` or both an image tag and
immutable `container_digest`.

For a local extracted directory, `engine_bundle_sha256` is a deterministic
digest over the engine manifest and its checksum manifest and
`engine_bundle_digest_kind` is `manifest-and-checksum-sha256`. Preserve the
catalog archive SHA-256 separately when a catalog archive is used.

`status` is `pass`, `fail`, or `skipped` for a case. A skipped required
case prevents the package-level status from becoming `complete`.

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
  "native_build_id": "rtx4070-ubuntu2404-release",
  "operation": "encode_decode_roundtrip",
  "command": "",
  "target_id": "rtx4070-ubuntu2404",
  "gpu_name": "NVIDIA GeForce RTX 4070",
  "compute_capability": "8.9",
  "multiprocessor_count": 46,
  "architecture": "x86_64",
  "os": "",
  "driver_version": "",
  "cuda_runtime_version": "",
  "tensorrt_version": "",
  "codec_id": "dcvc-rt",
  "model_set_id": "dcvcrt-cvpr2025",
  "provider_id": "tensorrt",
  "compatibility_class": "exact",
  "build_target_id": "rtx4070-ubuntu2404",
  "engine_profile_id": "720p",
  "engine_manifest_sha256": "",
  "engine_bundle_sha256": "",
  "engine_bundle_digest_kind": "manifest-and-checksum-sha256",
  "model_profile_sha256": "",
  "target_profile_sha256": "",
  "engine_profile_sha256": "",
  "sequence_id": "",
  "input_sha256": "",
  "input_bytes": 134092800,
  "input_redistribution": "local-only",
  "pixel_format": "yuv420p8",
  "resolution": "720p",
  "width": 1280,
  "height": 720,
  "source_fps": 60,
  "frames": 97,
  "qp": 32,
  "gop_size": 97,
  "mode": "ip",
  "warmup_runs": 1,
  "measured_runs": 3,
  "profiling_enabled": true,
  "profile_runs": 1,
  "performance_instrumentation": "disabled",
  "encode_fps_mean": null,
  "encode_fps_stddev": null,
  "decode_fps_mean": null,
  "decode_fps_stddev": null,
  "encode_latency_ms_median": null,
  "encode_latency_ms_p95": null,
  "decode_latency_ms_median": null,
  "decode_latency_ms_p95": null,
  "first_frame_latency_ms": null,
  "total_wall_time_ms": null,
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
  "compatibility_baseline_available": false,
  "exact_encode_fps_mean": null,
  "exact_decode_fps_mean": null,
  "encode_fps_ratio_vs_exact": null,
  "decode_fps_ratio_vs_exact": null,
  "status": "skipped",
  "error_message": "schema example; metrics were not measured",
  "commands": []
}
```

## Derivation notes

`bits_per_pixel = payload_bytes * 8 / (width * height * frames)` when the
payload covers the recorded frame range. The aggregate row keeps encode and
decode timing fields distinct.

`payload_bytes` is the sum of serialized `NVAU` access units. It includes
their NVAU headers and codec-private payloads and excludes outer `PacketIO` or
`NVCS` framing. Entropy or rANS bytes are a different boundary and may be
compared only with rows measured at that same boundary.

`encode_fps_mean`, `decode_fps_mean`, their standard deviations,
`total_wall_time_ms`, payload, and BPP come only from the normal measured
repetitions. Those commands omit verbose per-frame output, quality calculation,
and memory polling. Profile repetitions run afterward and contribute only
latency, first-frame latency, PSNR, and peak-memory fields. Their throughput and
process wall time are intentionally discarded.

The driver hashes exactly the measured YUV prefix
`width * height * 3 / 2 * frames`, even when the source file contains more
frames. During a `--profile` pass, host memory is sampled from the CLI process
and discrete-GPU memory is sampled through `nvidia-smi`. Unsupported samplers
produce `null` and fail the profiled case. A target-specific sampler must
populate the same field before that target can produce a complete package.

## Python reference rows

Input supplied through `--python-reference-jsonl` uses
`nvcr.softwarex.python-reference.v1`. The join fields are:

```text
sequence_id width height frames qp gop_size
```

Each row must contain `python_psnr_yuv`, `python_payload_bytes`,
`python_vs_nvcr_psnr_yuv`, `input_sha256`, `python_command`,
`python_source_commit`, `image_checkpoint_sha256`, and
`video_checkpoint_sha256`. The source/checkpoint values must match the current
model profile, and the input digest must match the measured YUV prefix. Rows
may also contain `python_encode_fps`, `python_decode_fps`, per-plane quality,
and reconstruction hashes.

## Exact baselines

Compatibility runs supply `--exact-baseline-jsonl`. Input rows must be passing
`nvcr.softwarex.result.v1` exact rows and join on target, sequence, dimensions,
frame count, QP, and GOP. The driver records encode/decode ratios. Every
same-compute or Ampere-plus case needs a match before its package can be
complete.

Same-compute baselines require identical numeric compute-capability major and
minor values. Both broader classes are desktop-only, remain bound to the
manifest's exact TensorRT `major.minor.patch`, and cannot be used to repair a
runtime-version mismatch. Jetson is exact-only.
