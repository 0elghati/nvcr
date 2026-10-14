# RTX 3050 TensorRT 10.9 engine publication

Date: 2026-10-12. Source: `6a66db7a98081f9210dedb49d768899bd4e9f2b7`.

The published RTX 3050 exact engines used TensorRT 10.7.0/CUDA runtime
12060 and could not be selected by the TensorRT 10.9.0/CUDA runtime 12080
container. The existing target-local exact bundles under
`build/engines/dcvcrt-3050-exact/` match live detection: Linux x86_64,
NVIDIA GeForce RTX 3050 Laptop GPU, SM 8.6, 16 multiprocessors.

## Admission evidence

- All six profiles pass bundle inventory, file checksum, and current model,
  engine-profile, and target-profile digest validation.
- All 84 plans independently load through TensorRT 10.9 `trtexec
  --loadEngine=<plan> --skipInference --device=0` without warnings/errors.
  Compact evidence: `plan-loads.json`; detailed logs remain local.
- Each profile passes four-frame synthetic YUV420P8 encode and decode at
  176x144, 352x288, 640x360, 960x540, 1280x720, or 1920x1080,
  respectively; QP 32, GOP 2, 30 FPS. Each run exercises I/P/I/P and
  reconstructs exactly four frames with the expected byte count.
  No runtime warnings/errors. See `round-trips.json`.
- Runtime image: `omarelghati/nvcr:1.0.7-amd64-cuda12.8-trt10.9`;
  immutable identity and labels are saved in `runtime-image.json`.
- `scripts/package_engine_bundle.sh` deterministically packages all six
  bundles, including NVCR LICENSE and upstream LICENSE.MIT/NOTICE.txt.
  Each archive is independently extracted and validated by the release
  admission helper. See `archive-digests.json`.
- Distribution basis: `MODEL_LICENSES.md` and
  `ASSET_DISTRIBUTION_POLICY.md`; archives contain derived plans and runtime
  assets, no checkpoint/ONNX files or NVIDIA SDK/runtime libraries.

## Publication status

Six archives and checksums are published to the public
[`engine-assets` release](https://github.com/0elghati/nvcr/releases/tag/engine-assets).
Local copies remain under `dist/rtx3050-trt109-20261012/`.
The published merged catalog retains 42 entries,
replacing the six exact RTX 3050 entries with the validated TensorRT 10.9
set. Other targets and the older SM 8.6 compatibility entries are preserved.
The upload script refreshes the live catalog immediately before merging,
uploads all archives first, checks remote sizes and SHA-256 digests, and
publishes the catalog last.

The user explicitly approved the model-derived payload and public destination.
The subsequent [upload-engine-assets workflow](https://github.com/0elghati/nvcr/actions/runs/38232945033)
also completed successfully using the six published archives.
All six remote archive sizes and SHA-256 digests match `archive-digests.json`;
release metadata is captured in `remote-assets.json`. The catalog was uploaded
after those checks passed.

A fresh anonymous catalog install through the 1.0.7 runtime image detected
the exact RTX 3050/TensorRT 10.9/CUDA runtime 12080 target, downloaded QCIF,
and installed it under `build/engines-publication-3050/`. A four-frame I/P
round trip using this downloaded bundle passed. Both encoded-stream and
reconstructed-YUV SHA-256 hashes equal the original local QCIF smoke hashes.
Detailed execution logs remain local as `published-encode.log` and
`published-decode.log`. These are artifact admission smokes, not performance
measurements or upstream compatibility tests.

Raw YUV/streams/plans, archives, and long logs remain outside Git.
