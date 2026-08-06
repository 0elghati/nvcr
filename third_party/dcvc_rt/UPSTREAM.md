# DCVC-RT upstream provenance

NVCR's first neural codec adapter is based on the pinned DCVC-RT source and
checkpoint/exporter lineage declared by `configs/models/dcvcrt-cvpr2025.json`.

- Repository: https://github.com/microsoft/DCVC.git
- Source commit: see `upstream-commit.txt`
- Source layout at the pinned commit: repository root (DCVC-RT was moved there
	by the pinned reorganization commit)
- Model profile: `dcvcrt-cvpr2025`
- Image checkpoint: `cvpr2025_image.pth.tar`
- Image checkpoint SHA-256: `555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9`
- Video checkpoint: `cvpr2025_video.pth.tar`
- Video checkpoint SHA-256: `b12e7faf4ddb6126d8e138a627ed6a349b8e1052d3ed9e343e1ba266466675d6`

NVCR does not vendor the upstream checkout or checkpoints in this directory.
The preparation scripts verify the source commit and checkpoint digests before
exporting model assets. TensorRT plans are derived target-local artifacts and
must be rebuilt on the final target/runtime.

The exporter and builder entry points are the backend-local scripts under
`scripts/backends/dcvcrt/`. Their exact inputs and selected engine profiles are
recorded in the generated model and engine manifests.

The official pinned checkout was inspected on 2026-08-06. Microsoft distributes
the repository's software under the MIT License in `LICENSE.txt`. Its
`NOTICE.txt` records third-party source attributions and license terms that must
remain with applicable source-derived distributions. The repository license is
not treated as a separate redistribution grant for pretrained checkpoints or
derived model and engine assets.

License and redistribution status for upstream source, checkpoints, generated
ONNX assets, and TensorRT plans is intentionally tracked separately in the root
`MODEL_LICENSES.md` and `ASSET_DISTRIBUTION_POLICY.md` files.
