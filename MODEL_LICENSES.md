# Model and checkpoint licensing

Status: review required before redistribution.

## Pinned DCVC-RT model set

The current model profile is `dcvcrt-cvpr2025`, defined in
`configs/models/dcvcrt-cvpr2025.json`. Its provenance is pinned to:

- Repository: https://github.com/microsoft/DCVC.git
- Commit: `1feb52a592a9ff2c4e4ba2e5122e2da49a211466`
- Image checkpoint: `cvpr2025_image.pth.tar`
- Video checkpoint: `cvpr2025_video.pth.tar`

The checkpoint SHA-256 values and source URLs are documented in
`docs/dcvcrt-artifacts.md` and the model profile. NVCR does not assume that
source-code licensing grants permission to redistribute checkpoints or derived
model assets.

## Observed review status

The official Microsoft repository was verified on 2026-08-06. Its
`LICENSE.txt` applies the MIT License to the repository software, and its
`NOTICE.txt` records licenses and attribution for incorporated third-party
software. The official README links the DCVC-RT checkpoints, but neither the
repository license nor the checkpoint download page states a separate grant to
redistribute pretrained weights. Source use is therefore governed by the MIT
License and applicable NOTICE obligations; checkpoint redistribution remains
unresolved.

## Derived assets

Exported ONNX graphs, entropy/quant assets, and TensorRT plans are derived from
the pinned model set. Their redistribution status depends on the upstream model
terms, checkpoint terms, target/runtime terms, and any applicable dataset or
vendor restrictions. They are excluded from generic NVCR binary packages.

## Decision rule

Until an explicit license review records permission and required notices, model
checkpoints and derived model/engine assets are treated as restricted to local
validation. Release automation must fail or exclude them rather than infer a
permission from a URL, filename, or source-code license.
