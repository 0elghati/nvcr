# Model and checkpoint licensing

Status: review required before redistribution.

## Pinned DCVC-RT model set

The current model profile is `dcvcrt-cvpr2025`, defined in
`configs/models/dcvcrt-cvpr2025.json`. Its provenance is pinned to:

- Repository: https://github.com/0elghati/DCVC-RT.git
- Commit: `48ab0ac5e5199d78fffb944bfbafafb2b6142f7b`
- Image checkpoint: `cvpr2025_image.pth.tar`
- Video checkpoint: `cvpr2025_video.pth.tar`

The checkpoint SHA-256 values and source URLs are documented in
`docs/dcvcrt-artifacts.md` and the model profile. NVCR does not assume that
source-code licensing grants permission to redistribute checkpoints or derived
model assets.

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
