# MLVC-S model bundle (`mlvc` engine archives)

Source repository: `microsoft/libmlvc`, MIT licensed (`LICENSE.MIT`, copied
unchanged at commit `eb5d60a24cbaa487b9ca33e232e645b511841b96`).

NVCR does not vendor the model. `mlvc` engine archives on the `engine-assets`
release carry Microsoft's published bundle
`data/model_bundles/dmc61sbr_mini_reglu-perceptual_v1-onnx-generic-v0.1.tar`
(SHA-256 `c37ee2f5aa418c1fbdd44e4bd95483faad84cbea592367b305e787042f845457`)
unchanged, beside TensorRT plans built from its ONNX parts by
`scripts/backends/mlvc/build_tensorrt_bundle.py`. The plans embed the same
weights. `scripts/package_engine_bundle.sh` copies this license into each
archive.
