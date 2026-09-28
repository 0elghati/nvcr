# Asset distribution policy

## Included in generic NVCR packages

- NVCR binaries and public headers
- NVCR CMake package metadata
- NVCR source-owned documentation and scripts
- Third-party notices required for included source or binary components

## Excluded by default

The following remain target-local or externally obtained and are not included in
generic binary packages:

- DCVC-RT checkpoints
- ONNX graphs and exported model bundles
- entropy and quant assets derived from the model set
- TensorRT engine plans
- test sequences and datasets
- CUDA and TensorRT runtime installations

## Engine bundles

A validated engine bundle is distributed as a separate, explicitly named asset.
The MIT licensing basis for NVCR and the DCVC-RT materials is recorded in
`MODEL_LICENSES.md`. Each bundle must retain its target identity, TensorRT/CUDA
compatibility, model provenance, manifests and checksums, together with the
applicable Microsoft licence, upstream notice and NVCR licence. It must not be
silently placed in a generic architecture package. NVIDIA runtime components
and third-party datasets retain their own terms.

## Release gate

Before publishing an asset, check its provenance and inclusion of the applicable
licences and notices. The recorded MIT licensing basis does not remove these
attribution requirements. Components outside that basis require their own
applicable terms. A local validation result or public download URL is not an
attribution check.
