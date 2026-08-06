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

A validated engine bundle may be distributed as a separate, explicitly named
asset only after its target identity, TensorRT/CUDA compatibility, model
provenance, checksums, and redistribution terms have been reviewed. The bundle
must retain its manifest and checksum file. It must not be silently placed in a
generic architecture package.

## Release gate

Unresolved licensing or provenance status is a release blocker for the affected
asset. Packaging and release automation must exclude restricted checkpoints,
model bundles, engine plans, and test data until the review is recorded. A local
validation result is not permission to redistribute an asset.
