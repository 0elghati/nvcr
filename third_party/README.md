# Third-party integrations

This directory contains third-party source, provenance records, and license
notices used by NVCR.

- `dcvc_rans/` contains the pinned native rANS sources used by the runtime.
- `dcvc_rt/` records the pinned DCVC-RT source and checkpoint provenance. The
  upstream checkout and checkpoints are obtained separately and are not vendored
  here.

Keep upstream source changes, licenses, notices, and pinned revisions documented
with the relevant integration. Do not treat an upstream source license as
permission to redistribute checkpoints or derived TensorRT assets.
