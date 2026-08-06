# Third-party notices

This file records third-party materials currently present in or required by the
NVCR source tree. It is a provenance index, not a grant of rights beyond the
upstream licenses and terms.

## NVCR

NVCR source is distributed under the MIT License in `LICENSE`.

## DCVC-RT

The pinned source repository and commit are recorded in
`third_party/dcvc_rt/UPSTREAM.md`, `upstream-repository.txt`, and
`upstream-commit.txt`. Microsoft distributes the official repository software
under the MIT License in its `LICENSE.txt`; its `NOTICE.txt` records applicable
CompressAI, BSD 3-Clause Clear, and MIT attributions. NVCR does not vendor that
checkout. Any distribution containing official DCVC source or a derivative
must retain the MIT notice and applicable upstream NOTICE content. These source
terms do not establish checkpoint or derived-engine redistribution rights.

## DCVC-RT native rANS import

The imported rANS sources under `third_party/dcvc_rans/` carry their own
`LICENSE.MIT`, `NOTICE.txt`, and upstream attribution. Those files must remain
with any distribution containing the imported sources or derived binaries.

## Runtime dependencies

CUDA and TensorRT are external NVIDIA components discovered by CMake. Their
licenses and redistribution terms are not replaced by NVCR documentation.
NVCR binary packaging does not include CUDA, TensorRT, checkpoints, ONNX graphs,
or TensorRT plans.

## Test sequences and datasets

Test sequence files are external assets. Their terms are not inferred from NVCR
source licensing and must be reviewed before copying them into releases or CI
artifacts.
