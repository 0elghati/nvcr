# Third-party integrations

NVCR vendors only the audited native rANS sources required by the runtime.
DCVC's coder, with its pinned origin, local optimization, and license notices,
is under `dcvc_rans/`. MLVC's `msrtc_rans` coder is under `msrtc_rans/`,
unmodified, with the MLVC license and notice. MLVC checkpoints and model
bundles are external, like DCVC-RT's; `mlvc_model/` holds only the libmlvc
license that `mlvc` engine archives carry with Microsoft's published bundle. The DCVC-RT Python checkout, checkpoints, exported model assets,
CUDA, and TensorRT remain external dependencies and are not vendored here.
