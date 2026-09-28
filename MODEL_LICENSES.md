# Model and checkpoint licensing

## Licensing basis

NVCR's licensing basis is MIT for its own code and the Microsoft DCVC-RT
source, pretrained checkpoints and derived model assets used by this project.
Retain the copyright and permission notices when distributing these materials.
Applicable third-party notices also remain in force. This declaration does not
apply the MIT licence to CUDA, TensorRT or third-party datasets.

Microsoft's pinned [LICENSE.txt](https://github.com/microsoft/DCVC/blob/1feb52a592a9ff2c4e4ba2e5122e2da49a211466/LICENSE.txt)
and [NOTICE .txt](https://github.com/microsoft/DCVC/blob/1feb52a592a9ff2c4e4ba2e5122e2da49a211466/NOTICE%20.txt)
are retained as [LICENSE.MIT](third_party/dcvc_rt/LICENSE.MIT) and
[NOTICE.txt](third_party/dcvc_rt/NOTICE.txt). The space in the upstream notice
filename is intentional. That notice records incorporated third-party material;
its individual terms are not replaced by NVCR's licence.

## Pinned DCVC-RT model set

The model profile is `dcvcrt-cvpr2025`, defined in
`configs/models/dcvcrt-cvpr2025.json`:

- Repository: https://github.com/microsoft/DCVC.git
- Commit: `1feb52a592a9ff2c4e4ba2e5122e2da49a211466`
- Image checkpoint: `cvpr2025_image.pth.tar`
- Video checkpoint: `cvpr2025_video.pth.tar`

Checkpoint SHA-256 values are recorded in the model profile. Acquisition and
conversion instructions are in [DCVC-RT artifacts](docs/dcvcrt-artifacts.md).

## Derived assets and distribution

ONNX graphs, entropy/quantization files and TensorRT plans retain the model
provenance and applicable attribution. TensorRT plans are distributed separately
from generic NVCR packages for specific GPU and CUDA/TensorRT configurations.
CUDA and TensorRT remain subject to NVIDIA's terms; distributing a generated
plan does not relicense NVIDIA's SDK or runtime libraries.

Include the applicable Microsoft licence, upstream notice and NVCR licence
with derived bundles. Keep their checkpoint, target and digest records intact.
The [asset distribution policy](ASSET_DISTRIBUTION_POLICY.md) records the package
requirements. Public availability and checksum verification do not demonstrate
that a particular package includes all required notices.
