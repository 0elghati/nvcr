# Code metadata

| Field | Value |
|---|---|
| Software | NVCR, native C++ runtime architecture for neural video codecs |
| Repository | https://github.com/0elghati/nvcr |
| Current release | [Latest release](https://github.com/0elghati/nvcr/releases/latest); the exact version is recorded in `version.txt` |
| Development identity | Commit from `git rev-parse HEAD` plus dirty state |
| Source license | MIT; see [`LICENSE`](../../LICENSE) |
| Languages | C++20, CUDA, Python 3, CMake, Bash, PowerShell, YAML, Dockerfile |
| Build tools | CMake 3.24+, CTest, a C++20 compiler; CUDA and TensorRT for GPU execution |
| Supported codec/provider | DCVC-RT with TensorRT FP16 on qualified Linux/NVIDIA targets |
| Conformance fixtures | Deterministic test codec and CPU provider |
| Public API/ABI | Pre-v1 and not frozen |
| Support | [`SUPPORT.md`](../../SUPPORT.md); security reports follow [`SECURITY.md`](../../SECURITY.md) |

Every retained GPU result must additionally record the platform, GPU, CUDA and
TensorRT versions, engine identity, input digest, command, and timing and byte
boundaries. Container results must record the resolved image digest.

The pinned DCVC-RT source/model identities and redistribution restrictions are
documented in [DCVC-RT artifacts](../dcvcrt-artifacts.md),
[`MODEL_LICENSES.md`](../../MODEL_LICENSES.md), and
[`ASSET_DISTRIBUTION_POLICY.md`](../../ASSET_DISTRIBUTION_POLICY.md).
