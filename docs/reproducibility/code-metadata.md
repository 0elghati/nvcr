# Code metadata

This is a living inventory for the release and publication metadata table.
Values that depend on the evaluated release are intentionally not fabricated.

| Field | Current value |
|---|---|
| Software | NVCR, native runtime architecture for neural video codecs |
| Current code version | `0.18.0` in `version.txt`; evaluated submission version: `TO BE FROZEN FOR SUBMISSION` |
| Permanent evaluated release link | `TO BE FROZEN FOR SUBMISSION` |
| Reproducible evidence/capsule link | `TO BE FROZEN FOR SUBMISSION` |
| Repository | https://github.com/0elghati/nvcr |
| Legal code license | MIT; see `LICENSE` |
| Version control | Git, hosted on GitHub |
| Languages | C++20, CUDA, Python 3, CMake, Bash, YAML, Dockerfile |
| Core tools/services | CMake, CTest, CUDA, TensorRT, Docker/Compose, GitHub Actions |
| Compilation requirements | CMake 3.24+, C++20 compiler; CUDA/TensorRT for the production vertical |
| Operating environments | CPU contract tests on Linux development environments; production evidence currently Linux x86_64 or aarch64 NVIDIA targets |
| Dependencies | Threads; optional spdlog; CUDA/TensorRT for production; Python tooling for artifact/evidence workflows |
| Developer documentation | `docs/README.md`, architecture, API, extension, stream, artifact, and experiment guides |
| Support contact | `TO BE FINALIZED FOR SUBMISSION`; use repository issue/support route meanwhile |
| Current production codec/provider | DCVC-RT adapter / TensorRT FP16 |
| Conformance fixtures | Deterministic test codec / CPU provider |

The pinned DCVC-RT source/model identities and redistribution restrictions are
recorded in [DCVC-RT artifacts](../dcvcrt-artifacts.md), `MODEL_LICENSES.md`,
and `ASSET_DISTRIBUTION_POLICY.md`. A source-code license does not grant
checkpoint or derived-engine redistribution permission.
