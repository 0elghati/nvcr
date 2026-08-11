# NVCR roadmap

Last reviewed: 2026-08-11

## Scope

```text
DCVC-RT dcvcrt-cvpr2025 -> TensorRT FP16 -> native I/P encode and decode
```

NVCR provides bounded `NVAU` access units, raw YUV420P8 CLI I/O,
provider-mediated execution, target-aware artifacts, Docker images, and
reproducible evaluation tooling. It is not yet a supported v1 release.

## Status

| Milestone | Status | Remaining gate |
|---|---|---|
| Runtime and stream contracts | Implemented | Maintain parser, reset, flush, and I/P coverage |
| TensorRT provider path | Implemented | Split model stages only when another production provider is added |
| Binary and container packaging | Implemented | Complete license and clean-package checks |
| Exact-target artifacts | In progress | Produce current warning-free profile sets |
| SoftwareX evaluation | In progress | Complete exact matrices and pinned Python comparisons |
| Compatibility classes | Experimental | Compare against complete exact baselines |
| FFmpeg and standard containers | Out of scope | No support claim |

Generic packages exclude checkpoints, exported model assets, TensorRT plans,
and datasets. Validated engine bundles use the separate rolling catalog and
remain bound to their recorded GPU, CUDA, TensorRT, model, and profile identity.

## Target state

| Target | Current gate |
|---|---|
| RTX 3050 Laptop | Clean exact artifacts and results required |
| RTX 4070 | Rebuild exact bundles from the current model profile |
| RTX 5060 Laptop | Clean exact artifacts and results required |
| Jetson Orin Nano | Complete the warning-free six-profile gate |

## Evidence

Publication results come from `scripts/benchmark_softwarex_matrix.py` and must
record clean source identity, detected hardware/software, artifact digests,
registered tests, exact inputs, FPS, wall time, separate profiling runs, BPP,
quality, and pinned Python comparisons.

The retained `evidence/live-release-20260806/resolution-matrix.jsonl` is
diagnostic, not publication evidence. Dirty, partial, skipped, plan-only, or
missing-metric runs remain diagnostic.

Verified implementation evidence includes CPU tests, TensorRT Release builds,
provider-selection contracts, package-consumer checks, focused SoftwareX
driver tests, and exact Orin validation through 720p. The public documentation
audit also passed link, shell-snippet, Compose, credential, and local-path
checks. Native installer tests cover default, selected multi-profile, and
explicit all-profile installation.

## Next action

Complete the Orin 1080p bundle and six-profile gate. Then rebuild RTX 4070
exact bundles and run the complete SoftwareX matrix with pinned Python rows.

## v1 exit criteria

1. Clean source and package builds.
2. Complete parser, reset, flush, malformed-input, and I/P tests.
3. Validated target-local artifacts and runtime round trips.
4. Complete reproducible SoftwareX results.
5. Final dependency, model, dataset, and package-license review.
6. Installation and external-consumer verification from published artifacts.

Additional codecs/providers, INT8 release support, a stable C ABI, FFmpeg,
standard containers, and universal TensorRT-plan portability remain outside v1.
