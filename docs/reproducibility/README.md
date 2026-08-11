# Reproducibility and review path

This directory is a compact landing area for an informed reviewer. It points
to the detailed user, architecture, artifact, and experiment documentation
without making publication metadata part of the software's identity.

## Software identity

NVCR is a native C++ runtime architecture for neural video codecs. Its first
complete production vertical is the DCVC-RT codec adapter with the TensorRT
FP16 provider on Linux/NVIDIA targets. The deterministic test codec and CPU
provider are conformance fixtures.

## Evaluated release

- Evaluated code version: `TO BE FROZEN FOR SUBMISSION`
- Permanent code-release link: `TO BE FROZEN FOR SUBMISSION`
- Reproducible evidence/capsule link: `TO BE FROZEN FOR SUBMISSION`
- Current repository: https://github.com/0elghati/nvcr

These placeholders are not publication metadata. An immutable tag, public
release assets, and a public evidence package are release gates.

## Five-minute smoke test

For a CPU-accessible contract check:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

This validates core contracts and fixtures; it does not run the production
DCVC-RT/TensorRT path. For the production path, install or prepare a validated
engine bundle and follow [Getting started](../getting-started.md).

## Review map

- [Identity and scope](../identity-and-scope.md)
- [Architecture](../architecture.md)
- [Extension guide](../extending-nvcr.md)
- [Elementary stream contract](../spec/nvcr-elementary-stream-v1.md)
- [Artifact preparation and selection](../dcvcrt-artifacts.md)
- [Compatibility](../compatibility.md)
- [Evaluation protocol](../experiments/evaluation-protocol.md)
- [Runbook](../experiments/runbook.md)
- [Result schema](../experiments/result-schema.md)
- [Claim-to-evidence matrix](claims-and-evidence.md)
- [Code metadata](code-metadata.md)
- [Pre-submission checklist](pre-submission-checklist.md)

## Known limitations

The public C++ API/ABI is not frozen. Only DCVC-RT/TensorRT is a production
vertical. TensorRT currently creates a provider-owned monolithic backend;
component-level executable loading is transitional. NVCR does not claim
upstream Python payload interchangeability, standard containers, FFmpeg
integration, universal TensorRT-plan portability, or unrestricted asset
redistribution.

## Licensing, citation, and support

NVCR source is MIT-licensed. Model, checkpoint, engine, runtime, and dataset
terms remain separate; see [the asset policy](../../ASSET_DISTRIBUTION_POLICY.md),
[model licensing](../../MODEL_LICENSES.md), and
[third-party notices](../../THIRD_PARTY_NOTICES.md).

Use [code metadata](code-metadata.md), [CITATION.cff](../../CITATION.cff), and
[SUPPORT.md](../../SUPPORT.md) for current software metadata and support routes.
