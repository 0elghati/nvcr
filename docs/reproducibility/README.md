# Reproducibility

Reproducible NVCR results bind software, hardware, artifacts, inputs, and the
measurement boundary. A successful command without those identities is useful
for diagnosis, but it is not a portable performance result.

## Record the software identity

- For a source build, record `git rev-parse HEAD`, `git status --short`, and the
  CMake configuration.
- For a release install, record the release asset name and checksum.
- For a container, record both the tag and resolved image digest.
- Record the codec/provider IDs and the engine manifest and bundle digests.

The current release is available from the repository's
[latest release page](https://github.com/0elghati/nvcr/releases/latest).
Historical results retain the revision that produced them; do not relabel them
with a newer release.

## CPU contract validation

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DNVCR_ENABLE_TENSORRT=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

This validates portable runtime, parser, artifact, and lifecycle contracts. It
does not execute DCVC-RT through TensorRT and does not measure neural-codec
performance.

## GPU and measurement records

Use target-local engine validation and a real encode/decode round trip before
collecting measurements. Keep codec-loop and complete-process timing separate,
and compare rate only at the same byte boundary.

- [Performance protocol](../performance.md)
- [Detailed evaluation protocol](../experiments/evaluation-protocol.md)
- [Result schema](../experiments/result-schema.md)
- [Execution runbook](../experiments/runbook.md)
- [Retained result inventory](../../results/README.md)
- [Code metadata](code-metadata.md)

The C++ API/ABI is not frozen. NVCR does not claim native Windows support,
universal TensorRT-plan portability, upstream Python payload interchangeability,
standard-container support, or unrestricted model and engine redistribution.

See [SUPPORT.md](../../SUPPORT.md) for questions and bug reports, and
[SECURITY.md](../../SECURITY.md) for vulnerability reporting.
