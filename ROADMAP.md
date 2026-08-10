# NVCR roadmap

## Product direction

NVCR is a focused Linux runtime for neural video codecs. The first production
path is DCVC-RT running with TensorRT on NVIDIA hardware.

The current scope is:

- C++20 runtime and command line tool;
- DCVC-RT model profile `dcvcrt-cvpr2025`;
- TensorRT FP16 execution;
- target-local model and engine bundles;
- planar 8-bit YUV420 input and output;
- stateful I/P encoding and decoding with reset and flush; and
- bounded, versioned `NVAU` access units.

This file describes project priorities. It is not a benchmark log or a record
of one machine's temporary state.

## Completed foundations

- Runtime, codec adapter, and execution-provider boundaries.
- Static codec and provider registration.
- Session lifecycle, configuration, reset, and flush contracts.
- `NVAU` framing, validation, and stream documentation.
- Model, target, engine, and artifact identity checks.
- Target-aware engine catalog installation and resolution.
- Native TensorRT DCVC-RT encode/decode and the command line interface.
- CPU contract tests and TensorRT integration tests.
- Architecture-specific binary packaging and Docker workflows.

## Current milestone: release readiness

The next release milestone is complete when the first supported target paths
are reproducible, validated, packaged, and documented.

### Priorities

1. Validate exact target-local engine bundles on the registered desktop and
   Jetson targets.
2. Run the complete correctness gates: I-frame, I/P, reset and reuse,
   malformed input, artifact validation, and round-trip quality.
3. Produce one complete performance and reference-comparison package using the
   documented SoftwareX protocol.
4. Verify binary packages, engine catalog assets, licenses, checksums, and
   exclusion of checkpoints and derived model files.
5. Keep installation, usage, compatibility, and release documentation aligned
   with the implemented commands.

### Release criteria

A target can be described as supported only after it has:

- a clean Release build;
- validated target-local engine bundles;
- passing correctness and negative-path tests;
- complete I/P round-trip coverage;
- recorded performance and reconstruction-quality results; and
- a reproducible package with source, artifact, and environment identities.

## Later work

- Expand execution-provider support to additional runtimes beyond TensorRT.
- Add further neural video codec integrations through the existing codec and
  provider boundaries.
- Split model-component executable loading as additional runtimes are added.
- Stabilize the public session API and consider a C ABI.
- Add FFmpeg and standard-container integration with complete timestamp,
  drain, reset, and container tests.
- Evaluate additional precision and TensorRT portability modes only after
  exact-target correctness and performance evidence exists.

## Boundaries

The project does not currently promise CPU inference, Windows or macOS
support, universal TensorRT-plan portability, FFmpeg integration, standard
video containers, or redistribution of checkpoints, exported model assets, or
TensorRT plans.

See [Scope and support](docs/scope-and-support.md), [Performance](docs/performance.md),
[SoftwareX experiments](docs/experiments/README.md), and [Release policy](docs/releasing.md)
for the detailed contracts.
