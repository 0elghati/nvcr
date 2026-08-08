# NVCR roadmap

Last reviewed: 2026-08-08

## Current position

NVCR is a Linux C++20 runtime architecture for neural video codecs. The first
working execution path is DCVC-RT with the TensorRT backend.

The current scope is intentionally small:

- DCVC-RT model profile `dcvcrt-cvpr2025`;
- NVIDIA Linux targets;
- TensorRT FP16;
- target-local model and engine bundles;
- native I/P encode/decode, reset, flush, and bounded `NVAU` access units.

This is development software, not a supported v1 release. A capability is implemented when code and contract tests exist; it is supported only after the relevant target evidence passes.

## Architecture work

The extensible-runtime contracts have landed. Production DCVC-RT component
creation now goes through provider management, but model-stage executable
loading is not split out of the TensorRT backend yet.

Landed:

- codec adapter and execution provider boundaries;
- static codec/provider registry;
- session-oriented contracts with compatibility one-shot calls;
- versioned `NVAU` format and stream specification;
- artifact provenance, catalog loading, and typed resolver;
- explicit separation of model components from provider engine profiles;
- provider-selected codec component creation through `RuntimeServices`;
- explicit runtime `provider_id` configuration and CLI `--provider` selection;
- deterministic test codec/provider contracts;
- CPU and TensorRT contract coverage.

Remaining provider closeout:

- split the current TensorRT codec backend into provider-loaded model-component
  executables when a second provider, such as ONNX Runtime, is added;
- prove that route with provider-level and exact-target I/P/reset tests;
- either make the component CMake targets own their implementations or document
  them strictly as interface/grouping targets.

Today the working DCVC-RT adapter no longer constructs `make_tensorrt_backend()`
directly. TensorRT remains usable as the only production provider, and the
lower-level factory remains available for direct TensorRT tests.

## Next phases

1. **Provider closeout:** rerun exact-target GPU suites through the
   provider-owned TensorRT path, then split provider-loaded model components when
   adding the next execution provider.
2. **CI and package hardening:** wire the header-dependency check and installed
   external-consumer test into CI, run the Clang sanitizer/parser jobs, and
   verify release package licenses and excluded model/engine assets.
3. **Reference-target refresh:** reconcile
   `rtx4070-ubuntu2404` with the actual TensorRT 10.9 reference environment and
   rebuild every engine from the current model/engine/target profile digests.
4. **SoftwareX exact evidence:** provide the five-profile input manifest and
   pinned Python rows, run the new publication driver on RTX 4070, then repeat
   on RTX 3050, RTX 5060, and Orin as access and target profiles permit.
5. **Compatibility evidence:** build same-compute and Ampere-plus bundles only
   after exact rows exist, then measure correctness and performance ratios
   against those exact baselines.

The next evaluation should cover correctness, I/P GOP behavior, reset/reuse,
malformed input rejection, runtime, memory, payload size, reconstruction
quality, and the mandatory RTX 4070 pinned-Python comparison.

## Evidence policy

Only the latest machine-readable matrix is retained at
[evidence/live-release-20260806/resolution-matrix.jsonl](evidence/live-release-20260806/resolution-matrix.jsonl).
It is a diagnostic record, not a post-closeout release result. Raw streams and
older run bundles were removed; the next clean run replaces this file.

No performance, energy, cross-runtime, or target-support claim should be made from the retained file alone.

The schema-producing driver now exists at
`scripts/benchmark_softwarex_matrix.py` and has focused unit coverage. Current
evaluation blockers are:

- RTX 3050's checked-in `rtx3050-laptop-ubuntu2404` profile and exact artifacts
  still need validation;
- RTX 4070's profile now matches the detected TensorRT 10.9 host, but local
  bundles still predate the current model profile digest;
- the local 2026-08-05 RTX 4070 bundles predate the current model-profile
  digest and must be rebuilt;
- the raw input manifest and pinned Python comparison rows are not yet
  available;
- RTX 3050, RTX 5060, and Orin require target access; Orin also needs a
  target-local GPU-memory sampler if `nvidia-smi` is unavailable.

Performance collection follows one explicit boundary: FPS, variation,
compatibility ratios, and total wall time come from repetitions without verbose
logging, quality calculation, or process/GPU polling. The `--profile` flag runs
separate repetitions for per-frame latency, PSNR, and memory. Performance-only
runs remain useful diagnostics but cannot become complete publication
packages.

## Verification evidence

The 2026-08-06 repository audit, updated on 2026-08-07, established:

- clean baseline `6938b87` matched `origin/main` before this documentation work;
- a fresh GCC Release CPU build passed all 10 tests registered at that baseline;
- after registering the SoftwareX driver test, the reconfigured CPU tree passed
  all 11 tests;
- `scripts/ci/check_header_deps.sh` passed;
- a temporary external CMake consumer found `NVCR::nvcr`, compiled, linked, and
  ran against the install tree;
- all 13 focused SoftwareX driver tests pass, including coverage that keeps
  profiled repetitions out of primary FPS and wall-time aggregates;
- on 2026-08-08, `scripts/benchmark_resolution_matrix.sh` was syntax-checked
  after aligning diagnostic JSONL output with the CSV/report overwrite behavior;
- on 2026-08-08, `scripts/build_orin_exact_engines.sh` was added and
  syntax/help checked as the Orin exact-engine rebuild wrapper; target-local
  GPU execution still has to run on the Jetson;
- on 2026-08-08, provider-managed DCVC-RT component creation landed:
  `RuntimeConfiguration.provider_id`, `RuntimeServices::create_components`, the
  DCVC-RT adapter's provider-owned construction path, the TensorRT provider
  component factory, CLI `--provider`, and contract coverage for provider-owned
  components;
- after that provider-management change, a fresh GCC Release CPU build passed
  all 11 tests, and a TensorRT-enabled Release build completed successfully;
- the TensorRT-enabled tree passed 12 non-CUDA-manager tests with
  `ctest --test-dir build-release --output-on-failure -E nvcr_cuda_ops`; the
  full 13-test run failed only at `nvcr_cuda_ops` because this local environment
  reported `NvRmMemInitNvmap failed`, so no target-local GPU execution claim is
  made from that run;
- a real RTX 4070 plan-only preflight previously failed before execution because
  the target version did not match detected TensorRT 10.9; the target and
  desktop build stack are now unified on TensorRT 10.9;
- the current model-profile SHA-256 is
  `3aaa4e2f3d309090e2947e065ce9d6be8456b01d14fde8a05f87f0daed348246`;
  the local RTX 4070 manifests record the stale
  `cc9477edd5724bfd3bbc2fbe1e34b73380ad3f338911b1ba42855e04ca23478c`.

The GPU matrix was not rerun: current target/artifact identity is invalid, so a
run would be diagnostic rather than release evidence.

## Release boundary

A v1 release must include a clean source build, target-local artifact validation, complete I/P round trips, format and negative-path tests, a reproducible evaluation record, and final licensing/package review.

The following remain outside v1:

- additional production codecs or providers;
- INT8 release support;
- a public C ABI;
- FFmpeg and standard container integration;
- universal TensorRT-plan portability;
- redistribution of checkpoints, exported model assets, or TensorRT plans.

See [Scope and support](docs/scope-and-support.md), [Performance](docs/performance.md), [SoftwareX experiments](docs/experiments/README.md), and [Release policy](docs/releasing.md) for the current contracts.
