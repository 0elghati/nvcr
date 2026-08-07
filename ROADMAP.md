# NVCR roadmap

Last reviewed: 2026-08-07

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

The extensible-runtime contracts have landed, but the production provider
handoff is not complete.

Landed:

- codec adapter and execution provider boundaries;
- static codec/provider registry;
- session-oriented contracts with compatibility one-shot calls;
- versioned `NVAU` format and stream specification;
- artifact provenance, catalog loading, and typed resolver;
- explicit separation of model components from provider engine profiles;
- deterministic test codec/provider contracts;
- CPU and TensorRT contract coverage.

Refactor closeout:

- pass `RuntimeServices` into the production DCVC-RT adapter and request
  component executables through it;
- replace the registered TensorRT provider's `not_implemented` load path with
  production executable loading;
- prove that route with provider-level and exact-target I/P/reset tests;
- either make the component CMake targets own their implementations or document
  them strictly as interface/grouping targets.

Today the working DCVC-RT path constructs `make_tensorrt_backend()` directly.
That remains usable, but it does not yet prove the intended provider-mediated
production architecture.

## Next phases

1. **Refactor closeout:** route production DCVC-RT execution through
   `RuntimeServices` and the TensorRT provider, then rerun the contract and
   exact-target GPU suites.
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

- RTX 3050 still needs a checked-in
  `rtx3050-laptop-ubuntu2404` profile and rebuilt exact artifacts;
- RTX 4070's profile records TensorRT 10.7 while the detected host and local
  bundles report 10.9;
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
- a real RTX 4070 plan-only preflight correctly failed before execution because
  target TensorRT 10.7 does not match detected TensorRT 10.9; the repeated
  preflight with `--profile` also recorded the intended profiling mode before
  preserving that strict failure;
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
