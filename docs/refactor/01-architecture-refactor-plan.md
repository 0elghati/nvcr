# M-EXT: extensible runtime architecture refactor plan

Date: 2026-08-06
Branch: `refactor/extensible-runtime-v1`
Prerequisite: [00-current-state-audit.md](00-current-state-audit.md)

This is the phased implementation plan for milestone `M-EXT` referenced from
`ROADMAP.md`. It exists so implementation can proceed across many separate
sessions against a single durable plan instead of re-deriving scope each time.
Each phase is its own build-test-verify cycle per `AGENTS.md`; do not start a
phase's code changes without first re-reading its "Depends on" phase's
verification evidence.

## Governing decisions

- M1–M4 evidence-gathering is paused (not abandoned) in favor of `M-EXT` at the
  user's explicit direction, recorded in `ROADMAP.md` under "Reprioritization
  decision, 2026-08-06".
- Preserve current wire format and CLI behavior via compatibility aliases
  through Phase 3; only intentionally break format/API in later phases with a
  documented migration path — never silently.
- Static registration only; no dynamic plugin ABI is introduced.
- No second real execution provider (e.g. ONNX Runtime) or second real codec
  is added; only the required test-only codec/provider implementations
  (Phase 5) prove the boundaries are real.
- Every phase must re-run `cmake --build build-release -j 8` (full CUDA/
  TensorRT) and a core-only (`NVCR_ENABLE_TENSORRT=OFF`) build, plus
  `ctest --test-dir build-release --output-on-failure`, and record pass/fail
  counts before being considered complete.
- Any phase touching `src/dcvcrt/**` or `include/nvcr/bitstream/**` must show
  byte-identical NVAU streams and unchanged PSNR-YUV versus the Phase 0
  baseline before merging.

## Phase 0 — Governance and baseline audit

Status: in progress (this document + the audit are the deliverable).

1. Branch `refactor/extensible-runtime-v1` created.
2. Baseline recorded in [00-current-state-audit.md](00-current-state-audit.md).
3. `ROADMAP.md` updated with the `M-EXT` milestone row and reprioritization
   note.

## Phase 1 — Core module/target boundaries

Depends on: Phase 0.

1. Restructure CMake into explicit targets while preserving current
   single-`nvcr` library behavior as default:
   - `nvcr_base` — `Frame`/`Error`/`Result<T>`/version/time types from
     `include/nvcr/common` and the non-session parts of `include/nvcr/runtime`.
   - `nvcr_format` — bitstream/access-unit code (already isolated under
     `include/nvcr/bitstream`, `src/bitstream`).
   - `nvcr_provider_api` — new headers for `IExecutionProvider`/`IExecutable`/
     `IArtifactCompiler` (bodies land in Phase 3/4).
   - `nvcr_codec_api` — generalized `include/nvcr/codec/backend.hpp` contract.
   - `nvcr_artifacts` — new, formalizes the manifest schema already described
     by `scripts/nvcr_artifacts.py` and `configs/`.
   - `nvcr_runtime` — registries plus `nvcr::Runtime` / `nvcr::codec::Runtime`.
   - `nvcr_codec_dcvc_rt` — DCVC-RT semantics, moved from `src/dcvcrt/*`
     (excluding the `backend/tensorrt` subtree).
   - `nvcr_provider_tensorrt` — moved from `src/dcvcrt/backend/tensorrt/*`.
   - `nvcr_cli`, `nvcr_inspect` (new stream/manifest inspection tool).
   - Keep `NVCR::nvcr` as an umbrella/compat target linking the above so
     existing consumers and `find_package(NVCR)` usage do not break.
2. Add build options: keep existing `NVCR_BUILD_CLI/TESTS/EXAMPLES`; add
   `NVCR_ENABLE_CUDA`, `NVCR_ENABLE_DCVC_RT`, `NVCR_ENABLE_TOOLS`,
   `NVCR_ENABLE_FUZZING`; keep `NVCR_ENABLE_TENSORRT` as-is or alias it.
3. Add a CI job asserting `nvcr_base`/`nvcr_format`/`nvcr_provider_api`/
   `nvcr_codec_api` public headers contain no `cuda`/`nvinfer`/DCVC-RT-
   implementation symbols (grep-based script under `scripts/ci/`).
4. Verification: core-only build (Clang + GCC) and full CUDA/TensorRT build
   both pass; full `ctest` suite passes; byte-identical NVAU streams and
   unchanged PSNR-YUV/perf versus the Phase 0 baseline.
5. Risk note: highest-regression-risk phase because it touches the whole
   build. Land as small incremental CMake changes with a full build+test after
   each sub-step, not one large restructuring commit.

## Phase 2 — Session-oriented codec API

Depends on: Phase 1.

1. Introduce `IEncoderSession` / `IDecoderSession` (`send_frame` /
   `receive_access_unit` / `flush` / `reset`, `send_access_unit` /
   `receive_frame` / `flush` / `reset`) in `nvcr_codec_api`, extending
   `include/nvcr/common/error.hpp`'s `Error`/`Result<T>` to cover success, "try
   again", EOS, invalid state, unsupported feature, malformed stream, missing
   artifact, incompatible provider/target, and internal failure — rather than
   inventing a parallel status vocabulary.
2. Adapt `nvcr::Runtime` / `nvcr::codec::Runtime` to implement the new session
   interface as a thin wrapper around today's one-shot `encode`/`decode`;
   DCVC-RT keeps its current one-frame/one-AU behavior underneath, explicitly
   documented as compliant-but-not-exercising delayed output.
3. Keep the old `encode(Frame)->Packet` / `decode(Packet)->Frame` calls as
   compatibility aliases on top of the new session API; CLI/examples do not
   change yet.
4. Verification: existing CLI/tests behave identically; new unit tests cover
   send/receive/try-again/flush/reset semantics using the DCVC-RT adapter.

## Phase 3 — Codec/provider registries and capability negotiation

Depends on: Phase 2.

1. Add `CodecDescriptor` / `CodecCapabilities` / `OptionSchema` to
   `nvcr_codec_api`, and `ProviderDescriptor` / `ProviderCapabilities` /
   `IExecutionProvider` / `IExecutable` / `IArtifactCompiler` to
   `nvcr_provider_api`.
2. Add a static registry in `nvcr_runtime`: codec/provider registration at
   static-init or explicit `register_*` calls, capability query, and an
   `nvcr::runtime::RuntimeServices` abstraction so codec adapters can request
   executable components without touching TensorRT types.
3. Namespaced options (`video.*`, `runtime.*`, `dcvc_rt.*`, `tensorrt.*`)
   replacing ad hoc `RuntimeConfiguration` fields; keep old fields as
   deprecated aliases.
4. CLI: add `nvcr codec list/describe`, `nvcr provider list/describe`,
   `nvcr compatibility check` (new `nvcr_inspect`/`nvcr` subcommands) driven by
   the registry instead of the hardcoded `--backend dcvcrt` string; keep
   `--backend` as a compat alias resolving through the registry.
5. Verification: CLI discovery commands list exactly `dcvc-rt`/`tensorrt`
   until Phase 5 registers the test implementations too; existing
   encode/decode commands are unchanged in behavior.

## Phase 4 — Isolate codec adapter and execution provider implementations

Depends on: Phase 3.

1. Move DCVC-RT coding semantics (GOP rules, quality-index interpretation,
   entropy, codec-private payload, model-component requests) fully behind
   `ICodecAdapter` in `nvcr_codec_dcvc_rt`; the adapter requests executables
   via `RuntimeServices` instead of calling into
   `src/dcvcrt/backend/tensorrt` directly.
2. Split the TensorRT provider into `IArtifactCompiler` (offline engine build —
   formalizes `scripts/backends/dcvcrt/build_tensorrt.sh` /
   `nvcr_artifacts.py build` as a provider-facing contract) versus
   `IExecutionProvider` / `IExecutable` (engine load + execute, from
   `src/dcvcrt/backend/tensorrt/backend.cpp`).
3. Preserve all current engine-bundle manifest fields/validation; this phase
   must not change wire format or manifest schema, only which module owns
   which code path.
4. Verification: byte-identical NVAU streams and PSNR-YUV vs the Phase 0
   baseline; TensorRT roundtrip/contract tests pass per profile; benchmark
   steady-state comparison against Phase 0 numbers (investigate before merging
   any regression greater than ~5%).

## Phase 5 — Extensibility proof: test codec + test provider + contract suites

Depends on: Phase 4 (may run in parallel with Phase 6).

1. Add `tests/support/test_codec/` — a deterministic lossless/trivial-
   transform codec adapter proving registration without modifying runtime
   core/CLI/TensorRT/DCVC-RT/the AU parser; exercise discovery, option schema,
   session lifecycle, delayed/multi-output behavior, flush/reset, and
   codec-private payload in the common envelope.
2. Add `tests/support/test_provider/` — a deterministic CPU execution provider
   proving provider registration without modifying runtime core/DCVC-RT/
   TensorRT/the AU parser.
3. Add contract test suites (parameterized over registered codecs/providers)
   per the codec/provider/format contract requirements, reusing/extending the
   existing `tests/parser_fuzz_tests.cpp`, `tests/rans_conformance_tests.cpp`,
   `tests/tensorrt_engine_tests.cpp`, `tests/tensorrt_roundtrip_tests.cpp`
   patterns.
4. Verification: contract suites pass for both DCVC-RT/TensorRT and the test
   codec/provider; CI runs the contract suites in the core-only
   (no CUDA/TensorRT) job.

## Phase 6 — Stream/AU formal specification and versioning contracts

Depends on: Phase 3 (may run in parallel with Phase 5).

1. Write `docs/spec/nvcr-elementary-stream-v1.md` formalizing the design
   already sketched in `docs/neural-bitstream-envelope.md` plus the current
   `NVAU` v1/v2 implementation: byte order, integer widths, alignment, magic/
   version semantics, max lengths, overflow-safe size calculations, unknown-
   extension behavior, checksum behavior, truncation/malformed rejection,
   stream/AU lifecycle, and hex examples.
2. Add explicit version types (software / codec-API / provider-API / stream-
   format / payload-syntax / model-set / manifest-schema) — likely a small
   `include/nvcr/common/versions.hpp` — wired into the existing
   `RuntimeConfiguration` / `AccessUnit` / engine-bundle fields rather than
   duplicating state.
3. Write ADRs:
   - `docs/adr/ADR-001-codec-provider-separation.md` — documents the existing
     `CodecBackend`/TensorRT confinement decision.
   - `docs/adr/ADR-002-stream-and-payload-contract.md` — documents the `NVAU`/
     `bitstream_model_id` decision made 2026-08-04.
   - `docs/adr/ADR-003-model-and-artifact-identity.md` — documents the triple-
     tuple manifest identity.
   - `docs/adr/ADR-004-extension-and-registration-model.md` — documents the
     static registry decision from Phase 3.
   - `docs/adr/ADR-005-conformance-levels.md` — defines the conformance-level
     vocabulary (runtime-compatible, self-conformant, cross-provider
     conformant, cross-device conformant, reference-fidelity validated,
     bit-exact).
4. Add golden byte-vector tests, malformed/truncated/oversized tests (extend
   `tests/parser_fuzz_tests.cpp`), and libFuzzer targets for stream/AU parsing
   under `tests/fuzz/`, gated by `NVCR_ENABLE_FUZZING`.
5. Verification: spec examples round-trip in tests; fuzz targets run for a
   bounded time in CI without crashes; ASan/UBSan core test job passes.

## Phase 7 — Artifact provenance, manifest formalization, resolver

Depends on: Phase 4.

1. Add `third_party/dcvc_rt/` (or repo-appropriate path) with `UPSTREAM.md`,
   `upstream-repository.txt`, `upstream-commit.txt`, `patches/`, `PATCHES.md`
   mirroring the existing `third_party/dcvc_rans/UPSTREAM.md` pattern, pinning
   the exact DCVC-RT source/checkpoint/exporter lineage currently referenced
   only in `scripts/backends/dcvcrt/prepare_artifacts.sh` and
   `configs/models/dcvcrt-cvpr2025.json`.
2. Formalize the artifact resolver in `nvcr_artifacts`: given codec + model
   set + component + device + provider preference, return the best local
   artifact or a structured failure reason (codec/provider/model-set/artifact
   missing, version/target/precision incompatible, digest mismatch, license-
   restricted). Build on the existing `nvcr_artifacts.py validate` / engine-
   bundle matching logic (`docs/dcvcrt-artifacts.md`), moving matching logic
   into C++ where the runtime needs it at load time.
3. Add root `THIRD_PARTY_NOTICES.md`, `MODEL_LICENSES.md`,
   `ASSET_DISTRIBUTION_POLICY.md` documenting NVCR/DCVC-RT/checkpoint/engine/
   test-sequence licensing; mark unresolved redistribution items explicitly
   and exclude them from release automation until reviewed.
4. Verification: `nvcr-artifacts validate`/resolver unit tests cover every
   failure category; license docs reviewed against actual current assets (no
   invented permissions).

## Phase 8 — CI/CD and release hardening

Depends on: Phases 1, 5, 6.

1. Extend `.github/workflows/ci.yml`: add a core-only Clang build (currently
   only GCC x86_64/arm64), an ASan/UBSan core test job, a format golden-
   vector/parser job, a CMake install + external-consumer
   `find_package(NVCR CONFIG REQUIRED)` job (new tiny example under
   `examples/external_consumer/`), docs/link/schema validation, and a
   license/third-party metadata check.
2. Extend `gpu-main.yml`/`release-assets.yml` so a trusted real-GPU release
   gate executes (not just compiles) the DCVC-RT/TensorRT conformance suite
   and attaches a machine-readable report before promoting a release; do not
   weaken the existing manual reference-gate discipline already documented in
   `release-assets.yml`.
3. Define the immutable release asset set: extend `scripts/package_release.sh`
   to also emit `SBOM.spdx.json`, `artifact-catalog-paper-<version>.json`,
   `paper-evidence-<version>.tar.gz`, `reproduction-instructions.md`,
   `release-validation-report.json` alongside the existing source/binary
   archives and checksums.
4. Verification: run each new CI job at least once (compile-only where no GPU
   is present); explicitly record which gates could not be executed in this
   environment versus the trusted GPU/self-hosted release host.

## Phase 9 — Documentation restructuring and SoftwareX framing

Depends on: Phases 6, 7.

1. Add `docs/softwarex/framing.md`, `contribution-claims.md`,
   `evaluation-protocol.md`, `reproducibility-checklist.md`,
   `release-checklist.md` using the claim/non-claim wording already
   established in `docs/scope-and-support.md` and `.github/copilot-
   instructions.md`.
2. Add `docs/codec-integration-guide.md` and
   `docs/provider-integration-guide.md` referencing the Phase 5 test
   codec/provider as worked examples.
3. Add `docs/limitations.md`; update `README.md` (implemented vs experimental
   vs planned, minimal immutable-release install, a small example, links to
   the spec/API, citation); add `CITATION.cff`, `CONTRIBUTING.md`,
   `SECURITY.md`, `SUPPORT.md` (`codemeta.json` optional).
4. Update `docs/architecture.md` with the new module diagram and decode
   workflow reflecting Phases 1–4; keep existing content that is still true
   (system boundary/ownership table) — do not rewrite wholesale.
5. Move genuinely historical dated log content that no longer reflects the
   current architecture into `docs/archive/development-history.md` /
   `historical-benchmarks.md`; keep `ROADMAP.md`'s evidence trail intact per
   repo convention (do not delete evidence).

## Phase 10 — Evaluation package for SoftwareX

Depends on: Phase 8. Resumes M1–M4-style evidence work under a new structure.

1. Add `evidence/softwarex-v1/{environment,conformance,quality,runtime,
   memory,robustness,tables,figures,commands}/` and
   `benchmarks/{configs,scripts,parsers}/`, building on the existing
   `scripts/benchmark_resolution_matrix.sh`, `scripts/
   benchmark_resolution_pair.sh`, `scripts/benchmark_orin_release.py`,
   `scripts/profile_energy.py` rather than replacing them.
2. Run/record the minimum evaluation matrix (360p/720p/1080p, at least two
   sequences per resolution, at least four quality points, low-delay I/P plus
   supplementary all-intra, one principal GPU plus optional Orin) with
   Python-vs-NVCR fidelity, runtime, memory, and robustness metrics, each
   traceable to release/commit/patch/checkpoint/artifact digests.
3. Energy profiling remains explicitly excluded as a release blocker.

## Phase 11 — Final acceptance and release checklist

Depends on: all prior phases.

1. Write `docs/refactor/99-final-acceptance-report.md` against the full
   acceptance checklist (architecture, extensibility proof, stream/AU,
   artifacts/provenance, API/packaging, CI/release, licensing/documentation,
   evaluation), marking each item implemented-and-tested /
   implemented-but-not-executable-here / designed-only / deferred, with exact
   reproduction commands.
2. Write a migration guide (API/CLI/manifest/stream version changes) and
   release checklists for `v1.0.0-rc1`/`v1.0.0`.
3. Update `ROADMAP.md` final status/evidence and architectural-decision
   pointers.

## See also

- [00-current-state-audit.md](00-current-state-audit.md)
- `ROADMAP.md` — milestone tracking (`M-EXT` row and reprioritization note)
- `docs/architecture.md`, `docs/neural-bitstream-envelope.md`,
  `docs/dcvcrt-artifacts.md`, `docs/dcvcrt-integration.md`,
  `docs/scope-and-support.md` — current implementation docs this refactor must
  stay consistent with until superseded by Phase 9
