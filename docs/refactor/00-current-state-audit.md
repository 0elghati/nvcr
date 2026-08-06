# Current-state audit: extensible runtime architecture refactor

Date: 2026-08-06
Branch: `refactor/extensible-runtime-v1`
Baseline commit: `9bf8478` (tag `v0.8.2`, `main`/`origin/main`, working tree clean
except this documentation)

This audit is the required baseline snapshot before the `M-EXT` refactor
described in [01-architecture-refactor-plan.md](01-architecture-refactor-plan.md).
It records what exists today so later phases can prove they preserved behavior
instead of guessing.

## 1. Baseline state recorded

- Version: `version.txt` = `0.8.2`; `CMakeLists.txt` project version matches
  (`x-release-please-version` marker).
- Submodules/vendored deps: no git submodules. `third_party/dcvc_rans/` is
  vendored with `UPSTREAM.md` pinning commit
  `dae827ffcc812566adbeaf4554f0fe2d9b4b9e0c` from the DCVC-RT rANS source. Full
  DCVC-RT model/exporter lineage is pinned only in
  `configs/models/dcvcrt-cvpr2025.json` and `scripts/backends/dcvcrt/
  prepare_artifacts.sh`, not under a `third_party/dcvc_rt/` provenance directory
  (see gap G7 below).
- Public exported CMake targets: `NVCR::nvcr` (library), `nvcr_cli` (if
  `NVCR_BUILD_CLI` and `NVCR_ENABLE_TENSORRT`), via `NVCRTargets.cmake` /
  `NVCRConfig.cmake`.
- CLI commands: `nvcr encode` and `nvcr decode` only; codec/provider selection
  is a hardcoded `--backend` string normalized to `dcvcrt`, not a registry
  lookup.
- File formats: `NVAU` v1/v2 codec access units (`include/nvcr/bitstream/
  access_unit.hpp`), `NVCR` packet envelope and `NVCS` sequence wrapper
  (development formats, `include/nvcr/bitstream/packet_io.hpp`),
  `nvcr.model-manifest.v2` and `nvcr.engine-bundle.v2` artifact manifests,
  `nvcr.engine-catalog.v1` rolling release catalog.
- Release assets today: generic engine-free `linux-x86_64-nvidia` /
  `linux-aarch64-jetson-l4t36` archives plus `SHA256SUMS`-style checksums (see
  `release-assets.yml`), and a separate rolling `engine-assets` GitHub Release
  carrying target-bound TensorRT bundles. No SBOM, no evidence bundle, no
  release-validation report exist yet.
- Generated/large artifacts that must stay untracked: `.plan`, `.engine`,
  `.onnx`, `.pth`/`.pth.tar` checkpoints, `i_entropy.bin`/`p_entropy.bin`,
  `engine_manifest.json`, `engine.sha256`. Already enforced by the smoke-package
  check in `.github/workflows/ci.yml`; no additional leakage found under
  `build*/`, `install-nvcr/`, `output/`, `assets/checkpoints/` during this audit
  (those directories are present locally but are build/output artifacts, not
  tracked sources — confirm `.gitignore` coverage before Phase 1 commits).
- Current build/test commands (for later phases to reproduce):
  `cmake --build build-release -j 8` and
  `ctest --test-dir build-release --output-on-failure`.
- Current DCVC-RT/TensorRT smoke behavior: documented and evidenced throughout
  `ROADMAP.md` (GOP-97 encode/decode fps and PSNR-YUV per resolution, e.g. 720p
  GOP-97 ≈ 94/44 fps, 1080p GOP-97 ≈ 44/20 fps on RTX 4070); this audit does not
  re-run those benchmarks — later phases must re-baseline immediately before
  any change that could affect them (Phase 4 in particular).

## 2. Gap matrix

| Area | Current implementation | Desired boundary | Risk | Planned change | Test proving completion |
|---|---|---|---|---|---|
| G1. Codec/provider discovery | Single hardcoded `dcvcrt` backend via `make_tensorrt_backend()` factory; CLI `--backend` is a string normalized to one value | Static codec/provider registry with capability query, driving CLI discovery commands | Medium — touches CLI and runtime init paths used by every test | Phase 3: add registry in `nvcr_runtime`, `nvcr codec/provider list/describe` | New registry unit tests; CLI discovery command tests |
| G2. Session API shape | One-shot `encode(Frame)->Packet` / `decode(Packet)->Frame` on `nvcr::Runtime` | `IEncoderSession`/`IDecoderSession` with `send_frame`/`receive_access_unit`/`flush`/`reset` and `send_access_unit`/`receive_frame`/`flush`/`reset` | Medium — public API surface, must keep DCVC-RT one-frame/one-AU behavior working underneath | Phase 2: introduce session interfaces as a wrapper, keep old calls as compat aliases | Session lifecycle tests (send/receive/try-again/flush/reset) |
| G3. Codec adapter isolation from execution | DCVC-RT semantics (`src/dcvcrt/payload.cpp`, `rans_codec.cpp`) already separate from TensorRT execution (`src/dcvcrt/backend/tensorrt/*`), confirmed by grep: no CUDA/TensorRT symbols in `include/nvcr/common`, `runtime`, `codec` | `ICodecAdapter` requesting executables from `RuntimeServices` instead of calling TensorRT code paths directly | Medium — must not change entropy/payload/manifest behavior | Phase 4: introduce `ICodecAdapter`/`IExecutionProvider`/`IExecutable` boundary | Byte-identical NVAU streams and PSNR-YUV vs this audit's baseline |
| G4. Offline compile vs execution split | TensorRT engine building lives in `scripts/backends/dcvcrt/build_tensorrt.sh` / `nvcr_artifacts.py build`; runtime-side loading in `src/dcvcrt/backend/tensorrt/backend.cpp` — already separated by process boundary, not by C++ interface | Explicit `IArtifactCompiler` vs `IExecutionProvider` C++ contracts | Low — mostly formalizing an existing process-level split | Phase 4 | Contract tests for both interfaces |
| G5. Stream/AU formal spec | `docs/neural-bitstream-envelope.md` already documents the target multi-codec envelope design; `NVAU` v1/v2 implemented and tested (`tests/parser_fuzz_tests.cpp`) | Normative `docs/spec/nvcr-elementary-stream-v1.md` with byte order, overflow-safe size rules, hex examples | Low — mostly documentation plus additional golden/fuzz coverage | Phase 6 | Golden byte-vector tests; fuzz targets run without crashes |
| G6. Version-type separation | `RuntimeConfiguration::bitstream_model_id` vs `model_id` already separates public stream identity from backend profile identity (2026-08-04 decision, undocumented as formal ADR) | Explicit version types for software/codec-API/provider-API/stream-format/payload-syntax/model-set/manifest-schema | Low | Phase 6: add `include/nvcr/common/versions.hpp`; write ADR-001..005 | Unit tests asserting version rejection rules |
| G7. DCVC-RT provenance chain | Pinned commit/checkpoint hashes exist in `configs/models/dcvcrt-cvpr2025.json`; no `third_party/dcvc_rt/UPSTREAM.md`-style pinned directory with patch series | `third_party/dcvc_rt/{UPSTREAM.md,upstream-commit.txt,patches/,PATCHES.md}` mirroring `third_party/dcvc_rans` | Low | Phase 7 | Manual review; digest match against `configs/models/dcvcrt-cvpr2025.json` |
| G8. Artifact resolver | Matching logic exists as Python (`nvcr_artifacts.py validate`) and as C++ runtime preflight checks in `backend.cpp`; no single resolver API returning structured failure reasons | `IArtifactResolver`-equivalent returning best artifact or a structured failure category | Medium | Phase 7 | Unit tests covering every listed failure category |
| G9. Extensibility proof | No test-only codec adapter or execution provider exists | Deterministic test codec + test provider proving registration without touching runtime core/CLI/DCVC-RT/TensorRT/AU parser | Low (additive) | Phase 5 | Contract suites pass against test implementations |
| G10. CI gates | Core-only CPU build (GCC, x86_64+arm64), portable CUDA compile-only build, lint, smoke-package validation already exist (`ci.yml`) | Add Clang core build, ASan/UBSan, `find_package` external-consumer test, license/SBOM checks | Low (additive) | Phase 8 | New CI jobs green |
| G11. Release evidence | Generic engine-free binary archives + checksums; manual reference-gate notes in `release-assets.yml` | SBOM, immutable evidence bundle, machine-readable validation report | Low (additive) | Phase 8 | Release dry run produces all listed assets |
| G12. Licensing docs | `LICENSE` (MIT) exists; no `THIRD_PARTY_NOTICES.md`, `MODEL_LICENSES.md`, `ASSET_DISTRIBUTION_POLICY.md`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md` | All of the above, with unresolved redistribution items explicitly excluded rather than guessed | Low (additive, but requires legal judgment) | Phase 7/9 | Manual review against actual current assets |
| G13. ADRs/spec directories | Architectural decisions recorded inline in `ROADMAP.md` prose (dated notes); no `docs/adr/` or `docs/spec/` directories | `docs/adr/ADR-001..005`, `docs/spec/nvcr-elementary-stream-v1.md` | Low (additive) | Phase 6 | N/A (documentation) |
| G14. SoftwareX framing docs | `docs/scope-and-support.md` already states non-claims (no new codec, no multi-provider, etc.); no `docs/softwarex/` directory | `docs/softwarex/{framing,contribution-claims,evaluation-protocol,reproducibility-checklist,release-checklist}.md` | Low (additive) | Phase 9 | N/A (documentation) |
| G15. Evaluation package structure | Ad hoc `docs/evidence/*.json`/`.jsonl` files and `scripts/benchmark_*` scripts; no `evidence/softwarex-v1/` structure | Structured evidence tree with environment/conformance/quality/runtime/memory/robustness/tables/figures/commands | Low (additive) | Phase 10 | Reproduced evaluation matrix committed under new structure |

## 3. Non-negotiables carried forward unchanged

- No wire-format, CLI, or public-behavior change is authorized without a
  documented compatibility alias or migration note (see the phased plan's
  Decisions section).
- No dynamic plugin ABI is introduced; codec/provider registration stays static
  (section 5.1 of the originating request).
- No second real codec or execution provider is added in this program beyond
  the required test-only implementations (G9); ONNX Runtime and a second codec
  remain explicitly deferred.
- M1–M4 evidence-gathering is paused, not deleted; existing `ROADMAP.md`
  evidence entries stay as historical record.

## See also

- [01-architecture-refactor-plan.md](01-architecture-refactor-plan.md) — full
  phased implementation plan.
- [../architecture.md](../architecture.md), [../neural-bitstream-envelope.md](../neural-bitstream-envelope.md),
  [../dcvcrt-artifacts.md](../dcvcrt-artifacts.md), [../dcvcrt-integration.md](../dcvcrt-integration.md)
  — current implementation docs this refactor must stay consistent with.
