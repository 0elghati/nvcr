# Version and identity contract inventory

Status: design inventory for v2, 2026-11-23. This document records the
contracts on `main`; it does not change a public API, stream byte, artifact
format, or compatibility rule.

NVCR already has most of the identities it needs. The problem is that some are
duplicated, some are only defaults in C++ structures, and some are names that
look like versions without an enforced compatibility rule. Before v2 changes
any format, this inventory gives each identity one purpose and identifies the
boundary that must enforce it.

## Vocabulary

- A **version** has an explicit compatibility rule between a producer and a
  consumer.
- An **identity** selects one exact thing, such as a codec, model set, target,
  or content digest. Ordering two identities has no meaning unless the owner
  defines it.
- A **schema identifier** selects the parser for a document. It is not the
  version of the model or engine described by that document.
- A **capability** says what an implementation can do. It should be proved by
  conformance evidence, not inferred from a version number.

## Current contract map

### Product and API identities

| Identity | Owner and current location | Current rule and consumers | Finding |
|---|---|---|---|
| NVCR software release | `version.txt`, generated `SoftwareVersion`, release-please, packages, CLI, containers, citation metadata | Semantic version used for the distribution and build identity | Keep. This is the product release, not a stream, model, or artifact version. |
| Shared-library/CMake package | CMake `VERSION`, major `SOVERSION`, and `SameMajorVersion` package compatibility | CMake accepts the same major; the documentation says minor and patch releases preserve the public API but cross-release ABI is not guaranteed | Clarify. CMake compatibility can be read as an ABI promise even though the published policy requires applications to rebuild. |
| Codec API | `CodecApiVersion{major}` in configuration, artifact requests, and descriptors; separate `CodecDescriptor::api_version` integer | The artifact resolver requires equal majors. Registry admission only checks that a codec ID exists and does not relate the descriptor integer to an artifact request. | Merge the duplicate representations and enforce the contract at codec registration/artifact admission. |
| Provider API | `ProviderApiVersion{major}` in configuration and artifact requests | The artifact resolver requires equal majors. The registered provider and experimental provider-session interface do not declare or negotiate this version. | Keep the axis, but attach it to provider registration and session creation before using it for artifact compatibility. |
| Provider implementation version | `ProviderDescriptor::version` string | Reports the installed provider runtime, such as TensorRT. It has no generic ordering rule. | Keep as implementation metadata. Provider-specific compatibility belongs in capabilities and artifact constraints. |

### Stream and codec identities

| Identity | Owner and current location | Current rule and consumers | Finding |
|---|---|---|---|
| `.nvcr` sequence wrapper | CLI-local `NVCS` header, version 1 and flags 0 | Exact version/flags match; followed by length-delimited packet records | Keep separate from the codec stream. It is a development/application wrapper, not a media container contract. |
| Packet envelope | `PacketIO`, `NVCR` magic, version 1 | Exact version match | Keep separate from NVAU. It carries packet metadata and an opaque data blob. |
| Common access unit | `AccessUnitIO`, `NVAU` wire versions 1 and 2 | Exact dispatch to the v1 or sectioned-v2 parser; unknown versions fail | This is the real common stream-format version. It should be the only common envelope version used for parser dispatch. |
| Configured stream version | `StreamFormatVersion{major, minor}` in `RuntimeConfiguration` and `AccessUnit` | Reconciled with legacy configuration fields, but not parsed from config files, validated, serialized, or used to select the NVAU writer | Deprecate the inert representation or connect it to an explicit writer/parser policy. Do not keep two stream-version systems. |
| Codec payload syntax | `PayloadSyntaxVersion{major}` in configuration and `AccessUnit` | Present as a default only; it is not serialized or checked | Make this codec-owned and carry it in the common envelope before relying on it. |
| DCVC-RT private payload | `NVI1` and `NVP1` magic values in the codec payload | Magic selects the only supported syntax; fields and rANS payload are bounded by codec parsers | Treat the trailing `1` as the current codec-private syntax identity and map it to an explicit payload contract during migration. |
| DCVC-UF private payload | `UFI1` and `UFP1` magic values in the codec adapter | Magic selects the only supported syntax; reserved and flag fields are validated | Same migration rule as DCVC-RT. The codec owns this version. |
| MLVC private payload in NVCR | MLVC record framing inside NVAU | No independent version field; compatibility is inferred from the NVAU model identity and accepted bundle layout | Add an explicit codec payload/model contract before more than one MLVC syntax is admitted. |
| NVAU section version | `AccessUnitSection::version` | Known required sections validate their own version; unknown required sections fail and unknown optional sections may be skipped | Keep. This is local extension versioning and should not replace the NVAU or codec payload version. |

### Model, numerics, and deployment identities

| Identity | Owner and current location | Current rule and consumers | Finding |
|---|---|---|---|
| Codec ID | Registry, configuration, NVAU v2, artifact descriptors | Exact string identity | Keep. Codec semantics own payload, state, entropy, and model interpretation. |
| Internal model/profile ID | `model_id` / `model_profile_id`, for example `dcvcrt-cvpr2025` | Selects artifacts and provenance; exact string match | Keep internal. It must not be substituted for the stable identity written into a stream. |
| Public stream model ID | `bitstream_model_id` written as NVAU `model_id` | Decoder requires an exact identity it knows how to satisfy | Keep the public/internal split. The stream names decoding requirements, not a GPU, provider, or compiled plan. |
| Model-set version | `ModelSetVersion{major, minor}` in configuration and artifact resolution | Equal major and candidate minor greater than or equal to requested minor | Rework. A generic numeric ordering cannot prove neural-model compatibility. Use an exact model identity plus a codec-owned compatibility rule where one actually exists. |
| Numerics profile | Strings such as `native`, `nvcr.intconv.v1`, and `nvcr.fx.v1` | Codec declares requirements; provider conformance proves support | Keep. The version is scoped to the named numerics contract, not to the codec or provider. |
| Normative program set | Digest in the NVAU `numerics_identity` section | Exact digest match for exact-numerics streams | Keep as content identity. A digest is not a version and has no ordering rule. |
| Artifact manifest version | `ManifestSchemaVersion{major, minor}` in configuration/artifact resolution | Equal major and candidate minor greater than or equal to requested minor | Replace with the actual document schema identity. Catalog candidates currently inherit the C++ default instead of declaring this value. |
| Artifact document schemas | Exact strings such as `nvcr.model-manifest.v2`, `nvcr.engine-bundle.v2`, and `nvcr.engine-catalog.v1`; some documents also contain numeric `format: 2` | Python tooling selects and validates exact schema strings; the C++ catalog recognizes its own schema | Keep one canonical schema ID per document family. Remove redundant numeric format fields unless they select a genuinely independent encoding. |
| Model, engine, and target profiles | `nvcr.model-profile.v1`, `nvcr.engine-profile.v1`, `nvcr.target-profile.v1` plus stable IDs | Schema selects the parser; profile ID selects the content | Keep schema and content identity distinct. Resolution coverage is a capability of an engine profile, not part of its name. |
| Provider/runtime/target requirements | Provider ID, precision, OS, architecture, CUDA/TensorRT versions, device and target profile | Provider-specific resolver and catalog rules | Keep as compatibility constraints. They are not NVCR contract-version axes. |
| Artifact content | File and bundle SHA-256 values | Exact verification | Keep. Digests provide integrity and reproducibility, not compatibility ordering. |
| Evidence schemas | `nvcr.measurement.*.v1`, `nvcr.softwarex.*.v1`, and other result schemas | Tool-specific exact parsing | Keep outside runtime compatibility. Evidence schema changes must not force a stream or codec version change. |

## What is enforced today

The artifact resolver has the clearest current compatibility policy:

- codec and provider API majors must match;
- model-set and manifest-schema majors must match, and the candidate minor must
  be at least the requested minor;
- provider, precision, target class, operating system, architecture,
  provider-runtime constraints, license, digest, and availability are checked
  separately.

That policy is useful, but its version inputs are usually defaults rather than
facts read from the catalog or bundle. A successful resolver result therefore
does not yet prove that the installed artifact declared those API and schema
versions.

The registry is looser. `Registry::compatible()` currently means that both IDs
are registered; it does not compare codec requirements with provider
capabilities or version contracts. Codec and provider registration is also
idempotent replacement by ID, with no version admission check.

Configuration exposes every version field in C++, including legacy duplicates,
but the configuration-file parser does not accept those versions and validation
does not check them. This makes them implementation defaults, not a usable
negotiation surface.

## Lessons from libmlvc

libmlvc is useful here because it keeps separate concerns separate. The split
is visible in its
[public API](https://github.com/microsoft/libmlvc/tree/main/include/libmlvc),
[bundle model](https://github.com/microsoft/libmlvc/tree/main/src/libmlvc/schema),
and [bitstream specification](https://github.com/microsoft/libmlvc/blob/main/docs/specification.md):

1. The libmlvc software release has its own semantic version. The MLVC model
   has a separate major/minor identity.
2. The bundle declares the model version it supplies. The manager indexes
   loaded bundles by that exact version and refuses unavailable versions.
3. Every SPS starts with the MLVC model major/minor. The decoder learns the
   required model from the stream before it selects model data and inference
   sessions.
4. Bitstream syntax belongs to the model major. The specification freezes the
   syntax within the current major and requires a new major for syntax changes.
5. Model selection by canvas is internal to a loaded bundle. A bundle can carry
   several fixed canvases without turning every canvas into a library release
   or a different codec.
6. Provider sessions and compiled caches are deployment concerns. They are
   selected after the model/stream contract is known and are not written as the
   codec identity.

NVCR's current MLVC adapter preserves Microsoft's model files, metadata,
interface type, and split type, but it derives stream model IDs from bundle and
canvas names rather than using libmlvc's explicit MLVC model version. That was
enough for one published bundle family; it is not a durable multi-version
contract.

NVCR should parse stable decoding requirements from the stream, resolve an
installed portable codec pack that satisfies them, then choose a conformant
provider and optional accelerator. A TensorRT plan, CUDA version, GPU name, or
resolution profile must never become the public codec identity.

## Proposed v2 axes

The smallest useful set is eight independent axes. Names and numeric values are
deliberately not assigned here.

| Axis | Changes when | Must not change when |
|---|---|---|
| Software release | NVCR is released | A stream is encoded or an artifact is rebuilt |
| Codec API contract | The runtime/codec adapter source contract breaks | A codec model, provider, or bundle changes |
| Provider API contract | The runtime/provider session contract breaks | A codec or stream syntax changes |
| Common envelope | NVAU parsing or common access-unit semantics break | A codec-private payload changes compatibly |
| Codec payload contract | A codec can no longer parse its private payload syntax | A provider or target changes |
| Codec model identity | Different learned parameters or codec semantics are required for decode | The same model is compiled for another provider, target, or shape range |
| Numerics contract | Normative arithmetic requirements change | A conformant provider implementation changes |
| Artifact document schema | A catalog, pack, or manifest parser contract changes | Artifact content or a product release changes |

The model identity does not need to be ordered. If a codec family later proves
that a minor model revision is backward-compatible, that codec may define a
major/minor rule. NVCR must not assume `candidate.minor >= requested.minor` for
all neural codecs.

Content digests, hardware targets, provider-runtime versions, precision, shape
coverage, and evidence-schema versions remain necessary metadata, but they are
not additional NVCR contract axes.

## Migration classification

### Keep

- the product semantic version and build identity;
- exact codec/provider/model/profile identifiers;
- NVAU's actual wire version and per-section versions;
- codec-owned private payload formats;
- named numerics profiles plus normative program digests;
- explicit schema IDs and content hashes;
- provider/runtime/target requirements as artifact constraints.

### Migrate or remove

- replace `CodecDescriptor::api_version` and `CodecApiVersion` with one codec
  API representation;
- attach the provider API version to provider registration/session admission;
- remove or serialize the inert `AccessUnit` stream, payload, and model version
  members;
- move compatibility versions out of user configuration and into declared
  producer/consumer capabilities;
- replace the generic `ModelSetVersion` ordering with codec-owned model
  compatibility;
- replace `ManifestSchemaVersion` defaults with the schema ID/version declared
  by each parsed document;
- choose one canonical schema/version field where manifests currently carry
  both `format: 2` and a `.v2` schema string;
- teach the catalog to carry the contract facts used by the resolver rather
  than letting candidates inherit default C++ values.

No field should be deleted until its replacement can read all supported v1
artifacts and streams or the project has explicitly chosen a v2-only break.

## Verification inventory

| Contract | Existing evidence | Required v2 addition |
|---|---|---|
| Product release identity | `version_identity_tests.py`, CLI/artifact version tests, package checks | Assert CMake package compatibility matches the documented source/ABI policy. |
| Codec/provider API | Contract and public-API compatibility tests | Registration tests for supported and rejected API versions; one canonical version field. |
| NVAU/packet parsing | Format contract, unit/smoke, parser boundary, and fuzz tests | Golden files for every retained version and explicit previous/current/unknown-version behavior. |
| Codec payloads | Codec round trips, malformed-input checks, pinned goldens | Per-codec payload-version fixtures and clean failure when a payload contract is unavailable. |
| Model identity | Decoder checks and codec-specific parity tests | Resolve stream requirements to an exact portable codec pack before provider selection. |
| Numerics | Numerics identity tests and cross-provider exact-profile evidence | Admission matrix generated from retained conformance results, including missing/mismatched program sets. |
| Artifact resolver | Resolver unit tests | Candidates populated from real parsed manifests/catalogs, not default version fields. |
| Artifact schemas and integrity | Artifact-tool, catalog, manifest, checksum, and install tests | Previous/current/unknown schema fixtures for each document family and migration tests for installed packs. |
| MLVC | Bundle unit tests and upstream-runtime parity | Preserve and expose the MLVC model version; test multiple installed versions and provider selection independently. |

Every compatibility test should state four things: producer identity, consumer
identity, expected decision, and expected failure category. A successful parse
alone is not evidence that a codec/provider cell is supported.

## Decisions this inventory does not make

- numeric values or names for v2 contracts;
- whether the v1 C++ API is retained alongside a new ABI;
- the portable codec-pack manifest shape;
- the registry transport or storage backend;
- the default artifact acquisition and offline-pack policy;
- NVAU v2 stabilization or a different future envelope;
- compatibility claims for untested codec/provider/target cells.

Each follow-up decision must say which contract it changes and leave the other
axes alone unless the change genuinely crosses a boundary.
