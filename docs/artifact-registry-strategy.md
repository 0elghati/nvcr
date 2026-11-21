# Artifact registry strategy

Status: deferred design revisit; no backend migration or roadmap reprioritization
is approved by this document.

This note separates two decisions that are currently coupled: where bundles are
stored and how NVCR discovers, validates, selects, publishes, and retires them.

## Current state

NVCR publishes DCVC-RT bundles to the rolling GitHub Release `engine-assets`
and MLVC bundles to `engine-assets-mlvc`. As checked on 2026-11-21, the releases
contained 85 assets and 9 assets respectively. The DCVC-RT release consists of
one catalog plus 42 archive/checksum pairs.

Publication currently:

1. creates the rolling release if needed;
2. downloads the current catalog;
3. validates staged archives, manifests, target metadata, and required profiles;
4. uploads archives and checksums with replacement enabled; and
5. replaces the catalog last.

Installation detects the target, ranks compatible entries, verifies size,
digest, manifest, and target metadata, installs into a content-addressed local
directory, then changes the active symlink atomically.

The current design has useful safety properties: admission is fail-closed, the
catalog is published last, and local activation is atomic. It is also simple
enough for existing clients.

Its limits are now visible:

- GitHub allows at most 1,000 assets per Release and each file must be smaller
  than 2 GiB. Six profiles consume 12 assets per target because archives and
  checksum files are separate.
- Stable remote filenames are replaced in place. Catalog-last publication
  avoids selecting an incomplete new entry, but a reader can still observe an
  old catalog with a newly replaced archive and fail its digest check.
- The catalog and assets have no independent signature or trust root.
- The flat catalog is tied to fixed backend/profile lists, and separate rolling
  releases are already needed for different backends.
- Publication has no compare-and-swap or single-writer rule, so concurrent
  publishers can overwrite catalog updates.
- Authentication, resumable pulls, remote content deduplication, lifecycle
  states, rollback history, mirroring, and garbage collection are not defined.
- Storage URLs and GitHub Release conventions leak into the registry mechanism.

## Workstream 1: distribution backend

| Option | Good fit | Main limitation |
|---|---|---|
| GitHub Releases | Simple public downloads and current-client compatibility | Mutable names, asset/file limits, weak lifecycle and deduplication |
| Hugging Face Hub | Model discovery, revisions, cards, and cached snapshots | Compiled target compatibility still needs an NVCR index |
| NVIDIA NGC | Versioned NVIDIA deployment assets | Vendor-specific workflow and client dependency |
| Object storage | Cheap scalable blobs, resumable delivery, CDN control | NVCR must build identity, signing, metadata, and lifecycle around it |
| OCI registry | Immutable digests, deduplicated layers, tags, standard clients, signatures and referrers | NVCR still needs a compatibility index and selection policy |

Ultralytics and PyTorch Hub show that GitHub Releases are reasonable for simple,
portable weights. Hugging Face adds model-oriented discovery and revisions.
Triton demonstrates repository delivery from object stores. NVCR's compiled,
target-specific bundles most closely match OCI artifacts.

### Smallest useful OCI prototype

Use OCI without redesigning bundle contents:

- publish one existing deterministic bundle archive as one OCI artifact layer;
- give the artifact and config NVCR-specific media types;
- use the OCI manifest digest as the immutable bundle identity;
- use tags only as human-readable aliases, never as lockfile identities; and
- attach provenance, conformance results, SBOMs, and signatures as referrers.

GitHub Container Registry is the first host to evaluate because it preserves the
current GitHub access model, but the format and client must remain registry
neutral. Splitting every TensorRT plan into a separate OCI layer is deferred
until measurements show that layer-level reuse materially reduces transfer or
storage cost.

## Workstream 2: registry and lifecycle mechanism

The registry mechanism should resolve capabilities, not filenames. A proposed
`nvcr.catalog.v2` index would be an immutable, signed artifact. A mutable channel
such as `stable` would point to an index digest; clients and evidence records
would pin the digest they actually used.

Each bundle record should describe at least:

- bundle and manifest digests;
- codec, model, provider, precision, and numerics profile;
- target and runtime compatibility;
- supported and preferred shape ranges and bundle role;
- provenance, license, conformance evidence, and lifecycle state; and
- one or more storage locations for mirrors.

Shape capability belongs in the record. Whether an exact plan or a broader
dynamic plan is preferred remains a separate measured selection-policy decision.

### Publication transaction

1. Build and validate the bundle locally.
2. Push it by digest, then attach provenance and conformance evidence.
3. Pull it back and run the required manifest and smoke checks.
4. Update the index with compare-and-swap or under a single-writer lock.
5. Sign and publish the new index by digest.
6. Move the channel tag last and retain the previous index for rollback.

An interrupted publication must leave the previous channel usable. Parallel
publishers must either serialize or retry from the new index, never silently
drop one another's entries.

### Installation and lifecycle

The client should resolve a channel to an index digest, verify its signature,
select a compatible record, pull the bundle by digest with resume and cache
support, verify it, install it content-addressed, and activate it atomically. It
should retain a compact receipt containing the index digest, bundle digest,
selection reason, target facts, and validation result.

Lifecycle states should be explicit:

- `candidate`: available only when requested;
- `published`: eligible for normal selection;
- `deprecated`: excluded from new automatic selections but still pin-installable;
- `revoked`: rejected unless a deliberate recovery override is used.

Garbage collection should mark everything reachable from retained signed
indexes, supported lockfiles, rollback points, and required evidence, then sweep
only unreferenced artifacts after a retention period.

## Migration path

1. Dual-publish one target/profile to GitHub Releases and OCI, then verify
   archive bytes, bundle manifest, pull-back validation, and runtime smoke.
2. Add an explicit experimental OCI source; do not change the default or remove
   Release support.
3. Dual-publish a complete target set and compare reliability, transfer size,
   cache reuse, publication time, and operational complexity.
4. Exercise signed-index rollback, parallel publication, mirrors, offline cache,
   deprecation, and revocation.
5. Consider changing the default only after the gates pass. Preserve the Release
   path for an announced compatibility window.

## Acceptance gates

- interrupted publication never exposes a broken candidate;
- parallel publishers cannot lose catalog entries;
- altered indexes, manifests, and layers fail signature or digest verification;
- the prior signed index can be restored without republishing bundles;
- anonymous public pull, authenticated private pull, resume, mirror, and offline
  cache behavior are tested;
- exact-target, same-compute-capability, broader compatibility, and shape-range
  resolution retain their documented behavior;
- installation receipts reproduce the selected index and bundle;
- existing Release clients continue to work during the migration window; and
- operational cost and complexity are recorded before adopting OCI as default.

## Open decisions

- OCI host and signing trust model;
- ORAS-based publishing versus a native client;
- one archive layer versus measured multi-layer reuse;
- index sharding threshold and mirror policy; and
- exact-plan versus canonical dynamic-plan preference after controlled tests.

## References

- [GitHub Release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [ORAS documentation](https://oras.land/docs/)
- [Hugging Face repositories](https://huggingface.co/docs/hub/en/repositories)
- [NVIDIA NGC Catalog](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
- [Triton model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [PyTorch Hub checkpoints](https://docs.pytorch.org/docs/stable/hub.html)
- [Ultralytics download implementation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/downloads.py)
