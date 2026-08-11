# Claims and evidence

This matrix separates architecture-level claims from claims about the first
production vertical. A claim is not release-ready until the listed code, tests,
and experimental evidence are retained for the evaluated revision.

| Claim | Code evidence | Test evidence | Experimental evidence | Current status |
|---|---|---|---|---|
| Stateful encoder/decoder session lifecycle | `include/nvcr/codec/session.hpp`, `Runtime`, sequence state | Contract, unit/smoke, reset/flush/drain tests | Required in every production run gate | Implemented; production evidence in progress |
| Codec/provider separation | Codec/provider headers, registry, `RuntimeServices` | Contract and resolver tests; header dependency check | Provider-mediated construction in target runs | Implemented core; component split transitional |
| Target-aware artifact resolution | Artifact descriptors, catalog, resolver, validator | Resolver/catalog and artifact-tool tests | Current manifest/catalog digests and target identity per row | Implemented; exact-target evidence in progress |
| Bounded access-unit contract | `AccessUnitIO`, packet I/O, stream specification | Format, parser fuzz/boundary, malformed-input tests | Stream identity recorded in production rows | Implemented core |
| Production DCVC-RT/TensorRT integration | DCVC-RT adapter, native rANS, TensorRT backend | Engine contracts, I/P round trips, golden/reference tests | Exact target matrix and deployment record | Implemented production path; release gates in progress |
| Reference consistency | DCVC-RT profile and comparison tooling | I-frame golden and payload tests | Pinned Python rows with matching inputs and tolerance | Required evidence; not payload interchangeability |
| Target deployment | Installer, package, Docker, catalog workflows | Installer/package/container checks | Target-local validated bundle and runtime result | Implemented infrastructure; target gates in progress |
| Reproducible evidence generation | `benchmark_softwarex_matrix.py`, schemas, runbook | Matrix-driver tests and Python syntax checks | Clean complete package with profile metrics and hashes | Implemented tooling; complete package pending |

The test codec and CPU provider appear in architecture tests only. They must
not be counted as a second production codec/provider or used as a production
performance baseline. Energy measurements are optional downstream research
data unless a separate controlled protocol elevates that claim.
