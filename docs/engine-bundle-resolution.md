# Capability-based engine bundle resolution

Status: design note; no runtime, catalog, or roadmap priority is changed by this
document.

## Problem

NVCR currently treats a named engine profile as both an optimization choice and
an artifact identity. The CLI maps visible dimensions to one of `qcif`, `cif`,
`360p`, `540p`, `720p`, or `1080p`, and the catalog admits only those
names. It then looks for that exact installed profile. It does not search other
installed bundles whose declared shape range covers the requested dimensions.

That makes optional performance specializations look mandatory. For example,
the fixed `540p` profile accepts only 960x540, but the dynamic `720p` and
`1080p` profiles declare ranges that also contain 960x540. An absent `540p`
bundle therefore blocks a workload that an installed dynamic bundle may be able
to execute.

The general problem is not the 540p mapping. A user should be able to request
any codec-supported dimensions without first knowing NVCR's profile names or
waiting for a shape-specific build.

## Desired contract

A named resolution is an optimization point, not a required artifact identity.
The request supplies visible width and height. Artifact resolution filters
bundles by hard compatibility first, then by shape coverage:

1. codec, model, provider, precision, numerics, operating system, architecture,
   target class, CUDA, TensorRT, manifest version, digest, and license match;
2. the declared visible-shape domain contains the requested dimensions and its
   alignment constraints are satisfied;
3. distinguish canonical coverage bundles from optional accelerator variants;
4. apply an explicit selection policy and deterministic tie-breaking.

The default selection policy remains an open decision. An exact optimization
point is not proof that a plan is faster, and the smallest covering range is not
proof that a plan is safer. Two policies need comparison:

- **exact-first:** preserve the current exact choice when it is available, then
  fall back to a covering dynamic bundle;
- **canonical-dynamic-first:** use a stable dynamic bundle for its preferred
  dimension region, and select a fixed accelerator only by explicit request or
  pinned deployment policy.

Whichever policy is accepted must record the chosen bundle and reason. Merely
installing an optional accelerator must not silently change output unless the
active policy permits that behavior. Decode must satisfy the stream's provider,
numerics, and artifact requirements rather than independently optimizing from
width and height.

An explicit bundle path remains an override. An explicit profile name remains
useful for benchmarking and debugging, but ordinary encode and decode should
not require one.

Resolution must never silently build a plan. If no validated installed bundle
covers the request, NVCR may offer a catalog download or an explicitly selected
provider fallback. Otherwise it fails before execution with the rejected
dimensions and available coverage.

## Registry and bundle model

Keep immutable bundles addressed by archive and manifest digests. Add the
minimum capability metadata needed to resolve before download:

- visible minimum, optimum, and maximum dimensions;
- preferred dimension region and canonical or accelerator role;
- fixed or dynamic shape profile;
- visible-dimension alignment constraints;
- codec, model, provider, precision, and numerics identities;
- target, CUDA, and provider-runtime compatibility;
- a digest binding the catalog record to the complete bundle manifest;
- retained conformance evidence identifying the shapes actually validated.

The remote catalog and local installed-bundle inventory should expose the same
candidate model. Catalog fields duplicated from the manifest must be checked
against it during installation, as target and runtime fields are today. The
CLI should query this inventory rather than reconstructing a bundle path from a
hard-coded profile label.

Shape coverage is a provider artifact property. Codec semantics continue to
own valid frame dimensions and padding. TensorRT owns optimization profiles,
plans, tactics, memory, and shape rejection. A codec-specific resolution table
must not be added to generic runtime code.

## MLVC precedent

MLVC separates learned weights from deployment shapes, but its exported graphs
are still fixed-shape. One checkpoint represents a model flavor and objective,
such as MLVC-S perceptual. Its
[bundle command](https://github.com/microsoft/mlvc#model-bundles) exports that
model at four default canvases: 960x544, 640x368, 432x240, and 320x192. Each
canvas has a separate encoder graph and decoder graph.

The upstream
[ONNX bundler](https://github.com/microsoft/mlvc/blob/main/video/conversion/_model_bundler.py)
hashes large tensors across all graphs. A tensor repeated in more than one
graph is stored once in a shared `weights.bin`; graph-specific tensors remain
inside their ONNX file. The result is one logical model bundle with shared
weights, multiple fixed graph pairs, per-canvas metadata, and a registry that
binds every part to both its graph and the shared weights.

At runtime MLVC chooses the smallest canvas that contains the visible frame,
tries the transposed orientation when necessary, edge-pads the input, and crops
the reconstruction. A new visible resolution therefore needs no export when it
fits an existing canvas. The tradeoff is extra computation on the padded area;
a frame outside every canvas still needs another export or fails.

NVCR's MLVC TensorRT packager preserves that logical bundle and builds one
fixed plan per canvas and part: eight plans for four canvases. This is a useful
packaging precedent, but DCVC-RT should not copy the fixed-canvas constraint
unnecessarily. Its existing dynamic profiles can provide true shape ranges.
The reusable lesson is one discoverable bundle with shared identity and
capability-based selection, with shape-specific plans kept as optional
performance specializations.

DCVC-RT's learned checkpoints are already independent of visible resolution.
Its duplication is in separately compiled TensorRT plans, whose weights and
tactics are self-contained. Those plans cannot directly share MLVC's external
`weights.bin` mechanism. Combining several fixed DCVC-RT profile directories
into one archive would simplify discovery but would not remove the duplicated
compiled weights. Dynamic coverage avoids more of that duplication.

## Fixed and dynamic plans

"Exact" can mean two different things:

- a fixed-shape plan accepts only one width and height;
- a dynamic plan can cover a range while using that width and height as its
  TensorRT optimization point.

Most performance value comes from optimizing tactics for the workload's shape;
it does not necessarily require excluding every other shape. A fixed plan can
also enable stronger shape specialization. Compared with an off-optimum dynamic
plan, it may improve throughput, latency, plan size, load or warm-up time, and
device memory. None of those improvements is guaranteed without measurement.

Separately built FP16 plans can select different tactics and accumulation
orders. They must not be assumed to produce byte-identical streams or
reconstructions merely because they cover the same visible dimensions.
Self-consistent encode/decode and the applicable cross-bundle gates remain
required.

The working hypothesis is a small set of validated dynamic coverage bundles,
with fixed plans or additional optimization points treated as accelerators.
That is not an accepted default until the controlled comparison establishes
the performance, memory, numerical, and reproducibility tradeoffs.

## Validation before advertising coverage

TensorRT accepting `setInputShape` is necessary but insufficient evidence.
Each advertised domain needs release-build tests at:

- minimum, optimum, and maximum shapes;
- boundary-adjacent aligned shapes;
- representative aspect ratios and intermediate sizes;
- encode/decode self-consistency, reset, drain, and repeated-session cases;
- malformed and out-of-range rejection;
- retained stream, reconstruction, quality, size, speed, and peak-memory
  measurements.

This matters because a broad profile can be valid at its optimization point yet
show numerical or tactic problems at smaller latent shapes. Coverage belongs in
the registry only after those cells pass, not merely because the profile bounds
contain them.

## First controlled comparison

Measure 960x540 on the same GPU, CUDA version, TensorRT version, model bundle,
precision, and source revision using:

1. the fixed 540p bundle;
2. the dynamic 720p bundle at 960x540.

Record build duration, archive and plan sizes, load and warm-up latency, process
and synchronized codec intervals, peak device memory, payload size, PSNR,
stream hash, and reconstruction hash. Use release builds and the performance
protocol. The current profiles also differ in builder optimization level, so a
causal fixed-versus-dynamic comparison needs a second pair with that setting
held constant.

## Smallest implementation sequence

1. Replace the CLI's single profile lookup with installed-manifest discovery
   and covering-range candidates. Retain explicit profile and directory
   overrides. Keep selection policy separate from capability discovery, and add
   deterministic resolver tests including 960x540 fallback.
2. Extend the catalog schema and installer with the same shape-capability
   fields so `install --size WIDTHxHEIGHT` can fetch the best candidate without
   downloading every bundle.
3. Run the coverage and fixed-versus-dynamic gates before reducing the
   published profile set or changing defaults.
4. Consider multiple TensorRT optimization profiles only if measurements show
   that a few dynamic coverage bundles do not provide acceptable performance.

This work affects artifact resolution, packaging, provider behavior, and
performance evidence. It should receive its own roadmap item when prioritized;
it is not implicitly part of the currently active provider-coverage work.
