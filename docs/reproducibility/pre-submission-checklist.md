# Release and publication checklist

Complete this checklist for the exact code and evidence release named by the
manuscript or other publication record.

## Publication and access

- [ ] The manuscript follows the selected current publication template and is
      accompanied by the open-source distribution and support material required
      by that venue.
- [ ] Repository is publicly visible, or the access plan explicitly records how
      reviewers receive access.
- [ ] Immutable version tag and public release assets exist.
- [ ] Permanent code-release and evidence/capsule links replace all placeholders.
- [ ] Open-source code license is present and compatible with submission policy.
- [ ] Citation metadata and preferred software citation are finalized.
- [ ] Support contact and problem-report route are finalized.

## Legal and distribution review

- [ ] Checkpoint and model terms are documented.
- [ ] ONNX/exported assets and TensorRT-plan redistribution decisions are
      recorded for every public asset.
- [ ] Dataset licenses and redistribution status are recorded.
- [ ] Third-party notices and dependency licenses are complete.
- [ ] No restricted checkpoint, ONNX graph, plan, raw dataset, token, or
      temporary asset URL is included in the public package.

## Reproduction

- [ ] README and developer documentation work from a clean checkout.
- [ ] Installation succeeds from the public release artifact.
- [ ] CPU contract tests pass and their scope is clearly reported.
- [ ] Exact target artifacts are validated and their hashes are retained.
- [ ] Production I/P, reset, flush/drain, malformed-input, and reference gates
      pass on the named targets.
- [ ] Result rows record clean commit identity, hardware/software,
      profile/model/artifact digests, inputs, commands, metrics, and statuses.
- [ ] A separate profiling pass supplies required quality, latency, and memory
      metrics without contaminating primary throughput timings.
- [ ] Required pinned reference rows and tolerances are retained.
- [ ] No dirty-commit, skipped, partial, plan-only, or missing-metric run is
      presented as publication evidence.

## Claim review

- [ ] The manuscript calls NVCR a runtime architecture, not merely a DCVC-RT
      implementation.
- [ ] DCVC-RT/TensorRT is identified as the first and only production vertical.
- [ ] The deterministic test codec and CPU provider are described as fixtures.
- [ ] `NVAU`, `.nvcr`, `NVCR`, and `NVCS` are described with their actual scope.
- [ ] No claim of universal portability, standard container support, FFmpeg
      integration, upstream byte/payload interchangeability, or ABI stability
      remains without evidence.
