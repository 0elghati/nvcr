# The first production codec integration: DCVC-RT

DCVC-RT is NVCR's first complete production codec integration, not the identity
of NVCR. The adapter owns GOP decisions, model-component meaning, entropy state,
codec-private payloads, and frame/feature references. TensorRT is an execution
provider, not part of the codec identity. NVCR core owns sessions, registry and
services, artifact selection, and bounded access-unit framing.

## Initialization

A session selects a model profile, target-local engine directory, device, QP, GOP, memory policy, and TensorRT execution mode. Before deserialization, NVCR validates the bundle schema, model and target identity, precision, runtime versions, required files, and checksums.

The supported artifact front end is `scripts/nvcr_artifacts.py`. The backend-specific exporters and builders under `scripts/backends/dcvcrt/` are implementation helpers, not separate user workflows.

## Frame paths

The I path runs image analysis, hyperprior, spatial priors, synthesis, and native rANS. The P path runs temporal/reference adaptation, analysis, hyperprior, temporal and spatial priors, synthesis, state updates, and native rANS.

An I-frame starts a GOP. P-frames require a valid reference. The encoder reports the effective QP used by the access unit. Reset clears references; flush completes work and resets the session.

## Payload and compatibility

The native rANS coder and `NVI1`/`NVP1` payload layouts are DCVC-RT adapter
implementation details. They are carried inside the bounded `NVAU` contract and
are not claimed to be upstream Python bitstreams.

Python is an offline export and reference-conformance dependency. Cross-runtime payload compatibility remains unclaimed until clean bidirectional golden tests pass.

## Current limits

Production component creation is provider-mediated through `RuntimeServices`.
TensorRT creates the codec backend as one provider-owned monolithic component
rather than exposing each model stage as an independent executable through
`IExecutionProvider::load`; that split is transitional. Stable public
plane/stride ownership, additional production providers, complete target
performance evidence, and the final v1 release gate remain future work.
