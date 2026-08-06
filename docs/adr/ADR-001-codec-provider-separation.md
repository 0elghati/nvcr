# ADR-001: Codec/Provider Separation

Status: accepted
Date: 2026-08-06

## Context

NVCR first vertical implementation uses DCVC-RT with TensorRT, but runtime architecture must keep codec semantics separate from execution provider mechanics.

## Decision

- Codec adapter owns codec semantics, payload syntax, temporal behavior, and model-component meaning.
- Execution provider owns executable loading, tensor bindings, device execution, and provider-specific failures.
- Runtime mediates via provider and codec APIs, not backend-specific concrete types.

## Consequences

- Core public headers avoid TensorRT/CUDA/DCVC-RT internals.
- Additional codecs/providers can be registered without changing runtime core.
