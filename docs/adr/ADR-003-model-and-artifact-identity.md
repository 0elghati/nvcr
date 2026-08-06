# ADR-003: Model and Artifact Identity

Status: accepted
Date: 2026-08-06

## Context

Runtime must resolve executable artifacts deterministically across model sets, components, provider targets, and precision.

## Decision

Artifact identity uses a tuple including at least:

- codec_id
- model_set_id
- component_id
- provider_id
- target_device
- precision

## Consequences

- Resolver behavior becomes explicit and testable.
- Missing/incompatible artifacts map to structured errors instead of ad-hoc failures.
