# ADR-005: Conformance Levels

Status: accepted
Date: 2026-08-06

## Context

NVCR needs precise language for compatibility and conformance claims in public documentation.

## Decision

Define levels:

1. runtime-compatible: implementation satisfies API contracts for local runtime use.
2. self-conformant: encode/decode cycle conforms within same codec/provider stack.
3. cross-provider-conformant: equivalent behavior across multiple providers for one codec.
4. cross-device-conformant: equivalent behavior across supported target devices.
5. reference-fidelity-validated: quality/performance validated against pinned reference workflow.
6. bit-exact: byte-identical outputs proven where contract requires exactness.

## Consequences

- Claims become measurable and auditable.
- CI/release evidence can map directly to explicit conformance levels.
