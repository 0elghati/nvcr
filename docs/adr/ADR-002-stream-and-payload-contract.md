# ADR-002: Stream and Payload Contract

Status: accepted
Date: 2026-08-06

## Context

NVCR uses a codec-neutral access-unit envelope while codec payload bytes remain codec-private.

## Decision

- Keep NVAU envelope parser/serializer in shared stream layer.
- Keep codec payload opaque to stream layer.
- Preserve current v1/v2 wire bytes in this phase; only document and test behavior.

## Consequences

- Stream tooling can inspect framing without understanding codec payload internals.
- Codec evolution can occur behind payload version/profile identifiers.
