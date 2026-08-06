# ADR-004: Extension and Registration Model

Status: accepted
Date: 2026-08-06

## Context

M-EXT requires extensibility proof without introducing unstable plugin ABI complexity.

## Decision

- Use static registration for codecs/providers.
- Registration can occur via explicit init entry points or static startup hooks.
- Dynamic plugin loading is out of scope for v1.

## Consequences

- Startup and deployment remain simple.
- Contract tests can verify extensibility boundaries with test codec/provider implementations.
