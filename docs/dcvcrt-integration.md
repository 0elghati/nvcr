# DCVC-RT integration

DCVC-RT is NVCR's first codec integration. This page explains how it fits into
NVCR; it is not the definition of NVCR itself. NVCR provides discovery,
construction, common lifecycle/error contracts, artifact selection, and bounded
access-unit framing. DCVC-RT sessions provide sequence policy, state, entropy,
codec-private payloads, and access-unit semantics. TensorRT is the current
execution provider.

## What happens when a session starts

The runtime resolves the requested codec adapter and provider. Artifact
selection finds a compatible engine for the machine and checks the complete
bundle before loading it. The DCVC-RT adapter then composes that provider
session into encoder and decoder sessions. The user-facing artifact tool is
`nvcr-artifacts`. Exporters and builders under `scripts/backends/dcvcrt/` are
source-maintainer tools, not a normal installation path.

## How frames are handled

An I-frame starts a group of pictures. Later P-frames use the previous decoded
state. The codec sessions own the GOP decision and separate encoder/decoder
`SequenceState`; TensorRT owns learned-model execution, and native rANS handles
entropy coding. Directional reset clears only that direction's state.
Directional flush finishes its pending work and makes it drainable to
`end_of_stream`.

## Stream compatibility

NVCR carries DCVC-RT data inside its bounded `NVAU` access-unit format. The
internal rANS, `NVI1`, and `NVP1` layouts are implementation details. NVCR does
not claim that its streams are byte-compatible with the upstream Python
implementation.

## Current limits

TensorRT currently creates the production provider session, and the DCVC-RT
adapter owns the codec sessions composed around it. Additional production
codecs and providers, stable public plane/stride ownership, and the timing and
portability prototypes in the
[cross-codec audit](cross-codec-requirements.md#open-questions-requiring-prototypes)
remain future work. The v1.x release line is already published.
