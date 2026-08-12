# DCVC-RT integration

DCVC-RT is NVCR's first codec integration. This page explains how it fits into
NVCR; it is not the definition of NVCR itself. NVCR provides the runtime,
engine selection, and stream handling. DCVC-RT provides the learned compression
logic. TensorRT is the current execution provider.

## What happens when a session starts

NVCR chooses the requested model profile, finds a compatible engine for the
machine, and then checks the bundle before loading it. The user-facing tool is
`nvcr-artifacts`. The exporters and builders under `scripts/backends/dcvcrt/`
are source-maintainer tools, not a normal installation path.

## How frames are handled

An I-frame starts a group of pictures. Later P-frames use the previous decoded
state. DCVC-RT performs learned-model processing with TensorRT and uses native
rANS entropy coding. Reset clears the stored reference state; flush finishes
pending work and resets the session.

## Stream compatibility

NVCR carries DCVC-RT data inside its bounded `NVAU` access-unit format. The
internal rANS, `NVI1`, and `NVP1` layouts are implementation details. NVCR does
not claim that its streams are byte-compatible with the upstream Python
implementation.

## Current limits

TensorRT currently creates the DCVC-RT backend as one provider-owned component.
Additional production codecs and providers, stable public plane/stride
ownership, and a final v1 release remain future work.
