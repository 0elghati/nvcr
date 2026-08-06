# SoftwareX experiments

This directory is the working protocol for the NVCR SoftwareX evaluation. It is deliberately separate from the short product documentation.

The evaluation covers the first complete NVCR vertical:

```text
codec adapter:       DCVC-RT
execution provider:  TensorRT
runtime:             C++20
artifacts:           target-local TensorRT engine bundles
stream layer:        bounded NVAU access units
pixels:              planar YUV420P8
```

The claim is narrow: NVCR provides a reproducible native runtime around this one codec/provider pair. It is not a claim of universal NVIDIA support, multi-codec production support, or Python bitstream compatibility.

## Read in this order

1. [SoftwareX evaluation protocol](softwarex-evaluation-protocol.md)
2. [Hardware targets](hardware-targets.md)
3. [Compatibility levels](compatibility-levels.md)
4. [Result schema](result-schema.md)
5. [Runbook](runbook.md)
6. [Missing-data checklist](missing-data-checklist.md)

## Current state

The repository already has device detection, target profiles, artifact validation, target-local engine building, Docker/Compose workflows, and `benchmark_resolution_matrix.sh`.

The checked-in target profiles currently cover RTX 4070, RTX 5060 Laptop, and Jetson Orin Nano. RTX 3050 exact is a planned evaluation target without a checked-in target profile. The SoftwareX result schema is defined here, but the existing shell benchmark does not yet emit every required field; that is an automation TODO, not evidence.

## Evidence rule

A clean evidence package contains metadata, hashes, commands, summaries, and small JSONL rows. Keep checkpoints, TensorRT plans, raw YUV, encoded streams, reconstructed video, and other large products outside Git.
