# Documentation

Start here if you are new to NVCR. The docs are arranged so users can first
install and run quickly, then go deeper into architecture and internals.

## Reading order

1. [Project README](../README.md) for the overview and scope.
2. [Binary Install Guide](install-binary.md) for fast user installation with
	published release artifacts.
3. [Getting Started](getting-started.md) for source build and local development setup.
4. [Native command-line interface](cli.md) for installed CLI usage on Linux.
5. [DCVC-RT artifact pipeline](dcvcrt-artifacts.md) for checkpoints, ONNX export,
	and target-local TensorRT engine generation.
6. [Architecture](architecture.md) for ownership, lifecycle, and module boundaries.
7. [Public API Reference](reference.md) for the exported runtime types.
8. [NVCR packet envelope](bitstream.md) for the wire format and sequence file layout.
9. [DCVC-RT integration contract](dcvcrt-integration.md) for codec-specific behavior.
10. [Performance](performance.md) for the current baseline and remaining work.

## Quick links

| Topic | Purpose |
|---|---|
| [Binary Install Guide](install-binary.md) | Fast install from published binary and engine bundles |
| [Getting Started](getting-started.md) | Build, install, and run NVCR from source |
| [Architecture](architecture.md) | Public runtime ownership and lifecycle |
| [API Reference](reference.md) | Public `nvcr` types and usage patterns |
| [Bitstream](bitstream.md) | Packet envelope and sequence file format |
| [CLI](cli.md) | `nvcr encode` and `nvcr decode` usage |
| [DCVC-RT artifacts](dcvcrt-artifacts.md) | Checkpoints, ONNX export, and TensorRT engines |
| [DCVC-RT integration](dcvcrt-integration.md) | TensorRT, rANS, and current codec status |
| [Performance](performance.md) | Baselines, targets, and remaining optimization work |

## Published-library notes

NVCR is intentionally split between a small public runtime API and detailed
codec-specific internals. The top-level README explains the package surface,
install flow, and scope. The topic docs below are where implementation-level
contracts live.

## Doc tree

```text
docs/
	README.md                 index and reading order
	install-binary.md         install from published binary/engine bundles
	getting-started.md        build, install, and first-run steps
	reference.md              public API summary
	architecture.md           ownership and lifecycle
	cli.md                    command-line usage
	dcvcrt-artifacts.md       checkpoint-to-engine artifact pipeline
	bitstream.md              wire format
	dcvcrt-integration.md     codec-specific integration contract
	performance.md            current benchmark baselines
```
