# Documentation

Start here if you are new to NVCR. The docs are arranged from the public
library boundary down to the codec-specific internals.

## Reading order

1. [Project README](../README.md) for the overview and scope.
2. [Getting Started](getting-started.md) for build, install, and first-run steps.
3. [Architecture](architecture.md) for ownership, lifecycle, and module boundaries.
4. [Public API Reference](reference.md) for the exported runtime types.
5. [Native command-line interface](cli.md) for installed CLI usage on Linux.
6. [NVCR packet envelope](bitstream.md) for the wire format and sequence file layout.
7. [DCVC-RT integration contract](dcvcrt-integration.md) for codec-specific behavior.
8. [Performance](performance.md) for the current baseline and remaining work.

## Quick links

| Topic | Purpose |
|---|---|
| [Getting Started](getting-started.md) | Build, install, and run NVCR on Linux |
| [Architecture](architecture.md) | Public runtime ownership and lifecycle |
| [API Reference](reference.md) | Public `nvcr` types and usage patterns |
| [Bitstream](bitstream.md) | Packet envelope and sequence file format |
| [CLI](cli.md) | `nvcr encode` and `nvcr decode` usage |
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
	getting-started.md        build, install, and first-run steps
	reference.md              public API summary
	architecture.md           ownership and lifecycle
	cli.md                    command-line usage
	bitstream.md              wire format
	dcvcrt-integration.md     codec-specific integration contract
	performance.md            current benchmark baselines
```

