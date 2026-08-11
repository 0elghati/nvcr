# Compatibility

Compatibility is multidimensional. A successful build or a matching target
profile does not imply that every layer is interchangeable.

## Current production matrix

| Profile | Production path | Artifact requirement | Current evidence status |
|---|---|---|---|
| `rtx3050-laptop-ubuntu2404` | Linux x86_64, RTX 3050 Laptop, TensorRT FP16 | Exact target-local plans | Profile registered; retained results are diagnostic |
| `rtx4070-ubuntu2404` | Linux x86_64, RTX 4070 SM 8.9, TensorRT FP16 | Exact plans and current TensorRT 10.9 identity | Profile registered; current bundles need model-profile refresh before publication evidence |
| `rtx5060-laptop-ubuntu2404` | Linux x86_64, RTX 5060 Laptop, TensorRT FP16 | Exact target-local plans | Profile registered; retained results are diagnostic |
| `orin-nano-l4t3647` | Linux aarch64, Jetson Orin Nano, TensorRT FP16 | Exact Jetson plans and CUDA/TensorRT runtime | Profile registered; warning-free six-profile gate remains required |

Other systems may build or run locally, but are not support targets until a
profile and the applicable evidence gates exist.

## Compatibility dimensions

| Dimension | Meaning in NVCR |
|---|---|
| Software/API | C++ headers, session contracts, codec/provider API versions, and CLI behavior; not ABI-frozen |
| Artifact schema | Manifest/catalog/model-set versions and required files |
| Provider runtime | CUDA/TensorRT/provider version and execution capabilities |
| Hardware target | Device name, compute capability, architecture, memory, and target profile |
| Stream format | `NVAU` version, IDs, sections, bounds, and parser rules |
| Codec payload | Private DCVC-RT syntax and model/profile behavior inside `NVAU`; not upstream Python interchange |
| Application/container | `Packet`, `.nvcr`, `NVCR`, and `NVCS` wrappers; not MP4, Matroska, or standard-container compatibility |

## Artifact classes

- **Exact:** built and validated for the named target profile. This is the
  primary production evidence mode.
- **Same compute capability:** explicitly cataloged desktop artifact for the
  same SM class, with representative correctness and comparison evidence.
- **Ampere-plus:** explicit desktop fallback class, separately validated and not
  applicable to Jetson.

TensorRT matching remains exact in the catalog policy. Jetson is exact-only.
CUDA helper-kernel coverage in a host binary does not make a TensorRT plan
portable across devices or runtimes. Always say what is portable: source code,
host binary, CUDA code, engine bundle, catalog entry, access unit, or container
userspace.

## Format and codec position

- `NVAU` is the versioned, bounded codec access-unit contract; v1 and v2 are
  implemented.
- `NVCR`/`NVCS` and `.nvcr` are development/application wrappers with no claim
  to be standard multimedia containers.
- The DCVC-RT inner payload is an NVCR implementation format, not an upstream
  Python bitstream contract.
- Reference consistency is a reconstruction/rate-distortion claim; payload
  interchangeability requires bidirectional cross-runtime golden tests.

See [compatibility levels](experiments/compatibility-levels.md) for the
experiment-specific acceptance rules and [scope and support](scope-and-support.md)
for the release support gate.
