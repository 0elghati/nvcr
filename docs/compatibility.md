# Compatibility

NVCR's current target matrix covers the runtime, engine, and hardware identity
needed for each completed platform.

| Profile | Runtime path | Engine requirement | Status |
|---|---|---|---|
| `rtx3050-laptop-ubuntu2404` | Linux x86_64, RTX 3050 Laptop, TensorRT FP16 | Exact target-local plans | Docker execution recorded; exact target validation remains open |
| `rtx4070-ubuntu2404` | Linux x86_64, RTX 4070 SM 8.9, TensorRT FP16 | Exact plans and TensorRT 10.9 identity | Completed execution results |
| `rtx5060-laptop-ubuntu2404` | Linux x86_64, RTX 5060 Laptop, TensorRT FP16 | Exact target-local plans | Portability execution recorded |
| `orin-nano-l4t3647` | Linux AArch64, Jetson Orin Nano, TensorRT FP16 | Exact Jetson plans and CUDA/TensorRT runtime | Completed native execution results |


## Compatibility dimensions

| Dimension | Meaning in NVCR |
|---|---|
| Software/API | Public C++ API compatibility follows semantic versioning: patch and minor releases preserve compatibility; breaking changes require a major release. Cross-release ABI compatibility is not guaranteed; rebuild applications when upgrading. |
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

## Cross-target decode

A DCVC-RT stream decodes correctly only on a decoder that reproduces the
encoder's entropy parameters bit for bit: the same execution provider and the
same exact plan set on the same GPU model. A different GPU (for example RTX
4070 to RTX 5060), a different plan set, or a different provider computes
slightly different FP16 scales. Any scale near one of the 128 entropy-table
boundaries then selects a neighbouring table and desynchronizes the rANS
decoder (issue #216).

Such a decode fails with `malformed_bitstream` ("DCVC-RT entropy decoding
desynchronized") instead of reconstructing wrong frames. Each frame's rANS
stream must end exactly at the encoder's initial state with every byte
consumed, and the decoder never reads past the stream. Detection is
probabilistic but practically certain for a desynchronized frame. Bit errors
that only touch raw bypass bits do not desynchronize the coder and are not
detected, because `NVAU` carries no checksum.

An encoder can instead request exact numerics (`--numerics nvcr.intconv.v1`).
Every decoder-side stage then runs in exact integer arithmetic from portable programs in the artifact bundle's
`normative/` directory, so every provider that implements the profile
reproduces the encoder's entropy decisions, reference state and decoded frames
bit for bit, I- and P-frames alike. The stream names the profile and a digest
of the programs on every random-access unit; a decoder without the profile or
with different programs fails before decoding. See
[cross-target decode](cross-target-decode.md).

### DCVC-RT capability matrix

Cells follow `AGENTS.md`: `exact` (bit-identical through a conformant numerics
profile), `native` (decodes with the provider's native arithmetic),
`unsupported` (fails, cleanly), `untested`. A cell is filled only from a
retained conformance result; support is never inferred from a GPU model or
from another pair.

Decoding a stream encoded on a different provider or target:

| Encoder | Decoder | Native numerics (default) | `nvcr.intconv.v1` |
|---|---|---|---|
| TensorRT, RTX 4070 | TensorRT, same RTX 4070 and plans | `native`, bit-identical | `exact` |
| TensorRT, RTX 4070 | ONNX Runtime CPU, same host | `unsupported` (desync detected) | `exact` |
| ONNX Runtime CPU | TensorRT, RTX 4070, same host | `unsupported` (desync detected) | `exact` |
| TensorRT, RTX 4070 | TensorRT, RTX 5060 | `untested` (issue #216 reports desync) | `untested` |
| Any | ONNX Runtime CUDA or QNN | `untested` | `unsupported` (profile not implemented; fails before decode) |

Exact numerics currently cost speed: at 1080p on the RTX 4070, decode runs at
18.4 fps against 48.7 fps native and encode at 30.1 against 64.5 fps (codec
interval, diagnostic measurement).
Evidence: [issue #216](../evidence/issue-216-cross-hardware-fix-20261104.md).

## Format and codec position

- `NVAU` is the versioned, bounded codec access-unit contract. Version 1 is the
  production default; version 2 is implemented but experimental, with no stable
  public format guarantee.
- `NVCR`/`NVCS` and `.nvcr` are development/application wrappers with no claim
  to be standard multimedia containers.
- The DCVC-RT inner payload is an NVCR implementation format, not an upstream
  Python bitstream contract.
- Reference consistency is a reconstruction/rate-distortion claim; payload
  interchangeability requires bidirectional cross-runtime golden tests.

## Current format compatibility

| Format | Writer | Current reader |
|---|---|---|
| NVAU v1 | Default writer | Accepted |
| NVAU v2 | Explicit sectioned writer; experimental | Accepted |
| Other NVAU versions | None | Rejected |
| NVCR packet envelope | Version 1 | Version 1 only |
| NVCS sequence wrapper | Version 1, flags 0 | Version 1, flags 0 |
| Private codec payload | NVI1 / NVP1 | Corresponding syntax and supported fields |

Acceptance means parsing valid, bounded data. It does not establish support for
every v2 feature in production decoding or interchangeability with upstream
Python streams.

See [compatibility levels](experiments/compatibility-levels.md) for the
experiment-specific acceptance rules and [scope and support](scope-and-support.md)
for the release support gate.
