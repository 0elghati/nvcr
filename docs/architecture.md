# Architecture

NVCR is a stateful DCVC-RT deployment runtime, not a model framework. The
supported boundary is defined in [Scope and support](scope-and-support.md); this
page describes the current implementation and clearly labels unfinished v1 work.

## System boundary

```text
CLI / benchmark / C++ application
               |
        nvcr::Runtime session
               |
  access-unit validation + sequence state
               |
       dcvcrt::CodecBackend
        /                  \
TensorRT/CUDA stages     native CPU rANS
        \                  /
         device/host DPB state
```

There is no generic codec plugin registry. DCVC-RT is the only supported codec,
so codec-specific orchestration stays in `nvcr::dcvcrt`. CUDA, TensorRT, and
entropy implementation types do not cross the public runtime boundary.

## Ownership

| Area | Owns | Does not own |
|---|---|---|
| `runtime` | Session lifecycle, serialized calls, `Frame`, `Packet`, statistics | Neural graph ordering or TensorRT objects |
| `dcvcrt` | I/P decisions, independent encoder/decoder state, TensorRT/CUDA/rANS orchestration | Training, checkpoints, model architecture |
| `bitstream` | Bounded `NVAU` access units and development packet framing | Standard containers or FFmpeg metadata |
| `configuration` | Per-session model, device, memory/execution, QP, GOP, and compatibility policy | Engine generation |
| `memory` | Host resource and reusable best-fit pool | Completed CUDA arena (v1 work) |
| `cli` | Raw YUV420P8 I/O and development sequence files | Codec state |
| `tools` / `scripts` | Offline reference and artifact helper implementation | Deployed runtime dependencies |
| `benchmarks` | Repeatable performance probes | Correctness authority |

## Session lifecycle and state

```text
create -> validate configuration/bundle -> initialize backend -> ready
                                                       |
                                  encode / decode / reset / flush
                                                       |
                                                     ready
```

`Runtime::create` validates configuration, initializes the injected backend, and
creates independent encoder and decoder `SequenceState` objects. Calls are
serialized because DCVC-RT mutates references and latent state. A failed backend
operation is not committed. `reset()` discards sequence state immediately;
`flush()` finishes backend work and resets both directions.

Each backend stores its selected TensorRT execution policy. `automatic` chooses a
conservative low-memory mode on constrained devices; `low_memory` and
`performance` are explicit per-session choices. The legacy environment override
is only a fallback and is not the v1 configuration contract.

A GOP begins with an I-frame. P-frames require a valid reference; later GOPs and
reset/reuse are covered by the registered engine roundtrip test. The backend DPB
may contain a reconstructed device frame and a learned feature reference.

## DCVC-RT execution

The backend owns 14 deserialized plans, their execution contexts, a nonblocking
CUDA stream, entropy tables, a 72-entry P-frame quantization cache, rANS state,
and encoder/decoder DPBs.

The I path executes analysis, hyper-analysis, hyper-synthesis, three spatial-prior
engines over four checkerboard passes, synthesis, and native rANS.

The P path executes frame/feature reference adaptation, analysis,
hyper-analysis, temporal prior, spatial prior, synthesis, adaptive QP shifts,
native rANS, and frame/feature DPB update. P-frame effective QP is bounded to
`0..71`; invalid derived indexes fail before cache access.

The P neural chain is substantially device-resident, while CPU entropy boundaries
and some prior/index handling still require transfers. The I chain and remaining
P state work have avoidable per-frame allocations. The reusable CUDA arena,
complete device-resident I path, and removal of avoidable context churn remain
active M1 work; current code is not described as performance-complete.

## Artifact preflight

Before TensorRT deserializes a plan, initialization requires an
`nvcr.engine-bundle.v2` manifest and verifies:

- configured model profile identity and FP16 precision;
- CUDA runtime, exact TensorRT version, GPU name, compute capability, and SM count;
- the digest of `engine.sha256`;
- the exact 14-plan/six-runtime-asset file set and every file digest;
- I/P model-manifest identities.

The offline `nvcr-artifacts validate` command additionally verifies source commit,
checkpoint hashes, graph/asset hashes, portable relative paths, and manifest
schemas. TensorRT plans are target-local and are never treated as portable.

## Access units and development framing

The codec boundary uses the versioned `NVAU` access unit. It contains model
identity, even dimensions, frame type, effective QP, reset state, and a bounded
codec payload. Parsers reject unknown flags, bad identities, invalid dimensions,
QP overflow, truncation, trailing bytes, and oversized lengths before allocation.

The TensorRT backend currently retains isolated `NVI1`/`NVP1` payload internals.
The CLI wraps access units in the older `NVCR` packet and `NVCS` sequence records
for timestamps and file streaming. Those outer formats are development formats,
not a v1 container or an upstream compatibility claim. See [Bitstream](bitstream.md).

## Public API status

The current C++ API uses owned `Frame`/`Packet` values, `RuntimeConfiguration`,
explicit reset/flush, integer timestamps, and structured `Result<T>` errors. It
is usable by the CLI and tests but is not ABI-frozen.

Before v1, the session API must stabilize explicit Y/U/V planes, strides,
dimensions, device selection, and host/device view ownership. A C ABI, FFmpeg
encoder/decoder wrapper, timestamp/container mapping, and hardware-frame
integration are post-v1 work.

## Incremental structure

Large directory moves are deferred until correctness tests are reliable. As code
is modified, application entry points move toward `apps/`, artifact/reference
logic toward `tools/`, and performance/energy logic toward `benchmarks/`. The
TensorRT backend should be split along manifest, engine/session, I-frame, P-frame,
and state/asset boundaries without rewriting working behavior.

## See also

- [Scope and support](scope-and-support.md)
- [Compatibility](compatibility.md)
- [Bitstream](bitstream.md)
- [Model and engine preparation](dcvcrt-artifacts.md)
- [Performance](performance.md)
