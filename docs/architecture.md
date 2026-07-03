# Architecture

Start with [Documentation](README.md) or return to the [project overview](../README.md).

## Design constraints

NVCR is a stateful deployment library, not a model framework. Its architecture
therefore favors explicit ownership, bounded allocations, narrow dependency
boundaries, and errors that cross module boundaries as values.

There is no generic codec base class or runtime plugin registry. Public codec
integration types live in `nvcr::dcvcrt`, because DCVC-RT is the only v0.1 codec.
This avoids prematurely freezing an abstraction around one implementation while
still keeping dependencies out of the application API.

## Module ownership

| Module | Owns | Does not own |
|---|---|---|
| `runtime` | Public lifecycle, serialization of calls, `Frame`, `Packet` | TensorRT or entropy implementation details |
| `dcvcrt` | I/P decisions, encoder/decoder state, backend orchestration, TensorRT I-frame plans, native rANS integration | Neural architecture, weights, entropy algorithms |
| `bitstream` | NVCR packet envelope, validation, metadata | Entropy-coded payload semantics |
| `cli` | Raw YUV I/O, sequence records, encode/decode commands | Codec state or TensorRT implementation |
| `memory` | Reusable allocations through memory resources | CUDA policy outside a CUDA resource |
| `configuration` | Parsing and validation | Backend-specific engine inspection |
| `logging` | Per-subsystem logger creation | Global logger state |
| `statistics` | Atomic counters and accumulated timings | Benchmark harnesses |
| `common` | Error codes and expected-style results | Exceptions as API control flow |

## Runtime lifecycle

```text
create
  |
  v
initializing ---- adapter/backend failure ----> failed
  |
  v
ready -- encode/decode/reset/flush --> ready
  |
  v
destruction / moved-from ------------> stopped
```

`nvcr::Runtime::create` validates configuration, creates subsystem loggers and
statistics, then initializes the codec backend. Calls are serialized
because a DCVC-RT sequence mutates references and latents. Encoder and decoder
state are independent, allowing a runtime to perform a round trip without one
direction corrupting the other. The current native path is I-frame only;
predicted frames return `not_implemented`, and `gop_size=1` is required.

No singleton holds CUDA, logger, sequence, or model state.

## Sequence state

`dcvcrt::SequenceState` contains:

- the reconstructed reference frame;
- opaque backend latent state;
- absolute frame index;
- GOP position and configured GOP size;
- a generation incremented on every reset.

The first frame and each GOP boundary are intra frames. Predicted frames are
rejected when a reference is unavailable or an intra boundary is expected.
Backend work is committed only after the codec backend succeeds, so a failed operation does not partially advance sequence state.
For the current runtime, configure `gop_size=1`.

`flush()` asks the backend to finish outstanding work and then resets encoder,
decoder, inference, and entropy state. `reset()` discards sequence state
immediately.

## Backend boundary

`dcvcrt::CodecBackend` owns the complete DCVC-RT operation order behind an
SDK-neutral C++ interface. A backend receives a source frame or compressed payload
and returns the payload/frame plus committed sequence state. The current TensorRT
implementation is an I-frame pipeline:

1. RGB24 to YCbCr conversion.
2. `i_analysis`.
3. `i_hyper_analysis`.
4. `i_hyper_synthesis`.
5. Four spatial-prior passes with `i_spatial_prior_1`, `i_spatial_prior_2`, and
   `i_spatial_prior_3`.
6. `i_synthesis`.
7. Native rANS packing into the `NVI1` payload wrapper.
8. YCbCr to RGB24 reconstruction.

The backend owns deserialized engines, execution contexts, one nonblocking CUDA
stream, entropy tables, and the native rANS wrapper. The current implementation
allocates device buffers per engine call and stages tensors through pageable host
memory between TensorRT calls for correctness and simplicity. No exception or TensorRT/CUDA type crosses the
public boundary.

## Memory

`MemoryResource` expresses host, pinned-host, or device memory without exposing
CUDA types. `MemoryPool` leases best-fit reusable blocks with RAII and a hard
capacity. A lease keeps the pool state and resource alive, which makes destruction
order safe.

The initial core provides the host resource. Pinned and device resources belong in
the CUDA/TensorRT module and use the same pool. Frame currently owns host bytes;
future device frame storage can be added as an optional resource-backed view
without changing the meaning of the existing host accessor.
The current YUV-to-RGB conversion used by the CLI lives in the application
layer, not in the runtime.

## FFmpeg compatibility

The long-term FFmpeg boundary should be a small C ABI layered over `nvcr::Runtime`.
The current design prepares for that by using owned/movable buffers, integer
timestamps, explicit reset/flush, no exceptions across subsystem boundaries, and
no process-global state. FFmpeg integration itself is out of scope until v0.1
codec and bitstream conformance are stable.

## See also

- [NVCR packet envelope](bitstream.md)
- [DCVC-RT integration contract](dcvcrt-integration.md)
- [Performance](performance.md)
