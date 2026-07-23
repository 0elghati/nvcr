# Public C++ API reference

The current API is a development C++ boundary, not a frozen ABI. v1 stabilization
still requires explicit plane/stride and host/device view contracts.

## Runtime and values

### `nvcr::Runtime`

Create with `Runtime::create(configuration, components)`, then use:

- `encode(const Frame&)` to produce a `Packet` containing one `NVAU`;
- `decode(const Packet&)` to reconstruct a frame;
- `reset()` to discard encoder/decoder state;
- `flush()` to finish backend work and reset state;
- `state()` and `statistics()` for lifecycle/counter inspection.

Calls return `Result<T>` and are serialized per runtime session.

### `nvcr::Frame`

An owned pixel buffer with width, height, format, and timestamp. `Frame::create`
checks storage arithmetic and format constraints; `copy_from` imports external
bytes. The CLI-supported v1 boundary is YUV420P8. Other enum formats are utility or
internal paths and are not independent v1 codec support claims.

### `nvcr::Packet`

An owned application packet with codec payload, timestamp, frame type, and bounded
metadata. `PacketIO` implements the development `NVCR` envelope.

### `nvcr::AccessUnit` / `nvcr::AccessUnitIO`

The versioned codec unit carries model ID, dimensions, effective QP, frame type,
reset flag, and payload. Serialization/deserialization are bounded and validate
all fields. See [Bitstream](bitstream.md).

## Configuration

`RuntimeConfiguration` includes:

- `model_id` and `device_id`;
- engine directory;
- GOP size and base intra QP;
- maximum packet bytes, host memory pool, and intended device-arena capacity;
- per-session `automatic`, `low_memory`, or `performance` TensorRT mode;
- legacy raw-access-unit policy;
- opt-in encoder reconstruction download for conformance tests;
- logging/profiling settings.

Configuration is validated before backend initialization. The encoder
reconstruction option is a test hook and should remain false in normal execution.

## DCVC-RT backend

`dcvcrt::Components` holds the explicit backend created by
`make_tensorrt_backend()`. `CodecBackend` owns initialization, I/P encode/decode,
reset, and flush. No CUDA/TensorRT type or exception crosses this interface.

This is a real codec boundary, not a promise of arbitrary backend plugins. Another
codec would require its own evidence and a justified abstraction change.

## Errors

Public operations return `nvcr::Result<T>`. `Error` supplies a stable category,
subsystem, and human-readable message. Expected invalid inputs, state errors,
malformed access units, unavailable dependencies, resource exhaustion, and backend
failures do not use exceptions as cross-module control flow.

## ABI and integration status

Headers may change before v1. A public C ABI and FFmpeg integration are post-v1.
Applications targeting the development tree should pin the exact NVCR revision,
model profile, and engine-bundle schema.
