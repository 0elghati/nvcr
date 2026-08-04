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

The versioned codec unit carries a public bitstream model ID, dimensions,
effective QP, frame type, reset flag, and payload. Serialization/deserialization
are bounded and validate all fields. `AccessUnitIO::deserialize` accepts v1 and
the sectioned v2 envelope.
`AccessUnitIO::serialize` writes v1 for compatibility;
`AccessUnitIO::serialize_sectioned` writes the explicit v2 envelope with codec
identity, ordering, dependencies, and typed sections. See [Bitstream](bitstream.md).

## Configuration

`RuntimeConfiguration` includes:

- `model_id`, for backend artifact/profile selection;
- `bitstream_model_id`, for the public access-unit identity written to streams;
- `device_id`;
- engine directory;
- GOP size and base intra QP;
- maximum packet bytes, host memory pool, and intended device-arena capacity;
- per-session `automatic`, `low_memory`, or `performance` TensorRT mode;
- legacy raw-access-unit policy;
- opt-in encoder reconstruction download for conformance tests;
- logging/profiling settings.

Configuration is validated before backend initialization. The encoder
reconstruction option is a test hook and should remain false in normal execution.

## Codec backend boundary

`codec::Components` holds the explicit backend injected into `Runtime::create`.
`codec::CodecBackend` owns initialization, frame encode/decode, reset, and flush.
No CUDA/TensorRT type or exception crosses this interface.

The current release provides one implementation, created by
`dcvcrt::make_tensorrt_backend()`. Additional neural codec backends would require
their own model profiles, payload contracts, conformance tests, and target
evidence before they could be described as supported.

## Errors

Public operations return `nvcr::Result<T>`. `Error` supplies a stable category,
subsystem, and human-readable message. Expected invalid inputs, state errors,
malformed access units, unavailable dependencies, resource exhaustion, and backend
failures do not use exceptions as cross-module control flow.

## ABI and integration status

Headers may change before v1. A public C ABI and FFmpeg integration are post-v1.
Applications targeting the development tree should pin the exact NVCR revision,
public bitstream model ID, backend model profile, and engine-bundle schema.
