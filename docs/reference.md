# Public API Reference

This page summarizes the exported types in the public `nvcr` headers. The API is
intentionally small and value-oriented.

## Core types

### `nvcr::Runtime`

The main library entry point. Create it with `Runtime::create`, then call:

- `encode(const Frame&)` to turn a frame into a packet.
- `decode(const Packet&)` to reconstruct a frame from a packet.
- `flush()` to drain outstanding codec work.
- `reset()` to clear sequence state.
- `state()` to inspect the runtime state.
- `statistics()` to read accumulated counters.

`Runtime` requires a `RuntimeConfiguration` and a codec backend implementation
wrapped in `nvcr::dcvcrt::Components`.

### `nvcr::Frame`

Owns pixel data plus dimensions, format, and timestamp. Use:

- `Frame::create(...)` to allocate a frame of the right size.
- `Frame::copy_from(...)` to copy external bytes into owned storage.
- `data()` to access the bytes.

The supported formats are `yuv420p8`, `rgb24`, `rgba32`, and `gray8`.

### `nvcr::Packet`

Owns a compressed payload, timestamp, frame type, and optional metadata. Packets
serialize through `PacketIO`.

### `nvcr::PacketIO`

Serializes and deserializes the versioned NVCR packet envelope. Use this for
wire storage or sequence-file streaming.

## Configuration and backend

### `nvcr::RuntimeConfiguration`

Controls engine paths, device selection, GOP size, packet limits, memory pool
size, and logging level.

### `nvcr::dcvcrt::Components`

Holds the codec backend. In v0.1 this should point to the DCVC-RT backend
implementation created by `make_tensorrt_backend()`.

### `nvcr::dcvcrt::CodecBackend`

The codec-level interface. It is responsible for initialization, encode, decode,
flush, and reset. Exceptions must not cross this boundary.

## Error handling

Public APIs return `nvcr::Result<T>` values instead of throwing across module
boundaries. On failure, the result carries a stable error code, subsystem, and
message.

## Suggested usage

1. Create or load a `RuntimeConfiguration`.
2. Create a codec backend and place it in `nvcr::dcvcrt::Components`.
3. Call `nvcr::Runtime::create(...)`.
4. Use `encode` and `decode` with owned `Frame` and `Packet` values.
5. Call `flush()` before shutdown when you need deterministic completion.

For the full ownership model, see [Architecture](architecture.md). For wire
format details, see [Bitstream](bitstream.md).
