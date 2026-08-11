# C++ API

The public C++ API is a runtime API for neural video codecs, not a DCVC-RT
wrapper. It is a development interface, not a frozen ABI; pin the NVCR revision,
codec/provider IDs, stream format, and artifact manifest when integrating it.

## Runtime

`nvcr::Runtime` owns one encoder state and one decoder state. The session surface
provides `send_frame`, `receive_access_unit`, `send_access_unit`,
`receive_frame`, `flush`, and `reset`. Convenience `encode` and `decode`
methods remain available for the current CLI path. Operations return
`nvcr::Result<T>` and calls are serialized per runtime session because codec
state is mutable.

```cpp
#include <nvcr/nvcr.hpp>
#include <iostream>

int encode_frame(nvcr::Runtime& runtime, const nvcr::Frame& frame) {
    if (auto submitted = runtime.send_frame(frame); !submitted) {
        std::cerr << submitted.error().describe() << '\n';
        return 1;
    }
    for (;;) {
        auto packet = runtime.receive_access_unit();
        if (packet) {
            // Pass packet.value() to the application/container mapping.
            continue;
        }
        if (packet.error().code() == nvcr::ErrorCode::try_again) return 0;
        std::cerr << packet.error().describe() << '\n';
        return 1;
    }
}
```

The application obtains `Runtime` components from its selected registered
codec/provider through `RuntimeServices`; that construction is intentionally
outside this architecture-neutral loop.

The caller must call `flush()` and continue receiving until
`ErrorCode::end_of_stream` when using a codec that delays output. `try_again`
is a session status, not a backend failure. `reset()` starts a new sequence and
clears codec state.

The compatibility surface also provides:

- `Runtime::create(configuration, components)`;
- `encode(const Frame&)` and `decode(const Packet&)`;
- `reset()` and `flush()`;
- `state()` and `statistics()`.

The current DCVC-RT path preserves one-frame/one-access-unit behavior and does
not return `try_again`, but the generic contract permits delayed output,
lookahead, and frame reordering.

## Values and format

- `Frame` owns a pixel buffer, dimensions, format, and timestamp.
- `Packet` owns one application-level payload, timestamp, frame type, and
  metadata.
- `AccessUnit` is the bounded codec unit. `AccessUnitIO` reads `NVAU` v1 and v2
  and writes the supported forms.

The CLI boundary is planar 8-bit YUV420. Other format enum values are not independent product support.

## Configuration

Runtime configuration covers model and public bitstream IDs, provider selection, device and engine selection, QP, GOP size, packet limits, memory policy, TensorRT execution mode, and diagnostic options. Configuration is checked before backend initialization.

## Codec and provider boundaries

`codec::ICodecAdapter` owns codec semantics. `provider::IExecutionProvider`
owns artifact loading and execution. `runtime::RuntimeServices` is the bridge
between them. TensorRT and CUDA types do not cross the public provider API.

The registry is static in v1. `dcvcrt` and `tensorrt` are the only production entries. Test codec/provider entries exist for contract coverage.

DCVC-RT asks `RuntimeServices` for provider-owned codec components, and the
TensorRT provider owns construction of the current monolithic backend. The
component-level `IExecutionProvider::load` path is still not split into
independent model-stage executables.

## Errors

`Error` provides a category, subsystem, and message. Structured categories include invalid state, malformed bitstream, missing artifact, missing provider, incompatible target/version/precision, digest mismatch, license restriction, and backend failure.

## ABI status

Headers and behavior may change before v1. A stable C ABI, FFmpeg integration, hardware-frame ownership contract, and standard container mapping are post-v1 work.
