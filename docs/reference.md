# C++ API

The public C++ surface is a runtime API for neural video codecs, not a
DCVC-RT-specific wrapper. It is transitional and not ABI-frozen. Pin the NVCR
revision, codec/provider IDs, stream versions, and artifact manifest when
integrating.

## Values and ownership

- `Frame` owns pixels, dimensions, format, and timestamp.
- `Packet` owns one application payload, timestamp, frame type, and metadata.
- `AccessUnit` is a bounded codec unit; `AccessUnitIO` reads NVAU v1/v2 and
  writes the selected supported form.
- `Result<T>` returns a value or a structured `Error`.

The current CLI boundary is planar 8-bit YUV420. Other format enum values are
not independent product support.

## Session lifecycle

Encoder and decoder sessions expose send/receive, flush, reset, state, and
statistics. Draining means receiving until `end_of_stream` after `flush()`;
there is no separate `drain()` method. The generic contract permits delayed output and frame
reordering even though the current concrete DCVC-RT runtime emits one output
per input.

```cpp
#include <nvcr/nvcr.hpp>
#include <iostream>

int submit(nvcr::Runtime& runtime, const nvcr::Frame& frame) {
    if (auto sent = runtime.send_frame(frame); !sent) {
        std::cerr << sent.error().describe() << '\n';
        return 1;
    }

    for (;;) {
        auto packet = runtime.receive_access_unit();
        if (packet) {
            // Forward packet.value() to the application/container mapping.
            continue;
        }
        if (packet.error().code() == nvcr::ErrorCode::try_again) {
            return 0;
        }
        std::cerr << packet.error().describe() << '\n';
        return 1;
    }
}
```

After the final input, call `flush()` and keep receiving until
`ErrorCode::end_of_stream`. `try_again` means output is not ready; it is
not a backend failure. Call `reset()` before reusing a session for a new
sequence so reference state and frame indexes are cleared.

Convenience `encode(const Frame&)` and `decode(const Packet&)` methods
remain available for the current immediate-output CLI path. Calls are
serialized per runtime because codec state is mutable.

## Construction and registration

Applications select registered codec `dcvc-rt` and provider `tensorrt`.
The selected adapter obtains provider-owned components through
`RuntimeServices`; that construction is separate from the lifecycle loop.

The registry is static in the current release. Test codec/provider entries are
linked only for contract coverage. They do not establish additional products
or CPU neural inference.

## Codec/provider ownership

`codec::ICodecAdapter` owns codec semantics: options, GOP/frame types,
reference state, entropy meaning, and codec-private payloads.

`provider::IExecutionProvider` owns executable artifacts, tensor bindings,
memory, synchronization, execution, and provider failures. CUDA, TensorRT, and
DCVC-RT implementation types do not cross the public provider headers.

The TensorRT provider currently constructs one provider-owned monolithic
DCVC-RT backend. Independent model-stage loading through
`IExecutionProvider::load` returns `not_implemented`; the interface is
transitional, not a completed production path.

## Configuration and errors

Configuration covers public model/bitstream IDs, provider and device, engine
selection, QP/GOP, packet bounds, memory policy, TensorRT mode, and diagnostics.
It is validated before backend initialization.

Structured errors cover invalid state, malformed stream, missing artifact or
provider, incompatible target/version/precision, digest mismatch, distribution
restriction, and backend failure. Preserve category, subsystem, and message
when mapping errors into an application.

## Stability boundary

Headers and behavior may change before v1. A stable C ABI, FFmpeg integration,
hardware-frame ownership contract, native Windows interface, and standard
container mapping are planned work, not current API guarantees. See
[Architecture](architecture.md) and [Scope and support](scope-and-support.md).
