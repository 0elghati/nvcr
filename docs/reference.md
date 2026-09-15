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

Encoder and decoder sessions expose send/receive, flush, and reset. The Runtime
facade also exposes aggregate state and statistics. Draining means receiving
until `end_of_stream` after `flush()`; there is no separate `drain()` method.
The generic contract permits delayed output, grouped access units, multiple
decoded frames per access unit, and frame reordering even though the current
DCVC-RT runtime emits one output per input.

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

After the final input, flush the relevant direction and keep receiving until
`ErrorCode::end_of_stream`. Use `flush_encoder()`, `reset_encoder()`,
`flush_decoder()`, and `reset_decoder()` when the directions must be controlled
independently. The facade's `flush()` and `reset()` remain shared compatibility
operations. `try_again` means output is not ready; it is not a backend failure.

Convenience `encode(const Frame&)` and `decode(const Packet&)` methods remain
available to callers that require exactly one immediate output. The CLI uses
the session lifecycle above so it can drive delayed and multi-output codecs.
Calls are serialized per runtime because codec state is mutable.

## Construction and registration

Applications register their built-in codec and provider entries, set
`RuntimeConfiguration::codec.id` and `provider.id`, then call
`Runtime::create(configuration)`. The runtime resolves the codec adapter and
provider-session factories through `RuntimeServices` and gives the selected
provider session to the adapter. The adapter returns complete encoder and
decoder sessions. Generic runtime code does not name DCVC-RT or TensorRT.

The registry is static in the current release. Test codec/provider entries are
linked only for contract coverage. They do not establish additional products
or CPU neural inference.

## Codec/provider ownership

`codec::ICodecAdapter` owns codec semantics through the sessions it creates:
options, GOP/frame types, reference state, buffering, output cardinality,
entropy meaning, and codec-private payloads.

`provider::experimental::IProviderSession` owns executable stages, buffers,
synchronization, execution, and provider failures. CUDA, TensorRT, and DCVC-RT
implementation types do not cross the public provider headers.

The TensorRT registry entry creates the real provider session used by
production. The selected DCVC-RT adapter composes that session with its codec
orchestration behind the current backend facade. The older component-level
`IExecutionProvider` API remains available to artifact clients and fixtures,
but TensorRT no longer registers an unused implementation of it.

## Configuration and errors

`RuntimeConfiguration` separates runtime, codec, provider, artifact-selection,
and stream-policy scopes. Together they cover public model/bitstream IDs,
provider and device, engine selection, QP/GOP, packet bounds, memory policy,
TensorRT mode, and diagnostics. The complete configuration is validated before
provider-session and codec-session initialization.

Structured errors cover invalid state, malformed stream, missing artifact or
provider, incompatible target/version/precision, digest mismatch, distribution
restriction, and backend failure. Preserve category, subsystem, and message
when mapping errors into an application.

## Stability boundary

NVCR v1.x is released, but cross-minor C++ API/ABI stability is not guaranteed.
A stable C ABI, FFmpeg integration, hardware-frame ownership contract, native
Windows interface, and standard container mapping remain planned work. See
[Architecture](architecture.md) and [Scope and support](scope-and-support.md).
