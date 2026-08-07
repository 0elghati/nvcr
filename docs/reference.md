# C++ API

The public C++ API is a development interface, not a frozen ABI. Pin the NVCR revision and manifest schema when integrating it.

## Runtime

`nvcr::Runtime` owns one encoder state and one decoder state. The compatibility surface provides:

- `Runtime::create(configuration, components)`;
- `encode(const Frame&)` and `decode(const Packet&)`;
- `reset()` and `flush()`;
- `state()` and `statistics()`.

Operations return `nvcr::Result<T>`. Calls are serialized per runtime session because codec state is mutable.

The codec API also defines session-oriented send/receive contracts for adapters that may delay output. The current DCVC-RT path preserves the one-frame/one-access-unit behavior underneath that boundary.

## Values and format

- `Frame` owns a pixel buffer, dimensions, format, and timestamp.
- `Packet` owns one application-level payload and timestamp.
- `AccessUnit` is the bounded codec unit. `AccessUnitIO` reads `NVAU` v1 and v2 and writes the supported forms.

The CLI boundary is planar 8-bit YUV420. Other format enum values are not independent product support.

## Configuration

Runtime configuration covers model and public bitstream IDs, device and engine selection, QP, GOP size, packet limits, memory policy, TensorRT execution mode, and diagnostic options. Configuration is checked before backend initialization.

## Codec and provider boundaries

`codec::ICodecAdapter` owns codec semantics. `provider::IExecutionProvider` owns artifact loading and execution. `runtime::RuntimeServices` is the bridge between them. TensorRT and CUDA types do not cross the public provider API.

The registry is static in v1. `dcvcrt` and `tensorrt` are the only production entries. Test codec/provider entries exist for contract coverage.

The API boundary is ahead of the production wiring: test providers execute
through `RuntimeServices`, but DCVC-RT currently constructs its TensorRT backend
directly and the registered TensorRT provider load path is a stub. Treat the
provider-mediated production flow as refactor closeout work, not a completed
support claim.

## Errors

`Error` provides a category, subsystem, and message. Structured categories include invalid state, malformed bitstream, missing artifact, missing provider, incompatible target/version/precision, digest mismatch, license restriction, and backend failure.

## ABI status

Headers and behavior may change before v1. A stable C ABI, FFmpeg integration, hardware-frame ownership contract, and standard container mapping are post-v1 work.
