# DCVC-UF rANS provenance

The implementation in `src/rans.cpp` and `src/rans.hpp` is derived from
`src/cpp/py_rans/rans.cpp` and `src/cpp/py_rans/rans.h` in
[microsoft/DCVC](https://github.com/microsoft/DCVC) at commit
`cbdae87a5445114cdc7f48816da63ea80bdeac40`.

NVCR's modification places the implementation in the codec-local
`nvcr::dcvcuf::upstream_rans` namespace and changes the header suffix to
`.hpp`. The entropy state machine, symbol mapping, and worker behavior remain
upstream DCVC-UF code.

NVCR also hardens the decoder (C13 cross-decode). It does not change the coding
rules or any valid stream:

- Reads stop at the end of the stream (`ByteReader`) and the decoder reports an
  overrun.
- The initial state must lie in `[2^23, 2^31)`, the range an encoder flushes.
- A bypass group count above 16 (more than a 32-bit raw value needs) and a
  decoded value outside `int8` mark the stream corrupt; the arithmetic that used
  to overflow `int32` on such values is done in `uint32`/`int64`.
- `at_initial_state()` and `consumed_bytes()` expose the end-of-stream state so
  NVCR can check that a frame's stream ended exactly where the encoder started
  (`UfRansCodec::finish_decode`).

This target is separate from `third_party/dcvc_rans`, which serves DCVC-RT.
The two codecs do not share entropy ABI or stream semantics.
