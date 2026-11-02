# DCVC-UF rANS provenance

The implementation in `src/rans.cpp` and `src/rans.hpp` is derived from
`src/cpp/py_rans/rans.cpp` and `src/cpp/py_rans/rans.h` in
[microsoft/DCVC](https://github.com/microsoft/DCVC) at commit
`cbdae87a5445114cdc7f48816da63ea80bdeac40`.

NVCR's modification places the implementation in the codec-local
`nvcr::dcvcuf::upstream_rans` namespace and changes the header suffix to
`.hpp`. The entropy state machine, symbol mapping, and worker behavior remain
upstream DCVC-UF code.

This target is separate from `third_party/dcvc_rans`, which serves DCVC-RT.
The two codecs do not share entropy ABI or stream semantics.
