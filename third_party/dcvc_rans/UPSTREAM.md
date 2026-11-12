# DCVC-RT native rANS core

Source repository: `microsoft/DCVC`.

Pinned reference commit: `dae827ffcc812566adbeaf4554f0fe2d9b4b9e0c`.

Imported files:

- `src/cpp/py_rans/rans.cpp`
- `src/cpp/py_rans/rans.h`
- `src/cpp/py_rans/rans_byte.h`

The imported files were taken from the pinned upstream commit. NVCR carries one audited local
performance change in `rans.cpp`: decoder symbol lookup uses `std::upper_bound`
instead of a linear CDF scan. The coding rules and bitstream are unchanged, and
the wide-CDF and golden-vector conformance tests cover the modified path.

NVCR also carries a memory-safety hardening (issue #216) in `rans.cpp` and
`rans.h`. It does not change the coding rules or any valid stream; the
golden vectors and the pinned Python/native golden pass unchanged:

- The decoder never reads past the stream. Renormalization and bypass reads
  stop at the end of the buffer and mark the decoder `corrupt()`.
- The initial state must lie in `[2^23, 2^31)`, the bypass group count is
  capped at the 16 groups a 32-bit raw value can need, and a decoded value
  outside `int8` marks the decoder corrupt.
- `at_initial_state()` and `consumed_bytes()` expose the end-of-stream state so
  NVCR can verify that a frame's stream ended exactly where the encoder started.
- The encoder sizes its output buffer for the worst case (8 bytes per symbol
  plus slack) instead of one byte per symbol, which high-cost symbols overflowed.

`rans.cpp` and `rans.h` carry the Apache-2.0 notice inherited from CompressAI;
`rans_byte.h` carries its original CC0 dedication. The parent DCVC repository license and notice are kept here.
