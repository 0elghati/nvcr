# DCVC-RT native rANS core

Source: `microsoft/DCVC`, working reference checkout used during import.

Pinned reference commit: `dae827ffcc812566adbeaf4554f0fe2d9b4b9e0c`.

Imported files:

- `src/cpp/py_rans/rans.cpp`
- `src/cpp/py_rans/rans.h`
- `src/cpp/py_rans/rans_byte.h`

The imported files were taken from that checkout. NVCR carries one audited local
performance change in `rans.cpp`: decoder symbol lookup uses `std::upper_bound`
instead of a linear CDF scan. The coding rules and bitstream are unchanged, and
the wide-CDF and golden-vector conformance tests cover the modified path.
`rans.cpp` and `rans.h`
carry the Apache-2.0 notice inherited from CompressAI; `rans_byte.h` carries its
original CC0 dedication. The parent DCVC repository license and notice are kept here.
