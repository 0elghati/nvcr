# MLVC native rANS coder (`msrtc_rans`)

Source repository: `microsoft/MLVC`.

Pinned reference commit: `1425880bf679bbafdaffb7be3fb61f80448fb0b6`. The imported
files are byte-identical at `7a85fee68545c47688aaece9f38e018c2cf0798f`.

Imported files, unmodified:

- `packages/msrtc_rans/include/msrtc_rans/EntropyCoder.h`
- `packages/msrtc_rans/include/msrtc_rans/span.h`
- `packages/msrtc_rans/private/include/msrtc_rans/rans.h`
- `packages/msrtc_rans/src/EntropyCoder.cpp`

Every file carries Microsoft's MIT header. The coder is adapted from ryg_rans
(CC0); the MLVC repository `LICENSE` and `NOTICE` are kept here unchanged.

MLVC's codec uses `RansVariant::RansByte` with 16 symbol bits and 2 bypass bits
(`video/conversion/_coder.py`). The C++ defaults differ (4 bypass bits), so NVCR
passes all three explicitly.
