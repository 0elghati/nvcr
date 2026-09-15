# Phase C4 grouped-access-unit timing evidence — 2026-09-14

## Scope

This evidence closes the grouped-access-unit timing prototype selected by
[the C0 audit](../docs/cross-codec-requirements.md). It tests the existing
registered session and development `PacketIO` boundary. It does not define a
chunk API, change NVAU, integrate another codec or provider, or claim an FFmpeg
or standard-container mapping.

- Branch: `codex/phase-c1-codec-sessions`
- Baseline revision: `fe6cb8e9db0d90cf2ac33a6f1030a2600bd413e2`
- DCVC source check: `microsoft/DCVC` `main` at
  `cbdae87a5445114cdc7f48816da63ea80bdeac40`
- MLVC source check: `microsoft/mlvc` `main` at
  `b881d799af62b8640c14c9ee6df3fb7fea679156`

## Contract result

The registered test codec groups eight input frames into one access unit and
decodes one access unit into eight ordered frames. Its full-group inputs use
these deliberately non-uniform timestamps in microseconds:

```text
1000, 2100, 3700, 5000, 8200, 8900, 12500, 17300
```

The encoder records the values under the codec-scoped
`test_codec.frame_timestamps_us` packet-metadata key. `PacketIO::serialize`
writes that bounded metadata with the packet, and `PacketIO::deserialize`
restores it before decode. The decoder validates the count and primary packet
timestamp, rejects the malformed value `1000,bad`, and reconstructs all eight
frames with their original bytes, order, and timestamps.

The same path preserves timestamp `20000` for a one-frame final group emitted
only by encoder flush. Encoder-only flush leaves the decoder open, and each
direction reaches end-of-stream only after its own flush and drain.

## Validation

The existing CUDA 12.8/TensorRT 10.9 Release tree was rebuilt and tested:

```text
rtk cmake --build build-c1-release-cuda128 -j2
rtk ctest --test-dir build-c1-release-cuda128 --output-on-failure
rtk ./build-c1-release-cuda128/tests/nvcr_contract_tests
```

Result: build passed, 22/22 tests passed, and the direct contract binary
reported `NVCR codec/provider contract tests passed`. The suite includes the
1080p TensorRT engine contract, registered I/P round trip, and pinned
Python/native I-frame golden.

A fresh CPU Release tree was configured under
`/tmp/nvcr-phase-c-verify.ykLgUQ`, built, tested, and installed to
`/tmp/nvcr-phase-c-install.rsEYw0` with TensorRT disabled and dependency fetches
disabled. Result: 18/18 tests passed and install completed.

A fresh Clang 18 ASan/UBSan and libFuzzer tree was configured under
`/tmp/nvcr-phase-c-sanitized.VLywP4`. Result: 15/15 tests passed, and
`nvcr_access_unit_fuzz -runs=1000 -timeout=5 -max_total_time=10` completed all
1,000 iterations without a sanitizer failure.

## Decision

The existing session verbs, codec-owned output queues, packet metadata, and
`PacketIO` framing preserve per-frame timing for the known grouped-output case.
C4 therefore does not justify a chunk API or NVAU change.

Packet metadata is an NVCR development/application boundary. A future FFmpeg or
standard-container integration must still define how one access unit maps to
multiple presentation timestamps and durations in that container.
