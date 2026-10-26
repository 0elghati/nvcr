#pragma once

// Session-oriented codec API.
//
// These interfaces support codecs with lookahead, frame reordering, delayed
// output, and flush-only output.  The current DCVC-RT adapter uses one-frame-
// per-access-unit, which is conformant; the contract does not require it.
//
// ErrorCode semantics on send/receive:
//   send success     – input accepted and consumed
//   receive success  – output available
//   try_again        – no output ready; more input may be required (receive only)
//   end_of_stream    – codec drained; no more output after flush()
//   invalid_state    – session is not in a state that accepts this call
//   malformed_stream – input access unit is corrupt or unrecognized
//   missing_artifact – required engine/model artifact is absent
//   incompatible_target – provider or target device is not compatible
//   backend_error    – provider or execution failure
//   internal_error   – unexpected implementation failure

#include "nvcr/common/error.hpp"
#include "nvcr/runtime/frame.hpp"
#include "nvcr/runtime/packet.hpp"
#include "nvcr/statistics/statistics.hpp"

#include <memory>

namespace nvcr {

class IEncoderSession {
public:
    virtual ~IEncoderSession() = default;

    // Submit one input frame.  Success means the frame was consumed; the
    // caller must not resubmit it.  Output readiness is reported by
    // receive_access_unit().  Never returns try_again in v1.
    [[nodiscard]] virtual Result<void> send_frame(const Frame& frame) = 0;

    // Retrieve the next encoded access unit.  Returns:
    //   Packet    – an access unit is available.
    //   try_again – no output is ready; more input may be required.
    //   end_of_stream – codec has been flushed and all output emitted.
    [[nodiscard]] virtual Result<Packet> receive_access_unit() = 0;

    // Signal that no more encoder input will arrive.  Drain buffered access
    // units by calling receive_access_unit() until end_of_stream.
    [[nodiscard]] virtual Result<void> flush() = 0;

    // Discard all encoder state.  The session returns to an initialized state
    // with frame_index == 0 so the next send_frame begins a new sequence.
    [[nodiscard]] virtual Result<void> reset() = 0;
};

class IDecoderSession {
public:
    virtual ~IDecoderSession() = default;

    // Submit one encoded access unit.  Success means the unit was consumed;
    // the caller must not resubmit it.  Output may remain unavailable until
    // additional units arrive; use receive_frame() to determine readiness.
    // Never returns try_again in v1.
    [[nodiscard]] virtual Result<void> send_access_unit(const Packet& packet) = 0;

    // Retrieve the next reconstructed frame.  Returns:
    //   Frame     – a decoded frame is available.
    //   try_again – no output is ready; more access units may be required.
    //   end_of_stream – codec has been flushed and all frames emitted.
    [[nodiscard]] virtual Result<Frame> receive_frame() = 0;

    // Signal that no more decoder input will arrive.  Drain buffered frames by
    // calling receive_frame() until end_of_stream.
    [[nodiscard]] virtual Result<void> flush() = 0;

    // Discard all decoder state.  The session accepts the next intra access
    // unit as the start of a new sequence.
    [[nodiscard]] virtual Result<void> reset() = 0;
};

}  // namespace nvcr

namespace nvcr::codec {

// Complete codec-owned session pair returned by a registered adapter.
// Statistics is shared with the generic Runtime facade when supplied.
struct Sessions final {
    std::unique_ptr<IEncoderSession> encoder;
    std::unique_ptr<IDecoderSession> decoder;
    std::shared_ptr<Statistics> statistics;
};

}  // namespace nvcr::codec
