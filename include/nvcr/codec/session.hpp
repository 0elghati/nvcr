#pragma once

// IEncoderSession / IDecoderSession: session-oriented codec API (Phase 2).
//
// These interfaces support codecs with lookahead, frame reordering, delayed
// output, and flush-only output.  The current DCVC-RT adapter uses one-frame-
// per-access-unit, which is conformant; the contract does not require it.
//
// ErrorCode semantics on send/receive:
//   success          – frame accepted / output available
//   try_again        – no output yet; send more input (not an error)
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

#include <memory>

namespace nvcr {

class IEncoderSession {
public:
    virtual ~IEncoderSession() = default;

    // Submit one input frame.  Returns success or an error.  Never returns
    // try_again; encoders always consume their input immediately in v1.
    [[nodiscard]] virtual Result<void> send_frame(const Frame& frame) = 0;

    // Retrieve the next encoded access unit.  Returns:
    //   Packet    – an access unit is available.
    //   try_again – no output yet; call send_frame again.
    //   end_of_stream – codec has been flushed and all output emitted.
    [[nodiscard]] virtual Result<Packet> receive_access_unit() = 0;

    // Signal end of input.  Codecs with lookahead push buffered access units
    // into the output queue; drain by calling receive_access_unit() until
    // end_of_stream.
    [[nodiscard]] virtual Result<void> flush() = 0;

    // Discard all encoder state.  The session returns to an initialized state
    // with frame_index == 0 so the next send_frame begins a new sequence.
    [[nodiscard]] virtual Result<void> reset() = 0;
};

class IDecoderSession {
public:
    virtual ~IDecoderSession() = default;

    // Submit one encoded access unit for decoding.  Returns success or an
    // error.  Returns try_again when the decoder needs additional units before
    // it can produce output (e.g. B-frame reordering).
    [[nodiscard]] virtual Result<void> send_access_unit(const Packet& packet) = 0;

    // Retrieve the next reconstructed frame.  Returns:
    //   Frame     – a decoded frame is available.
    //   try_again – send another access unit first.
    //   end_of_stream – codec has been flushed and all frames emitted.
    [[nodiscard]] virtual Result<Frame> receive_frame() = 0;

    // Signal end of input; drain remaining frames via receive_frame().
    [[nodiscard]] virtual Result<void> flush() = 0;

    // Discard all decoder state.  The session accepts the next intra access
    // unit as the start of a new sequence.
    [[nodiscard]] virtual Result<void> reset() = 0;
};

}  // namespace nvcr
