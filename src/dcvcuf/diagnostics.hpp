#pragma once

// Test-only observation of DCVC-UF encoder and decoder state (C13 G1).
//
// Not installed and not part of the public API. With no observer installed the
// execution sessions do no extra work. With one installed they report, per
// coding step, the tensors that define the decoder-side reconstruction:
//
//   intra step      encoder: reconstructed frame; decoder: decoded frame
//   HT-S group      encoder: memory and context used for the group (before),
//                   y_hat and feature (after), and the eight frames the
//                   reconstruction head produces from a copy of the feature
//                   decoder: memory, context, y_hat, feature, decoded frames
//
// The encoder never runs the reconstruction head for HT-S groups; the
// observed encoder frames come from an extra, side-effect-free head run on a
// copy of the encoder's feature, so the stream is unchanged.
//
// Records follow the computation where the path allows it, so a step whose
// decode fails still reports the tensors computed before the failure. The
// fused path decodes a group in one upstream call and reports its tensors only
// after the call returns; a failed fused group therefore records no tensors.
//
// Tensor bytes are the provider's device bytes in its own layout. Each path
// compares only with itself, so layouts match between encoder and decoder.

#include "nvcr/runtime/frame.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace nvcr::dcvcuf::diagnostics {

enum class Direction : std::uint8_t { encoder, decoder };
enum class StepKind : std::uint8_t { intra, predicted };

class Observer {
public:
    virtual ~Observer() = default;
    // Starts a coding step; the records that follow belong to it.
    virtual void step(Direction direction, StepKind kind, bool reset_feature_memory) = 0;
    virtual void tensor(Direction direction, std::string_view name,
                        std::span<const std::int64_t> shape,
                        std::span<const std::byte> bytes) = 0;
    virtual void frames(Direction direction, std::span<const Frame> frames) = 0;
};

// Process-wide; set by a test before creating sessions and cleared after.
void set_observer(Observer* observer) noexcept;
[[nodiscard]] Observer* observer() noexcept;

}  // namespace nvcr::dcvcuf::diagnostics
