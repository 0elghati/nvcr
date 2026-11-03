#pragma once

#include "nvcr/configuration/configuration.hpp"
#include "nvcr/provider/experimental/session.hpp"
#include "nvcr/runtime/frame.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace nvcr::dcvcuf {

struct Encoded final {
    std::vector<std::byte> bitstream;
    std::uint8_t entropy_parallel{1};
};

// What the DCVC-UF adapter needs from an execution path: one intra frame or one
// eight-frame HT-S group per call, with the reference state kept inside.
class ExecutionSession : public provider::experimental::IProviderSession {
public:
    [[nodiscard]] virtual Result<Encoded> encode_intra(const Frame& frame, std::uint32_t qp) = 0;
    [[nodiscard]] virtual Result<Frame> decode_intra(
        std::span<const std::byte> bitstream, std::uint8_t entropy_parallel,
        std::uint32_t qp, std::uint32_t width, std::uint32_t height,
        Timestamp timestamp) = 0;
    [[nodiscard]] virtual Result<Encoded> encode_predicted(
        std::span<const Frame> frames, std::uint32_t qp, bool reset_feature_memory) = 0;
    [[nodiscard]] virtual Result<std::vector<Frame>> decode_predicted(
        std::span<const std::byte> bitstream, std::uint8_t entropy_parallel,
        std::uint32_t qp, std::uint32_t width, std::uint32_t height,
        std::span<const Timestamp> timestamps, bool reset_feature_memory) = 0;
    [[nodiscard]] virtual Result<void> reset_encoder_state() = 0;
    [[nodiscard]] virtual Result<void> reset_decoder_state() = 0;
};

}  // namespace nvcr::dcvcuf
