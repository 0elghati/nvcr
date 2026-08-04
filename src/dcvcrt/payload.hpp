#pragma once

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvcr::dcvcrt {

struct IntraPayload final {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t qp;
    bool two_coders;
    std::span<const std::byte> rans;
};

struct PredictedPayload final {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t qp;
    bool two_coders;
    bool use_frame_reference;
    std::span<const std::byte> rans;
};

[[nodiscard]] std::vector<std::byte> make_intra_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, std::span<const std::byte> rans);

[[nodiscard]] Result<IntraPayload> parse_intra_payload(
    std::span<const std::byte> payload);

[[nodiscard]] std::vector<std::byte> make_predicted_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, bool use_frame_reference, std::span<const std::byte> rans);

[[nodiscard]] Result<PredictedPayload> parse_predicted_payload(
    std::span<const std::byte> payload);

}  // namespace nvcr::dcvcrt
