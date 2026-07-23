#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/runtime/packet.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nvcr {

struct AccessUnit final {
    std::string model_id;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t qp{};
    FrameType frame_type{FrameType::intra};
    bool reset_state{};
    std::vector<std::byte> payload;
};

class AccessUnitIO final {
public:
    static constexpr std::uint16_t format_version = 1;
    static constexpr std::size_t maximum_model_id_bytes = 128;
    static constexpr std::uint32_t maximum_dimension = 16'384;
    static constexpr std::uint64_t maximum_pixels = 256ULL * 1024ULL * 1024ULL;

    [[nodiscard]] static bool has_magic(std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] static Result<std::vector<std::byte>> serialize(
        const AccessUnit& access_unit,
        std::size_t maximum_access_unit_bytes = 64U * 1024U * 1024U);
    [[nodiscard]] static Result<AccessUnit> deserialize(
        std::span<const std::byte> bytes,
        std::size_t maximum_access_unit_bytes = 64U * 1024U * 1024U);
};

}  // namespace nvcr
