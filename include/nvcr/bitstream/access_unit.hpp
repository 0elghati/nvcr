#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/common/versions.hpp"
#include "nvcr/runtime/packet.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nvcr {

enum class AccessUnitPixelFormat : std::uint8_t {
    yuv420p8 = 1,
};

enum class AccessUnitSectionType : std::uint16_t {
    codec_payload = 1,
    codec_configuration = 2,
    dependency_map = 3,
    quality_rate_side_data = 4,
    color_metadata = 5,
    encoder_statistics = 6,
    private_extension = 0x8000,
};

struct AccessUnitSection final {
    std::uint16_t type{};
    std::uint16_t version{1};
    std::uint32_t flags{};
    std::vector<std::byte> payload;
};

struct AccessUnit final {
    AccessUnit() = default;
    AccessUnit(
        std::string model_identity,
        std::uint32_t visible_width,
        std::uint32_t visible_height,
        std::uint32_t effective_qp,
        FrameType type,
        bool resets_state,
        std::vector<std::byte> codec_payload)
        : model_id(std::move(model_identity)),
          width(visible_width),
          height(visible_height),
          qp(effective_qp),
          frame_type(type),
          reset_state(resets_state),
          payload(std::move(codec_payload)) {}

    std::string model_id;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t qp{};
    FrameType frame_type{FrameType::intra};
    bool reset_state{};
    std::vector<std::byte> payload;

    std::string codec_id;
    std::string codec_profile_id;
    StreamFormatVersion stream_format_version{1, 0};
    PayloadSyntaxVersion payload_syntax_version{1};
    ModelSetVersion model_set_version{1, 0};
    AccessUnitPixelFormat pixel_format{AccessUnitPixelFormat::yuv420p8};
    std::uint8_t bit_depth{8};
    bool random_access{};
    bool discontinuity{};
    std::uint64_t decode_order_index{};
    std::uint64_t presentation_order_index{};
    std::vector<std::uint64_t> dependencies;
    std::vector<AccessUnitSection> sections;
};

class AccessUnitIO final {
public:
    static constexpr std::uint16_t format_version = 1;
    static constexpr std::uint16_t sectioned_format_version = 2;
    static constexpr std::size_t maximum_model_id_bytes = 128;
    static constexpr std::size_t maximum_codec_id_bytes = 128;
    static constexpr std::size_t maximum_codec_profile_id_bytes = 128;
    static constexpr std::uint16_t maximum_dependencies = 32;
    static constexpr std::uint16_t maximum_sections = 64;
    static constexpr std::uint32_t required_section_flag = 1U;
    static constexpr std::uint32_t maximum_dimension = 16'384;
    static constexpr std::uint64_t maximum_pixels = 256ULL * 1024ULL * 1024ULL;

    [[nodiscard]] static bool has_magic(std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] static Result<std::vector<std::byte>> serialize(
        const AccessUnit& access_unit,
        std::size_t maximum_access_unit_bytes = 64U * 1024U * 1024U);
    [[nodiscard]] static Result<std::vector<std::byte>> serialize_sectioned(
        const AccessUnit& access_unit,
        std::size_t maximum_access_unit_bytes = 64U * 1024U * 1024U);
    [[nodiscard]] static Result<AccessUnit> deserialize(
        std::span<const std::byte> bytes,
        std::size_t maximum_access_unit_bytes = 64U * 1024U * 1024U);
};

}  // namespace nvcr
