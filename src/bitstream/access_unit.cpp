#include "nvcr/bitstream/access_unit.hpp"

#include "nvcr/bitstream/packet_io.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <new>
#include <optional>
#include <string_view>

namespace nvcr {
namespace {

constexpr std::byte magic[] = {
    std::byte{'N'}, std::byte{'V'}, std::byte{'A'}, std::byte{'U'}};
constexpr std::uint8_t reset_state_flag = 1U;
constexpr std::uint8_t supported_v1_flags = reset_state_flag;
constexpr std::uint32_t v2_reset_state_flag = 1U;
constexpr std::uint32_t v2_random_access_flag = 2U;
constexpr std::uint32_t v2_discontinuity_flag = 4U;
constexpr std::uint32_t supported_v2_flags =
    v2_reset_state_flag | v2_random_access_flag | v2_discontinuity_flag;
constexpr std::size_t fixed_header_bytes_v1 = 32U;
constexpr std::size_t fixed_header_bytes_v2 = 64U;
constexpr std::size_t section_table_entry_bytes_v2 = 16U;
constexpr std::uint16_t codec_payload_section_type =
    static_cast<std::uint16_t>(AccessUnitSectionType::codec_payload);

struct SectionHeader final {
    std::uint16_t type{};
    std::uint16_t version{};
    std::uint32_t flags{};
    std::uint64_t size{};
};

Error malformed(std::string message) {
    return Error(ErrorCode::malformed_bitstream, std::move(message), "access-unit");
}

bool valid_identifier(std::string_view value, std::size_t maximum_size) {
    if (value.empty() || value.size() > maximum_size) return false;
    return std::ranges::all_of(value, [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '.' || character == '_' ||
            character == '-';
    });
}

bool valid_model_id(std::string_view model_id) {
    return valid_identifier(model_id, AccessUnitIO::maximum_model_id_bytes);
}

Result<void> validate_common_fields(const AccessUnit& access_unit) {
    if (!valid_model_id(access_unit.model_id)) {
        return Error(
            ErrorCode::invalid_argument,
            "model_id must contain 1-128 ASCII letters, digits, '.', '_' or '-'",
            "access-unit");
    }
    if (access_unit.width == 0 || access_unit.height == 0 ||
        access_unit.width > AccessUnitIO::maximum_dimension ||
        access_unit.height > AccessUnitIO::maximum_dimension ||
        (access_unit.width & 1U) != 0U || (access_unit.height & 1U) != 0U ||
        static_cast<std::uint64_t>(access_unit.width) * access_unit.height >
            AccessUnitIO::maximum_pixels) {
        return Error(ErrorCode::invalid_argument, "access-unit dimensions are out of range",
                     "access-unit");
    }
    const auto maximum_qp = access_unit.frame_type == FrameType::intra ? 63U : 71U;
    if (access_unit.qp > maximum_qp) {
        return Error(ErrorCode::invalid_argument, "access-unit QP is out of range",
                     "access-unit");
    }
    if (access_unit.reset_state != (access_unit.frame_type == FrameType::intra)) {
        return Error(
            ErrorCode::invalid_argument,
            "intra access units must carry reset_state and predicted access units must not",
            "access-unit");
    }
    if (access_unit.payload.empty()) {
        return Error(ErrorCode::invalid_argument, "access-unit payload is empty", "access-unit");
    }
    return {};
}

Result<void> validate_v1_fields(const AccessUnit& access_unit) {
    return validate_common_fields(access_unit);
}

bool known_section_type(std::uint16_t type) noexcept {
    switch (static_cast<AccessUnitSectionType>(type)) {
    case AccessUnitSectionType::codec_payload:
    case AccessUnitSectionType::codec_configuration:
    case AccessUnitSectionType::dependency_map:
    case AccessUnitSectionType::quality_rate_side_data:
    case AccessUnitSectionType::color_metadata:
    case AccessUnitSectionType::encoder_statistics:
    case AccessUnitSectionType::private_extension:
        return true;
    }
    return false;
}

Result<std::vector<AccessUnitSection>> normalized_sections(const AccessUnit& access_unit) {
    std::vector<AccessUnitSection> sections = access_unit.sections;
    if (sections.empty()) {
        sections.push_back(AccessUnitSection{
            codec_payload_section_type,
            1U,
            AccessUnitIO::required_section_flag,
            access_unit.payload});
    }

    std::optional<std::size_t> codec_payload_index;
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto& section = sections[index];
        if (section.payload.empty()) {
            return Error(ErrorCode::invalid_argument, "access-unit section payload is empty",
                         "access-unit");
        }
        if (section.type == codec_payload_section_type) {
            if (codec_payload_index.has_value()) {
                return Error(ErrorCode::invalid_argument,
                             "access-unit has multiple codec payload sections", "access-unit");
            }
            codec_payload_index = index;
        }
    }
    if (!codec_payload_index.has_value()) {
        return Error(ErrorCode::invalid_argument,
                     "access-unit has no codec payload section", "access-unit");
    }
    if (!std::equal(
            sections[*codec_payload_index].payload.begin(),
            sections[*codec_payload_index].payload.end(),
            access_unit.payload.begin(), access_unit.payload.end())) {
        return Error(ErrorCode::invalid_argument,
                     "access-unit payload must match the codec payload section", "access-unit");
    }
    return sections;
}

Result<std::vector<AccessUnitSection>> validate_v2_fields(const AccessUnit& access_unit) {
    auto valid = validate_common_fields(access_unit);
    if (!valid) return valid.error();
    if (!valid_identifier(access_unit.codec_id, AccessUnitIO::maximum_codec_id_bytes)) {
        return Error(ErrorCode::invalid_argument, "access-unit codec_id is invalid", "access-unit");
    }
    if (!valid_identifier(
            access_unit.codec_profile_id, AccessUnitIO::maximum_codec_profile_id_bytes)) {
        return Error(ErrorCode::invalid_argument, "access-unit codec_profile_id is invalid",
                     "access-unit");
    }
    if (access_unit.pixel_format != AccessUnitPixelFormat::yuv420p8 ||
        access_unit.bit_depth != 8U) {
        return Error(ErrorCode::invalid_argument,
                     "access-unit v2 currently supports only YUV420P8", "access-unit");
    }
    if (access_unit.dependencies.size() > AccessUnitIO::maximum_dependencies) {
        return Error(ErrorCode::invalid_argument, "access-unit has too many dependencies",
                     "access-unit");
    }
    if (access_unit.sections.size() > AccessUnitIO::maximum_sections) {
        return Error(ErrorCode::invalid_argument, "access-unit has too many sections",
                     "access-unit");
    }
    return normalized_sections(access_unit);
}

Result<AccessUnit> deserialize_v1(
    ByteReader& reader, std::span<const std::byte> bytes, std::size_t maximum_access_unit_bytes) {
    auto frame_type = reader.read_u8();
    auto flags = reader.read_u8();
    auto model_id_size = reader.read_u16();
    auto reserved = reader.read_u16();
    auto width = reader.read_u32();
    auto height = reader.read_u32();
    auto qp = reader.read_u32();
    auto payload_size = reader.read_u64();
    if (!frame_type) return frame_type.error();
    if (!flags) return flags.error();
    if (!model_id_size) return model_id_size.error();
    if (!reserved) return reserved.error();
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!payload_size) return payload_size.error();

    if (frame_type.value() > static_cast<std::uint8_t>(FrameType::predicted)) {
        return malformed("invalid access-unit frame type");
    }
    if ((flags.value() & ~supported_v1_flags) != 0U || reserved.value() != 0U) {
        return malformed("unsupported access-unit flags or reserved fields");
    }
    if (model_id_size.value() == 0 || model_id_size.value() > AccessUnitIO::maximum_model_id_bytes) {
        return malformed("access-unit model identity is out of range");
    }
    if (payload_size.value() == 0 || payload_size.value() > reader.remaining() ||
        payload_size.value() > std::numeric_limits<std::size_t>::max()) {
        return malformed("access-unit payload length is invalid");
    }
    if (bytes.size() > maximum_access_unit_bytes) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }

    auto model_id = reader.read_string(model_id_size.value());
    if (!model_id) return model_id.error();
    auto payload = reader.read_bytes(static_cast<std::size_t>(payload_size.value()));
    if (!payload) return payload.error();
    if (reader.remaining() != 0) return malformed("trailing access-unit data");

    AccessUnit output{
        std::move(model_id.value()), width.value(), height.value(), qp.value(),
        static_cast<FrameType>(frame_type.value()),
        (flags.value() & reset_state_flag) != 0U,
        std::vector<std::byte>(payload.value().begin(), payload.value().end())};
    auto valid = validate_v1_fields(output);
    if (!valid) return malformed(std::string(valid.error().message()));
    return output;
}

Result<AccessUnit> deserialize_v2(
    ByteReader& reader, std::span<const std::byte> bytes, std::size_t maximum_access_unit_bytes) {
    auto header_size = reader.read_u16();
    auto flags = reader.read_u32();
    auto frame_type = reader.read_u8();
    auto pixel_format = reader.read_u8();
    auto bit_depth = reader.read_u8();
    auto reserved8 = reader.read_u8();
    auto width = reader.read_u32();
    auto height = reader.read_u32();
    auto qp = reader.read_u32();
    auto decode_order_index = reader.read_u64();
    auto presentation_order_index = reader.read_u64();
    auto dependency_count = reader.read_u16();
    auto section_count = reader.read_u16();
    auto codec_id_size = reader.read_u16();
    auto codec_profile_id_size = reader.read_u16();
    auto model_id_size = reader.read_u16();
    auto reserved16 = reader.read_u16();
    auto total_size = reader.read_u64();
    if (!header_size) return header_size.error();
    if (!flags) return flags.error();
    if (!frame_type) return frame_type.error();
    if (!pixel_format) return pixel_format.error();
    if (!bit_depth) return bit_depth.error();
    if (!reserved8) return reserved8.error();
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!decode_order_index) return decode_order_index.error();
    if (!presentation_order_index) return presentation_order_index.error();
    if (!dependency_count) return dependency_count.error();
    if (!section_count) return section_count.error();
    if (!codec_id_size) return codec_id_size.error();
    if (!codec_profile_id_size) return codec_profile_id_size.error();
    if (!model_id_size) return model_id_size.error();
    if (!reserved16) return reserved16.error();
    if (!total_size) return total_size.error();

    if (header_size.value() != fixed_header_bytes_v2) return malformed("invalid NVAU v2 header size");
    if (total_size.value() != bytes.size() || total_size.value() > maximum_access_unit_bytes) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }
    if ((flags.value() & ~supported_v2_flags) != 0U || reserved8.value() != 0U ||
        reserved16.value() != 0U) {
        return malformed("unsupported NVAU v2 flags or reserved fields");
    }
    if (frame_type.value() > static_cast<std::uint8_t>(FrameType::predicted)) {
        return malformed("invalid access-unit frame type");
    }
    if (pixel_format.value() != static_cast<std::uint8_t>(AccessUnitPixelFormat::yuv420p8) ||
        bit_depth.value() != 8U) {
        return malformed("unsupported NVAU v2 pixel format");
    }
    if (dependency_count.value() > AccessUnitIO::maximum_dependencies ||
        section_count.value() == 0 || section_count.value() > AccessUnitIO::maximum_sections) {
        return malformed("NVAU v2 dependency or section count is out of range");
    }
    if (codec_id_size.value() == 0 ||
        codec_id_size.value() > AccessUnitIO::maximum_codec_id_bytes ||
        codec_profile_id_size.value() == 0 ||
        codec_profile_id_size.value() > AccessUnitIO::maximum_codec_profile_id_bytes ||
        model_id_size.value() == 0 ||
        model_id_size.value() > AccessUnitIO::maximum_model_id_bytes) {
        return malformed("NVAU v2 identity length is out of range");
    }

    auto codec_id = reader.read_string(codec_id_size.value());
    auto codec_profile_id = reader.read_string(codec_profile_id_size.value());
    auto model_id = reader.read_string(model_id_size.value());
    if (!codec_id) return codec_id.error();
    if (!codec_profile_id) return codec_profile_id.error();
    if (!model_id) return model_id.error();

    std::vector<std::uint64_t> dependencies;
    try {
        dependencies.reserve(dependency_count.value());
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "dependency allocation failed", "access-unit");
    }
    for (std::uint16_t index = 0; index < dependency_count.value(); ++index) {
        auto dependency = reader.read_u64();
        if (!dependency) return dependency.error();
        dependencies.push_back(dependency.value());
    }

    std::vector<SectionHeader> section_headers;
    try {
        section_headers.reserve(section_count.value());
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "section allocation failed", "access-unit");
    }
    std::uint64_t aggregate_section_bytes = 0;
    for (std::uint16_t index = 0; index < section_count.value(); ++index) {
        auto type = reader.read_u16();
        auto section_version = reader.read_u16();
        auto section_flags = reader.read_u32();
        auto section_size = reader.read_u64();
        if (!type) return type.error();
        if (!section_version) return section_version.error();
        if (!section_flags) return section_flags.error();
        if (!section_size) return section_size.error();
        if (section_size.value() == 0 ||
            section_size.value() > std::numeric_limits<std::size_t>::max()) {
            return malformed("NVAU v2 section length is invalid");
        }
        if (!known_section_type(type.value()) &&
            (section_flags.value() & AccessUnitIO::required_section_flag) != 0U) {
            return malformed("unknown required NVAU v2 section");
        }
        if (aggregate_section_bytes >
            std::numeric_limits<std::uint64_t>::max() - section_size.value()) {
            return malformed("NVAU v2 aggregate section length overflow");
        }
        aggregate_section_bytes += section_size.value();
        section_headers.push_back(
            SectionHeader{type.value(), section_version.value(), section_flags.value(),
                          section_size.value()});
    }
    if (aggregate_section_bytes != reader.remaining()) {
        return malformed("NVAU v2 section table and payload sizes are inconsistent");
    }

    std::vector<AccessUnitSection> sections;
    try {
        sections.reserve(section_headers.size());
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "section allocation failed", "access-unit");
    }
    std::vector<std::byte> codec_payload;
    bool saw_codec_payload = false;
    for (const auto& header : section_headers) {
        auto payload = reader.read_bytes(static_cast<std::size_t>(header.size));
        if (!payload) return payload.error();
        sections.push_back(AccessUnitSection{
            header.type, header.version, header.flags,
            std::vector<std::byte>(payload.value().begin(), payload.value().end())});
        if (header.type == codec_payload_section_type) {
            if (saw_codec_payload) return malformed("multiple NVAU v2 codec payload sections");
            saw_codec_payload = true;
            codec_payload = sections.back().payload;
        }
    }
    if (reader.remaining() != 0) {
        return malformed("trailing NVAU v2 section payload data");
    }
    if (!saw_codec_payload) return malformed("NVAU v2 has no codec payload section");

    AccessUnit output{
        std::move(model_id.value()),
        width.value(),
        height.value(),
        qp.value(),
        static_cast<FrameType>(frame_type.value()),
        (flags.value() & v2_reset_state_flag) != 0U,
        std::move(codec_payload)};
    output.codec_id = std::move(codec_id.value());
    output.codec_profile_id = std::move(codec_profile_id.value());
    output.pixel_format = static_cast<AccessUnitPixelFormat>(pixel_format.value());
    output.bit_depth = bit_depth.value();
    output.random_access = (flags.value() & v2_random_access_flag) != 0U;
    output.discontinuity = (flags.value() & v2_discontinuity_flag) != 0U;
    output.decode_order_index = decode_order_index.value();
    output.presentation_order_index = presentation_order_index.value();
    output.dependencies = std::move(dependencies);
    output.sections = std::move(sections);

    auto valid = validate_v2_fields(output);
    if (!valid) return malformed(std::string(valid.error().message()));
    return output;
}

}  // namespace

bool AccessUnitIO::has_magic(std::span<const std::byte> bytes) noexcept {
    return bytes.size() >= std::size(magic) &&
        std::equal(std::begin(magic), std::end(magic), bytes.begin());
}

Result<std::vector<std::byte>> AccessUnitIO::serialize(
    const AccessUnit& access_unit, std::size_t maximum_access_unit_bytes) {
    auto valid = validate_v1_fields(access_unit);
    if (!valid) return valid.error();
    if (access_unit.model_id.size() > std::numeric_limits<std::uint16_t>::max() ||
        access_unit.payload.size() > std::numeric_limits<std::uint64_t>::max() ||
        fixed_header_bytes_v1 > maximum_access_unit_bytes ||
        access_unit.model_id.size() > maximum_access_unit_bytes - fixed_header_bytes_v1 ||
        access_unit.payload.size() >
            maximum_access_unit_bytes - fixed_header_bytes_v1 - access_unit.model_id.size()) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }

    try {
        ByteWriter writer(
            fixed_header_bytes_v1 + access_unit.model_id.size() + access_unit.payload.size());
        writer.write_bytes(magic);
        writer.write_u16(format_version);
        writer.write_u8(static_cast<std::uint8_t>(access_unit.frame_type));
        writer.write_u8(access_unit.reset_state ? reset_state_flag : 0U);
        writer.write_u16(static_cast<std::uint16_t>(access_unit.model_id.size()));
        writer.write_u16(0U);
        writer.write_u32(access_unit.width);
        writer.write_u32(access_unit.height);
        writer.write_u32(access_unit.qp);
        writer.write_u64(static_cast<std::uint64_t>(access_unit.payload.size()));
        writer.write_string(access_unit.model_id);
        writer.write_bytes(access_unit.payload);
        return writer.take();
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted,
                     "access-unit allocation failed", "access-unit");
    }
}

Result<std::vector<std::byte>> AccessUnitIO::serialize_sectioned(
    const AccessUnit& access_unit, std::size_t maximum_access_unit_bytes) {
    auto sections = validate_v2_fields(access_unit);
    if (!sections) return sections.error();

    std::uint64_t section_payload_bytes = 0;
    for (const auto& section : sections.value()) {
        if (section.payload.size() > std::numeric_limits<std::uint64_t>::max() - section_payload_bytes) {
            return Error(ErrorCode::resource_exhausted,
                         "access-unit section size overflow", "access-unit");
        }
        section_payload_bytes += static_cast<std::uint64_t>(section.payload.size());
    }

    const std::uint64_t dependency_bytes = access_unit.dependencies.size() * sizeof(std::uint64_t);
    const std::uint64_t section_table_bytes =
        sections.value().size() * section_table_entry_bytes_v2;
    const std::uint64_t total_size = fixed_header_bytes_v2 +
        access_unit.codec_id.size() + access_unit.codec_profile_id.size() +
        access_unit.model_id.size() + dependency_bytes + section_table_bytes + section_payload_bytes;
    if (total_size > maximum_access_unit_bytes ||
        total_size > std::numeric_limits<std::size_t>::max()) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }

    try {
        ByteWriter writer(static_cast<std::size_t>(total_size));
        std::uint32_t flags = access_unit.reset_state ? v2_reset_state_flag : 0U;
        if (access_unit.random_access) flags |= v2_random_access_flag;
        if (access_unit.discontinuity) flags |= v2_discontinuity_flag;
        writer.write_bytes(magic);
        writer.write_u16(sectioned_format_version);
        writer.write_u16(static_cast<std::uint16_t>(fixed_header_bytes_v2));
        writer.write_u32(flags);
        writer.write_u8(static_cast<std::uint8_t>(access_unit.frame_type));
        writer.write_u8(static_cast<std::uint8_t>(access_unit.pixel_format));
        writer.write_u8(access_unit.bit_depth);
        writer.write_u8(0U);
        writer.write_u32(access_unit.width);
        writer.write_u32(access_unit.height);
        writer.write_u32(access_unit.qp);
        writer.write_u64(access_unit.decode_order_index);
        writer.write_u64(access_unit.presentation_order_index);
        writer.write_u16(static_cast<std::uint16_t>(access_unit.dependencies.size()));
        writer.write_u16(static_cast<std::uint16_t>(sections.value().size()));
        writer.write_u16(static_cast<std::uint16_t>(access_unit.codec_id.size()));
        writer.write_u16(static_cast<std::uint16_t>(access_unit.codec_profile_id.size()));
        writer.write_u16(static_cast<std::uint16_t>(access_unit.model_id.size()));
        writer.write_u16(0U);
        writer.write_u64(total_size);
        writer.write_string(access_unit.codec_id);
        writer.write_string(access_unit.codec_profile_id);
        writer.write_string(access_unit.model_id);
        for (const auto dependency : access_unit.dependencies) {
            writer.write_u64(dependency);
        }
        for (const auto& section : sections.value()) {
            writer.write_u16(section.type);
            writer.write_u16(section.version);
            writer.write_u32(section.flags);
            writer.write_u64(static_cast<std::uint64_t>(section.payload.size()));
        }
        for (const auto& section : sections.value()) {
            writer.write_bytes(section.payload);
        }
        return writer.take();
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted,
                     "access-unit allocation failed", "access-unit");
    }
}

Result<AccessUnit> AccessUnitIO::deserialize(
    std::span<const std::byte> bytes, std::size_t maximum_access_unit_bytes) {
    if (bytes.size() > maximum_access_unit_bytes) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }
    if (!has_magic(bytes)) return malformed("invalid access-unit magic");

    ByteReader reader(bytes);
    auto wire_magic = reader.read_bytes(std::size(magic));
    if (!wire_magic) return wire_magic.error();
    auto version = reader.read_u16();
    if (!version) return version.error();

    if (version.value() == format_version) {
        return deserialize_v1(reader, bytes, maximum_access_unit_bytes);
    }
    if (version.value() == sectioned_format_version) {
        return deserialize_v2(reader, bytes, maximum_access_unit_bytes);
    }
    return malformed("unsupported access-unit version");
}

}  // namespace nvcr
