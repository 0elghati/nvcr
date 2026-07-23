#include "nvcr/bitstream/access_unit.hpp"

#include "nvcr/bitstream/packet_io.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <new>

namespace nvcr {
namespace {

constexpr std::byte magic[] = {
    std::byte{'N'}, std::byte{'V'}, std::byte{'A'}, std::byte{'U'}};
constexpr std::uint8_t reset_state_flag = 1U;
constexpr std::uint8_t supported_flags = reset_state_flag;
constexpr std::size_t fixed_header_bytes = 32U;

Error malformed(std::string message) {
    return Error(ErrorCode::malformed_bitstream, std::move(message), "access-unit");
}

bool valid_model_id(std::string_view model_id) {
    if (model_id.empty() || model_id.size() > AccessUnitIO::maximum_model_id_bytes) {
        return false;
    }
    return std::ranges::all_of(model_id, [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '.' || character == '_' ||
            character == '-';
    });
}

Result<void> validate_fields(const AccessUnit& access_unit) {
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

}  // namespace

bool AccessUnitIO::has_magic(std::span<const std::byte> bytes) noexcept {
    return bytes.size() >= std::size(magic) &&
        std::equal(std::begin(magic), std::end(magic), bytes.begin());
}

Result<std::vector<std::byte>> AccessUnitIO::serialize(
    const AccessUnit& access_unit, std::size_t maximum_access_unit_bytes) {
    auto valid = validate_fields(access_unit);
    if (!valid) return valid.error();
    if (access_unit.model_id.size() > std::numeric_limits<std::uint16_t>::max() ||
        access_unit.payload.size() > std::numeric_limits<std::uint64_t>::max() ||
        fixed_header_bytes > maximum_access_unit_bytes ||
        access_unit.model_id.size() > maximum_access_unit_bytes - fixed_header_bytes ||
        access_unit.payload.size() >
            maximum_access_unit_bytes - fixed_header_bytes - access_unit.model_id.size()) {
        return Error(ErrorCode::resource_exhausted,
                     "access unit exceeds configured limit", "access-unit");
    }

    try {
        ByteWriter writer(
            fixed_header_bytes + access_unit.model_id.size() + access_unit.payload.size());
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
    auto frame_type = reader.read_u8();
    auto flags = reader.read_u8();
    auto model_id_size = reader.read_u16();
    auto reserved = reader.read_u16();
    auto width = reader.read_u32();
    auto height = reader.read_u32();
    auto qp = reader.read_u32();
    auto payload_size = reader.read_u64();
    if (!version) return version.error();
    if (!frame_type) return frame_type.error();
    if (!flags) return flags.error();
    if (!model_id_size) return model_id_size.error();
    if (!reserved) return reserved.error();
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!payload_size) return payload_size.error();

    if (version.value() != format_version) return malformed("unsupported access-unit version");
    if (frame_type.value() > static_cast<std::uint8_t>(FrameType::predicted)) {
        return malformed("invalid access-unit frame type");
    }
    if ((flags.value() & ~supported_flags) != 0U || reserved.value() != 0U) {
        return malformed("unsupported access-unit flags or reserved fields");
    }
    if (model_id_size.value() == 0 || model_id_size.value() > maximum_model_id_bytes) {
        return malformed("access-unit model identity is out of range");
    }
    if (payload_size.value() == 0 || payload_size.value() > reader.remaining() ||
        payload_size.value() > std::numeric_limits<std::size_t>::max()) {
        return malformed("access-unit payload length is invalid");
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
    auto valid = validate_fields(output);
    if (!valid) return malformed(std::string(valid.error().message()));
    return output;
}

}  // namespace nvcr
