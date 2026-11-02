#include "record.hpp"

#include <limits>
#include <new>
#include <string>

namespace nvcr::mlvc {
namespace {

constexpr std::string_view subsystem = "mlvc.record";
constexpr std::size_t header_bytes = 8;

void write_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

std::uint32_t read_u32(std::span<const std::byte> bytes) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4U; ++index) {
        value |= std::to_integer<std::uint32_t>(bytes[index]) << (8U * index);
    }
    return value;
}

}  // namespace

Result<std::vector<std::byte>> serialize_record(
    std::int32_t q_index, std::span<const std::byte> payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        return Error(ErrorCode::resource_exhausted, "MLVC record payload is too large",
                     std::string(subsystem));
    }
    if (q_index == dropped_frame_q_index && !payload.empty()) {
        return Error(ErrorCode::invalid_argument, "a dropped MLVC frame has no payload",
                     std::string(subsystem));
    }
    try {
        std::vector<std::byte> output;
        output.reserve(header_bytes + payload.size());
        write_u32(output, static_cast<std::uint32_t>(q_index));
        write_u32(output, static_cast<std::uint32_t>(payload.size()));
        output.insert(output.end(), payload.begin(), payload.end());
        return output;
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "MLVC record allocation failed",
                     std::string(subsystem));
    }
}

Result<FrameRecord> parse_record(std::span<const std::byte> bytes) {
    if (bytes.size() < header_bytes) {
        return Error(ErrorCode::malformed_bitstream, "MLVC record header is truncated",
                     std::string(subsystem));
    }
    const auto q_index = static_cast<std::int32_t>(read_u32(bytes.first(4)));
    const auto length = read_u32(bytes.subspan(4, 4));
    if (length != bytes.size() - header_bytes) {
        return Error(ErrorCode::malformed_bitstream,
                     "MLVC record length does not match the access unit",
                     std::string(subsystem));
    }
    if (q_index < dropped_frame_q_index ||
        (q_index == dropped_frame_q_index && length != 0U)) {
        return Error(ErrorCode::malformed_bitstream, "MLVC record has an invalid Q index",
                     std::string(subsystem));
    }
    return FrameRecord{q_index, bytes.subspan(header_bytes)};
}

}  // namespace nvcr::mlvc
