#include "payload.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace nvcr::dcvcrt {
namespace {

constexpr char subsystem[] = "dcvcrt.payload";
constexpr std::size_t image_qp_count = 64U;
constexpr std::size_t video_qp_count = 72U;
constexpr std::uint32_t minimum_frame_dimension = 64U;
constexpr std::uint32_t maximum_frame_width = 1920U;
constexpr std::uint32_t maximum_frame_height = 1080U;

bool supported_payload_dimensions(std::uint32_t width, std::uint32_t height) {
    return width >= minimum_frame_dimension && height >= minimum_frame_dimension &&
        width <= maximum_frame_width && height <= maximum_frame_height &&
        (width & 1U) == 0U && (height & 1U) == 0U;
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

Result<std::uint32_t> read_u32(
    std::span<const std::byte> input, std::size_t& offset) {
    if (offset > input.size() || input.size() - offset < 4) {
        return Error(ErrorCode::malformed_bitstream, "truncated DCVC-RT payload header", subsystem);
    }
    std::uint32_t value = 0;
    for (std::uint32_t index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(
            std::to_integer<unsigned char>(input[offset + index])) << (index * 8U);
    }
    offset += 4;
    return value;
}

}  // namespace

std::vector<std::byte> make_intra_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, std::span<const std::byte> rans) {
    std::vector<std::byte> output;
    output.reserve(20 + rans.size());
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x49}, std::byte{0x31}});
    append_u32(output, width);
    append_u32(output, height);
    append_u32(output, qp);
    append_u32(output, two_coders ? 1U : 0U);
    output.insert(output.end(), rans.begin(), rans.end());
    return output;
}

Result<IntraPayload> parse_intra_payload(std::span<const std::byte> payload) {
    constexpr std::array<std::byte, 4> magic{
        std::byte{0x4e}, std::byte{0x56}, std::byte{0x49}, std::byte{0x31}};
    if (payload.size() < 24 || !std::equal(magic.begin(), magic.end(), payload.begin())) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR I-frame payload", subsystem);
    }
    std::size_t offset = 4;
    auto width = read_u32(payload, offset);
    auto height = read_u32(payload, offset);
    auto qp = read_u32(payload, offset);
    auto flags = read_u32(payload, offset);
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!flags) return flags.error();
    if (!supported_payload_dimensions(width.value(), height.value()) ||
        qp.value() >= image_qp_count || flags.value() > 1) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR I-frame fields", subsystem);
    }
    return IntraPayload{width.value(), height.value(), qp.value(), flags.value() == 1,
                        payload.subspan(offset)};
}

std::vector<std::byte> make_predicted_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, bool use_frame_reference, std::span<const std::byte> rans) {
    std::vector<std::byte> output;
    output.reserve(20 + rans.size());
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x50}, std::byte{0x31}});
    append_u32(output, width);
    append_u32(output, height);
    append_u32(output, qp);
    append_u32(output, (two_coders ? 1U : 0U) | (use_frame_reference ? 2U : 0U));
    output.insert(output.end(), rans.begin(), rans.end());
    return output;
}

Result<PredictedPayload> parse_predicted_payload(std::span<const std::byte> payload) {
    constexpr std::array<std::byte, 4> magic{
        std::byte{0x4e}, std::byte{0x56}, std::byte{0x50}, std::byte{0x31}};
    if (payload.size() < 24 || !std::equal(magic.begin(), magic.end(), payload.begin())) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR P-frame payload", subsystem);
    }
    std::size_t offset = 4;
    auto width = read_u32(payload, offset);
    auto height = read_u32(payload, offset);
    auto qp = read_u32(payload, offset);
    auto flags = read_u32(payload, offset);
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!flags) return flags.error();
    if (!supported_payload_dimensions(width.value(), height.value()) ||
        qp.value() >= video_qp_count || flags.value() > 3) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR P-frame fields", subsystem);
    }
    return PredictedPayload{width.value(), height.value(), qp.value(),
                            (flags.value() & 1U) != 0, (flags.value() & 2U) != 0,
                            payload.subspan(offset)};
}

}  // namespace nvcr::dcvcrt
