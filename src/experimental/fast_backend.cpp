#include "nvcr/experimental/fast_backend.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <utility>

namespace nvcr::experimental {
namespace {

constexpr std::string_view subsystem = "experimental.fast";
constexpr std::array<std::byte, 4> payload_magic{
    std::byte{'N'}, std::byte{'V'}, std::byte{'F'}, std::byte{'0'}};
constexpr std::uint16_t payload_version = 1;
constexpr std::size_t header_bytes = 24;

void write_u16(std::vector<std::byte>& output, std::uint16_t value) {
    output.push_back(static_cast<std::byte>(value & 0xFFU));
    output.push_back(static_cast<std::byte>((value >> 8U) & 0xFFU));
}

void write_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
    }
}

void write_u64(std::vector<std::byte>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
    }
}

std::uint16_t read_u16(std::span<const std::byte> bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[offset]) |
        (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U));
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        value |= static_cast<std::uint32_t>(bytes[offset + shift / 8U]) << shift;
    }
    return value;
}

std::uint64_t read_u64(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        value |= static_cast<std::uint64_t>(bytes[offset + shift / 8U]) << shift;
    }
    return value;
}

class FastBackend final : public codec::CodecBackend {
public:
    Result<void> initialize(const RuntimeConfiguration&) override { return {}; }

    Result<codec::CodecEncodeResult> encode(
        const Frame& frame,
        FrameType,
        const codec::SequenceStateView&) override {
        if (frame.pixel_format() != PixelFormat::yuv420p8) {
            return Error(
                ErrorCode::invalid_argument,
                "experimental fast backend currently accepts only YUV420P8",
                std::string(subsystem));
        }
        if (frame.data().size() > std::numeric_limits<std::uint64_t>::max()) {
            return Error(
                ErrorCode::resource_exhausted,
                "frame is too large",
                std::string(subsystem));
        }
        try {
            std::vector<std::byte> payload;
            payload.reserve(header_bytes + frame.data().size());
            payload.insert(payload.end(), payload_magic.begin(), payload_magic.end());
            write_u16(payload, payload_version);
            write_u16(payload, 0);
            write_u32(payload, frame.width());
            write_u32(payload, frame.height());
            write_u64(payload, static_cast<std::uint64_t>(frame.data().size()));
            payload.insert(payload.end(), frame.data().begin(), frame.data().end());

            auto reconstructed = Frame::copy_from(
                frame.width(), frame.height(), frame.pixel_format(),
                frame.data(), frame.timestamp());
            if (!reconstructed) return reconstructed.error();
            return codec::CodecEncodeResult{
                std::move(payload), std::move(reconstructed.value()), {}, 0};
        } catch (const std::bad_alloc&) {
            return Error(
                ErrorCode::resource_exhausted,
                "experimental fast encode allocation failed",
                std::string(subsystem));
        }
    }

    Result<codec::CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType,
        Timestamp timestamp,
        const codec::SequenceStateView&) override {
        if (payload.size() < header_bytes ||
            !std::equal(payload_magic.begin(), payload_magic.end(), payload.begin())) {
            return Error(
                ErrorCode::malformed_bitstream,
                "invalid experimental fast payload",
                std::string(subsystem));
        }
        if (read_u16(payload, 4) != payload_version || read_u16(payload, 6) != 0) {
            return Error(
                ErrorCode::malformed_bitstream,
                "unsupported experimental fast payload version or flags",
                std::string(subsystem));
        }
        const auto width = read_u32(payload, 8);
        const auto height = read_u32(payload, 12);
        const auto data_size = read_u64(payload, 16);
        if (data_size > payload.size() - header_bytes ||
            data_size > std::numeric_limits<std::size_t>::max()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "experimental fast payload size is invalid",
                std::string(subsystem));
        }
        auto expected = frame_size_bytes(width, height, PixelFormat::yuv420p8);
        if (!expected) return expected.error();
        if (data_size != expected.value() || header_bytes + expected.value() != payload.size()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "experimental fast payload dimensions and size disagree",
                std::string(subsystem));
        }
        auto frame = Frame::copy_from(
            width, height, PixelFormat::yuv420p8,
            payload.subspan(header_bytes, expected.value()), timestamp);
        if (!frame) return frame.error();
        return codec::CodecDecodeResult{std::move(frame.value()), {}};
    }

    Result<void> flush() override { return {}; }
    void reset() noexcept override {}
};

}  // namespace

Result<std::unique_ptr<codec::CodecBackend>> make_fast_backend() {
    try {
        return std::unique_ptr<codec::CodecBackend>(std::make_unique<FastBackend>());
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate experimental fast backend",
            std::string(subsystem));
    }
}

}  // namespace nvcr::experimental
