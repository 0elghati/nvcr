#pragma once

#include "nvcr/runtime/frame.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace nvcr {

enum class FrameType : std::uint8_t {
    intra = 0,
    predicted = 1,
};

using PacketMetadata = std::map<std::string, std::string, std::less<>>;

class Packet final {
public:
    Packet() = default;
    Packet(
        std::vector<std::byte> bytes,
        Timestamp timestamp,
        FrameType frame_type,
        PacketMetadata metadata = {});

    [[nodiscard]] std::span<std::byte> data() noexcept;
    [[nodiscard]] std::span<const std::byte> data() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] Timestamp timestamp() const noexcept;
    [[nodiscard]] FrameType frame_type() const noexcept;
    [[nodiscard]] PacketMetadata& metadata() noexcept;
    [[nodiscard]] const PacketMetadata& metadata() const noexcept;

private:
    std::vector<std::byte> bytes_;
    Timestamp timestamp_{};
    FrameType frame_type_{FrameType::intra};
    PacketMetadata metadata_;
};

}  // namespace nvcr

