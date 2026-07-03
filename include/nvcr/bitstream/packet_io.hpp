#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/runtime/packet.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr {

class ByteWriter final {
public:
    explicit ByteWriter(std::size_t reserve_bytes = 0);

    void write_u8(std::uint8_t value);
    void write_u16(std::uint16_t value);
    void write_u32(std::uint32_t value);
    void write_u64(std::uint64_t value);
    void write_bytes(std::span<const std::byte> bytes);
    void write_string(std::string_view value);

    [[nodiscard]] std::span<const std::byte> data() const noexcept;
    [[nodiscard]] std::vector<std::byte> take() noexcept;

private:
    std::vector<std::byte> data_;
};

class ByteReader final {
public:
    explicit ByteReader(std::span<const std::byte> data) noexcept;

    [[nodiscard]] Result<std::uint8_t> read_u8();
    [[nodiscard]] Result<std::uint16_t> read_u16();
    [[nodiscard]] Result<std::uint32_t> read_u32();
    [[nodiscard]] Result<std::uint64_t> read_u64();
    [[nodiscard]] Result<std::span<const std::byte>> read_bytes(std::size_t count);
    [[nodiscard]] Result<std::string> read_string(std::size_t count);
    [[nodiscard]] std::size_t remaining() const noexcept;

private:
    std::span<const std::byte> data_;
    std::size_t offset_{};
};

class PacketIO final {
public:
    static constexpr std::uint16_t format_version = 1;

    [[nodiscard]] static Result<std::vector<std::byte>> serialize(const Packet& packet);
    [[nodiscard]] static Result<Packet> deserialize(
        std::span<const std::byte> bytes,
        std::size_t maximum_packet_bytes = 64U * 1024U * 1024U);
};

}  // namespace nvcr

