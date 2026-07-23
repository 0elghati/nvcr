#include "nvcr/bitstream/packet_io.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>

namespace nvcr {
namespace {

constexpr std::byte magic[] = {
    std::byte{'N'}, std::byte{'V'}, std::byte{'C'}, std::byte{'R'}};

Error truncated_error() {
    return Error(ErrorCode::malformed_bitstream, "unexpected end of packet", "bitstream");
}

}  // namespace

ByteWriter::ByteWriter(std::size_t reserve_bytes) { data_.reserve(reserve_bytes); }

void ByteWriter::write_u8(std::uint8_t value) {
    data_.push_back(static_cast<std::byte>(value));
}

void ByteWriter::write_u16(std::uint16_t value) {
    write_u8(static_cast<std::uint8_t>(value & 0xffU));
    write_u8(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

void ByteWriter::write_u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        write_u8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void ByteWriter::write_u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U) {
        write_u8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void ByteWriter::write_bytes(std::span<const std::byte> bytes) {
    data_.insert(data_.end(), bytes.begin(), bytes.end());
}

void ByteWriter::write_string(std::string_view value) {
    const auto bytes = std::as_bytes(std::span(value.data(), value.size()));
    write_bytes(bytes);
}

std::span<const std::byte> ByteWriter::data() const noexcept { return data_; }
std::vector<std::byte> ByteWriter::take() noexcept { return std::move(data_); }

ByteReader::ByteReader(std::span<const std::byte> data) noexcept : data_(data) {}

Result<std::uint8_t> ByteReader::read_u8() {
    if (remaining() < 1U) {
        return truncated_error();
    }
    return std::to_integer<std::uint8_t>(data_[offset_++]);
}

Result<std::uint16_t> ByteReader::read_u16() {
    auto bytes = read_bytes(2);
    if (!bytes) return bytes.error();
    return static_cast<std::uint16_t>(
        std::to_integer<std::uint16_t>(bytes.value()[0]) |
        (std::to_integer<std::uint16_t>(bytes.value()[1]) << 8U));
}

Result<std::uint32_t> ByteReader::read_u32() {
    auto bytes = read_bytes(4);
    if (!bytes) return bytes.error();
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4U; ++index) {
        value |= std::to_integer<std::uint32_t>(bytes.value()[index]) << (index * 8U);
    }
    return value;
}

Result<std::uint64_t> ByteReader::read_u64() {
    auto bytes = read_bytes(8);
    if (!bytes) return bytes.error();
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8U; ++index) {
        value |= std::to_integer<std::uint64_t>(bytes.value()[index]) << (index * 8U);
    }
    return value;
}

Result<std::span<const std::byte>> ByteReader::read_bytes(std::size_t count) {
    if (count > remaining()) {
        return truncated_error();
    }
    const auto result = data_.subspan(offset_, count);
    offset_ += count;
    return result;
}

Result<std::string> ByteReader::read_string(std::size_t count) {
    auto bytes = read_bytes(count);
    if (!bytes) return bytes.error();
    try {
        return std::string(
            reinterpret_cast<const char*>(bytes.value().data()), bytes.value().size());
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "metadata allocation failed",
            "bitstream");
    }
}

std::size_t ByteReader::remaining() const noexcept { return data_.size() - offset_; }

Result<std::vector<std::byte>> PacketIO::serialize(const Packet& packet) {
    if (packet.metadata().size() > maximum_metadata_entries) {
        return Error(ErrorCode::invalid_argument, "too many metadata entries", "bitstream");
    }
    std::size_t metadata_bytes = 0;
    for (const auto& [key, value] : packet.metadata()) {
        if (key.size() > std::numeric_limits<std::uint16_t>::max() ||
            value.size() > std::numeric_limits<std::uint16_t>::max()) {
            return Error(ErrorCode::invalid_argument, "metadata entry is too large", "bitstream");
        }
        constexpr std::size_t length_fields = 4U;
        if (key.size() > maximum_metadata_bytes - length_fields ||
            value.size() > maximum_metadata_bytes - length_fields - key.size() ||
            metadata_bytes > maximum_metadata_bytes - length_fields - key.size() - value.size()) {
            return Error(
                ErrorCode::invalid_argument,
                "aggregate metadata exceeds configured format limit",
                "bitstream");
        }
        metadata_bytes += length_fields + key.size() + value.size();
    }

    try {
        if (packet.size() > std::numeric_limits<std::size_t>::max() - metadata_bytes - 32U) {
            return Error(ErrorCode::resource_exhausted, "packet size overflow", "bitstream");
        }
        ByteWriter writer(packet.size() + metadata_bytes + 32U);
        writer.write_bytes(magic);
        writer.write_u16(format_version);
        writer.write_u8(static_cast<std::uint8_t>(packet.frame_type()));
        writer.write_u8(0);
        writer.write_u64(std::bit_cast<std::uint64_t>(packet.timestamp().count()));
        writer.write_u16(static_cast<std::uint16_t>(packet.metadata().size()));
        for (const auto& [key, value] : packet.metadata()) {
            writer.write_u16(static_cast<std::uint16_t>(key.size()));
            writer.write_u16(static_cast<std::uint16_t>(value.size()));
            writer.write_string(key);
            writer.write_string(value);
        }
        writer.write_u64(static_cast<std::uint64_t>(packet.size()));
        writer.write_bytes(packet.data());
        return writer.take();
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "packet allocation failed", "bitstream");
    }
}

Result<Packet> PacketIO::deserialize(
    std::span<const std::byte> bytes, std::size_t maximum_packet_bytes) {
    if (bytes.size() > maximum_packet_bytes) {
        return Error(
            ErrorCode::resource_exhausted, "packet exceeds configured limit", "bitstream");
    }
    ByteReader reader(bytes);
    auto wire_magic = reader.read_bytes(sizeof(magic));
    if (!wire_magic) return wire_magic.error();
    if (!std::equal(wire_magic.value().begin(), wire_magic.value().end(), std::begin(magic))) {
        return Error(ErrorCode::malformed_bitstream, "invalid packet magic", "bitstream");
    }

    auto version = reader.read_u16();
    if (!version) return version.error();
    if (version.value() != format_version) {
        return Error(ErrorCode::malformed_bitstream, "unsupported packet version", "bitstream");
    }
    auto frame_type = reader.read_u8();
    if (!frame_type) return frame_type.error();
    if (frame_type.value() > static_cast<std::uint8_t>(FrameType::predicted)) {
        return Error(ErrorCode::malformed_bitstream, "invalid frame type", "bitstream");
    }
    auto flags = reader.read_u8();
    if (!flags) return flags.error();
    if (flags.value() != 0) {
        return Error(ErrorCode::malformed_bitstream, "unsupported packet flags", "bitstream");
    }
    auto timestamp = reader.read_u64();
    if (!timestamp) return timestamp.error();
    auto metadata_count = reader.read_u16();
    if (!metadata_count) return metadata_count.error();
    if (metadata_count.value() > maximum_metadata_entries) {
        return Error(ErrorCode::resource_exhausted, "too many metadata entries", "bitstream");
    }

    PacketMetadata metadata;
    std::size_t metadata_bytes = 0;
    try {
        for (std::uint16_t index = 0; index < metadata_count.value(); ++index) {
            auto key_size = reader.read_u16();
            if (!key_size) return key_size.error();
            auto value_size = reader.read_u16();
            if (!value_size) return value_size.error();
            constexpr std::size_t length_fields = 4U;
            const auto entry_bytes = length_fields + static_cast<std::size_t>(key_size.value()) +
                static_cast<std::size_t>(value_size.value());
            if (entry_bytes > maximum_metadata_bytes ||
                metadata_bytes > maximum_metadata_bytes - entry_bytes) {
                return Error(
                    ErrorCode::resource_exhausted,
                    "aggregate metadata exceeds configured format limit",
                    "bitstream");
            }
            metadata_bytes += entry_bytes;
            auto key = reader.read_string(key_size.value());
            if (!key) return key.error();
            auto value = reader.read_string(value_size.value());
            if (!value) return value.error();
            const auto [unused, inserted] =
                metadata.emplace(std::move(key.value()), std::move(value.value()));
            static_cast<void>(unused);
            if (!inserted) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "duplicate metadata key",
                    "bitstream");
            }
        }
        auto payload_size = reader.read_u64();
        if (!payload_size) return payload_size.error();
        if (payload_size.value() > maximum_packet_bytes ||
            payload_size.value() > std::numeric_limits<std::size_t>::max()) {
            return Error(ErrorCode::resource_exhausted, "packet exceeds configured limit", "bitstream");
        }
        auto payload = reader.read_bytes(static_cast<std::size_t>(payload_size.value()));
        if (!payload) return payload.error();
        if (reader.remaining() != 0) {
            return Error(ErrorCode::malformed_bitstream, "trailing packet data", "bitstream");
        }
        std::vector<std::byte> owned_payload(payload.value().begin(), payload.value().end());
        return Packet(
            std::move(owned_payload),
            Timestamp(std::bit_cast<std::int64_t>(timestamp.value())),
            static_cast<FrameType>(frame_type.value()),
            std::move(metadata));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "packet allocation failed", "bitstream");
    }
}

}  // namespace nvcr
