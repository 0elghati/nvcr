#include "nvcr/runtime/packet.hpp"

#include <utility>

namespace nvcr {

Packet::Packet(
    std::vector<std::byte> bytes,
    Timestamp timestamp,
    FrameType frame_type,
    PacketMetadata metadata)
    : bytes_(std::move(bytes)),
      timestamp_(timestamp),
      frame_type_(frame_type),
      metadata_(std::move(metadata)) {}

std::span<std::byte> Packet::data() noexcept { return bytes_; }
std::span<const std::byte> Packet::data() const noexcept { return bytes_; }
std::size_t Packet::size() const noexcept { return bytes_.size(); }
Timestamp Packet::timestamp() const noexcept { return timestamp_; }
FrameType Packet::frame_type() const noexcept { return frame_type_; }
PacketMetadata& Packet::metadata() noexcept { return metadata_; }
const PacketMetadata& Packet::metadata() const noexcept { return metadata_; }

}  // namespace nvcr
