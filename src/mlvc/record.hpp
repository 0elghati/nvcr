#pragma once

// One MLVC frame record exactly as MLVC's conversion tooling writes it
// (video/conversion/utils.py save_mlvc_bitstreams): little-endian signed
// int32 Q index, little-endian uint32 payload length, payload. Q index -1 marks
// a dropped frame with an empty payload. NVCR carries one record as the codec
// payload of each NVAU access unit, so the payloads of an NVCR MLVC stream
// concatenate to an upstream `.mlvc` file.

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvcr::mlvc {

inline constexpr std::int32_t dropped_frame_q_index = -1;

struct FrameRecord final {
    std::int32_t q_index{};
    std::span<const std::byte> payload;
};

[[nodiscard]] Result<std::vector<std::byte>> serialize_record(
    std::int32_t q_index, std::span<const std::byte> payload);

// The record must span exactly the given bytes.
[[nodiscard]] Result<FrameRecord> parse_record(std::span<const std::byte> bytes);

}  // namespace nvcr::mlvc
