#pragma once

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nvcr::dcvcrt {

struct RansCdfTable final {
    std::vector<std::vector<std::int32_t>> values;
    std::vector<std::int32_t> sizes;
    std::vector<std::int32_t> offsets;
};

// Native DCVC-RT entropy session. This is a typed ownership and validation
// boundary around the pinned upstream rANS implementation; it does not alter
// the codec's coding rules or bitstream.
class RansCodec final {
public:
    RansCodec();
    ~RansCodec();
    RansCodec(RansCodec&&) noexcept;
    RansCodec& operator=(RansCodec&&) noexcept;
    RansCodec(const RansCodec&) = delete;
    RansCodec& operator=(const RansCodec&) = delete;

    [[nodiscard]] Result<void> set_use_two_coders(bool enabled);
    [[nodiscard]] bool uses_two_coders() const noexcept;

    [[nodiscard]] Result<std::size_t> add_cdf(RansCdfTable table);
    void clear_cdfs() noexcept;

    // Each int16 contains the signed symbol in the high byte and the CDF row
    // index in the low byte, exactly as produced by DCVC-RT build_index_enc.
    [[nodiscard]] Result<void> encode_y(
        std::span<const std::int16_t> combined_symbols,
        std::size_t cdf_group);
    [[nodiscard]] Result<void> encode_z(
        std::span<const std::int8_t> symbols,
        std::size_t cdf_group,
        std::size_t start_offset,
        std::size_t per_channel_size);
    [[nodiscard]] Result<std::vector<std::byte>> finish_encode();
    void reset_encoder() noexcept;

    [[nodiscard]] Result<void> set_stream(std::span<const std::byte> bitstream);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_y(
        std::span<const std::uint8_t> indexes,
        std::size_t cdf_group);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_z(
        std::size_t symbol_count,
        std::size_t cdf_group,
        std::size_t start_offset,
        std::size_t per_channel_size);
    // Call after the last symbol of a stream. A correctly decoded stream ends
    // at the encoder's initial rANS state with every byte consumed; anything
    // else means the stream is corrupt or was decoded with different entropy
    // parameters (indexes) than it was encoded with, and returns
    // malformed_bitstream. The stream must be set again before more decoding.
    [[nodiscard]] Result<void> finish_decode();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr::dcvcrt
