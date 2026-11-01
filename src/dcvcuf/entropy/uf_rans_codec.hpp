#pragma once

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nvcr::dcvcuf {

// UF keeps its entropy partitioning and stream framing inside the codec. This
// wrapper intentionally has no dependency on NVCR runtime or provider types.
struct UfRansCdfTable final {
    std::vector<std::vector<std::int32_t>> values;
    std::vector<std::int32_t> sizes;
    std::vector<std::int32_t> offsets;
};

class UfRansCodec final {
public:
    UfRansCodec();
    ~UfRansCodec();
    UfRansCodec(UfRansCodec&&) noexcept;
    UfRansCodec& operator=(UfRansCodec&&) noexcept;
    UfRansCodec(const UfRansCodec&) = delete;
    UfRansCodec& operator=(const UfRansCodec&) = delete;

    [[nodiscard]] Result<void> set_entropy_coder_parallel(std::size_t count);
    [[nodiscard]] std::size_t entropy_coder_parallel() const noexcept;
    [[nodiscard]] Result<std::size_t> add_cdf(UfRansCdfTable table);
    void clear_cdfs() noexcept;

    // Symbols use the same packed int16 representation as the upstream UF
    // encoder: signed value in the high byte, CDF row in the low byte.
    [[nodiscard]] Result<void> encode_y(std::span<const std::int16_t> symbols,
                                        std::size_t cdf_group);
    [[nodiscard]] Result<void> encode_z(std::span<const std::int8_t> symbols,
                                        std::size_t cdf_group,
                                        std::size_t start_offset,
                                        std::size_t per_channel_size);
    [[nodiscard]] Result<std::vector<std::byte>> finish_encode();
    void reset_encoder() noexcept;

    [[nodiscard]] Result<void> set_stream(std::span<const std::byte> stream);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_y(
        std::span<const std::uint8_t> indexes, std::size_t cdf_group);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_z(
        std::size_t symbol_count, std::size_t cdf_group,
        std::size_t start_offset, std::size_t per_channel_size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr::dcvcuf
