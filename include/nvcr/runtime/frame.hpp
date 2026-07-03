#pragma once

#include "nvcr/common/error.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvcr {

using Timestamp = std::chrono::microseconds;

enum class PixelFormat : std::uint8_t {
    yuv420p8,
    rgb24,
    rgba32,
    gray8,
};

class Frame final {
public:
    Frame() = default;

    [[nodiscard]] static Result<Frame> create(
        std::uint32_t width,
        std::uint32_t height,
        PixelFormat format,
        Timestamp timestamp = {});

    [[nodiscard]] static Result<Frame> copy_from(
        std::uint32_t width,
        std::uint32_t height,
        PixelFormat format,
        std::span<const std::byte> bytes,
        Timestamp timestamp = {});

    [[nodiscard]] std::uint32_t width() const noexcept;
    [[nodiscard]] std::uint32_t height() const noexcept;
    [[nodiscard]] PixelFormat pixel_format() const noexcept;
    [[nodiscard]] Timestamp timestamp() const noexcept;
    [[nodiscard]] std::span<std::byte> data() noexcept;
    [[nodiscard]] std::span<const std::byte> data() const noexcept;
    [[nodiscard]] std::size_t size_bytes() const noexcept;

private:
    Frame(
        std::uint32_t width,
        std::uint32_t height,
        PixelFormat format,
        Timestamp timestamp,
        std::vector<std::byte> data);

    std::uint32_t width_{};
    std::uint32_t height_{};
    PixelFormat format_{PixelFormat::yuv420p8};
    Timestamp timestamp_{};
    std::vector<std::byte> data_;
};

[[nodiscard]] Result<std::size_t> frame_size_bytes(
    std::uint32_t width, std::uint32_t height, PixelFormat format);

}  // namespace nvcr

