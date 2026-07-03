#include "nvcr/runtime/frame.hpp"

#include <algorithm>
#include <limits>

namespace nvcr {
namespace {

Result<std::size_t> checked_pixels(std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0) {
        return Error(ErrorCode::invalid_argument, "frame dimensions must be non-zero", "frame");
    }
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h) {
        return Error(ErrorCode::resource_exhausted, "frame dimensions overflow", "frame");
    }
    return w * h;
}

}  // namespace

Result<std::size_t> frame_size_bytes(
    std::uint32_t width, std::uint32_t height, PixelFormat format) {
    auto pixels = checked_pixels(width, height);
    if (!pixels) {
        return pixels.error();
    }
    const auto count = pixels.value();
    switch (format) {
    case PixelFormat::yuv420p8:
        if ((width % 2U) != 0U || (height % 2U) != 0U) {
            return Error(
                ErrorCode::invalid_argument,
                "YUV420 dimensions must both be even",
                "frame");
        }
        return count + count / 2U;
    case PixelFormat::rgb24:
        if (count > std::numeric_limits<std::size_t>::max() / 3U) {
            return Error(ErrorCode::resource_exhausted, "frame size overflow", "frame");
        }
        return count * 3U;
    case PixelFormat::rgba32:
        if (count > std::numeric_limits<std::size_t>::max() / 4U) {
            return Error(ErrorCode::resource_exhausted, "frame size overflow", "frame");
        }
        return count * 4U;
    case PixelFormat::gray8:
        return count;
    }
    return Error(ErrorCode::invalid_argument, "unsupported pixel format", "frame");
}

Frame::Frame(
    std::uint32_t width,
    std::uint32_t height,
    PixelFormat format,
    Timestamp timestamp,
    std::vector<std::byte> data)
    : width_(width),
      height_(height),
      format_(format),
      timestamp_(timestamp),
      data_(std::move(data)) {}

Result<Frame> Frame::create(
    std::uint32_t width,
    std::uint32_t height,
    PixelFormat format,
    Timestamp timestamp) {
    auto size = frame_size_bytes(width, height, format);
    if (!size) {
        return size.error();
    }
    try {
        return Frame(width, height, format, timestamp, std::vector<std::byte>(size.value()));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "unable to allocate frame", "frame");
    }
}

Result<Frame> Frame::copy_from(
    std::uint32_t width,
    std::uint32_t height,
    PixelFormat format,
    std::span<const std::byte> bytes,
    Timestamp timestamp) {
    auto expected_size = frame_size_bytes(width, height, format);
    if (!expected_size) {
        return expected_size.error();
    }
    if (bytes.size() != expected_size.value()) {
        return Error(ErrorCode::invalid_argument, "input buffer has the wrong size", "frame");
    }
    try {
        return Frame(
            width, height, format, timestamp, std::vector<std::byte>(bytes.begin(), bytes.end()));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "unable to allocate frame", "frame");
    }
}

std::uint32_t Frame::width() const noexcept { return width_; }
std::uint32_t Frame::height() const noexcept { return height_; }
PixelFormat Frame::pixel_format() const noexcept { return format_; }
Timestamp Frame::timestamp() const noexcept { return timestamp_; }
std::span<std::byte> Frame::data() noexcept { return data_; }
std::span<const std::byte> Frame::data() const noexcept { return data_; }
std::size_t Frame::size_bytes() const noexcept { return data_.size(); }

}  // namespace nvcr

