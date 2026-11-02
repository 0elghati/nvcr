#pragma once

// Host-side MLVC transforms between the provider-executed ONNX parts. Each
// function names the upstream code it reproduces; float rounding points follow
// the numpy reference exactly.

#include "bundle.hpp"

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvcr::mlvc::host_ops {

using Half = std::uint16_t;

// IEEE binary16 conversion with round-to-nearest-even, matching numpy's
// float32 -> float16 astype.
[[nodiscard]] Half to_half(float value) noexcept;
[[nodiscard]] float to_float(Half value) noexcept;

// Where a source frame sits on the model canvas
// (_base_split_model.py _prepare_input_frame, BOTTOM_RIGHT padding).
struct Placement final {
    std::uint32_t frame_width{};
    std::uint32_t frame_height{};
    std::uint32_t canvas_width{};
    std::uint32_t canvas_height{};
    bool transposed{};  // H/W swapped when the frame only fits that way
};

[[nodiscard]] Result<Placement> place(
    std::uint32_t frame_width, std::uint32_t frame_height, const ModelParameters& model);

// 8-bit I420 -> [1,3,H,W] YUV444 model input: u8 / 255 in float32, chroma
// nearest-neighbour upsampling, optional H/W transpose, edge padding to the
// canvas, then a cast to the model precision.
void prepare_input(
    std::span<const std::byte> i420,
    const Placement& placement,
    Precision precision,
    std::span<std::byte> output);

// [1,3,H,W] model output -> 8-bit I420: crop, undo the transpose, 2x2 chroma
// mean in float32 cast back to the model precision, then
// round-half-even(255 * v) clipped to [0, 255] (_frame_loop.py _save_yuv_output).
void finish_output(
    std::span<const std::byte> x_hat,
    const Placement& placement,
    Precision precision,
    std::span<std::byte> i420);

// Upsample scale decoder (_scale_decoder.py UpsampleScaleDecoder): Gaussian
// distribution indices for the two checkerboard halves of y, read from the
// first latent_channels / y_scale_repeat channels of z.
struct ScaleIndices final {
    std::vector<std::int32_t> scales_0;
    std::vector<std::int32_t> scales_1;
};

struct LatentShape final {
    std::uint32_t y_channels{};  // latent_channels / 2 per half
    std::uint32_t y_height{};
    std::uint32_t y_width{};
    std::uint32_t z_channels{};
    std::uint32_t z_height{};
    std::uint32_t z_width{};

    [[nodiscard]] std::size_t y_elements() const noexcept {
        return static_cast<std::size_t>(y_channels) * y_height * y_width;
    }
    [[nodiscard]] std::size_t z_elements() const noexcept {
        return static_cast<std::size_t>(z_channels) * z_height * z_width;
    }
};

[[nodiscard]] LatentShape latent_shape(const ModelParameters& model) noexcept;

// z values are the float values the encoder produced, or decoded integers.
[[nodiscard]] ScaleIndices extract_scales(
    std::span<const float> z, const ModelParameters& model, std::uint32_t scale_levels);

}  // namespace nvcr::mlvc::host_ops
