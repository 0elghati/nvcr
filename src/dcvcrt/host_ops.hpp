#pragma once

// Host implementations of the DCVC-RT inter-stage transforms in
// backend/tensorrt/cuda_ops.cu. Each function computes in float and rounds to
// FP16 at the same points as its CUDA counterpart, so providers without CUDA
// run the same codec flow. Functions return false when spans are too small or
// shapes are invalid; no work is done in that case.

#include <cstddef>
#include <cstdint>
#include <span>

namespace nvcr::dcvcrt::host_ops {

using Half = std::uint16_t;  // IEEE 754 binary16 bit pattern

struct Shape4D final {
    std::int32_t batch{};
    std::int32_t channels{};
    std::int32_t height{};
    std::int32_t width{};
};

// Round-to-nearest-even, matching __float2half_rn for finite values.
[[nodiscard]] Half to_half(float value) noexcept;
[[nodiscard]] float to_float(Half value) noexcept;

// Rounds values in place and writes the int8 symbols.
[[nodiscard]] bool round_to_int8(
    std::span<Half> values, std::span<std::int8_t> quantized) noexcept;
[[nodiscard]] bool int8_to_half(
    std::span<const std::int8_t> values, std::span<Half> output) noexcept;

[[nodiscard]] bool replicate_pad(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape,
    std::int32_t padded_height, std::int32_t padded_width) noexcept;
[[nodiscard]] bool crop_spatial(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape,
    std::int32_t output_height, std::int32_t output_width) noexcept;

[[nodiscard]] bool apply_image_prior(
    std::span<Half> params, std::size_t spatial_count) noexcept;
[[nodiscard]] bool apply_image_prior_and_quantize(
    std::span<Half> params, std::span<Half> values, std::size_t spatial_count) noexcept;

[[nodiscard]] bool rgb24_to_ycbcr_padded(
    std::span<const std::uint8_t> input, std::int32_t source_width,
    std::int32_t source_height, std::span<Half> output,
    std::int32_t padded_height, std::int32_t padded_width) noexcept;
[[nodiscard]] bool yuv420p8_to_ycbcr_padded(
    std::span<const std::uint8_t> input, std::int32_t source_width,
    std::int32_t source_height, std::span<Half> output,
    std::int32_t padded_height, std::int32_t padded_width) noexcept;
[[nodiscard]] bool ycbcr_padded_to_yuv420p8(
    std::span<const Half> input, Shape4D input_shape, std::int32_t visible_width,
    std::int32_t visible_height, std::span<std::uint8_t> output) noexcept;

[[nodiscard]] bool make_four_way_mask(
    std::span<Half> mask, Shape4D shape, std::int32_t stage) noexcept;
[[nodiscard]] bool make_two_way_mask(
    std::span<Half> mask, Shape4D shape, std::int32_t stage) noexcept;

// Processes values.size() elements; residual may be empty.
[[nodiscard]] bool process_with_mask(
    std::span<const Half> values, std::span<const Half> scales,
    std::span<const Half> means, std::span<const Half> mask,
    std::span<Half> residual, std::span<Half> symbols,
    std::span<Half> reconstructed, std::span<Half> masked_scales,
    float force_zero_threshold) noexcept;

[[nodiscard]] bool reduce_channel_quarters(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape) noexcept;
[[nodiscard]] bool reduce_masked_quarters(
    std::span<const Half> input, std::span<const Half> mask, std::span<Half> output,
    Shape4D input_shape) noexcept;
[[nodiscard]] bool restore_four_way(
    std::span<const Half> symbols, std::span<const Half> means,
    std::span<const Half> mask, std::span<Half> output, Shape4D output_shape) noexcept;

[[nodiscard]] bool reduce_channel_halves(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape) noexcept;
[[nodiscard]] bool reduce_masked_halves(
    std::span<const Half> input, std::span<const Half> mask, std::span<Half> output,
    Shape4D input_shape) noexcept;
[[nodiscard]] bool restore_two_way(
    std::span<const Half> symbols, std::span<const Half> means,
    std::span<const Half> mask, std::span<Half> output, Shape4D output_shape) noexcept;

[[nodiscard]] bool clamp_min(std::span<Half> values, float minimum) noexcept;
[[nodiscard]] bool clamp_reciprocal_with_quant(
    std::span<Half> quant_step, std::span<Half> values) noexcept;
[[nodiscard]] bool add_in_place(std::span<Half> first, std::span<const Half> second) noexcept;
[[nodiscard]] bool multiply_broadcast_in_place(
    std::span<Half> values, std::span<const Half> factors, std::size_t spatial_count,
    std::int32_t channels) noexcept;
[[nodiscard]] bool add_and_multiply(
    std::span<Half> first, std::span<const Half> second,
    std::span<const Half> quant_step) noexcept;

[[nodiscard]] bool build_encode_indexes(
    std::span<const Half> symbols, std::span<const Half> scales,
    std::span<std::int16_t> output, float force_zero_threshold) noexcept;
[[nodiscard]] bool build_decode_indexes(
    std::span<const Half> scales, std::span<std::uint8_t> output,
    float force_zero_threshold) noexcept;

}  // namespace nvcr::dcvcrt::host_ops
