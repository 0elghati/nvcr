#include "host_ops.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace nvcr::dcvcrt::host_ops {
namespace {

constexpr float scale_min = 0.11F;
constexpr float scale_max = 16.0F;
constexpr float log_scale_min = -2.2072749131897207F;
constexpr float log_step_recip = 25.516414672012925F;

bool invalid_shape(Shape4D shape) {
    return shape.batch <= 0 || shape.channels <= 0 || shape.height <= 0 || shape.width <= 0;
}

std::size_t volume(Shape4D shape) {
    return static_cast<std::size_t>(shape.batch) *
        static_cast<std::size_t>(shape.channels) *
        static_cast<std::size_t>(shape.height) *
        static_cast<std::size_t>(shape.width);
}

float load(std::span<const Half> values, std::size_t index) {
    return to_float(values[index]);
}

void store(std::span<Half> values, std::size_t index, float value) {
    values[index] = to_half(value);
}

std::uint8_t quantize_luma(float value) {
    return static_cast<std::uint8_t>(
        std::round(std::fmin(1.0F, std::fmax(0.0F, value)) * 255.0F));
}

std::uint8_t quantize_chroma(float value) {
    return static_cast<std::uint8_t>(std::fmin(255.0F, std::fmax(0.0F, value * 255.0F)));
}

std::uint8_t scale_index(float scale) {
    scale = std::fmin(scale_max, std::fmax(scale_min, scale));
    const float index = (std::log(scale) - log_scale_min) * log_step_recip;
    return static_cast<std::uint8_t>(index);
}

float sigmoid_quant(float value) {
    return (1.0F / (1.0F + std::exp(-value))) * 1.5F + 0.5F;
}

}  // namespace

Half to_half(float value) noexcept {
    // Branch structure from F. Giesen's public-domain float_to_half_fast3_rtne;
    // it relies on round-to-nearest-even float addition.
    constexpr std::uint32_t f32_infinity = 255U << 23U;
    constexpr std::uint32_t f16_overflow = (127U + 16U) << 23U;
    constexpr std::uint32_t denormal_magic = ((127U - 15U) + (23U - 10U) + 1U) << 23U;
    auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = bits & 0x80000000U;
    bits ^= sign;
    std::uint32_t output = 0;
    if (bits >= f16_overflow) {
        output = bits > f32_infinity ? 0x7e00U : 0x7c00U;
    } else if (bits < (113U << 23U)) {
        const float aligned =
            std::bit_cast<float>(bits) + std::bit_cast<float>(denormal_magic);
        output = std::bit_cast<std::uint32_t>(aligned) - denormal_magic;
    } else {
        const std::uint32_t mantissa_odd = (bits >> 13U) & 1U;
        bits += (static_cast<std::uint32_t>(15 - 127) << 23U) + 0xfffU;
        bits += mantissa_odd;
        output = bits >> 13U;
    }
    return static_cast<Half>(output | (sign >> 16U));
}

float to_float(Half value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    const std::uint32_t mantissa = value & 0x3ffU;
    if (exponent == 0U) {
        const float magnitude = static_cast<float>(mantissa) * 0x1p-24F;
        return std::bit_cast<float>(std::bit_cast<std::uint32_t>(magnitude) | sign);
    }
    if (exponent == 31U) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    }
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

bool round_to_int8(std::span<Half> values, std::span<std::int8_t> quantized) noexcept {
    if (values.empty() || quantized.size() < values.size()) return false;
    for (std::size_t index = 0; index < values.size(); ++index) {
        float value = std::round(to_float(values[index]));
        value = std::fmin(127.0F, std::fmax(-128.0F, value));
        values[index] = to_half(value);
        quantized[index] = static_cast<std::int8_t>(value);
    }
    return true;
}

bool int8_to_half(std::span<const std::int8_t> values, std::span<Half> output) noexcept {
    if (values.empty() || output.size() < values.size()) return false;
    for (std::size_t index = 0; index < values.size(); ++index) {
        output[index] = to_half(static_cast<float>(values[index]));
    }
    return true;
}

bool replicate_pad(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape,
    std::int32_t padded_height, std::int32_t padded_width) noexcept {
    if (invalid_shape(input_shape) || padded_height < input_shape.height ||
        padded_width < input_shape.width) return false;
    const auto planes = static_cast<std::size_t>(input_shape.batch) * input_shape.channels;
    if (input.size() < volume(input_shape) ||
        output.size() < planes * padded_height * padded_width) return false;
    std::size_t index = 0;
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::int32_t y = 0; y < padded_height; ++y) {
            const auto source_y = std::min(y, input_shape.height - 1);
            for (std::int32_t x = 0; x < padded_width; ++x) {
                const auto source_x = std::min(x, input_shape.width - 1);
                output[index++] = input[(plane * input_shape.height + source_y) *
                    static_cast<std::size_t>(input_shape.width) + source_x];
            }
        }
    }
    return true;
}

bool crop_spatial(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape,
    std::int32_t output_height, std::int32_t output_width) noexcept {
    if (invalid_shape(input_shape) || output_height <= 0 || output_width <= 0 ||
        output_height > input_shape.height || output_width > input_shape.width) return false;
    const auto planes = static_cast<std::size_t>(input_shape.batch) * input_shape.channels;
    if (input.size() < volume(input_shape) ||
        output.size() < planes * output_height * output_width) return false;
    std::size_t index = 0;
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::int32_t y = 0; y < output_height; ++y) {
            const auto row = (plane * input_shape.height + y) *
                static_cast<std::size_t>(input_shape.width);
            for (std::int32_t x = 0; x < output_width; ++x) {
                output[index++] = input[row + x];
            }
        }
    }
    return true;
}

bool apply_image_prior(std::span<Half> params, std::size_t spatial_count) noexcept {
    if (spatial_count == 0 || params.size() < 2 * spatial_count) return false;
    for (std::size_t index = 0; index < spatial_count; ++index) {
        const float q_enc = sigmoid_quant(to_float(params[index]));
        const float q_dec = sigmoid_quant(to_float(params[spatial_count + index]));
        store(params, index, q_enc);
        store(params, spatial_count + index, q_dec);
    }
    return true;
}

bool apply_image_prior_and_quantize(
    std::span<Half> params, std::span<Half> values, std::size_t spatial_count) noexcept {
    if (spatial_count == 0 || params.size() < 2 * spatial_count ||
        values.size() < 256 * spatial_count) return false;
    for (std::size_t index = 0; index < spatial_count; ++index) {
        const float q_enc = sigmoid_quant(to_float(params[index]));
        const float q_dec = sigmoid_quant(to_float(params[spatial_count + index]));
        store(params, index, q_enc);
        store(params, spatial_count + index, q_dec);
        // The unrounded scale multiplies the latent, as in the CUDA kernel.
        for (std::size_t channel = 0; channel < 256; ++channel) {
            const auto offset = channel * spatial_count + index;
            store(values, offset, to_float(values[offset]) * q_enc);
        }
    }
    return true;
}

bool rgb24_to_ycbcr_padded(
    std::span<const std::uint8_t> input, std::int32_t source_width,
    std::int32_t source_height, std::span<Half> output,
    std::int32_t padded_height, std::int32_t padded_width) noexcept {
    if (source_width <= 0 || source_height <= 0 || padded_height < source_height ||
        padded_width < source_width) return false;
    const auto plane = static_cast<std::size_t>(padded_height) * padded_width;
    if (input.size() < static_cast<std::size_t>(source_width) * source_height * 3U ||
        output.size() < 3 * plane) return false;
    constexpr float kr = 0.2126F;
    constexpr float kg = 0.7152F;
    constexpr float kb = 0.0722F;
    for (std::int32_t y = 0; y < padded_height; ++y) {
        const auto source_y = y < source_height ? y : source_height - 1;
        for (std::int32_t x = 0; x < padded_width; ++x) {
            const auto source_x = x < source_width ? x : source_width - 1;
            const auto source =
                (static_cast<std::size_t>(source_y) * source_width + source_x) * 3U;
            const float r = static_cast<float>(input[source]) / 255.0F;
            const float g = static_cast<float>(input[source + 1]) / 255.0F;
            const float b = static_cast<float>(input[source + 2]) / 255.0F;
            const float luma = kr * r + kg * g + kb * b;
            const float cb = 0.5F * (b - luma) / (1.0F - kb) + 0.5F;
            const float cr = 0.5F * (r - luma) / (1.0F - kr) + 0.5F;
            const auto offset = static_cast<std::size_t>(y) * padded_width + x;
            output[offset] = to_half(luma);
            output[plane + offset] = to_half(cb);
            output[2 * plane + offset] = to_half(cr);
        }
    }
    return true;
}

bool yuv420p8_to_ycbcr_padded(
    std::span<const std::uint8_t> input, std::int32_t source_width,
    std::int32_t source_height, std::span<Half> output,
    std::int32_t padded_height, std::int32_t padded_width) noexcept {
    if (source_width <= 0 || source_height <= 0 || (source_width & 1) != 0 ||
        (source_height & 1) != 0 || padded_height < source_height ||
        padded_width < source_width) return false;
    const auto y_size = static_cast<std::size_t>(source_width) * source_height;
    const auto uv_width = source_width / 2;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (source_height / 2);
    const auto plane = static_cast<std::size_t>(padded_height) * padded_width;
    if (input.size() < y_size + 2 * uv_size || output.size() < 3 * plane) return false;
    const auto u_plane = input.subspan(y_size, uv_size);
    const auto v_plane = input.subspan(y_size + uv_size, uv_size);
    for (std::int32_t y = 0; y < padded_height; ++y) {
        const auto source_y = y < source_height ? y : source_height - 1;
        for (std::int32_t x = 0; x < padded_width; ++x) {
            const auto source_x = x < source_width ? x : source_width - 1;
            const auto y_index = static_cast<std::size_t>(source_y) * source_width + source_x;
            const auto uv_index =
                static_cast<std::size_t>(source_y / 2) * uv_width + source_x / 2;
            const auto offset = static_cast<std::size_t>(y) * padded_width + x;
            output[offset] = to_half(static_cast<float>(input[y_index]) / 255.0F);
            output[plane + offset] = to_half(static_cast<float>(u_plane[uv_index]) / 255.0F);
            output[2 * plane + offset] =
                to_half(static_cast<float>(v_plane[uv_index]) / 255.0F);
        }
    }
    return true;
}

bool ycbcr_padded_to_yuv420p8(
    std::span<const Half> input, Shape4D input_shape, std::int32_t visible_width,
    std::int32_t visible_height, std::span<std::uint8_t> output) noexcept {
    if (invalid_shape(input_shape) || input_shape.batch != 1 || input_shape.channels != 3 ||
        visible_width <= 0 || visible_height <= 0 || (visible_width & 1) != 0 ||
        (visible_height & 1) != 0 || visible_width > input_shape.width ||
        visible_height > input_shape.height) return false;
    const auto y_size = static_cast<std::size_t>(visible_width) * visible_height;
    const auto uv_size = y_size / 4;
    if (input.size() < volume(input_shape) || output.size() < y_size + 2 * uv_size) {
        return false;
    }
    const auto plane = static_cast<std::size_t>(input_shape.height) * input_shape.width;
    for (std::int32_t y = 0; y < visible_height; ++y) {
        for (std::int32_t x = 0; x < visible_width; ++x) {
            output[static_cast<std::size_t>(y) * visible_width + x] = quantize_luma(
                load(input, static_cast<std::size_t>(y) * input_shape.width + x));
        }
    }
    const auto uv_width = visible_width / 2;
    for (std::int32_t uv_y = 0; uv_y < visible_height / 2; ++uv_y) {
        for (std::int32_t uv_x = 0; uv_x < uv_width; ++uv_x) {
            float cb_sum = 0.0F;
            float cr_sum = 0.0F;
            for (std::int32_t dy = 0; dy < 2; ++dy) {
                for (std::int32_t dx = 0; dx < 2; ++dx) {
                    const auto spatial =
                        static_cast<std::size_t>(uv_y * 2 + dy) * input_shape.width +
                        uv_x * 2 + dx;
                    cb_sum += load(input, plane + spatial);
                    cr_sum += load(input, 2 * plane + spatial);
                }
            }
            const auto index = static_cast<std::size_t>(uv_y) * uv_width + uv_x;
            output[y_size + index] = quantize_chroma(cb_sum * 0.25F);
            output[y_size + uv_size + index] = quantize_chroma(cr_sum * 0.25F);
        }
    }
    return true;
}

bool make_four_way_mask(std::span<Half> mask, Shape4D shape, std::int32_t stage) noexcept {
    if (invalid_shape(shape) || shape.channels % 4 != 0 || stage < 0 || stage > 3 ||
        mask.size() < volume(shape)) return false;
    constexpr std::int32_t patterns[4][4] = {
        {0, 1, 2, 3},
        {3, 2, 1, 0},
        {2, 3, 0, 1},
        {1, 0, 3, 2},
    };
    std::size_t index = 0;
    for (std::int32_t batch = 0; batch < shape.batch; ++batch) {
        for (std::int32_t channel = 0; channel < shape.channels; ++channel) {
            const auto quarter = channel / (shape.channels / 4);
            for (std::int32_t y = 0; y < shape.height; ++y) {
                for (std::int32_t x = 0; x < shape.width; ++x) {
                    const auto parity = (y & 1) * 2 + (x & 1);
                    store(mask, index++, patterns[stage][quarter] == parity ? 1.0F : 0.0F);
                }
            }
        }
    }
    return true;
}

bool make_two_way_mask(std::span<Half> mask, Shape4D shape, std::int32_t stage) noexcept {
    if (invalid_shape(shape) || shape.channels % 2 != 0 || stage < 0 || stage > 1 ||
        mask.size() < volume(shape)) return false;
    std::size_t index = 0;
    for (std::int32_t batch = 0; batch < shape.batch; ++batch) {
        for (std::int32_t channel = 0; channel < shape.channels; ++channel) {
            const bool first_half = channel < shape.channels / 2;
            for (std::int32_t y = 0; y < shape.height; ++y) {
                for (std::int32_t x = 0; x < shape.width; ++x) {
                    const bool diagonal = (y & 1) == (x & 1);
                    const bool selected =
                        stage == 0 ? first_half == diagonal : first_half != diagonal;
                    store(mask, index++, selected ? 1.0F : 0.0F);
                }
            }
        }
    }
    return true;
}

bool process_with_mask(
    std::span<const Half> values, std::span<const Half> scales,
    std::span<const Half> means, std::span<const Half> mask,
    std::span<Half> residual, std::span<Half> symbols,
    std::span<Half> reconstructed, std::span<Half> masked_scales,
    float force_zero_threshold) noexcept {
    const auto count = values.size();
    if (count == 0 || scales.size() < count || means.size() < count ||
        mask.size() < count || symbols.size() < count || reconstructed.size() < count ||
        masked_scales.size() < count || (!residual.empty() && residual.size() < count)) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        const float mask_value = load(mask, index);
        const float scale = load(scales, index) * mask_value;
        const float mean = load(means, index) * mask_value;
        const float value_residual = (load(values, index) - mean) * mask_value;
        float symbol = std::round(value_residual);
        if (force_zero_threshold > 0.0F && scale <= force_zero_threshold) symbol = 0.0F;
        symbol = std::fmin(127.0F, std::fmax(-128.0F, symbol));
        if (!residual.empty()) store(residual, index, value_residual);
        store(symbols, index, symbol);
        store(reconstructed, index, symbol + mean);
        store(masked_scales, index, scale);
    }
    return true;
}

bool reduce_channel_quarters(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape) noexcept {
    if (invalid_shape(input_shape) || input_shape.channels % 4 != 0) return false;
    const auto quarter_size = volume(input_shape) / 4;
    if (input.size() < 4 * quarter_size || output.size() < quarter_size) return false;
    for (std::size_t index = 0; index < quarter_size; ++index) {
        float value = 0.0F;
        for (std::size_t quarter = 0; quarter < 4; ++quarter) {
            value += load(input, index + quarter * quarter_size);
        }
        store(output, index, value);
    }
    return true;
}

bool reduce_masked_quarters(
    std::span<const Half> input, std::span<const Half> mask, std::span<Half> output,
    Shape4D input_shape) noexcept {
    if (invalid_shape(input_shape) || input_shape.channels % 4 != 0) return false;
    const auto quarter_size = volume(input_shape) / 4;
    if (input.size() < 4 * quarter_size || mask.size() < 4 * quarter_size ||
        output.size() < quarter_size) return false;
    for (std::size_t index = 0; index < quarter_size; ++index) {
        float value = 0.0F;
        for (std::size_t quarter = 0; quarter < 4; ++quarter) {
            const auto source = index + quarter * quarter_size;
            value += load(input, source) * load(mask, source);
        }
        store(output, index, value);
    }
    return true;
}

bool restore_four_way(
    std::span<const Half> symbols, std::span<const Half> means,
    std::span<const Half> mask, std::span<Half> output, Shape4D output_shape) noexcept {
    if (invalid_shape(output_shape) || output_shape.channels % 4 != 0) return false;
    const auto quarter_size = volume(output_shape) / 4;
    if (symbols.size() < quarter_size || means.size() < 4 * quarter_size ||
        mask.size() < 4 * quarter_size || output.size() < 4 * quarter_size) return false;
    for (std::size_t index = 0; index < quarter_size; ++index) {
        const float symbol = load(symbols, index);
        for (std::size_t quarter = 0; quarter < 4; ++quarter) {
            const auto destination = index + quarter * quarter_size;
            store(output, destination,
                  (symbol + load(means, destination)) * load(mask, destination));
        }
    }
    return true;
}

bool reduce_channel_halves(
    std::span<const Half> input, std::span<Half> output, Shape4D input_shape) noexcept {
    if (invalid_shape(input_shape) || input_shape.channels % 2 != 0) return false;
    const auto half_size = volume(input_shape) / 2;
    if (input.size() < 2 * half_size || output.size() < half_size) return false;
    for (std::size_t index = 0; index < half_size; ++index) {
        store(output, index, load(input, index) + load(input, index + half_size));
    }
    return true;
}

bool reduce_masked_halves(
    std::span<const Half> input, std::span<const Half> mask, std::span<Half> output,
    Shape4D input_shape) noexcept {
    if (invalid_shape(input_shape) || input_shape.channels % 2 != 0) return false;
    const auto half_size = volume(input_shape) / 2;
    if (input.size() < 2 * half_size || mask.size() < 2 * half_size ||
        output.size() < half_size) return false;
    for (std::size_t index = 0; index < half_size; ++index) {
        const float first = load(input, index) * load(mask, index);
        const float second = load(input, index + half_size) * load(mask, index + half_size);
        store(output, index, first + second);
    }
    return true;
}

bool restore_two_way(
    std::span<const Half> symbols, std::span<const Half> means,
    std::span<const Half> mask, std::span<Half> output, Shape4D output_shape) noexcept {
    if (invalid_shape(output_shape) || output_shape.channels % 2 != 0) return false;
    const auto half_size = volume(output_shape) / 2;
    if (symbols.size() < half_size || means.size() < 2 * half_size ||
        mask.size() < 2 * half_size || output.size() < 2 * half_size) return false;
    for (std::size_t index = 0; index < half_size; ++index) {
        const float symbol = load(symbols, index);
        for (std::size_t half = 0; half < 2; ++half) {
            const auto destination = index + half * half_size;
            store(output, destination,
                  (symbol + load(means, destination)) * load(mask, destination));
        }
    }
    return true;
}

bool clamp_min(std::span<Half> values, float minimum) noexcept {
    if (values.empty()) return false;
    for (auto& value : values) value = to_half(std::fmax(to_float(value), minimum));
    return true;
}

bool clamp_reciprocal_with_quant(std::span<Half> quant_step, std::span<Half> values) noexcept {
    if (values.empty() || quant_step.size() < values.size()) return false;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const Half quant = to_half(std::fmax(to_float(quant_step[index]), 0.5F));
        quant_step[index] = quant;
        const Half reciprocal = to_half(1.0F / to_float(quant));
        values[index] = to_half(to_float(values[index]) * to_float(reciprocal));
    }
    return true;
}

bool add_in_place(std::span<Half> first, std::span<const Half> second) noexcept {
    if (first.empty() || second.size() < first.size()) return false;
    for (std::size_t index = 0; index < first.size(); ++index) {
        first[index] = to_half(to_float(first[index]) + load(second, index));
    }
    return true;
}

bool multiply_broadcast_in_place(
    std::span<Half> values, std::span<const Half> factors, std::size_t spatial_count,
    std::int32_t channels) noexcept {
    if (spatial_count == 0 || channels <= 0 || factors.size() < spatial_count) return false;
    const auto count = spatial_count * static_cast<std::size_t>(channels);
    if (values.size() < count) return false;
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = to_half(to_float(values[index]) * load(factors, index % spatial_count));
    }
    return true;
}

bool add_and_multiply(
    std::span<Half> first, std::span<const Half> second,
    std::span<const Half> quant_step) noexcept {
    if (first.empty() || second.size() < first.size() ||
        quant_step.size() < first.size()) return false;
    for (std::size_t index = 0; index < first.size(); ++index) {
        const Half sum = to_half(to_float(first[index]) + load(second, index));
        first[index] = to_half(to_float(sum) * load(quant_step, index));
    }
    return true;
}

bool build_encode_indexes(
    std::span<const Half> symbols, std::span<const Half> scales,
    std::span<std::int16_t> output, float force_zero_threshold) noexcept {
    const auto count = output.size();
    if (count == 0 || symbols.size() < count || scales.size() < count) return false;
    for (std::size_t index = 0; index < count; ++index) {
        const float scale = load(scales, index);
        if (force_zero_threshold > 0.0F && scale <= force_zero_threshold) {
            output[index] = 0;
            continue;
        }
        const auto symbol = static_cast<std::int16_t>(load(symbols, index));
        output[index] = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(symbol) << 8U | scale_index(scale));
    }
    return true;
}

bool build_decode_indexes(
    std::span<const Half> scales, std::span<std::uint8_t> output,
    float force_zero_threshold) noexcept {
    const auto count = output.size();
    if (count == 0 || scales.size() < count) return false;
    for (std::size_t index = 0; index < count; ++index) {
        const float scale = load(scales, index);
        output[index] = force_zero_threshold > 0.0F && scale <= force_zero_threshold
            ? 0
            : scale_index(scale);
    }
    return true;
}

}  // namespace nvcr::dcvcrt::host_ops
