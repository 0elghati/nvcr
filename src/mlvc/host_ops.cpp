#include "host_ops.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>

namespace nvcr::mlvc::host_ops {
namespace {

constexpr std::string_view subsystem = "mlvc.host";

float load(std::span<const std::byte> tensor, Precision precision, std::size_t index) noexcept {
    if (precision == Precision::fp16) {
        Half value{};
        std::memcpy(&value, tensor.data() + index * sizeof(Half), sizeof(Half));
        return to_float(value);
    }
    float value{};
    std::memcpy(&value, tensor.data() + index * sizeof(float), sizeof(float));
    return value;
}

std::byte to_u8(float value) noexcept {
    // np.round(255 * v).clip(0, 255) in float32; nearbyint rounds half to even.
    const float scaled = std::nearbyint(255.0F * value);
    const float clipped = std::clamp(scaled, 0.0F, 255.0F);
    return static_cast<std::byte>(static_cast<std::uint8_t>(clipped));
}

// Every 8-bit sample maps to one model value, and every fp16 model value to one
// 8-bit sample, so tables built with the scalar conversions above are exact.
template <class Value>
const std::array<Value, 256>& input_table() {
    static const auto table = [] {
        std::array<Value, 256> values{};
        for (unsigned sample = 0; sample < 256U; ++sample) {
            const float value = static_cast<float>(sample) / 255.0F;
            if constexpr (std::is_same_v<Value, Half>) {
                values[sample] = to_half(value);
            } else {
                values[sample] = value;
            }
        }
        return values;
    }();
    return table;
}

const std::array<std::uint8_t, 65536>& half_to_u8_table() {
    static const auto table = [] {
        std::array<std::uint8_t, 65536> values{};
        for (std::uint32_t half = 0; half < 65536U; ++half) {
            values[half] = std::to_integer<std::uint8_t>(to_u8(to_float(static_cast<Half>(half))));
        }
        return values;
    }();
    return table;
}

const std::array<float, 65536>& half_to_float_table() {
    static const auto table = [] {
        std::array<float, 65536> values{};
        for (std::uint32_t half = 0; half < 65536U; ++half) {
            values[half] = to_float(static_cast<Half>(half));
        }
        return values;
    }();
    return table;
}

template <class Value>
void prepare_planes(
    std::span<const std::byte> i420, const Placement& placement, std::span<std::byte> output) {
    const auto& table = input_table<Value>();
    const std::size_t width = placement.frame_width;
    const std::size_t height = placement.frame_height;
    const std::size_t canvas_width = placement.canvas_width;
    const std::size_t canvas_height = placement.canvas_height;
    const std::size_t plane = canvas_width * canvas_height;
    const auto* luma = i420.data();
    const auto* cb = luma + width * height;
    const auto* cr = cb + (width / 2U) * (height / 2U);
    // The image on the canvas before padding (transposed if needed).
    const std::size_t image_width = placement.transposed ? height : width;
    const std::size_t image_height = placement.transposed ? width : height;
    auto* out = output.data();
    const auto put = [&](std::size_t index, std::byte sample) {
        std::memcpy(out + index * sizeof(Value), &table[std::to_integer<unsigned>(sample)],
                    sizeof(Value));
    };

    for (std::size_t y = 0; y < canvas_height; ++y) {
        // Edge padding repeats the last row and column of the placed image.
        const std::size_t image_y = std::min(y, image_height - 1U);
        for (std::size_t x = 0; x < canvas_width; ++x) {
            const std::size_t image_x = std::min(x, image_width - 1U);
            const std::size_t row = placement.transposed ? image_x : image_y;
            const std::size_t column = placement.transposed ? image_y : image_x;
            const std::size_t chroma = (row / 2U) * (width / 2U) + column / 2U;
            const std::size_t index = y * canvas_width + x;
            put(index, luma[row * width + column]);
            put(plane + index, cb[chroma]);
            put(2U * plane + index, cr[chroma]);
        }
    }
}

}  // namespace

Half to_half(float value) noexcept {
    // F. Giesen's public-domain float_to_half_fast3_rtne; relies on
    // round-to-nearest-even float addition.
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
        const float aligned = std::bit_cast<float>(bits) + std::bit_cast<float>(denormal_magic);
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
        return sign != 0U ? -magnitude : magnitude;
    }
    if (exponent == 0x1fU) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    }
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

Result<Placement> place(
    std::uint32_t frame_width, std::uint32_t frame_height, const ModelParameters& model) {
    if (frame_width == 0U || frame_height == 0U || frame_width % 2U != 0U ||
        frame_height % 2U != 0U) {
        return Error(ErrorCode::invalid_argument,
                     "MLVC requires positive, even I420 frame dimensions",
                     std::string(subsystem));
    }
    Placement placement{frame_width, frame_height, model.width, model.height, false};
    if (frame_width <= model.width && frame_height <= model.height) return placement;
    if (frame_height <= model.width && frame_width <= model.height) {
        placement.transposed = true;
        return placement;
    }
    return Error(ErrorCode::invalid_argument,
                 "frame " + std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                     " is larger than the MLVC canvas " + std::to_string(model.width) + "x" +
                     std::to_string(model.height),
                 std::string(subsystem));
}

void prepare_input(
    std::span<const std::byte> i420,
    const Placement& placement,
    Precision precision,
    std::span<std::byte> output) {
    if (precision == Precision::fp16) {
        prepare_planes<Half>(i420, placement, output);
    } else {
        prepare_planes<float>(i420, placement, output);
    }
}

void finish_output(
    std::span<const std::byte> x_hat,
    const Placement& placement,
    Precision precision,
    std::span<std::byte> i420) {
    const std::size_t width = placement.frame_width;
    const std::size_t height = placement.frame_height;
    const std::size_t canvas_width = placement.canvas_width;
    const std::size_t plane = canvas_width * placement.canvas_height;
    const auto sample = [&](std::size_t channel, std::size_t row, std::size_t column) {
        const std::size_t y = placement.transposed ? column : row;
        const std::size_t x = placement.transposed ? row : column;
        return load(x_hat, precision, channel * plane + y * canvas_width + x);
    };

    auto* luma = i420.data();
    if (precision == Precision::fp16) {
        const auto& table = half_to_u8_table();
        for (std::size_t row = 0; row < height; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                const std::size_t y = placement.transposed ? column : row;
                const std::size_t x = placement.transposed ? row : column;
                Half half{};
                std::memcpy(&half, x_hat.data() + (y * canvas_width + x) * sizeof(Half),
                            sizeof(Half));
                luma[row * width + column] = static_cast<std::byte>(table[half]);
            }
        }
    } else {
        for (std::size_t row = 0; row < height; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                luma[row * width + column] = to_u8(sample(0, row, column));
            }
        }
    }
    // _downsample_uv: float32 reshape(h/2, 2, w/2, 2).mean(axis=(1, 3)),
    // accumulated top-left, top-right, bottom-left, bottom-right.
    const auto* halves = precision == Precision::fp16 ? &half_to_float_table() : nullptr;
    const auto* u8 = precision == Precision::fp16 ? &half_to_u8_table() : nullptr;
    for (std::size_t channel = 1; channel < 3U; ++channel) {
        auto* chroma = luma + width * height + (channel - 1U) * (width / 2U) * (height / 2U);
        const auto value = [&](std::size_t row, std::size_t column) {
            if (halves == nullptr) return sample(channel, row, column);
            const std::size_t y = placement.transposed ? column : row;
            const std::size_t x = placement.transposed ? row : column;
            Half half{};
            std::memcpy(&half, x_hat.data() + (channel * plane + y * canvas_width + x) * sizeof(Half),
                        sizeof(Half));
            return (*halves)[half];
        };
        for (std::size_t row = 0; row < height / 2U; ++row) {
            for (std::size_t column = 0; column < width / 2U; ++column) {
                float sum = value(2U * row, 2U * column);
                sum += value(2U * row, 2U * column + 1U);
                sum += value(2U * row + 1U, 2U * column);
                sum += value(2U * row + 1U, 2U * column + 1U);
                // fp16 tensors stay fp16 between numpy steps: cast the mean back.
                chroma[row * (width / 2U) + column] = u8 != nullptr ?
                    static_cast<std::byte>((*u8)[to_half(sum / 4.0F)]) : to_u8(sum / 4.0F);
            }
        }
    }
}

LatentShape latent_shape(const ModelParameters& model) noexcept {
    LatentShape shape;
    shape.y_channels = model.latent_channels / 2U;
    shape.y_height = model.height / model.downsample_latent;
    shape.y_width = model.width / model.downsample_latent;
    const auto z_stride = model.downsample_latent * model.downsample_hyperprior;
    shape.z_channels = model.hyperprior_channels;
    shape.z_height = (model.height + z_stride - 1U) / z_stride;
    shape.z_width = (model.width + z_stride - 1U) / z_stride;
    return shape;
}

ScaleIndices extract_scales(
    std::span<const float> z, const ModelParameters& model, std::uint32_t scale_levels) {
    const auto shape = latent_shape(model);
    const std::size_t half = shape.y_channels;
    const std::size_t repeat = model.y_scale_repeat;
    const std::size_t spatial = model.downsample_hyperprior;
    const float maximum = static_cast<float>(scale_levels - 1U);
    const auto scale = [&](std::size_t channel, std::size_t y, std::size_t x) {
        const std::size_t source = ((channel / repeat) * shape.z_height + y / spatial) *
            shape.z_width + x / spatial;
        return std::clamp(std::fabs(z[source]), 0.0F, maximum);
    };

    ScaleIndices indices;
    indices.scales_0.resize(shape.y_elements());
    indices.scales_1.resize(shape.y_elements());
    for (std::size_t channel = 0; channel < half; ++channel) {
        for (std::size_t y = 0; y < shape.y_height; ++y) {
            for (std::size_t x = 0; x < shape.y_width; ++x) {
                const bool even = (y + x) % 2U == 0U;
                const float first = scale(channel, y, x);
                const float second = scale(channel + half, y, x);
                const std::size_t index = (channel * shape.y_height + y) * shape.y_width + x;
                // astype(np.int32) truncates toward zero.
                indices.scales_0[index] = static_cast<std::int32_t>(even ? first : second);
                indices.scales_1[index] = static_cast<std::int32_t>(even ? second : first);
            }
        }
    }
    return indices;
}

}  // namespace nvcr::mlvc::host_ops
