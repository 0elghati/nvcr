#include "host_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace ops = nvcr::dcvcrt::host_ops;
using ops::Half;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<Half> to_half(const std::vector<float>& values) {
    std::vector<Half> output;
    output.reserve(values.size());
    for (float value : values) output.push_back(ops::to_half(value));
    return output;
}

std::vector<float> to_float(const std::vector<Half>& values) {
    std::vector<float> output;
    output.reserve(values.size());
    for (Half value : values) output.push_back(ops::to_float(value));
    return output;
}

void half_conversion_test() {
    expect(ops::to_half(1.0F) == 0x3c00U, "1.0 converts to 0x3c00");
    expect(ops::to_half(-2.0F) == 0xc000U, "-2.0 converts to 0xc000");
    expect(ops::to_half(65504.0F) == 0x7bffU, "largest finite half is preserved");
    expect(ops::to_half(65520.0F) == 0x7c00U, "values rounding past 65504 become infinity");
    expect(ops::to_half(0x1p-24F) == 0x0001U, "smallest subnormal half is preserved");
    expect(ops::to_half(0x1p-25F) == 0x0000U, "half of the smallest subnormal rounds to even");
    expect(ops::to_half(1.0F + 0x1p-11F) == 0x3c00U, "ties round to the even mantissa");
    expect(ops::to_half(1.0F + 3.0F * 0x1p-11F) == 0x3c02U, "odd ties round up to even");
    expect(ops::to_half(std::numeric_limits<float>::infinity()) == 0x7c00U,
           "infinity is preserved");

    // Every finite half survives a float round trip.
    bool exact = true;
    for (std::uint32_t bits = 0; bits <= 0xffffU; ++bits) {
        const auto value = static_cast<Half>(bits);
        if ((value & 0x7c00U) == 0x7c00U) continue;
        if (ops::to_half(ops::to_float(value)) != value) exact = false;
    }
    expect(exact, "all finite halves round-trip through float");
}

void round_test() {
    auto values = to_half({-129.2F, -1.6F, 0.49F, 126.6F, 200.0F});
    std::array<std::int8_t, 5> quantized{};
    expect(ops::round_to_int8(values, quantized), "round_to_int8 accepts input");
    expect(quantized == std::array<std::int8_t, 5>{-128, -2, 0, 127, 127},
           "rounding and int8 clamping match DCVC-RT");
    expect(to_float(values) == std::vector<float>{-128.0F, -2.0F, 0.0F, 127.0F, 127.0F},
           "rounded values are written back in place");
}

void int8_to_half_test() {
    constexpr std::array<std::int8_t, 5> input{-128, -2, 0, 64, 127};
    std::vector<Half> output(input.size());
    expect(ops::int8_to_half(input, output), "int8_to_half accepts input");
    expect(to_float(output) == std::vector<float>{-128.0F, -2.0F, 0.0F, 64.0F, 127.0F},
           "int8 to half conversion preserves entropy symbols");
}

void padding_and_crop_test() {
    const auto input = to_half({1.0F, 2.0F, 3.0F, 4.0F});
    std::vector<Half> padded(12);
    expect(ops::replicate_pad(input, padded, {1, 1, 2, 2}, 3, 4), "replicate_pad accepts input");
    expect(to_float(padded) == std::vector<float>{
        1.0F, 2.0F, 2.0F, 2.0F,
        3.0F, 4.0F, 4.0F, 4.0F,
        3.0F, 4.0F, 4.0F, 4.0F},
        "replicate padding matches upstream semantics");
    std::vector<Half> cropped(4);
    expect(ops::crop_spatial(padded, cropped, {1, 1, 3, 4}, 2, 2), "crop_spatial accepts input");
    expect(cropped == input, "cropping inverts replicate padding");
    std::vector<Half> too_small(11);
    expect(!ops::replicate_pad(input, too_small, {1, 1, 2, 2}, 3, 4),
           "replicate_pad rejects an undersized output");
}

void image_prior_test() {
    auto params = to_half({0.0F, 0.0F, 0.0F, 0.0F});
    expect(ops::apply_image_prior(params, 2), "apply_image_prior accepts input");
    expect(to_float(params) == std::vector<float>{1.25F, 1.25F, 1.25F, 1.25F},
           "image prior transforms q-encode and q-decode logits");
}

void mask_test() {
    constexpr ops::Shape4D shape{1, 4, 2, 2};
    constexpr std::int32_t pattern[4][4] = {
        {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2}};
    std::vector<Half> mask(16);
    for (std::int32_t stage = 0; stage < 4; ++stage) {
        expect(ops::make_four_way_mask(mask, shape, stage), "make_four_way_mask accepts input");
        const auto values = to_float(mask);
        for (std::int32_t channel = 0; channel < 4; ++channel) {
            for (std::int32_t pixel = 0; pixel < 4; ++pixel) {
                const float expected = pattern[stage][channel] == pixel ? 1.0F : 0.0F;
                expect(values[static_cast<std::size_t>(channel * 4 + pixel)] == expected,
                       "four-way mask pattern matches upstream");
            }
        }
    }

    std::vector<Half> first(16);
    std::vector<Half> second(16);
    expect(ops::make_two_way_mask(first, shape, 0) && ops::make_two_way_mask(second, shape, 1),
           "make_two_way_mask accepts input");
    bool complementary = true;
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (ops::to_float(first[index]) + ops::to_float(second[index]) != 1.0F) {
            complementary = false;
        }
    }
    expect(complementary, "two-way masks partition every element");
}

void ycbcr_to_yuv420_test() {
    constexpr std::int32_t visible_width = 4;
    constexpr std::int32_t visible_height = 2;
    constexpr std::int32_t padded_width = 6;
    constexpr std::int32_t padded_height = 4;
    constexpr std::size_t plane = static_cast<std::size_t>(padded_width) * padded_height;
    std::vector<float> input(3 * plane, 0.875F);
    const float luma[] = {-0.25F, 0.0F, 0.49F / 255.0F, 0.5F / 255.0F,
                          127.49F / 255.0F, 127.5F / 255.0F, 1.0F, 1.25F};
    for (std::int32_t y = 0; y < visible_height; ++y) {
        for (std::int32_t x = 0; x < visible_width; ++x) {
            input[static_cast<std::size_t>(y * padded_width + x)] =
                luma[y * visible_width + x];
        }
    }
    std::vector<std::uint8_t> output(12);
    expect(ops::ycbcr_padded_to_yuv420p8(
               to_half(input), {1, 3, padded_height, padded_width}, visible_width,
               visible_height, output),
           "ycbcr_padded_to_yuv420p8 accepts input");
    // Half rounding lifts 127.49/255 to exactly 0.5, and 127.5 rounds away from zero.
    const std::vector<std::uint8_t> expected{0, 0, 0, 0, 128, 128, 255, 255, 223, 223, 223, 223};
    expect(output == expected, "YCbCr to YUV420P8 rounds luma and truncates chroma");
}

void color_round_trip_test() {
    constexpr std::int32_t width = 4;
    constexpr std::int32_t height = 2;
    std::vector<std::uint8_t> yuv(12);
    for (std::size_t index = 0; index < yuv.size(); ++index) {
        yuv[index] = static_cast<std::uint8_t>(index * 20U);
    }
    std::vector<Half> ycbcr(3 * 4 * 8);
    expect(ops::yuv420p8_to_ycbcr_padded(yuv, width, height, ycbcr, 4, 8),
           "yuv420p8_to_ycbcr_padded accepts input");
    std::vector<std::uint8_t> restored(12);
    expect(ops::ycbcr_padded_to_yuv420p8(ycbcr, {1, 3, 4, 8}, width, height, restored),
           "padded YCbCr converts back");
    expect(std::equal(restored.begin(), restored.begin() + 8, yuv.begin()),
           "luma survives a YUV420P8 round trip");

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U, 255U);
    expect(ops::rgb24_to_ycbcr_padded(rgb, width, height, ycbcr, 4, 8),
           "rgb24_to_ycbcr_padded accepts input");
    expect(std::abs(ops::to_float(ycbcr[0]) - 1.0F) < 1e-3F &&
               std::abs(ops::to_float(ycbcr[32]) - 0.5F) < 1e-3F,
           "white RGB maps to full luma and neutral chroma");
    expect(!ops::rgb24_to_ycbcr_padded(
               std::span<const std::uint8_t>(rgb).first(10), width, height, ycbcr, 4, 8),
           "rgb24 conversion rejects a truncated frame");
}

void quarter_reduction_test() {
    const auto input = to_half({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    std::vector<Half> output(2);
    expect(ops::reduce_channel_quarters(input, output, {1, 4, 1, 2}),
           "reduce_channel_quarters accepts input");
    expect(to_float(output) == std::vector<float>{16.0F, 20.0F},
           "quarter reduction matches channel chunk summation");
}

// The decoder must derive the same entropy indexes from the scales alone, and
// restore the encoder's reconstruction from the decoded symbols.
void encoder_decoder_symmetry_test() {
    constexpr ops::Shape4D four_way{1, 8, 3, 4};
    constexpr ops::Shape4D two_way{1, 4, 3, 4};
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> latent(-12.0F, 12.0F);
    std::uniform_real_distribution<float> scale(0.01F, 20.0F);
    for (const bool quarters : {true, false}) {
        const auto shape = quarters ? four_way : two_way;
        const auto count =
            static_cast<std::size_t>(shape.channels * shape.height * shape.width);
        const auto reduced = quarters ? count / 4 : count / 2;
        const auto stages = quarters ? 4 : 2;
        std::vector<Half> values(count), scales(count), means(count);
        for (std::size_t index = 0; index < count; ++index) {
            values[index] = ops::to_half(latent(generator));
            scales[index] = ops::to_half(scale(generator));
            means[index] = ops::to_half(latent(generator) * 0.25F);
        }
        for (std::int32_t stage = 0; stage < stages; ++stage) {
            std::vector<Half> mask(count), symbols_full(count), reconstructed(count);
            std::vector<Half> masked_scales(count), symbols(reduced), reduced_scales(reduced);
            std::vector<std::int16_t> encode_indexes(reduced);
            std::vector<std::uint8_t> decode_indexes(reduced);
            std::vector<Half> decoder_scales(reduced), restored(count);
            const bool prepared = quarters ?
                ops::make_four_way_mask(mask, shape, stage) &&
                    ops::process_with_mask(values, scales, means, mask, {}, symbols_full,
                                           reconstructed, masked_scales, 0.0F) &&
                    ops::reduce_channel_quarters(symbols_full, symbols, shape) &&
                    ops::reduce_channel_quarters(masked_scales, reduced_scales, shape) &&
                    ops::reduce_masked_quarters(scales, mask, decoder_scales, shape) :
                ops::make_two_way_mask(mask, shape, stage) &&
                    ops::process_with_mask(values, scales, means, mask, {}, symbols_full,
                                           reconstructed, masked_scales, 0.0F) &&
                    ops::reduce_channel_halves(symbols_full, symbols, shape) &&
                    ops::reduce_channel_halves(masked_scales, reduced_scales, shape) &&
                    ops::reduce_masked_halves(scales, mask, decoder_scales, shape);
            expect(prepared, "mask processing accepts input");
            expect(ops::build_encode_indexes(symbols, reduced_scales, encode_indexes, 0.0F) &&
                       ops::build_decode_indexes(decoder_scales, decode_indexes, 0.0F),
                   "index building accepts input");
            bool indexes_match = true;
            for (std::size_t index = 0; index < reduced; ++index) {
                if (static_cast<std::uint8_t>(encode_indexes[index] & 0xff) !=
                    decode_indexes[index]) {
                    indexes_match = false;
                }
            }
            expect(indexes_match, "decoder scale indexes match the encoder's");
            const bool restored_ok = quarters ?
                ops::restore_four_way(symbols, means, mask, restored, shape) :
                ops::restore_two_way(symbols, means, mask, restored, shape);
            expect(restored_ok, "symbol restoration accepts input");
            bool reconstruction_match = true;
            for (std::size_t index = 0; index < count; ++index) {
                if (ops::to_float(restored[index]) != ops::to_float(reconstructed[index])) {
                    reconstruction_match = false;
                }
            }
            expect(reconstruction_match, "decoder restoration matches encoder reconstruction");
        }
    }
}

void quant_step_test() {
    auto quant = to_half({0.25F, 2.0F});
    auto values = to_half({3.0F, 3.0F});
    expect(ops::clamp_reciprocal_with_quant(quant, values), "clamp_reciprocal accepts input");
    expect(to_float(quant) == std::vector<float>{0.5F, 2.0F}, "quant step is clamped to 0.5");
    expect(to_float(values) == std::vector<float>{6.0F, 1.5F}, "values are divided by the step");

    auto decoder_quant = to_half({0.25F, 2.0F});
    expect(ops::clamp_min(decoder_quant, 0.5F) && decoder_quant == quant,
           "decoder clamp matches the encoder's quant step");

    auto first = to_half({1.0F, 2.0F});
    const auto second = to_half({1.0F, 1.0F});
    expect(ops::add_and_multiply(first, second, quant) &&
               to_float(first) == std::vector<float>{1.0F, 6.0F},
           "add_and_multiply sums then scales");
}

}  // namespace

int main() {
    half_conversion_test();
    round_test();
    int8_to_half_test();
    padding_and_crop_test();
    image_prior_test();
    mask_test();
    ycbcr_to_yuv420_test();
    color_round_trip_test();
    quarter_reduction_test();
    encoder_decoder_symmetry_test();
    quant_step_test();
    if (failures == 0) std::cout << "DCVC-RT host operators passed\n";
    return failures == 0 ? 0 : 1;
}
