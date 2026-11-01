#include "cuda_ops.hpp"
#include "host_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expect_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        std::cerr << "FAIL: " << operation << ": " << cudaGetErrorString(status) << '\n';
        ++failures;
    }
}

struct DeviceBuffer final {
    explicit DeviceBuffer(std::size_t bytes) {
        expect_cuda(cudaMalloc(&data, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data != nullptr) static_cast<void>(cudaFree(data));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data{};
};

std::vector<__half> to_half(const std::vector<float>& values) {
    std::vector<__half> output;
    output.reserve(values.size());
    for (float value : values) output.push_back(__float2half(value));
    return output;
}

std::vector<float> to_float(const std::vector<__half>& values) {
    std::vector<float> output;
    output.reserve(values.size());
    for (__half value : values) output.push_back(__half2float(value));
    return output;
}

void round_test(cudaStream_t stream) {
    auto input = to_half({-129.2F, -1.6F, 0.49F, 126.6F, 200.0F});
    DeviceBuffer values(input.size() * sizeof(__half));
    DeviceBuffer quantized(input.size() * sizeof(std::int8_t));
    expect_cuda(cudaMemcpyAsync(
        values.data, input.data(), input.size() * sizeof(__half),
        cudaMemcpyHostToDevice, stream), "copy round input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::round_to_int8(
        values.data, static_cast<std::int8_t*>(quantized.data), input.size(), stream),
        "round_to_int8");
    std::array<std::int8_t, 5> actual{};
    expect_cuda(cudaMemcpyAsync(
        actual.data(), quantized.data, actual.size(), cudaMemcpyDeviceToHost, stream),
        "copy rounded symbols");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize round test");
    expect(actual == std::array<std::int8_t, 5>{-128, -2, 0, 127, 127},
           "rounding and int8 clamping match DCVC-RT");
}

void int8_to_half_test(cudaStream_t stream) {
    constexpr std::array<std::int8_t, 5> input{-128, -2, 0, 64, 127};
    DeviceBuffer values(input.size() * sizeof(std::int8_t));
    DeviceBuffer converted(input.size() * sizeof(__half));
    expect_cuda(cudaMemcpyAsync(
        values.data, input.data(), input.size() * sizeof(std::int8_t),
        cudaMemcpyHostToDevice, stream), "copy int8 input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::int8_to_half(
        static_cast<const std::int8_t*>(values.data), converted.data, input.size(), stream),
        "int8_to_half");
    std::vector<__half> output(input.size());
    expect_cuda(cudaMemcpyAsync(
        output.data(), converted.data, output.size() * sizeof(__half),
        cudaMemcpyDeviceToHost, stream), "copy converted symbols");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize int8 conversion test");
    expect(to_float(output) == std::vector<float>{-128.0F, -2.0F, 0.0F, 64.0F, 127.0F},
           "int8 to half conversion preserves entropy symbols");
}

void padding_test(cudaStream_t stream) {
    auto input = to_half({1.0F, 2.0F, 3.0F, 4.0F});
    DeviceBuffer source(input.size() * sizeof(__half));
    DeviceBuffer padded(12 * sizeof(__half));
    expect_cuda(cudaMemcpyAsync(
        source.data, input.data(), input.size() * sizeof(__half),
        cudaMemcpyHostToDevice, stream), "copy padding input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::replicate_pad(
        source.data, padded.data, {1, 1, 2, 2}, 3, 4, stream), "replicate_pad");
    std::vector<__half> output(12);
    expect_cuda(cudaMemcpyAsync(
        output.data(), padded.data, output.size() * sizeof(__half),
        cudaMemcpyDeviceToHost, stream), "copy padded output");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize padding test");
    expect(to_float(output) == std::vector<float>{
        1.0F, 2.0F, 2.0F, 2.0F,
        3.0F, 4.0F, 4.0F, 4.0F,
        3.0F, 4.0F, 4.0F, 4.0F},
        "replicate padding matches upstream semantics");
}

void image_prior_test(cudaStream_t stream) {
    auto input = to_half({0.0F, 0.0F, 0.0F, 0.0F});
    DeviceBuffer params(input.size() * sizeof(__half));
    expect_cuda(cudaMemcpyAsync(
        params.data, input.data(), input.size() * sizeof(__half),
        cudaMemcpyHostToDevice, stream), "copy image prior input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::apply_image_prior(
        params.data, 2, stream), "apply_image_prior");
    std::vector<__half> output(input.size());
    expect_cuda(cudaMemcpyAsync(
        output.data(), params.data, output.size() * sizeof(__half),
        cudaMemcpyDeviceToHost, stream), "copy image prior output");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize image prior test");
    expect(to_float(output) == std::vector<float>{1.25F, 1.25F, 1.25F, 1.25F},
           "image prior transforms q-encode and q-decode logits");
}

void mask_test(cudaStream_t stream) {
    constexpr nvcr::dcvcrt::cuda_ops::Shape4D shape{1, 4, 2, 2};
    DeviceBuffer device_mask(16 * sizeof(__half));
    constexpr std::int32_t pattern[4][4] = {
        {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2}};
    for (std::int32_t stage = 0; stage < 4; ++stage) {
        expect_cuda(nvcr::dcvcrt::cuda_ops::make_four_way_mask(
            device_mask.data, shape, stage, stream), "make_four_way_mask");
        std::vector<__half> mask(16);
        expect_cuda(cudaMemcpyAsync(
            mask.data(), device_mask.data, mask.size() * sizeof(__half),
            cudaMemcpyDeviceToHost, stream), "copy mask");
        expect_cuda(cudaStreamSynchronize(stream), "synchronize mask test");
        const auto values = to_float(mask);
        for (std::int32_t channel = 0; channel < 4; ++channel) {
            for (std::int32_t pixel = 0; pixel < 4; ++pixel) {
                const float expected = pattern[stage][channel] == pixel ? 1.0F : 0.0F;
                expect(values[static_cast<std::size_t>(channel * 4 + pixel)] == expected,
                       "four-way mask pattern matches upstream");
            }
        }
    }
}

std::vector<std::uint8_t> reference_ycbcr_to_yuv420p8(
    const std::vector<__half>& values,
    std::int32_t padded_height,
    std::int32_t padded_width,
    std::int32_t visible_width,
    std::int32_t visible_height) {
    const auto y_size = static_cast<std::size_t>(visible_width) * visible_height;
    const auto uv_width = visible_width / 2;
    std::vector<std::uint8_t> output(y_size + y_size / 2);
    const auto plane = static_cast<std::size_t>(padded_height) * padded_width;
    const auto offset = [&](std::int32_t channel, std::int32_t y, std::int32_t x) {
        return static_cast<std::size_t>(channel) * plane +
            static_cast<std::size_t>(y) * padded_width + x;
    };
    const auto quantize_y = [](float value) {
        return static_cast<std::uint8_t>(
            std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
    };
    const auto quantize_uv = [](float value) {
        return static_cast<std::uint8_t>(std::clamp(value * 255.0F, 0.0F, 255.0F));
    };
    for (std::int32_t y = 0; y < visible_height; ++y) {
        for (std::int32_t x = 0; x < visible_width; ++x) {
            output[static_cast<std::size_t>(y) * visible_width + x] =
                quantize_y(__half2float(values[offset(0, y, x)]));
        }
    }
    for (std::int32_t y = 0; y < visible_height; y += 2) {
        for (std::int32_t x = 0; x < visible_width; x += 2) {
            float cb_sum = 0.0F;
            float cr_sum = 0.0F;
            for (std::int32_t dy = 0; dy < 2; ++dy) {
                for (std::int32_t dx = 0; dx < 2; ++dx) {
                    cb_sum += __half2float(values[offset(1, y + dy, x + dx)]);
                    cr_sum += __half2float(values[offset(2, y + dy, x + dx)]);
                }
            }
            const auto chroma = static_cast<std::size_t>(y / 2) * uv_width + x / 2;
            output[y_size + chroma] = quantize_uv(cb_sum * 0.25F);
            output[y_size + y_size / 4 + chroma] = quantize_uv(cr_sum * 0.25F);
        }
    }
    return output;
}

void ycbcr_to_yuv420_test(cudaStream_t stream) {
    constexpr std::int32_t visible_width = 4;
    constexpr std::int32_t visible_height = 2;
    constexpr std::int32_t padded_width = 6;
    constexpr std::int32_t padded_height = 4;
    constexpr std::size_t plane = static_cast<std::size_t>(padded_width) * padded_height;
    auto input = to_half(std::vector<float>(3 * plane, 0.875F));
    const auto offset = [](std::int32_t channel, std::int32_t y, std::int32_t x) {
        return static_cast<std::size_t>(channel) * plane +
            static_cast<std::size_t>(y) * padded_width + x;
    };

    const auto y_values = to_half({
        -0.25F, 0.0F, 0.49F / 255.0F, 0.5F / 255.0F,
        127.49F / 255.0F, 127.5F / 255.0F, 1.0F, 1.25F});
    for (std::int32_t y = 0; y < visible_height; ++y) {
        for (std::int32_t x = 0; x < visible_width; ++x) {
            input[offset(0, y, x)] = y_values[static_cast<std::size_t>(y) * visible_width + x];
        }
    }

    const auto u_values = to_half({-1.0F, 0.5F, 1.0F, 2.0F, 1.5F, 0.5F, 1.0F, 1.0F});
    const auto v_values = to_half({0.25F, 0.25F, -1.0F, -1.0F, 0.25F, 0.25F, -1.0F, -1.0F});
    for (std::int32_t y = 0; y < visible_height; ++y) {
        for (std::int32_t x = 0; x < visible_width; ++x) {
            const auto source = static_cast<std::size_t>(y) * visible_width + x;
            input[offset(1, y, x)] = u_values[source];
            input[offset(2, y, x)] = v_values[source];
        }
    }

    const auto expected = reference_ycbcr_to_yuv420p8(
        input, padded_height, padded_width, visible_width, visible_height);
    DeviceBuffer source(input.size() * sizeof(__half));
    DeviceBuffer output(expected.size());
    expect_cuda(cudaMemcpyAsync(
        source.data, input.data(), input.size() * sizeof(__half),
        cudaMemcpyHostToDevice, stream), "copy YCbCr input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::ycbcr_padded_to_yuv420p8(
        source.data, {1, 3, padded_height, padded_width}, visible_width, visible_height,
        static_cast<std::uint8_t*>(output.data), stream), "ycbcr_padded_to_yuv420p8");
    std::vector<std::uint8_t> actual(expected.size());
    expect_cuda(cudaMemcpyAsync(
        actual.data(), output.data, actual.size(), cudaMemcpyDeviceToHost, stream),
        "copy YUV420P8 output");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize YCbCr output test");
    expect(actual == expected,
           "device YCbCr to YUV420P8 matches CPU rounding, clamping, and 2x2 chroma averaging");
}

void quarter_reduction_test(cudaStream_t stream) {
    auto input = to_half({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F});
    DeviceBuffer source(input.size() * sizeof(__half));
    DeviceBuffer reduced(2 * sizeof(__half));
    expect_cuda(cudaMemcpyAsync(
        source.data, input.data(), input.size() * sizeof(__half),
        cudaMemcpyHostToDevice, stream), "copy reduction input");
    expect_cuda(nvcr::dcvcrt::cuda_ops::reduce_channel_quarters(
        source.data, reduced.data, {1, 4, 1, 2}, stream), "reduce_channel_quarters");
    std::vector<__half> output(2);
    expect_cuda(cudaMemcpyAsync(
        output.data(), reduced.data, output.size() * sizeof(__half),
        cudaMemcpyDeviceToHost, stream), "copy reduction output");
    expect_cuda(cudaStreamSynchronize(stream), "synchronize reduction test");
    expect(to_float(output) == std::vector<float>{16.0F, 20.0F},
           "quarter reduction matches channel chunk summation");
}

// Host transforms replace these kernels for providers without CUDA. Kernels
// built only from IEEE arithmetic must match bit for bit; kernels that call
// expf/logf or that nvcc may contract into FMA report their mismatch count and
// must stay within a 1% bound.
namespace host = nvcr::dcvcrt::host_ops;

template <typename T>
void upload(DeviceBuffer& buffer, const std::vector<T>& values, cudaStream_t stream) {
    expect_cuda(cudaMemcpyAsync(
        buffer.data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream),
        "parity upload");
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count, cudaStream_t stream) {
    std::vector<T> values(count);
    expect_cuda(cudaMemcpyAsync(
        values.data(), buffer.data, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
        "parity download");
    expect_cuda(cudaStreamSynchronize(stream), "parity synchronize");
    return values;
}

std::vector<host::Half> random_halves(
    std::size_t count, float low, float high, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<host::Half> values(count);
    for (auto& value : values) value = host::to_half(distribution(generator));
    return values;
}

template <typename T>
void compare(
    std::string_view name, const std::vector<T>& device, const std::vector<T>& host_values,
    bool exact) {
    std::size_t different = 0;
    for (std::size_t index = 0; index < device.size(); ++index) {
        if (device[index] != host_values[index]) ++different;
    }
    std::cout << "host parity " << name << ": " << different << '/' << device.size()
              << " differ\n";
    expect(exact ? different == 0 : different * 100 <= device.size(),
           exact ? "host transform matches the CUDA kernel bit for bit" :
                   "host transform stays within the 1% transcendental/FMA bound");
}

void host_parity_test(cudaStream_t stream) {
    namespace cuda = nvcr::dcvcrt::cuda_ops;
    std::mt19937 generator(2026);
    constexpr cuda::Shape4D shape{1, 256, 8, 8};
    constexpr host::Shape4D host_shape{1, 256, 8, 8};
    constexpr std::size_t count = 256 * 8 * 8;
    constexpr std::size_t spatial = 8 * 8;
    constexpr std::size_t bytes = count * sizeof(host::Half);

    {
        auto values = random_halves(count, -200.0F, 200.0F, generator);
        DeviceBuffer device_values(bytes);
        DeviceBuffer device_symbols(count);
        upload(device_values, values, stream);
        expect_cuda(cuda::round_to_int8(
            device_values.data, static_cast<std::int8_t*>(device_symbols.data), count, stream),
            "parity round_to_int8");
        std::vector<std::int8_t> symbols(count);
        expect(host::round_to_int8(values, symbols), "host round_to_int8");
        compare("round_to_int8 symbols", download<std::int8_t>(device_symbols, count, stream),
                symbols, true);
        compare("round_to_int8 values", download<host::Half>(device_values, count, stream),
                values, true);
    }

    const auto values = random_halves(count, -20.0F, 20.0F, generator);
    const auto scales = random_halves(count, 0.0F, 24.0F, generator);
    const auto means = random_halves(count, -4.0F, 4.0F, generator);
    DeviceBuffer device_values(bytes);
    DeviceBuffer device_scales(bytes);
    DeviceBuffer device_means(bytes);
    upload(device_values, values, stream);
    upload(device_scales, scales, stream);
    upload(device_means, means, stream);

    for (std::int32_t stage = 0; stage < 4; ++stage) {
        std::vector<host::Half> mask(count);
        std::vector<host::Half> symbols(count);
        std::vector<host::Half> reconstructed(count);
        std::vector<host::Half> masked_scales(count);
        std::vector<host::Half> reduced_symbols(count / 4);
        std::vector<host::Half> reduced_scales(count / 4);
        std::vector<host::Half> decoder_scales(count / 4);
        std::vector<std::int16_t> encode_indexes(count / 4);
        std::vector<std::uint8_t> decode_indexes(count / 4);
        expect(host::make_four_way_mask(mask, host_shape, stage) &&
                   host::process_with_mask(values, scales, means, mask, {}, symbols,
                                           reconstructed, masked_scales, 0.0F) &&
                   host::reduce_channel_quarters(symbols, reduced_symbols, host_shape) &&
                   host::reduce_channel_quarters(masked_scales, reduced_scales, host_shape) &&
                   host::reduce_masked_quarters(scales, mask, decoder_scales, host_shape) &&
                   host::build_encode_indexes(
                       reduced_symbols, reduced_scales, encode_indexes, 0.0F) &&
                   host::build_decode_indexes(decoder_scales, decode_indexes, 0.0F),
               "host mask pipeline");

        DeviceBuffer device_mask(bytes);
        DeviceBuffer device_symbols(bytes);
        DeviceBuffer device_reconstructed(bytes);
        DeviceBuffer device_masked_scales(bytes);
        DeviceBuffer device_reduced_symbols(bytes / 4);
        DeviceBuffer device_reduced_scales(bytes / 4);
        DeviceBuffer device_decoder_scales(bytes / 4);
        DeviceBuffer device_encode_indexes(count / 4 * sizeof(std::int16_t));
        DeviceBuffer device_decode_indexes(count / 4);
        expect_cuda(cuda::make_four_way_mask(device_mask.data, shape, stage, stream),
                    "parity mask");
        expect_cuda(cuda::process_with_mask(
            device_values.data, device_scales.data, device_means.data, device_mask.data,
            nullptr, device_symbols.data, device_reconstructed.data,
            device_masked_scales.data, count, 0.0F, stream), "parity process_with_mask");
        expect_cuda(cuda::reduce_channel_quarters(
            device_symbols.data, device_reduced_symbols.data, shape, stream),
            "parity reduce symbols");
        expect_cuda(cuda::reduce_channel_quarters(
            device_masked_scales.data, device_reduced_scales.data, shape, stream),
            "parity reduce scales");
        expect_cuda(cuda::reduce_masked_quarters(
            device_scales.data, device_mask.data, device_decoder_scales.data, shape, stream),
            "parity reduce masked scales");
        expect_cuda(cuda::build_encode_indexes(
            device_reduced_symbols.data, device_reduced_scales.data,
            static_cast<std::int16_t*>(device_encode_indexes.data), count / 4, 0.0F, stream),
            "parity encode indexes");
        expect_cuda(cuda::build_decode_indexes(
            device_decoder_scales.data, static_cast<std::uint8_t*>(device_decode_indexes.data),
            count / 4, 0.0F, stream), "parity decode indexes");

        compare("four_way_mask", download<host::Half>(device_mask, count, stream), mask, true);
        compare("process_with_mask symbols",
                download<host::Half>(device_symbols, count, stream), symbols, true);
        compare("process_with_mask reconstruction",
                download<host::Half>(device_reconstructed, count, stream), reconstructed, true);
        compare("reduce_masked_quarters",
                download<host::Half>(device_decoder_scales, count / 4, stream),
                decoder_scales, true);
        compare("build_encode_indexes",
                download<std::int16_t>(device_encode_indexes, count / 4, stream),
                encode_indexes, false);
        compare("build_decode_indexes",
                download<std::uint8_t>(device_decode_indexes, count / 4, stream),
                decode_indexes, false);
    }

    {
        auto params = random_halves(2 * spatial, -6.0F, 6.0F, generator);
        auto latent = values;
        DeviceBuffer device_params(params.size() * sizeof(host::Half));
        DeviceBuffer device_latent(bytes);
        upload(device_params, params, stream);
        upload(device_latent, latent, stream);
        expect_cuda(cuda::apply_image_prior_and_quantize(
            device_params.data, device_latent.data, spatial, stream), "parity image prior");
        expect(host::apply_image_prior_and_quantize(params, latent, spatial),
               "host image prior");
        compare("apply_image_prior_and_quantize params",
                download<host::Half>(device_params, params.size(), stream), params, false);
        compare("apply_image_prior_and_quantize latent",
                download<host::Half>(device_latent, count, stream), latent, false);
    }

    {
        auto quant = random_halves(count, 0.0F, 3.0F, generator);
        auto latent = values;
        auto first = means;
        DeviceBuffer device_quant(bytes);
        DeviceBuffer device_latent(bytes);
        DeviceBuffer device_first(bytes);
        upload(device_quant, quant, stream);
        upload(device_latent, latent, stream);
        upload(device_first, first, stream);
        expect_cuda(cuda::clamp_reciprocal_with_quant(
            device_quant.data, device_latent.data, count, stream), "parity clamp reciprocal");
        expect_cuda(cuda::add_and_multiply(
            device_first.data, device_values.data, device_quant.data, count, stream),
            "parity add_and_multiply");
        expect(host::clamp_reciprocal_with_quant(quant, latent) &&
                   host::add_and_multiply(first, values, quant),
               "host quant step ops");
        compare("clamp_reciprocal_with_quant",
                download<host::Half>(device_latent, count, stream), latent, true);
        compare("add_and_multiply", download<host::Half>(device_first, count, stream), first,
                true);
    }

    {
        constexpr std::int32_t width = 34;
        constexpr std::int32_t height = 18;
        constexpr std::int32_t padded_width = 48;
        constexpr std::int32_t padded_height = 32;
        constexpr std::size_t plane = static_cast<std::size_t>(padded_width) * padded_height;
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3U);
        std::vector<std::uint8_t> yuv(static_cast<std::size_t>(width) * height * 3U / 2U);
        for (auto& value : rgb) value = static_cast<std::uint8_t>(byte(generator));
        for (auto& value : yuv) value = static_cast<std::uint8_t>(byte(generator));
        std::vector<host::Half> from_rgb(3 * plane);
        std::vector<host::Half> from_yuv(3 * plane);
        std::vector<std::uint8_t> back(yuv.size());
        expect(host::rgb24_to_ycbcr_padded(
                   rgb, width, height, from_rgb, padded_height, padded_width) &&
                   host::yuv420p8_to_ycbcr_padded(
                       yuv, width, height, from_yuv, padded_height, padded_width) &&
                   host::ycbcr_padded_to_yuv420p8(
                       from_rgb, {1, 3, padded_height, padded_width}, width, height, back),
               "host color conversion");
        DeviceBuffer device_rgb(rgb.size());
        DeviceBuffer device_yuv(yuv.size());
        DeviceBuffer device_from_rgb(3 * plane * sizeof(host::Half));
        DeviceBuffer device_from_yuv(3 * plane * sizeof(host::Half));
        DeviceBuffer device_back(yuv.size());
        upload(device_rgb, rgb, stream);
        upload(device_yuv, yuv, stream);
        expect_cuda(cuda::rgb24_to_ycbcr_padded(
            device_rgb.data, width, height, device_from_rgb.data, padded_height,
            padded_width, stream), "parity rgb24 conversion");
        expect_cuda(cuda::yuv420p8_to_ycbcr_padded(
            device_yuv.data, width, height, device_from_yuv.data, padded_height,
            padded_width, stream), "parity yuv420p8 conversion");
        // Feed the host RGB result to both so the output conversion is isolated.
        upload(device_from_rgb, from_rgb, stream);
        expect_cuda(cuda::ycbcr_padded_to_yuv420p8(
            device_from_rgb.data, {1, 3, padded_height, padded_width}, width, height,
            static_cast<std::uint8_t*>(device_back.data), stream), "parity YUV420P8 output");
        compare("yuv420p8_to_ycbcr_padded",
                download<host::Half>(device_from_yuv, 3 * plane, stream), from_yuv, true);
        compare("ycbcr_padded_to_yuv420p8",
                download<std::uint8_t>(device_back, yuv.size(), stream), back, true);
        DeviceBuffer device_rgb_again(3 * plane * sizeof(host::Half));
        expect_cuda(cuda::rgb24_to_ycbcr_padded(
            device_rgb.data, width, height, device_rgb_again.data, padded_height,
            padded_width, stream), "parity rgb24 conversion repeat");
        compare("rgb24_to_ycbcr_padded",
                download<host::Half>(device_rgb_again, 3 * plane, stream), from_rgb, false);
    }
}

}  // namespace

int main() {
    cudaStream_t stream{};
    expect_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    if (stream != nullptr) {
        round_test(stream);
        int8_to_half_test(stream);
        padding_test(stream);
        image_prior_test(stream);
        mask_test(stream);
        ycbcr_to_yuv420_test(stream);
        quarter_reduction_test(stream);
        host_parity_test(stream);
        expect_cuda(cudaStreamDestroy(stream), "destroy stream");
    }
    if (failures == 0) std::cout << "Standalone DCVC-RT CUDA operators passed\n";
    return failures == 0 ? 0 : 1;
}
