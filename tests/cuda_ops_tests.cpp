#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
        expect_cuda(cudaStreamDestroy(stream), "destroy stream");
    }
    if (failures == 0) std::cout << "Standalone DCVC-RT CUDA operators passed\n";
    return failures == 0 ? 0 : 1;
}
