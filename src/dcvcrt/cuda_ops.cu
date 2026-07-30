#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace nvcr::dcvcrt::cuda_ops {
namespace {

constexpr std::size_t threads_per_block = 256;
constexpr float scale_min = 0.11F;
constexpr float scale_max = 16.0F;
constexpr float log_scale_min = -2.2072749131897207F;
constexpr float log_step_recip = 25.516414672012925F;

std::size_t volume(Shape4D shape) {
    return static_cast<std::size_t>(shape.batch) *
        static_cast<std::size_t>(shape.channels) *
        static_cast<std::size_t>(shape.height) *
        static_cast<std::size_t>(shape.width);
}

std::size_t blocks(std::size_t count) {
    return (count + threads_per_block - 1) / threads_per_block;
}

__device__ float load_half(const __half* values, std::size_t index) {
    return __half2float(values[index]);
}

__device__ void store_half(__half* values, std::size_t index, float value) {
    values[index] = __float2half_rn(value);
}

__global__ void round_to_int8_kernel(
    __half* values, std::int8_t* quantized, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = roundf(load_half(values, index));
    value = fminf(127.0F, fmaxf(-128.0F, value));
    store_half(values, index, value);
    quantized[index] = static_cast<std::int8_t>(value);
}

__global__ void replicate_pad_kernel(
    const __half* input,
    __half* output,
    Shape4D shape,
    std::int32_t padded_height,
    std::int32_t padded_width,
    std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % padded_width);
    const auto row = index / static_cast<std::size_t>(padded_width);
    const auto y = static_cast<std::int32_t>(row % padded_height);
    const auto channel_batch = row / static_cast<std::size_t>(padded_height);
    const auto source_x = min(x, shape.width - 1);
    const auto source_y = min(y, shape.height - 1);
    const auto source =
        (channel_batch * static_cast<std::size_t>(shape.height) + source_y) *
            static_cast<std::size_t>(shape.width) + source_x;
    output[index] = input[source];
}

__global__ void crop_spatial_kernel(
    const __half* input,
    __half* output,
    Shape4D input_shape,
    std::int32_t output_height,
    std::int32_t output_width,
    std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % output_width);
    const auto row = index / static_cast<std::size_t>(output_width);
    const auto y = static_cast<std::int32_t>(row % output_height);
    const auto channel_batch = row / static_cast<std::size_t>(output_height);
    const auto source =
        (channel_batch * static_cast<std::size_t>(input_shape.height) + y) *
            static_cast<std::size_t>(input_shape.width) + x;
    output[index] = input[source];
}

__global__ void image_prior_quant_kernel(__half* params, __half* values, std::size_t spatial_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= spatial_count) return;
    const float q_enc_sigmoid = 1.0F / (1.0F + expf(-load_half(params, index)));
    const float q_dec_sigmoid = 1.0F / (1.0F + expf(-load_half(params, spatial_count + index)));
    const float q_enc = q_enc_sigmoid * 1.5F + 0.5F;
    const float q_dec = q_dec_sigmoid * 1.5F + 0.5F;
    store_half(params, index, q_enc);
    store_half(params, spatial_count + index, q_dec);
    for (std::size_t channel = 0; channel < 256; ++channel) {
        const auto offset = channel * spatial_count + index;
        store_half(values, offset, load_half(values, offset) * q_enc);
    }
}

__global__ void rgb24_to_ycbcr_padded_kernel(
    const std::uint8_t* input,
    std::int32_t source_width,
    std::int32_t source_height,
    __half* output,
    std::int32_t padded_height,
    std::int32_t padded_width,
    std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % static_cast<std::size_t>(padded_width));
    const auto y = static_cast<std::int32_t>(index / static_cast<std::size_t>(padded_width));
    const auto source_x = x < source_width ? x : source_width - 1;
    const auto source_y = y < source_height ? y : source_height - 1;
    const auto source =
        (static_cast<std::size_t>(source_y) * source_width + source_x) * 3U;

    constexpr float kr = 0.2126F;
    constexpr float kg = 0.7152F;
    constexpr float kb = 0.0722F;
    const float r = static_cast<float>(input[source]) / 255.0F;
    const float g = static_cast<float>(input[source + 1]) / 255.0F;
    const float b = static_cast<float>(input[source + 2]) / 255.0F;
    const float luma = kr * r + kg * g + kb * b;
    const float cb = 0.5F * (b - luma) / (1.0F - kb) + 0.5F;
    const float cr = 0.5F * (r - luma) / (1.0F - kr) + 0.5F;

    const auto plane = static_cast<std::size_t>(padded_height) * padded_width;
    const auto offset = static_cast<std::size_t>(y) * padded_width + x;
    output[offset] = __float2half_rn(luma);
    output[plane + offset] = __float2half_rn(cb);
    output[2 * plane + offset] = __float2half_rn(cr);
}

__global__ void yuv420p8_to_ycbcr_padded_kernel(
    const std::uint8_t* input,
    std::int32_t source_width,
    std::int32_t source_height,
    __half* output,
    std::int32_t padded_height,
    std::int32_t padded_width,
    std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % static_cast<std::size_t>(padded_width));
    const auto y = static_cast<std::int32_t>(index / static_cast<std::size_t>(padded_width));
    const auto source_x = x < source_width ? x : source_width - 1;
    const auto source_y = y < source_height ? y : source_height - 1;

    const auto y_size = static_cast<std::size_t>(source_width) * source_height;
    const auto uv_width = source_width / 2;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (source_height / 2);
    const auto y_index = static_cast<std::size_t>(source_y) * source_width + source_x;
    const auto uv_index = static_cast<std::size_t>(source_y / 2) * uv_width + source_x / 2;

    const auto* u_plane = input + y_size;
    const auto* v_plane = input + y_size + uv_size;
    const float luma = static_cast<float>(input[y_index]) / 255.0F;
    const float cb = static_cast<float>(u_plane[uv_index]) / 255.0F;
    const float cr = static_cast<float>(v_plane[uv_index]) / 255.0F;

    const auto plane = static_cast<std::size_t>(padded_height) * padded_width;
    const auto offset = static_cast<std::size_t>(y) * padded_width + x;
    output[offset] = __float2half_rn(luma);
    output[plane + offset] = __float2half_rn(cb);
    output[2 * plane + offset] = __float2half_rn(cr);
}

__global__ void make_four_way_mask_kernel(
    __half* mask, Shape4D shape, std::int32_t stage, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % shape.width);
    const auto row = index / static_cast<std::size_t>(shape.width);
    const auto y = static_cast<std::int32_t>(row % shape.height);
    const auto channel = static_cast<std::int32_t>(
        (row / static_cast<std::size_t>(shape.height)) % shape.channels);
    const auto quarter = channel / (shape.channels / 4);
    constexpr std::int32_t patterns[4][4] = {
        {0, 1, 2, 3},
        {3, 2, 1, 0},
        {2, 3, 0, 1},
        {1, 0, 3, 2},
    };
    const auto parity = (y & 1) * 2 + (x & 1);
    mask[index] = __float2half_rn(patterns[stage][quarter] == parity ? 1.0F : 0.0F);
}

__global__ void process_with_mask_kernel(
    const __half* values,
    const __half* scales,
    const __half* means,
    const __half* mask,
    __half* residual,
    __half* symbols,
    __half* reconstructed,
    __half* masked_scales,
    std::size_t count,
    float force_zero_threshold) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float mask_value = load_half(mask, index);
    const float scale = load_half(scales, index) * mask_value;
    const float mean = load_half(means, index) * mask_value;
    const float value_residual = (load_half(values, index) - mean) * mask_value;
    float symbol = roundf(value_residual);
    if (force_zero_threshold > 0.0F && scale <= force_zero_threshold) symbol = 0.0F;
    symbol = fminf(127.0F, fmaxf(-128.0F, symbol));
    if (residual != nullptr) store_half(residual, index, value_residual);
    store_half(symbols, index, symbol);
    store_half(reconstructed, index, symbol + mean);
    store_half(masked_scales, index, scale);
}

__global__ void reduce_channel_quarters_kernel(
    const __half* input, __half* output, std::size_t quarter_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= quarter_size) return;
    float value = 0.0F;
    for (std::size_t quarter = 0; quarter < 4; ++quarter) {
        value += load_half(input, index + quarter * quarter_size);
    }
    store_half(output, index, value);
}

__global__ void reduce_masked_quarters_kernel(
    const __half* input, const __half* mask, __half* output, std::size_t quarter_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= quarter_size) return;
    float value = 0.0F;
    for (std::size_t quarter = 0; quarter < 4; ++quarter) {
        const auto source = index + quarter * quarter_size;
        value += load_half(input, source) * load_half(mask, source);
    }
    store_half(output, index, value);
}

__global__ void restore_four_way_kernel(
    const __half* symbols,
    const __half* means,
    const __half* mask,
    __half* output,
    std::size_t quarter_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= quarter_size) return;
    const float symbol = load_half(symbols, index);
    for (std::size_t quarter = 0; quarter < 4; ++quarter) {
        const auto destination = index + quarter * quarter_size;
        const float value = (symbol + load_half(means, destination)) *
            load_half(mask, destination);
        store_half(output, destination, value);
    }
}

__global__ void make_two_way_mask_kernel(
    __half* mask, Shape4D shape, std::int32_t stage, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto x = static_cast<std::int32_t>(index % shape.width);
    const auto row = index / static_cast<std::size_t>(shape.width);
    const auto y = static_cast<std::int32_t>(row % shape.height);
    const auto channel = static_cast<std::int32_t>(
        (row / static_cast<std::size_t>(shape.height)) % shape.channels);
    const bool first_half = channel < shape.channels / 2;
    const bool diagonal = (y & 1) == (x & 1);
    const bool selected = stage == 0 ? first_half == diagonal : first_half != diagonal;
    mask[index] = __float2half_rn(selected ? 1.0F : 0.0F);
}

__global__ void clamp_min_kernel(__half* values, std::size_t count, float minimum) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    values[index] = __float2half_rn(fmaxf(load_half(values, index), minimum));
}

__global__ void clamp_reciprocal_with_quant_kernel(
    __half* quant_step, __half* values, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const __half quant = __float2half_rn(fmaxf(load_half(quant_step, index), 0.5F));
    quant_step[index] = quant;
    const __half reciprocal = __float2half_rn(1.0F / __half2float(quant));
    values[index] = __float2half_rn(
        load_half(values, index) * __half2float(reciprocal));
}

__global__ void reduce_channel_halves_kernel(
    const __half* input, __half* output, std::size_t half_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= half_size) return;
    store_half(output, index,
               load_half(input, index) + load_half(input, index + half_size));
}

__global__ void reduce_masked_halves_kernel(
    const __half* input, const __half* mask, __half* output, std::size_t half_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= half_size) return;
    const float first = load_half(input, index) * load_half(mask, index);
    const float second = load_half(input, index + half_size) *
        load_half(mask, index + half_size);
    store_half(output, index, first + second);
}

__global__ void restore_two_way_kernel(
    const __half* symbols, const __half* means, const __half* mask,
    __half* output, std::size_t half_size) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= half_size) return;
    const float symbol = load_half(symbols, index);
    for (std::size_t half = 0; half < 2; ++half) {
        const auto destination = index + half * half_size;
        store_half(output, destination,
                   (symbol + load_half(means, destination)) *
                       load_half(mask, destination));
    }
}

__global__ void add_in_place_kernel(
    __half* first, const __half* second, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    first[index] = __float2half_rn(load_half(first, index) + load_half(second, index));
}

__global__ void multiply_broadcast_kernel(
    __half* values, const __half* factors, std::size_t spatial_count,
    std::int32_t channels, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto spatial = index % spatial_count;
    const auto channel = static_cast<std::int32_t>(index / spatial_count);
    if (channel >= channels) return;
    values[index] = __float2half_rn(load_half(values, index) * load_half(factors, spatial));
}

__global__ void add_and_multiply_kernel(
    __half* first, const __half* second, const __half* quant_step, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const __half sum = __float2half_rn(
        load_half(first, index) + load_half(second, index));
    first[index] = __float2half_rn(
        __half2float(sum) * load_half(quant_step, index));
}

__device__ std::uint8_t scale_index(float scale) {
    scale = fminf(scale_max, fmaxf(scale_min, scale));
    const float index = (logf(scale) - log_scale_min) * log_step_recip;
    return static_cast<std::uint8_t>(index);
}

__global__ void build_encode_indexes_kernel(
    const __half* symbols,
    const __half* scales,
    std::int16_t* output,
    std::size_t count,
    float force_zero_threshold) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float scale = load_half(scales, index);
    if (force_zero_threshold > 0.0F && scale <= force_zero_threshold) {
        output[index] = 0;
        return;
    }
    const auto symbol = static_cast<std::int16_t>(load_half(symbols, index));
    output[index] = static_cast<std::int16_t>(
        static_cast<std::uint16_t>(symbol) << 8U | scale_index(scale));
}

__global__ void build_decode_indexes_kernel(
    const __half* scales,
    std::uint8_t* output,
    std::size_t count,
    float force_zero_threshold) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float scale = load_half(scales, index);
    output[index] = force_zero_threshold > 0.0F && scale <= force_zero_threshold
        ? 0
        : scale_index(scale);
}

bool invalid_shape(Shape4D shape) {
    return shape.batch <= 0 || shape.channels <= 0 || shape.height <= 0 || shape.width <= 0;
}

}  // namespace

cudaError_t round_to_int8(
    void* values,
    std::int8_t* quantized,
    std::size_t count,
    cudaStream_t stream) noexcept {
    if (values == nullptr || quantized == nullptr || count == 0) return cudaErrorInvalidValue;
    round_to_int8_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(values), quantized, count);
    return cudaPeekAtLastError();
}

cudaError_t replicate_pad(
    const void* input,
    void* output,
    Shape4D input_shape,
    std::int32_t padded_height,
    std::int32_t padded_width,
    cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || invalid_shape(input_shape) ||
        padded_height < input_shape.height || padded_width < input_shape.width) {
        return cudaErrorInvalidValue;
    }
    const auto count = static_cast<std::size_t>(input_shape.batch) * input_shape.channels *
        padded_height * padded_width;
    replicate_pad_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<__half*>(output), input_shape,
        padded_height, padded_width, count);
    return cudaPeekAtLastError();
}

cudaError_t make_four_way_mask(
    void* mask, Shape4D shape, std::int32_t stage, cudaStream_t stream) noexcept {
    if (mask == nullptr || invalid_shape(shape) || shape.channels % 4 != 0 ||
        stage < 0 || stage > 3) return cudaErrorInvalidValue;
    const auto count = volume(shape);
    make_four_way_mask_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(mask), shape, stage, count);
    return cudaPeekAtLastError();
}

cudaError_t crop_spatial(
    const void* input, void* output, Shape4D input_shape, std::int32_t output_height,
    std::int32_t output_width, cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || invalid_shape(input_shape) ||
        output_height <= 0 || output_width <= 0 || output_height > input_shape.height ||
        output_width > input_shape.width) return cudaErrorInvalidValue;
    const auto count = static_cast<std::size_t>(input_shape.batch) * input_shape.channels *
        output_height * output_width;
    crop_spatial_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<__half*>(output), input_shape,
        output_height, output_width, count);
    return cudaPeekAtLastError();
}

cudaError_t apply_image_prior_and_quantize(
    void* params, void* values, std::size_t spatial_count, cudaStream_t stream) noexcept {
    if (params == nullptr || values == nullptr || spatial_count == 0) return cudaErrorInvalidValue;
    image_prior_quant_kernel<<<blocks(spatial_count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(params), static_cast<__half*>(values), spatial_count);
    return cudaPeekAtLastError();
}

cudaError_t rgb24_to_ycbcr_padded(
    const void* input,
    std::int32_t source_width,
    std::int32_t source_height,
    void* output,
    std::int32_t padded_height,
    std::int32_t padded_width,
    cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || source_width <= 0 || source_height <= 0 ||
        padded_height < source_height || padded_width < source_width) {
        return cudaErrorInvalidValue;
    }
    const auto count = static_cast<std::size_t>(padded_height) * padded_width;
    rgb24_to_ycbcr_padded_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const std::uint8_t*>(input), source_width, source_height,
        static_cast<__half*>(output), padded_height, padded_width, count);
    return cudaPeekAtLastError();
}

cudaError_t yuv420p8_to_ycbcr_padded(
    const void* input,
    std::int32_t source_width,
    std::int32_t source_height,
    void* output,
    std::int32_t padded_height,
    std::int32_t padded_width,
    cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || source_width <= 0 || source_height <= 0 ||
        (source_width & 1) != 0 || (source_height & 1) != 0 ||
        padded_height < source_height || padded_width < source_width) {
        return cudaErrorInvalidValue;
    }
    const auto count = static_cast<std::size_t>(padded_height) * padded_width;
    yuv420p8_to_ycbcr_padded_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const std::uint8_t*>(input), source_width, source_height,
        static_cast<__half*>(output), padded_height, padded_width, count);
    return cudaPeekAtLastError();
}

cudaError_t process_with_mask(
    const void* values,
    const void* scales,
    const void* means,
    const void* mask,
    void* residual,
    void* symbols,
    void* reconstructed,
    void* masked_scales,
    std::size_t count,
    float force_zero_threshold,
    cudaStream_t stream) noexcept {
    if (values == nullptr || scales == nullptr || means == nullptr || mask == nullptr ||
        symbols == nullptr || reconstructed == nullptr || masked_scales == nullptr ||
        count == 0) return cudaErrorInvalidValue;
    process_with_mask_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(values), static_cast<const __half*>(scales),
        static_cast<const __half*>(means), static_cast<const __half*>(mask),
        static_cast<__half*>(residual), static_cast<__half*>(symbols),
        static_cast<__half*>(reconstructed), static_cast<__half*>(masked_scales), count,
        force_zero_threshold);
    return cudaPeekAtLastError();
}

cudaError_t reduce_channel_quarters(
    const void* input, void* output, Shape4D input_shape, cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || invalid_shape(input_shape) ||
        input_shape.channels % 4 != 0) return cudaErrorInvalidValue;
    const auto quarter_size = volume(input_shape) / 4;
    reduce_channel_quarters_kernel<<<blocks(quarter_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<__half*>(output), quarter_size);
    return cudaPeekAtLastError();
}

cudaError_t reduce_masked_quarters(
    const void* input,
    const void* mask,
    void* output,
    Shape4D input_shape,
    cudaStream_t stream) noexcept {
    if (input == nullptr || mask == nullptr || output == nullptr ||
        invalid_shape(input_shape) || input_shape.channels % 4 != 0) {
        return cudaErrorInvalidValue;
    }
    const auto quarter_size = volume(input_shape) / 4;
    reduce_masked_quarters_kernel<<<blocks(quarter_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<const __half*>(mask),
        static_cast<__half*>(output), quarter_size);
    return cudaPeekAtLastError();
}

cudaError_t restore_four_way(
    const void* symbols,
    const void* means,
    const void* mask,
    void* output,
    Shape4D output_shape,
    cudaStream_t stream) noexcept {
    if (symbols == nullptr || means == nullptr || mask == nullptr || output == nullptr ||
        invalid_shape(output_shape) || output_shape.channels % 4 != 0) {
        return cudaErrorInvalidValue;
    }
    const auto quarter_size = volume(output_shape) / 4;
    restore_four_way_kernel<<<blocks(quarter_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(symbols), static_cast<const __half*>(means),
        static_cast<const __half*>(mask), static_cast<__half*>(output), quarter_size);
    return cudaPeekAtLastError();
}

cudaError_t make_two_way_mask(
    void* mask, Shape4D shape, std::int32_t stage, cudaStream_t stream) noexcept {
    if (mask == nullptr || invalid_shape(shape) || shape.channels % 2 != 0 ||
        stage < 0 || stage > 1) return cudaErrorInvalidValue;
    const auto count = volume(shape);
    make_two_way_mask_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(mask), shape, stage, count);
    return cudaPeekAtLastError();
}

cudaError_t clamp_min(
    void* values, std::size_t count, float minimum, cudaStream_t stream) noexcept {
    if (values == nullptr || count == 0) return cudaErrorInvalidValue;
    clamp_min_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(values), count, minimum);
    return cudaPeekAtLastError();
}

cudaError_t clamp_reciprocal_with_quant(
    void* quant_step, void* values, std::size_t count, cudaStream_t stream) noexcept {
    if (quant_step == nullptr || values == nullptr || count == 0) return cudaErrorInvalidValue;
    clamp_reciprocal_with_quant_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(quant_step), static_cast<__half*>(values), count);
    return cudaPeekAtLastError();
}

cudaError_t reduce_channel_halves(
    const void* input, void* output, Shape4D input_shape, cudaStream_t stream) noexcept {
    if (input == nullptr || output == nullptr || invalid_shape(input_shape) ||
        input_shape.channels % 2 != 0) return cudaErrorInvalidValue;
    const auto half_size = volume(input_shape) / 2;
    reduce_channel_halves_kernel<<<blocks(half_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<__half*>(output), half_size);
    return cudaPeekAtLastError();
}

cudaError_t reduce_masked_halves(
    const void* input, const void* mask, void* output, Shape4D input_shape,
    cudaStream_t stream) noexcept {
    if (input == nullptr || mask == nullptr || output == nullptr || invalid_shape(input_shape) ||
        input_shape.channels % 2 != 0) return cudaErrorInvalidValue;
    const auto half_size = volume(input_shape) / 2;
    reduce_masked_halves_kernel<<<blocks(half_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(input), static_cast<const __half*>(mask),
        static_cast<__half*>(output), half_size);
    return cudaPeekAtLastError();
}

cudaError_t restore_two_way(
    const void* symbols, const void* means, const void* mask, void* output,
    Shape4D output_shape, cudaStream_t stream) noexcept {
    if (symbols == nullptr || means == nullptr || mask == nullptr || output == nullptr ||
        invalid_shape(output_shape) || output_shape.channels % 2 != 0) {
        return cudaErrorInvalidValue;
    }
    const auto half_size = volume(output_shape) / 2;
    restore_two_way_kernel<<<blocks(half_size), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(symbols), static_cast<const __half*>(means),
        static_cast<const __half*>(mask), static_cast<__half*>(output), half_size);
    return cudaPeekAtLastError();
}

cudaError_t add_in_place(
    void* first, const void* second, std::size_t count, cudaStream_t stream) noexcept {
    if (first == nullptr || second == nullptr || count == 0) return cudaErrorInvalidValue;
    add_in_place_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(first), static_cast<const __half*>(second), count);
    return cudaPeekAtLastError();
}

cudaError_t multiply_broadcast_in_place(
    void* values, const void* factors, std::size_t spatial_count,
    std::int32_t channels, cudaStream_t stream) noexcept {
    if (values == nullptr || factors == nullptr || spatial_count == 0 || channels <= 0) {
        return cudaErrorInvalidValue;
    }
    const auto count = spatial_count * static_cast<std::size_t>(channels);
    multiply_broadcast_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(values), static_cast<const __half*>(factors),
        spatial_count, channels, count);
    return cudaPeekAtLastError();
}

cudaError_t add_and_multiply(
    void* first, const void* second, const void* quant_step, std::size_t count,
    cudaStream_t stream) noexcept {
    if (first == nullptr || second == nullptr || quant_step == nullptr || count == 0) {
        return cudaErrorInvalidValue;
    }
    add_and_multiply_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<__half*>(first), static_cast<const __half*>(second),
        static_cast<const __half*>(quant_step), count);
    return cudaPeekAtLastError();
}

cudaError_t build_encode_indexes(
    const void* symbols,
    const void* scales,
    std::int16_t* output,
    std::size_t count,
    float force_zero_threshold,
    cudaStream_t stream) noexcept {
    if (symbols == nullptr || scales == nullptr || output == nullptr || count == 0) {
        return cudaErrorInvalidValue;
    }
    build_encode_indexes_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(symbols), static_cast<const __half*>(scales), output,
        count, force_zero_threshold);
    return cudaPeekAtLastError();
}

cudaError_t build_decode_indexes(
    const void* scales,
    std::uint8_t* output,
    std::size_t count,
    float force_zero_threshold,
    cudaStream_t stream) noexcept {
    if (scales == nullptr || output == nullptr || count == 0) return cudaErrorInvalidValue;
    build_decode_indexes_kernel<<<blocks(count), threads_per_block, 0, stream>>>(
        static_cast<const __half*>(scales), output, count, force_zero_threshold);
    return cudaPeekAtLastError();
}

}  // namespace nvcr::dcvcrt::cuda_ops
