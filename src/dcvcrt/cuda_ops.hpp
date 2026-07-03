#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace nvcr::dcvcrt::cuda_ops {

struct Shape4D final {
    std::int32_t batch;
    std::int32_t channels;
    std::int32_t height;
    std::int32_t width;
};

cudaError_t round_to_int8(
    void* values,
    std::int8_t* quantized,
    std::size_t count,
    cudaStream_t stream) noexcept;

cudaError_t replicate_pad(
    const void* input,
    void* output,
    Shape4D input_shape,
    std::int32_t padded_height,
    std::int32_t padded_width,
    cudaStream_t stream) noexcept;

cudaError_t make_four_way_mask(
    void* mask,
    Shape4D shape,
    std::int32_t stage,
    cudaStream_t stream) noexcept;

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
    cudaStream_t stream) noexcept;

cudaError_t reduce_channel_quarters(
    const void* input,
    void* output,
    Shape4D input_shape,
    cudaStream_t stream) noexcept;

cudaError_t restore_four_way(
    const void* symbols,
    const void* means,
    const void* mask,
    void* output,
    Shape4D output_shape,
    cudaStream_t stream) noexcept;

cudaError_t make_two_way_mask(
    void* mask,
    Shape4D shape,
    std::int32_t stage,
    cudaStream_t stream) noexcept;

cudaError_t clamp_min(
    void* values,
    std::size_t count,
    float minimum,
    cudaStream_t stream) noexcept;

cudaError_t clamp_reciprocal_with_quant(
    void* quant_step,
    void* values,
    std::size_t count,
    cudaStream_t stream) noexcept;

cudaError_t reduce_channel_halves(
    const void* input,
    void* output,
    Shape4D input_shape,
    cudaStream_t stream) noexcept;

cudaError_t reduce_masked_halves(
    const void* input,
    const void* mask,
    void* output,
    Shape4D input_shape,
    cudaStream_t stream) noexcept;

cudaError_t restore_two_way(
    const void* symbols,
    const void* means,
    const void* mask,
    void* output,
    Shape4D output_shape,
    cudaStream_t stream) noexcept;

cudaError_t add_and_multiply(
    void* first,
    const void* second,
    const void* quant_step,
    std::size_t count,
    cudaStream_t stream) noexcept;

cudaError_t build_encode_indexes(
    const void* symbols,
    const void* scales,
    std::int16_t* output,
    std::size_t count,
    float force_zero_threshold,
    cudaStream_t stream) noexcept;

cudaError_t build_decode_indexes(
    const void* scales,
    std::uint8_t* output,
    std::size_t count,
    float force_zero_threshold,
    cudaStream_t stream) noexcept;

}  // namespace nvcr::dcvcrt::cuda_ops
