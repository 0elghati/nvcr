#pragma once

// DCVC-UF codec glue between TensorRT stages: masks, masked quantization,
// entropy index construction, skip compaction, padding, crops and shuffles.
// Kernels are pinned upstream device code (see glue_kernels.inc); these
// launchers replace upstream's at::Tensor wrappers with the same launch shapes.
//
// FP16 tensors are raw device pointers holding IEEE half bits. Latent and
// feature tensors are NHWC, which is TensorRT's hwc8 layout when C % 8 == 0 and
// upstream's channels-last layout; frame-space tensors may be NCHW. Kernels
// that load channel pairs require NHWC and reject other layouts.

#include <cuda_runtime_api.h>

#include <cstdint>

namespace nvcr::dcvcuf::glue {

enum class Layout : std::uint8_t { nhwc, nchw };

template <typename T>
struct Tensor final {
    T* data{};
    int channels{};
    int height{};
    int width{};
    Layout layout{Layout::nhwc};
};

using HalfTensor = Tensor<std::uint16_t>;
using ConstHalfTensor = Tensor<const std::uint16_t>;
using MaskTensor = Tensor<const bool>;

// Prefix-sum scratch for conditional_index_part1 / conditional_recover:
// COND_KERNEL_THREAD_NUM2 blocks of COND_KERNEL_THREAD_NUM1 counts.
constexpr int conditional_index_capacity = 1024 * 1024 * 8;
constexpr int conditional_scratch_ints = 1024 * 1024;

[[nodiscard]] cudaError_t add_and_multiply_broadcast(
    HalfTensor out, ConstHalfTensor x0, ConstHalfTensor x1, const std::uint16_t* q,
    cudaStream_t stream);
[[nodiscard]] cudaError_t add_and_multiply_with_clamp_min_inplace(
    HalfTensor x0, ConstHalfTensor x1, ConstHalfTensor q, cudaStream_t stream);
[[nodiscard]] cudaError_t build_index_dec(
    ConstHalfTensor scales, float skip_threshold, std::uint8_t* out, bool* condition,
    cudaStream_t stream);
[[nodiscard]] cudaError_t build_index_enc(
    ConstHalfTensor symbols, ConstHalfTensor scales, float skip_threshold, std::int16_t* out,
    bool* condition, cudaStream_t stream);
// element_size 2 (int16), 1 (uint8 or int8); total receives the kept count.
[[nodiscard]] cudaError_t conditional_index_part1(
    const void* x, int element_size, bool element_signed, const bool* mask, int count,
    void* out, int* total, int* scratch, cudaStream_t stream);
// out_half selects Half output, otherwise int8.
[[nodiscard]] cudaError_t conditional_recover(
    const std::int8_t* x, const bool* mask, const int* scratch, int count, bool out_half,
    void* out, cudaStream_t stream);
[[nodiscard]] cudaError_t divide_with_clamp_min_inplace(
    HalfTensor y, ConstHalfTensor q, cudaStream_t stream);
// out_half selects Half output, otherwise int16.
[[nodiscard]] cudaError_t int8_to_dtype(
    const std::int8_t* x, int count, bool out_half, void* out, cudaStream_t stream);
[[nodiscard]] cudaError_t multiply_with_broadcast(
    HalfTensor out, ConstHalfTensor a, const std::uint16_t* b, cudaStream_t stream);
[[nodiscard]] cudaError_t process_with_mask(
    HalfTensor y_q, HalfTensor y_hat, HalfTensor s_hat, ConstHalfTensor y,
    ConstHalfTensor scales, ConstHalfTensor means, MaskTensor mask, float force_zero_threshold,
    cudaStream_t stream);
[[nodiscard]] cudaError_t process_with_mask_no_scale(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, float force_zero_threshold, cudaStream_t stream);
[[nodiscard]] cudaError_t process_with_mask_no_scale_add_inplace(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, float force_zero_threshold, cudaStream_t stream);
[[nodiscard]] cudaError_t process_with_mask_no_scale_add_and_multiply_inplace(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, ConstHalfTensor quant, float force_zero_threshold,
    cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y_and_add_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y_and_add_multiply_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    ConstHalfTensor q, cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y_4x(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y_4x_and_add_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream);
[[nodiscard]] cudaError_t restore_y_4x_add_and_multiply_broadcast(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    ConstHalfTensor y_hat_so_far, const std::uint16_t* q, cudaStream_t stream);
// z is read in memory order; for an NHWC tensor that is upstream's symbol order.
[[nodiscard]] cudaError_t round_z(
    const std::uint16_t* z, int count, std::uint16_t* z_hat, std::int8_t* z_int8,
    cudaStream_t stream);
[[nodiscard]] cudaError_t single_part_for_reading_4x(
    HalfTensor out, ConstHalfTensor x, MaskTensor mask, cudaStream_t stream);
[[nodiscard]] cudaError_t single_part_for_writing_4x(
    HalfTensor out, ConstHalfTensor x, cudaStream_t stream);
[[nodiscard]] cudaError_t pad_and_unshuffle_8(
    HalfTensor out, ConstHalfTensor x, int pad_bottom, int pad_right, cudaStream_t stream);
[[nodiscard]] cudaError_t replicate_pad(
    HalfTensor out, ConstHalfTensor x, int pad_bottom, int pad_right, cudaStream_t stream);
// Copies the top-left out.height x out.width window of x (crop_hyper_params).
[[nodiscard]] cudaError_t slice(HalfTensor out, ConstHalfTensor x, cudaStream_t stream);
[[nodiscard]] cudaError_t pixel_shuffle_2(HalfTensor out, ConstHalfTensor x, cudaStream_t stream);
[[nodiscard]] cudaError_t pixel_shuffle_8(
    HalfTensor out, ConstHalfTensor x, bool clamp, cudaStream_t stream);
[[nodiscard]] cudaError_t pixel_unshuffle(
    HalfTensor out, ConstHalfTensor x, int downscale_factor, cudaStream_t stream);

}  // namespace nvcr::dcvcuf::glue
