#include "glue.hpp"

#include "glue_types.cuh"

namespace nvcr::dcvcuf::glue {

// Upstream kernels see CUDA's integer min/max alongside the Half overloads;
// without these, integer index arithmetic would resolve to the Half versions.
using ::max;
using ::min;

#include "glue_kernels.inc"

namespace {

template <typename T, typename U>
GPUTensor4D<T> accessor(const Tensor<U>& tensor) {
    const int C = tensor.channels;
    const int H = tensor.height;
    const int W = tensor.width;
    GPUTensor4D<T> view{};
    view.data = const_cast<T*>(reinterpret_cast<const T*>(tensor.data));
    view.sizes[0] = 1;
    view.sizes[1] = C;
    view.sizes[2] = H;
    view.sizes[3] = W;
    if (tensor.layout == Layout::nhwc) {
        const int P = tensor.stride != 0 ? tensor.stride : C;
        view.strides[0] = H * W * P;
        view.strides[1] = 1;
        view.strides[2] = W * P;
        view.strides[3] = P;
    } else {
        view.strides[0] = C * H * W;
        view.strides[1] = H * W;
        view.strides[2] = W;
        view.strides[3] = 1;
    }
    return view;
}

GPUTensor4D<Half> half4(const HalfTensor& tensor) { return accessor<Half>(tensor); }
GPUTensor4D<Half> half4(const ConstHalfTensor& tensor) { return accessor<Half>(tensor); }
GPUTensor4D<bool> mask4(const MaskTensor& tensor) { return accessor<bool>(tensor); }

template <typename T>
GPUTensor1D<T> flat(const void* data) {
    return GPUTensor1D<T>{const_cast<T*>(static_cast<const T*>(data))};
}

GPUTensor1D<Half> half1(const std::uint16_t* data) { return flat<Half>(data); }

// Upstream get_kernel_config_4D<factor, BX, BY, BZ>.
template <int factor = 1, int BX = 64, int BY = 2, int BZ = 2>
std::pair<dim3, dim3> config_4d(int C, int H, int W, CudaCHW& chw) {
    chw = {C / factor, H, W};
    return {dim3((C / factor + BX - 1) / BX, (W + BY - 1) / BY, (H + BZ - 1) / BZ),
            dim3(BX, BY, BZ)};
}

// Upstream get_kernel_config_1D<factor>.
template <int factor = 1>
std::pair<dim3, dim3> config_1d(int count, int& n) {
    n = count / factor;
    return {dim3((n + 255) / 256), dim3(256)};
}

template <typename T>
bool paired(const Tensor<T>& tensor) {
    return tensor.layout == Layout::nhwc && tensor.channels % 2 == 0;
}

cudaError_t invalid() { return cudaErrorInvalidValue; }

}  // namespace

cudaError_t add_and_multiply_broadcast(
    HalfTensor out, ConstHalfTensor x0, ConstHalfTensor x1, const std::uint16_t* q,
    cudaStream_t stream) {
    if (!paired(out) || !paired(x0) || !paired(x1)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(x0.channels, x0.height, x0.width, chw);
    add_and_multiply_broadcast_kernel<<<grid, block, 0, stream>>>(
        half4(out), half4(x0), half4(x1), half1(q), chw);
    return cudaGetLastError();
}

cudaError_t add_and_multiply_with_clamp_min_inplace(
    HalfTensor x0, ConstHalfTensor x1, ConstHalfTensor q, cudaStream_t stream) {
    if (!paired(x0) || !paired(x1) || !paired(q)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(x0.channels, x0.height, x0.width, chw);
    add_and_multiply_with_clamp_min_inplace_kernel<<<grid, block, 0, stream>>>(
        half4(x0), half4(x1), half4(q), chw);
    return cudaGetLastError();
}

cudaError_t build_index_dec(
    ConstHalfTensor scales, float skip_threshold, std::uint8_t* out, bool* condition,
    cudaStream_t stream) {
    if (!paired(scales)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(scales.channels, scales.height, scales.width, chw);
    build_index_dec_kernel<<<grid, block, 0, stream>>>(
        flat<std::uint8_t>(out), flat<bool>(condition), half4(scales), Half(skip_threshold),
        chw);
    return cudaGetLastError();
}

cudaError_t build_index_enc(
    ConstHalfTensor symbols, ConstHalfTensor scales, float skip_threshold, std::int16_t* out,
    bool* condition, cudaStream_t stream) {
    if (!paired(symbols) || !paired(scales)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(scales.channels, scales.height, scales.width, chw);
    build_index_enc_kernel<<<grid, block, 0, stream>>>(
        flat<std::int16_t>(out), flat<bool>(condition), half4(symbols), half4(scales),
        Half(skip_threshold), chw);
    return cudaGetLastError();
}

cudaError_t conditional_index_part1(
    const void* x, int element_size, bool element_signed, const bool* mask, int count,
    void* out, int* total, int* scratch, cudaStream_t stream) {
    constexpr int per_block = COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM;
    const int block_num = (count + per_block - 1) / per_block;
    if (block_num > COND_KERNEL_THREAD_NUM2 || count <= 0) return invalid();
    GPUTensor2D<int> num{};
    num.data = scratch;
    num.sizes[0] = block_num;
    num.sizes[1] = COND_KERNEL_THREAD_NUM1;
    num.strides[0] = COND_KERNEL_THREAD_NUM1;
    num.strides[1] = 1;
    conditional_index_step1_kernel<COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>
        <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(num, flat<bool>(mask), count);
    conditional_index_step2_kernel<COND_KERNEL_THREAD_NUM1, COND_KERNEL_THREAD_NUM2>
        <<<1, COND_KERNEL_THREAD_NUM2, 0, stream>>>(num, flat<int>(total), block_num);
    if (element_size == 2) {
        conditional_index_step3_kernel<int16_t, COND_KERNEL_THREAD_NUM1,
                                       COND_KERNEL_PER_THREAD_NUM>
            <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(
                flat<int16_t>(out), num, flat<int16_t>(x), flat<bool>(mask), count);
    } else if (element_size == 1 && !element_signed) {
        conditional_index_step3_kernel<uint8_t, COND_KERNEL_THREAD_NUM1,
                                       COND_KERNEL_PER_THREAD_NUM>
            <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(
                flat<uint8_t>(out), num, flat<uint8_t>(x), flat<bool>(mask), count);
    } else if (element_size == 1) {
        conditional_index_step3_kernel<int8_t, COND_KERNEL_THREAD_NUM1,
                                       COND_KERNEL_PER_THREAD_NUM>
            <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(
                flat<int8_t>(out), num, flat<int8_t>(x), flat<bool>(mask), count);
    } else {
        return invalid();
    }
    return cudaGetLastError();
}

cudaError_t conditional_recover(
    const std::int8_t* x, const bool* mask, const int* scratch, int count, bool out_half,
    void* out, cudaStream_t stream) {
    constexpr int per_block = COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM;
    const int block_num = (count + per_block - 1) / per_block;
    if (block_num > COND_KERNEL_THREAD_NUM2 || count <= 0) return invalid();
    GPUTensor2D<int> num{};
    num.data = const_cast<int*>(scratch);
    num.sizes[0] = block_num;
    num.sizes[1] = COND_KERNEL_THREAD_NUM1;
    num.strides[0] = COND_KERNEL_THREAD_NUM1;
    num.strides[1] = 1;
    if (out_half) {
        conditional_recover_with_type_conversion_kernel<Half, COND_KERNEL_THREAD_NUM1,
                                                        COND_KERNEL_PER_THREAD_NUM>
            <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(
                flat<Half>(out), num, flat<int8_t>(x), flat<bool>(mask), count);
    } else {
        conditional_recover_with_type_conversion_kernel<int8_t, COND_KERNEL_THREAD_NUM1,
                                                        COND_KERNEL_PER_THREAD_NUM>
            <<<block_num, COND_KERNEL_THREAD_NUM1, 0, stream>>>(
                flat<int8_t>(out), num, flat<int8_t>(x), flat<bool>(mask), count);
    }
    return cudaGetLastError();
}

cudaError_t divide_with_clamp_min_inplace(HalfTensor y, ConstHalfTensor q, cudaStream_t stream) {
    if (!paired(y) || !paired(q)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(y.channels, y.height, y.width, chw);
    divide_with_clamp_inplace_kernel<<<grid, block, 0, stream>>>(half4(y), half4(q), chw);
    return cudaGetLastError();
}

cudaError_t int8_to_dtype(
    const std::int8_t* x, int count, bool out_half, void* out, cudaStream_t stream) {
    int n = 0;
    auto [grid, block] = config_1d<2>(count, n);
    if (out_half) {
        int8_to_dtype_kernel<Half><<<grid, block, 0, stream>>>(flat<Half>(out), flat<int8_t>(x), n);
    } else {
        int8_to_dtype_kernel<int16_t><<<grid, block, 0, stream>>>(
            flat<int16_t>(out), flat<int8_t>(x), n);
    }
    return cudaGetLastError();
}

cudaError_t multiply_with_broadcast(
    HalfTensor out, ConstHalfTensor a, const std::uint16_t* b, cudaStream_t stream) {
    // Upstream indexes [0][c][0][n] over H*W, so a and out must be NCHW-contiguous
    // per channel; this mirrors its use on 1x1 or channel-major tensors.
    const int C = a.channels;
    int N = a.height * a.width;
    int factor = 1;
    if (N % 4 == 0) {
        factor = 4;
        N /= 4;
    } else if (N % 2 == 0) {
        factor = 2;
        N /= 2;
    }
    auto flat_hw = [](auto tensor) {
        auto view = half4(tensor);
        // Collapse H and W into one axis as upstream's [0][c][0][n] access assumes.
        view.sizes[2] = 1;
        view.sizes[3] = tensor.height * tensor.width;
        if (tensor.layout == Layout::nhwc) {
            view.strides[3] = tensor.channels;
        } else {
            view.strides[3] = 1;
        }
        return view;
    };
    const dim3 block(256, 1);
    const dim3 grid((C + 255) / 256, N);
    if (factor == 4) {
        multiply_with_broadcast_kernel<4><<<grid, block, 0, stream>>>(
            flat_hw(out), flat_hw(a), half1(b), C, N);
    } else if (factor == 2) {
        multiply_with_broadcast_kernel<2><<<grid, block, 0, stream>>>(
            flat_hw(out), flat_hw(a), half1(b), C, N);
    } else {
        multiply_with_broadcast_kernel<1><<<grid, block, 0, stream>>>(
            flat_hw(out), flat_hw(a), half1(b), C, N);
    }
    return cudaGetLastError();
}

namespace {

template <bool scale_out, bool add_inplace, bool multiply>
cudaError_t launch_process(
    HalfTensor y_q, HalfTensor y_hat, GPUTensor4D<Half> s_hat, ConstHalfTensor y,
    ConstHalfTensor scales, ConstHalfTensor means, MaskTensor mask, GPUTensor4D<Half> quant,
    float force_zero_threshold, cudaStream_t stream) {
    if (!paired(y_q) || !paired(y_hat) || !paired(y) || !paired(scales) || !paired(means) ||
        !paired(mask)) {
        return invalid();
    }
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(y.channels, y.height, y.width, chw);
    process_with_mask_kernel<scale_out, add_inplace, multiply><<<grid, block, 0, stream>>>(
        half4(y_q), half4(y_hat), s_hat, half4(y), half4(scales), half4(means), mask4(mask),
        quant, Half(force_zero_threshold), chw);
    return cudaGetLastError();
}

GPUTensor4D<Half> null_half4() { return GPUTensor4D<Half>{}; }

}  // namespace

cudaError_t process_with_mask(
    HalfTensor y_q, HalfTensor y_hat, HalfTensor s_hat, ConstHalfTensor y,
    ConstHalfTensor scales, ConstHalfTensor means, MaskTensor mask, float force_zero_threshold,
    cudaStream_t stream) {
    if (!paired(s_hat)) return invalid();
    return launch_process<true, false, false>(
        y_q, y_hat, half4(s_hat), y, scales, means, mask, null_half4(), force_zero_threshold,
        stream);
}

cudaError_t process_with_mask_no_scale(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, float force_zero_threshold, cudaStream_t stream) {
    return launch_process<false, false, false>(
        y_q, y_hat, null_half4(), y, scales, means, mask, null_half4(), force_zero_threshold,
        stream);
}

cudaError_t process_with_mask_no_scale_add_inplace(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, float force_zero_threshold, cudaStream_t stream) {
    return launch_process<false, true, false>(
        y_q, y_hat, null_half4(), y, scales, means, mask, null_half4(), force_zero_threshold,
        stream);
}

cudaError_t process_with_mask_no_scale_add_and_multiply_inplace(
    HalfTensor y_q, HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor scales,
    ConstHalfTensor means, MaskTensor mask, ConstHalfTensor quant, float force_zero_threshold,
    cudaStream_t stream) {
    if (!paired(quant)) return invalid();
    return launch_process<false, true, true>(
        y_q, y_hat, null_half4(), y, scales, means, mask, half4(quant), force_zero_threshold,
        stream);
}

namespace {

template <bool add_inplace, bool multiply>
cudaError_t launch_restore(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    GPUTensor4D<Half> q, cudaStream_t stream) {
    if (!paired(out) || !paired(y) || !paired(means)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(y.channels, y.height, y.width, chw);
    restore_y_kernel<add_inplace, multiply><<<grid, block, 0, stream>>>(
        half4(out), half4(y), half4(means), mask4(mask), q, chw);
    return cudaGetLastError();
}

}  // namespace

cudaError_t restore_y(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream) {
    return launch_restore<false, false>(out, y, means, mask, null_half4(), stream);
}

cudaError_t restore_y_and_add_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream) {
    return launch_restore<true, false>(y_hat, y, means, mask, null_half4(), stream);
}

cudaError_t restore_y_and_add_multiply_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    ConstHalfTensor q, cudaStream_t stream) {
    if (!paired(q)) return invalid();
    return launch_restore<true, true>(y_hat, y, means, mask, half4(q), stream);
}

cudaError_t restore_y_4x(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d(y.channels, y.height, y.width, chw);
    restore_y_4x_kernel<false><<<grid, block, 0, stream>>>(
        half4(out), half4(y), half4(means), mask4(mask), chw);
    return cudaGetLastError();
}

cudaError_t restore_y_4x_and_add_inplace(
    HalfTensor y_hat, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d(y.channels, y.height, y.width, chw);
    restore_y_4x_kernel<true><<<grid, block, 0, stream>>>(
        half4(y_hat), half4(y), half4(means), mask4(mask), chw);
    return cudaGetLastError();
}

cudaError_t restore_y_4x_add_and_multiply_broadcast(
    HalfTensor out, ConstHalfTensor y, ConstHalfTensor means, MaskTensor mask,
    ConstHalfTensor y_hat_so_far, const std::uint16_t* q, cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d(y.channels, y.height, y.width, chw);
    restore_y_4x_add_and_multiply_broadcast_kernel<<<grid, block, 0, stream>>>(
        half4(out), half4(y_hat_so_far), half4(y), half4(means), mask4(mask), half1(q), chw);
    return cudaGetLastError();
}

cudaError_t round_z(
    const std::uint16_t* z, int count, std::uint16_t* z_hat, std::int8_t* z_int8,
    cudaStream_t stream) {
    int n = 0;
    auto [grid, block] = config_1d<2>(count, n);
    round_z_kernel<<<grid, block, 0, stream>>>(
        flat<Half>(z_hat), flat<int8_t>(z_int8), flat<Half>(z), n);
    return cudaGetLastError();
}

cudaError_t single_part_for_reading_4x(
    HalfTensor out, ConstHalfTensor x, MaskTensor mask, cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d(out.channels, out.height, out.width, chw);
    single_part_for_reading_4x_kernel<<<grid, block, 0, stream>>>(
        half4(out), half4(x), mask4(mask), chw);
    return cudaGetLastError();
}

cudaError_t single_part_for_writing_4x(HalfTensor out, ConstHalfTensor x, cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d(out.channels, out.height, out.width, chw);
    single_part_for_writing_4x_kernel<<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    return cudaGetLastError();
}

cudaError_t pad_and_unshuffle_8(
    HalfTensor out, ConstHalfTensor x, int pad_bottom, int pad_right, cudaStream_t stream) {
    if (!paired(out)) return invalid();
    const int C = x.channels * 8 * 8;
    const int H = (x.height + pad_bottom) / 8;
    const int W = (x.width + pad_right) / 8;
    CudaCHW chw{};
    auto [grid, block] = C == 3 * 8 * 8 ? config_4d<2, 16, 4, 4>(C, H, W, chw) :
                                          config_4d<2, 256, 1, 1>(C, H, W, chw);
    pad_and_unshuffle_8_kernel<<<grid, block, 0, stream>>>(
        half4(out), half4(x), pad_bottom, pad_right, chw);
    return cudaGetLastError();
}

cudaError_t replicate_pad(
    HalfTensor out, ConstHalfTensor x, int pad_bottom, int pad_right, cudaStream_t stream) {
    const int C = x.channels;
    const int H = x.height + pad_bottom;
    const int W = x.width + pad_right;
    CudaCHW chw{};
    auto [grid, block] = C <= 4 ? config_4d<1, 4, 16, 16>(C, H, W, chw) :
                                  config_4d<1>(C, H, W, chw);
    replicate_pad_kernel<<<grid, block, 0, stream>>>(
        half4(out), half4(x), pad_bottom, pad_right, chw);
    return cudaGetLastError();
}

cudaError_t slice(HalfTensor out, ConstHalfTensor x, cudaStream_t stream) {
    if (!paired(out) || !paired(x)) return invalid();
    CudaCHW chw{};
    auto [grid, block] = config_4d<2>(x.channels, out.height, out.width, chw);
    slice_kernel<<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    return cudaGetLastError();
}

cudaError_t pixel_shuffle_2(HalfTensor out, ConstHalfTensor x, cudaStream_t stream) {
    CudaCHW chw{};
    auto [grid, block] = config_4d<1, 4, 8, 8>(x.channels / 4, x.height * 2, x.width * 2, chw);
    pixel_shuffle_2_kernel<<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    return cudaGetLastError();
}

cudaError_t pixel_shuffle_8(HalfTensor out, ConstHalfTensor x, bool clamp, cudaStream_t stream) {
    const int C = x.channels / 64;
    const int H = x.height * 8;
    const int W = x.width * 8;
    CudaCHW chw{};
    if (C == 3) {
        auto [grid, block] = config_4d<1, 1, 16, 16>(C / 3, H, W, chw);
        if (clamp) {
            pixel_shuffle_8_out_3_kernel<true><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
        } else {
            pixel_shuffle_8_out_3_kernel<false><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
        }
    } else {
        auto [grid, block] = config_4d<1, 4, 8, 8>(C, H, W, chw);
        if (clamp) {
            pixel_shuffle_8_kernel<true><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
        } else {
            pixel_shuffle_8_kernel<false><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
        }
    }
    return cudaGetLastError();
}

cudaError_t pixel_unshuffle(
    HalfTensor out, ConstHalfTensor x, int downscale_factor, cudaStream_t stream) {
    if (!paired(out)) return invalid();
    const int C = x.channels * downscale_factor * downscale_factor;
    const int H = x.height / downscale_factor;
    const int W = x.width / downscale_factor;
    CudaCHW chw{};
    auto [grid, block] = config_4d<1, 16, 4, 4>(C / 2, H, W, chw);
    if (downscale_factor == 2) {
        pixel_unshuffle_kernel<2><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    } else if (downscale_factor == 4) {
        pixel_unshuffle_kernel<4><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    } else if (downscale_factor == 8) {
        pixel_unshuffle_kernel<8><<<grid, block, 0, stream>>>(half4(out), half4(x), chw);
    } else {
        return invalid();
    }
    return cudaGetLastError();
}

}  // namespace nvcr::dcvcuf::glue
