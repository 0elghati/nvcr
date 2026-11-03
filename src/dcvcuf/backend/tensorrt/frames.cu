// YUV420P8 <-> model tensors, with the arithmetic of the native path's torch ops
// (native_session.cpp frames_to_tensor / tensors_to_frames): u8 -> half,
// x * (1/255) and - 0.5 rounded to half each step; back: + 0.5, luma * 255
// rounded half-to-even, chroma 2x2 mean in float then * 255 truncated.
#include "glue.hpp"

#include <cuda_fp16.h>

namespace nvcr::dcvcuf::glue {
namespace {

__device__ std::uint16_t from_u8(std::uint8_t value) {
    const __half scaled = __float2half_rn(static_cast<float>(value) * (1.0F / 255.0F));
    return __half_as_ushort(__float2half_rn(__half2float(scaled) - 0.5F));
}

__device__ float to_float(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

__device__ float round_half(float value) { return __half2float(__float2half_rn(value)); }

__device__ std::size_t index(const HalfTensor& t, int c, int y, int x) {
    return t.layout == Layout::nhwc
        ? (static_cast<std::size_t>(y) * t.width + x) * t.channels + c
        : (static_cast<std::size_t>(c) * t.height + y) * t.width + x;
}

__global__ void yuv420_to_tensor_kernel(
    HalfTensor out, const std::uint8_t* yuv, int width, int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int c = static_cast<int>(blockIdx.z);
    if (x >= out.width || y >= out.height) return;
    // Replicate padding: clamp to the last source row and column.
    const int sx = min(x, width - 1);
    const int sy = min(y, height - 1);
    const std::size_t frame = static_cast<std::size_t>(width) * height * 3 / 2;
    const std::uint8_t* planes = yuv + static_cast<std::size_t>(c / 3) * frame;
    const std::size_t luma = static_cast<std::size_t>(width) * height;
    const int plane = c % 3;
    const std::uint8_t value = plane == 0
        ? planes[static_cast<std::size_t>(sy) * width + sx]
        : planes[luma + (plane - 1) * luma / 4 + static_cast<std::size_t>(sy / 2) * (width / 2) + sx / 2];
    out.data[index(out, c, y, x)] = from_u8(value);
}

__global__ void tensor_to_yuv420_kernel(
    std::uint8_t* yuv, HalfTensor frame, int width, int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;
    const auto sample = [&](int c, int sy, int sx) {
        return round_half(to_float(frame.data[index(frame, c, sy, sx)]) + 0.5F);
    };
    const float luma = fminf(fmaxf(round_half(sample(0, y, x) * 255.0F), 0.0F), 255.0F);
    yuv[static_cast<std::size_t>(y) * width + x] = static_cast<std::uint8_t>(rintf(luma));
    if ((x & 1) != 0 || (y & 1) != 0) return;
    const std::size_t luma_size = static_cast<std::size_t>(width) * height;
    for (int c = 1; c < 3; ++c) {
        const float mean = round_half(
            (sample(c, y, x) + sample(c, y, x + 1) + sample(c, y + 1, x) + sample(c, y + 1, x + 1)) /
            4.0F);
        const float scaled = fminf(fmaxf(round_half(mean * 255.0F), 0.0F), 255.0F);
        yuv[luma_size + (c - 1) * luma_size / 4 + static_cast<std::size_t>(y / 2) * (width / 2) +
            x / 2] = static_cast<std::uint8_t>(scaled);
    }
}

dim3 grid_for(int width, int height, int depth) {
    return dim3((width + 15) / 16, (height + 15) / 16, depth);
}

}  // namespace

cudaError_t yuv420_to_tensor(
    HalfTensor out, const std::uint8_t* yuv, int width, int height, cudaStream_t stream) {
    yuv420_to_tensor_kernel<<<grid_for(out.width, out.height, out.channels), dim3(16, 16), 0,
                              stream>>>(out, yuv, width, height);
    return cudaGetLastError();
}

cudaError_t tensor_to_yuv420(
    std::uint8_t* yuv, ConstHalfTensor frame, int width, int height, cudaStream_t stream) {
    const HalfTensor view{const_cast<std::uint16_t*>(frame.data), frame.channels, frame.height,
                          frame.width, frame.layout, frame.stride};
    tensor_to_yuv420_kernel<<<grid_for(width, height, 1), dim3(16, 16), 0, stream>>>(
        yuv, view, width, height);
    return cudaGetLastError();
}

}  // namespace nvcr::dcvcuf::glue
