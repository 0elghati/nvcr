#include "ops.hpp"

#include <map>
#include <mutex>

namespace nvcr::tensorrt::fx {
namespace {

constexpr int threads = 256;
constexpr std::int64_t stored_max = std::int64_t{1} << 30;

unsigned blocks_for(std::int64_t count) {
    const auto blocks = (count + threads - 1) / threads;
    return static_cast<unsigned>(blocks < 1 ? 1 : (blocks > 65535 * 64 ? 65535 * 64 : blocks));
}

__device__ std::int64_t rsr(std::int64_t a, int s) {
    return s == 0 ? a : (a + (std::int64_t{1} << (s - 1))) >> s;
}

// Stores v; a value beyond the profile's stored bound sets the failure word.
__device__ void store(std::int32_t* out, std::int64_t index, std::int64_t v, std::uint32_t* word) {
    if (v > stored_max || v < -stored_max) atomicOr(word, 1U);
    out[index] = static_cast<std::int32_t>(v);
}

#define FX_GRID_LOOP(index, count)                                                                       \
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < (count); \
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x)

__global__ void check_kernel(const std::int32_t* amax, int index, std::int32_t limit, std::uint32_t* word) {
    if (amax[index] > limit) atomicOr(word, 1U);
}

__global__ void mul_kernel(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::int64_t plane,
                           std::int64_t count, bool b_channels, bool b_plane, int shift, std::uint32_t* word) {
    FX_GRID_LOOP(i, count) {
        const std::int64_t c = b_channels ? i / plane : 0;
        const std::int64_t p = b_plane ? i % plane : 0;
        store(out, i, rsr(static_cast<std::int64_t>(a[i]) * b[c * (b_plane ? plane : 1) + p], shift), word);
    }
}

__global__ void affine_kernel(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int64_t alpha,
                              std::int64_t beta, int shift, std::uint32_t* word) {
    FX_GRID_LOOP(i, count) store(out, i, rsr(alpha * x[i] + beta, shift), word);
}

__global__ void add_kernel(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::int64_t count,
                           std::uint32_t* word) {
    FX_GRID_LOOP(i, count) store(out, i, static_cast<std::int64_t>(a[i]) + b[i], word);
}

__global__ void sat_table_kernel(const std::int32_t* x, std::int32_t* out, std::int64_t count,
                                 const std::int32_t* table, std::int64_t n, std::int32_t top) {
    FX_GRID_LOOP(i, count) {
        const std::int64_t v = x[i];
        out[i] = v >= n ? top : v < -n ? 0 : table[v + n];
    }
}

__global__ void delta_kernel(const std::int32_t* x, std::int32_t* out, std::int64_t count, const std::int16_t* dc,
                             std::int64_t h, int shift) {
    FX_GRID_LOOP(i, count) {
        const std::int64_t r = x[i];
        const std::int64_t base = r > 0 ? r : 0;
        out[i] = static_cast<std::int32_t>(r < -h || r > h ? base : base + dc[(r + h) >> shift]);
    }
}

__global__ void clamp_kernel(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int32_t lo,
                             std::int32_t hi) {
    FX_GRID_LOOP(i, count) out[i] = min(max(x[i], lo), hi);
}

__global__ void param_row_kernel(const std::int32_t* qp, const std::int64_t* rows, std::int32_t* out, int channels,
                                 int row_count, std::uint32_t* word) {
    const std::int32_t q = qp[0];
    if (q < 0 || q >= row_count) {
        if (threadIdx.x == 0 && blockIdx.x == 0) atomicOr(word, 1U);
        return;
    }
    FX_GRID_LOOP(c, channels) out[c] = static_cast<std::int32_t>(rows[static_cast<std::int64_t>(q) * channels + c]);
}

__global__ void crop_kernel(const std::int32_t* x, std::int32_t* out, Shape in, int h, int w) {
    const std::int64_t count = static_cast<std::int64_t>(in.channels) * h * w;
    FX_GRID_LOOP(i, count) {
        const std::int64_t c = i / (static_cast<std::int64_t>(h) * w);
        const std::int64_t y = (i / w) % h;
        const std::int64_t col = i % w;
        out[i] = x[(c * in.height + y) * in.width + col];
    }
}

// deep: (C r^2, H, W); wide: (C, H r, W r). up copies deep -> wide.
__global__ void shuffle_kernel(const std::int32_t* x, std::int32_t* out, Shape deep, int r, bool up) {
    const std::int64_t count = static_cast<std::int64_t>(deep.channels) * deep.height * deep.width;
    FX_GRID_LOOP(d, count) {
        const std::int64_t c = d / (static_cast<std::int64_t>(deep.height) * deep.width);
        const std::int64_t y = (d / deep.width) % deep.height;
        const std::int64_t col = d % deep.width;
        const std::int64_t base = c / (r * r);
        const std::int64_t i = (c / r) % r;
        const std::int64_t j = c % r;
        const std::int64_t wide_w = static_cast<std::int64_t>(deep.width) * r;
        const std::int64_t wide_h = static_cast<std::int64_t>(deep.height) * r;
        const std::int64_t s = (base * wide_h + y * r + i) * wide_w + col * r + j;
        if (up) {
            out[s] = x[d];
        } else {
            out[d] = x[s];
        }
    }
}

__global__ void im2col_kernel(const std::int32_t* x, std::int32_t* out, Shape in, int k, int stride, int pad, int h,
                              int w) {
    const std::int64_t count = static_cast<std::int64_t>(in.channels) * k * k * h * w;
    FX_GRID_LOOP(index, count) {
        const std::int64_t plane = static_cast<std::int64_t>(h) * w;
        const std::int64_t tap = index / plane;
        const std::int64_t c = tap / (k * k);
        const int i = static_cast<int>((tap / k) % k);
        const int j = static_cast<int>(tap % k);
        const int oy = static_cast<int>((index % plane) / w);
        const int ox = static_cast<int>(index % w);
        const int iy = oy * stride - pad + i;
        const int ix = ox * stride - pad + j;
        out[index] = iy < 0 || iy >= in.height || ix < 0 || ix >= in.width
            ? 0
            : x[(c * in.height + iy) * in.width + ix];
    }
}

std::mutex word_mutex;
std::map<int, std::uint32_t*> words;

}  // namespace

cudaError_t failure_word(std::uint32_t** word) {
    int device = 0;
    if (const auto status = cudaGetDevice(&device); status != cudaSuccess) return status;
    std::lock_guard<std::mutex> lock(word_mutex);
    if (auto found = words.find(device); found != words.end()) {
        *word = found->second;
        return cudaSuccess;
    }
    std::uint32_t* created = nullptr;
    if (const auto status = cudaMalloc(&created, sizeof(std::uint32_t)); status != cudaSuccess) return status;
    if (const auto status = cudaMemset(created, 0, sizeof(std::uint32_t)); status != cudaSuccess) {
        cudaFree(created);
        return status;
    }
    words[device] = created;
    *word = created;
    return cudaSuccess;
}

cudaError_t check_max(const std::int32_t* amax, int index, std::int32_t limit, std::uint32_t* word,
                      cudaStream_t stream) {
    check_kernel<<<1, 1, 0, stream>>>(amax, index, limit, word);
    return cudaGetLastError();
}

cudaError_t mul(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, Shape a_shape, Shape b_shape,
                int shift, std::uint32_t* word, cudaStream_t stream) {
    const std::int64_t plane = static_cast<std::int64_t>(a_shape.height) * a_shape.width;
    const std::int64_t count = plane * a_shape.channels;
    mul_kernel<<<blocks_for(count), threads, 0, stream>>>(a, b, out, plane, count, b_shape.channels != 1,
                                                          b_shape.height * b_shape.width != 1, shift, word);
    return cudaGetLastError();
}

cudaError_t affine(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int64_t alpha,
                   std::int64_t beta, int shift, std::uint32_t* word, cudaStream_t stream) {
    affine_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, count, alpha, beta, shift, word);
    return cudaGetLastError();
}

cudaError_t add(const std::int32_t* a, const std::int32_t* b, std::int32_t* out, std::int64_t count,
                std::uint32_t* word, cudaStream_t stream) {
    add_kernel<<<blocks_for(count), threads, 0, stream>>>(a, b, out, count, word);
    return cudaGetLastError();
}

cudaError_t sat_table(const std::int32_t* x, std::int32_t* out, std::int64_t count, const std::int32_t* table,
                      std::int64_t n, std::int32_t top, cudaStream_t stream) {
    sat_table_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, count, table, n, top);
    return cudaGetLastError();
}

cudaError_t delta(const std::int32_t* x, std::int32_t* out, std::int64_t count, const std::int16_t* dc,
                  std::int64_t h, int shift, cudaStream_t stream) {
    delta_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, count, dc, h, shift);
    return cudaGetLastError();
}

cudaError_t clamp(const std::int32_t* x, std::int32_t* out, std::int64_t count, std::int32_t lo, std::int32_t hi,
                  cudaStream_t stream) {
    clamp_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, count, lo, hi);
    return cudaGetLastError();
}

cudaError_t param_row(const std::int32_t* qp, const std::int64_t* rows, std::int32_t* out, int channels,
                      int row_count, std::uint32_t* word, cudaStream_t stream) {
    param_row_kernel<<<blocks_for(channels), threads, 0, stream>>>(qp, rows, out, channels, row_count, word);
    return cudaGetLastError();
}

cudaError_t crop(const std::int32_t* x, std::int32_t* out, Shape in, int out_height, int out_width,
                 cudaStream_t stream) {
    const std::int64_t count = static_cast<std::int64_t>(in.channels) * out_height * out_width;
    crop_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, in, out_height, out_width);
    return cudaGetLastError();
}

cudaError_t shuffle(const std::int32_t* x, std::int32_t* out, Shape in, int r, bool up, cudaStream_t stream) {
    const Shape deep = up ? in : Shape{in.channels * r * r, in.height / r, in.width / r};
    const std::int64_t count = static_cast<std::int64_t>(deep.channels) * deep.height * deep.width;
    shuffle_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, deep, r, up);
    return cudaGetLastError();
}

cudaError_t im2col(const std::int32_t* x, std::int32_t* out, Shape in, int k, int stride, cudaStream_t stream) {
    const int pad = k == 3 ? 1 : 0;
    const int h = (in.height + 2 * pad - k) / stride + 1;
    const int w = (in.width + 2 * pad - k) / stride + 1;
    const std::int64_t count = static_cast<std::int64_t>(in.channels) * k * k * h * w;
    im2col_kernel<<<blocks_for(count), threads, 0, stream>>>(x, out, in, k, stride, pad, h, w);
    return cudaGetLastError();
}

}  // namespace nvcr::tensorrt::fx
