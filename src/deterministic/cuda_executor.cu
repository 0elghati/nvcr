#include "cuda_executor.hpp"

#include "det_math.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace nvcr::deterministic {
namespace {

constexpr int threads_per_block = 256;

// 1x1 convolution as a tiled GEMM: out[m][n] = bias[m] + sum_k w[m][k] * x[k][n].
// Each output accumulates k = 0, 1, ..., K-1 in order with one fma per step, the
// exact sequence of the CPU reference. Tiles only change which thread owns an
// output, never the order of its accumulation.
constexpr int tile_m = 64;
constexpr int tile_n = 64;
constexpr int tile_k = 16;
constexpr int thread_m = 4;
constexpr int thread_n = 4;

__global__ void pointwise_conv_kernel(
    const float* __restrict__ weights, const float* __restrict__ bias,
    const float* __restrict__ input, float* __restrict__ output,
    int out_channels, int positions, int in_channels) {
    __shared__ float weight_tile[tile_k][tile_m];
    __shared__ float input_tile[tile_k][tile_n];
    const int thread = static_cast<int>(threadIdx.x);
    const int row = thread / (tile_n / thread_n);
    const int column = thread % (tile_n / thread_n);
    const int m0 = static_cast<int>(blockIdx.y) * tile_m;
    const int n0 = static_cast<int>(blockIdx.x) * tile_n;

    float accumulator[thread_m][thread_n];
    for (int i = 0; i < thread_m; ++i) {
        const int m = m0 + row * thread_m + i;
        const float initial = m < out_channels ? bias[m] : 0.0F;
        for (int j = 0; j < thread_n; ++j) accumulator[i][j] = initial;
    }

    for (int k0 = 0; k0 < in_channels; k0 += tile_k) {
        for (int index = thread; index < tile_k * tile_m; index += threads_per_block) {
            const int local_m = index / tile_k;
            const int local_k = index % tile_k;
            const int m = m0 + local_m;
            const int k = k0 + local_k;
            weight_tile[local_k][local_m] =
                m < out_channels && k < in_channels
                ? weights[static_cast<std::size_t>(m) * in_channels + k]
                : 0.0F;
        }
        for (int index = thread; index < tile_k * tile_n; index += threads_per_block) {
            const int local_k = index / tile_n;
            const int local_n = index % tile_n;
            const int k = k0 + local_k;
            const int n = n0 + local_n;
            input_tile[local_k][local_n] =
                k < in_channels && n < positions
                ? input[static_cast<std::size_t>(k) * positions + n]
                : 0.0F;
        }
        __syncthreads();
        // Only real k values are accumulated, so the sequence matches the CPU's.
        const int k_count = min(tile_k, in_channels - k0);
        for (int kk = 0; kk < k_count; ++kk) {
            float a[thread_m];
            float b[thread_n];
            for (int i = 0; i < thread_m; ++i) a[i] = weight_tile[kk][row * thread_m + i];
            for (int j = 0; j < thread_n; ++j) b[j] = input_tile[kk][column * thread_n + j];
            for (int i = 0; i < thread_m; ++i) {
                for (int j = 0; j < thread_n; ++j) {
                    accumulator[i][j] = det_fma(a[i], b[j], accumulator[i][j]);
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < thread_m; ++i) {
        const int m = m0 + row * thread_m + i;
        if (m >= out_channels) continue;
        for (int j = 0; j < thread_n; ++j) {
            const int n = n0 + column * thread_n + j;
            if (n < positions) output[static_cast<std::size_t>(m) * positions + n] = accumulator[i][j];
        }
    }
}

struct ConvGeometry {
    int in_channels;
    int out_channels;
    int group_inputs;
    int depthwise;
    int in_height;
    int in_width;
    int out_height;
    int out_width;
    int kernel_height;
    int kernel_width;
    int stride_y;
    int stride_x;
    int pad_top;
    int pad_left;
};

// General convolution, one thread per output, in the reference order
// (input channel, kernel row, kernel column); padding taps fma a zero.
__global__ void conv_kernel(
    const float* __restrict__ weights, const float* __restrict__ bias,
    const float* __restrict__ input, float* __restrict__ output, ConvGeometry g) {
    const auto count = static_cast<std::size_t>(g.out_channels) * g.out_height * g.out_width;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int ox = static_cast<int>(index % g.out_width);
    const int oy = static_cast<int>((index / g.out_width) % g.out_height);
    const int out_channel = static_cast<int>(index / (static_cast<std::size_t>(g.out_width) * g.out_height));
    float accumulator = bias[out_channel];
    for (int local = 0; local < g.group_inputs; ++local) {
        const int in_channel = g.depthwise ? out_channel : local;
        const float* plane = input + static_cast<std::size_t>(in_channel) * g.in_height * g.in_width;
        const float* kernel = weights +
            (static_cast<std::size_t>(out_channel) * g.group_inputs + local) *
                g.kernel_height * g.kernel_width;
        for (int ky = 0; ky < g.kernel_height; ++ky) {
            const int iy = oy * g.stride_y - g.pad_top + ky;
            for (int kx = 0; kx < g.kernel_width; ++kx) {
                const int ix = ox * g.stride_x - g.pad_left + kx;
                const float value = iy >= 0 && iy < g.in_height && ix >= 0 && ix < g.in_width
                    ? plane[static_cast<std::size_t>(iy) * g.in_width + ix]
                    : 0.0F;
                accumulator = det_fma(kernel[ky * g.kernel_width + kx], value, accumulator);
            }
        }
    }
    output[index] = accumulator;
}

__global__ void binary_kernel(
    const float* first, const float* second, float* output, std::size_t count, int multiply) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = multiply ? det_mul(first[index], second[index])
                             : det_add(first[index], second[index]);
}

__global__ void mul_scalar_kernel(const float* input, float* output, std::size_t count, float scalar) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = det_mul(input[index], scalar);
}

__global__ void sigmoid_kernel(const float* input, float* output, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = det_sigmoid(input[index]);
}

__global__ void depth_to_space_kernel(
    const float* input, float* output, int channels, int out_height, int out_width,
    int in_height, int in_width, int block) {
    const auto count = static_cast<std::size_t>(channels) * out_height * out_width;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int x = static_cast<int>(index % out_width);
    const int y = static_cast<int>((index / out_width) % out_height);
    const int channel = static_cast<int>(index / (static_cast<std::size_t>(out_width) * out_height));
    const int source = channel * block * block + (y % block) * block + x % block;
    output[index] = input[(static_cast<std::size_t>(source) * in_height + y / block) * in_width + x / block];
}

__global__ void scale_channels_kernel(
    const float* input, const float* scale, float* output, std::size_t count, std::size_t plane) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = det_mul(input[index], scale[index / plane]);
}

__global__ void clamp_kernel(const float* input, float* output, std::size_t count, float lo, float hi) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = input[index];
    output[index] = value < lo ? lo : (value > hi ? hi : value);
}

__global__ void crop_kernel(
    const float* input, float* output, int channels, int out_height, int out_width,
    int in_height, int in_width) {
    const auto count = static_cast<std::size_t>(channels) * out_height * out_width;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int x = static_cast<int>(index % out_width);
    const int y = static_cast<int>((index / out_width) % out_height);
    const int channel = static_cast<int>(index / (static_cast<std::size_t>(out_width) * out_height));
    output[index] = input[(static_cast<std::size_t>(channel) * in_height + y) * in_width + x];
}

__global__ void space_to_depth_kernel(
    const float* input, float* output, int channels, int out_height, int out_width,
    int in_height, int in_width, int block) {
    const auto count = static_cast<std::size_t>(channels) * out_height * out_width;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int x = static_cast<int>(index % out_width);
    const int y = static_cast<int>((index / out_width) % out_height);
    const int channel = static_cast<int>(index / (static_cast<std::size_t>(out_width) * out_height));
    const int source = channel / (block * block);
    const int offset_y = (channel / block) % block;
    const int offset_x = channel % block;
    output[index] = input[(static_cast<std::size_t>(source) * in_height + y * block + offset_y) * in_width +
                          x * block + offset_x];
}

__global__ void im2col_kernel(
    const float* input, float* output, int channels, int out_height, int out_width,
    int in_height, int in_width, int kernel_height, int kernel_width, int stride_y, int stride_x,
    int pad_top, int pad_left) {
    const auto count = static_cast<std::size_t>(channels) * out_height * out_width;
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int x = static_cast<int>(index % out_width);
    const int y = static_cast<int>((index / out_width) % out_height);
    const int channel = static_cast<int>(index / (static_cast<std::size_t>(out_width) * out_height));
    const int taps = kernel_height * kernel_width;
    const int source = channel / taps;
    const int ky = (channel % taps) / kernel_width;
    const int kx = channel % kernel_width;
    const int iy = y * stride_y - pad_top + ky;
    const int ix = x * stride_x - pad_left + kx;
    output[index] = iy >= 0 && iy < in_height && ix >= 0 && ix < in_width
        ? input[(static_cast<std::size_t>(source) * in_height + iy) * in_width + ix]
        : 0.0F;
}

// det-int-v1 pointwise conv on INT8 tensor cores.
//
// An 8-bit value is one signed limb. A 16-bit value q splits into
// q = 256 h + l with h = q >> 8 (signed byte) and l = q & 255, stored as
// l' = l - 128 (signed byte). With weights W and activations X split this way
// (Aw, Ax = 256 for two limbs, 1 for one):
//   sum W X = Aw Ax sum(Wh Xh)
//           + [X two limbs] Aw (sum(Wh Xl') + 128 sum Wh)
//           + [W two limbs] Ax (sum(Wl' Xh) + 128 sum Xh)
//           + [both]        sum(Wl' Xl') + 128 sum Xl' + 128 sum Wl' + 16384 K
// Every term is an exact integer, so the result equals the CPU reference
// whatever the tiling or tensor-core order. In mean mode the last line is the
// constant floor(65025 K / 4) (program.hpp).
// Matrices are stored row-major with K contiguous: weights as [rows][K] and
// activations as [positions][K], both zero-padded to whole blocks.
constexpr int tile = 16;

// Tiled INT8 tensor-core GEMM. A block computes 64 output channels x 128
// positions with eight warps (2 x 4), each owning 32 x 32 outputs as 2 x 4
// mma.m16n8k32 tiles. Each stage's weight and activation rows are copied to
// shared memory with cp.async into a multi-stage ring (IntegerShape). Integer
// products are exact in any order, so the tiling cannot change a result.
// Weight rows are padded to whole blocks (64), positions to whole blocks (128),
// and K to 64, all with zeros.
constexpr int block_row_tiles = 4;
constexpr int block_column_tiles = 8;
constexpr int block_k_tiles = 4;  // K padding granule
constexpr int block_rows = block_row_tiles * tile;
constexpr int block_columns = block_column_tiles * tile;
constexpr int integer_warps = 8;  // 2 x 4
constexpr int integer_threads = integer_warps * 32;

// Stage geometry per limb count: as wide and deep as 48 KiB of shared memory
// allows, two blocks per SM (one block per SM with a deeper K measured slower
// for every layer shape, despite the 16 x 16-bit variant's register spills).
// Rows carry 16 bytes of padding, so the eight rows one ldmatrix phase reads
// fall in distinct 16-byte bank groups. Shared memory is dynamic.
template <int WeightLimbs, int ActivationLimbs>
struct IntegerShape {
    static constexpr int blocks_per_sm = 2;
    static constexpr int shared_limit = 48 * 1024;
    static constexpr int k = ActivationLimbs == 1 ? 64 : 32;
    static constexpr int stride = k + 16;
    static constexpr int a_bytes = block_rows * stride;     // per limb
    static constexpr int b_bytes = block_columns * stride;  // per limb
    static constexpr int stage_bytes = WeightLimbs * a_bytes + ActivationLimbs * b_bytes;
    static constexpr int stages = 3 * stage_bytes <= shared_limit ? 3 : 2;
    static constexpr int shared_bytes = stages * stage_bytes;
    static_assert(shared_bytes <= shared_limit);
};

// One block quantizes 16 positions; 16 lanes split the channels. Max and
// integer sums are order-independent, so the split does not affect any value.
// The block also writes its padded positions and channels as zeros.
constexpr int quantize_positions = 16;
constexpr int quantize_lanes = 16;
constexpr int quantize_chunk = 64;  // channels per transposed write

// Also writes each position's exact correction constant for the given weight
// limb count (see integer_pointwise_kernel).
template <int Limbs>
__global__ void quantize_activations_kernel(
    const float* __restrict__ input, int channels, int positions, int k_padded,
    signed char* __restrict__ high, signed char* __restrict__ low, long long high_multiplier,
    long long low_multiplier, long long* __restrict__ column_constants, int* __restrict__ exponents) {
    constexpr int bits = Limbs == 2 ? 16 : 8;
    __shared__ float maxima[quantize_lanes][quantize_positions];
    __shared__ int finite_flags[quantize_lanes][quantize_positions];
    __shared__ int block_exponent[quantize_positions];
    __shared__ int block_finite[quantize_positions];
    __shared__ int partial_high[quantize_lanes][quantize_positions];
    __shared__ int partial_low[quantize_lanes][quantize_positions];
    __shared__ __align__(16) signed char staged_high[quantize_positions][quantize_chunk + 4];
    __shared__ __align__(16) signed char staged_low[quantize_positions][quantize_chunk + 4];
    const int lane = static_cast<int>(threadIdx.y);
    const int local = static_cast<int>(threadIdx.x);
    const int block_position = static_cast<int>(blockIdx.x) * quantize_positions;
    const int position = block_position + local;
    const bool valid = position < positions;
    float maximum = 0.0F;
    bool finite = true;
    if (valid) {
        for (int channel = lane; channel < channels; channel += quantize_lanes) {
            const float value = input[static_cast<std::size_t>(channel) * positions + position];
            finite = finite && det_isfinite(value);
            maximum = fmaxf(maximum, fabsf(value));
        }
    }
    maxima[lane][local] = maximum;
    finite_flags[lane][local] = finite ? 1 : 0;
    __syncthreads();
    if (lane == 0) {
        for (int other = 1; other < quantize_lanes; ++other) {
            maximum = fmaxf(maximum, maxima[other][local]);
            finite = finite && finite_flags[other][local] != 0;
        }
        block_exponent[local] = det_int_exponent(maximum, bits);
        block_finite[local] = finite ? 1 : 0;
    }
    __syncthreads();
    const int exponent = block_exponent[local];
    finite = block_finite[local] != 0;
    int high_sum = 0;
    int low_sum = 0;
    const int thread = lane * quantize_positions + local;
    const int write_position = thread / (quantize_chunk / 4);
    const int write_part = thread % (quantize_chunk / 4);
    for (int chunk = 0; chunk < k_padded; chunk += quantize_chunk) {
#pragma unroll
        for (int step = 0; step < quantize_chunk / quantize_lanes; ++step) {
            const int within = lane + step * quantize_lanes;
            const int channel = chunk + within;
            const int q = valid && finite && channel < channels
                ? det_int_quantize(input[static_cast<std::size_t>(channel) * positions + position],
                                   exponent, bits)
                : 0;
            if constexpr (Limbs == 2) {
                const bool real = valid && channel < channels;
                const int h = q >> 8;
                const int l = real ? (q & 255) - 128 : 0;
                staged_high[local][within] = static_cast<signed char>(h);
                staged_low[local][within] = static_cast<signed char>(l);
                high_sum += h;
                low_sum += l;
            } else {
                staged_high[local][within] = static_cast<signed char>(q);
                high_sum += q;
            }
        }
        __syncthreads();
        const auto offset = static_cast<std::size_t>(block_position + write_position) * k_padded +
            chunk + write_part * 4;
        *reinterpret_cast<int*>(high + offset) =
            *reinterpret_cast<const int*>(&staged_high[write_position][write_part * 4]);
        if constexpr (Limbs == 2) {
            *reinterpret_cast<int*>(low + offset) =
                *reinterpret_cast<const int*>(&staged_low[write_position][write_part * 4]);
        }
        __syncthreads();
    }
    partial_high[lane][local] = high_sum;
    partial_low[lane][local] = low_sum;
    __syncthreads();
    if (lane == 0 && valid) {
        for (int other = 1; other < quantize_lanes; ++other) {
            high_sum += partial_high[other][local];
            low_sum += partial_low[other][local];
        }
        // [weights two limbs] Ax * 128 * sum Xh + [both two limbs] 128 * sum Xl'
        column_constants[position] = high_multiplier * high_sum + low_multiplier * low_sum;
        exponents[position] = finite ? exponent : INT_MIN;
    }
}

struct IntegerEpilogue {
    int wsilu;           // apply x * sigmoid(scale * x) after the conv
    float wsilu_scale;
    int chunk_add;       // weight rows are paired (see prepare_integer_weights): add each pair
    const float* residual;  // added last when not null
};

struct IntegerOperands {
    const signed char* weight_high;
    const signed char* weight_low;
    const long long* row_constants;
    const float* scales;
    const float* bias;
    const signed char* input_high;
    const signed char* input_low;
    const long long* column_constants;
    const int* exponents;
    float* output;
    int out_channels;
    int positions;
    int in_channels;
    int k_padded;
};

__device__ __forceinline__ void copy16(void* shared, const void* global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(address), "l"(global));
#else
    *static_cast<uint4*>(shared) = *static_cast<const uint4*>(global);
#endif
}

__device__ __forceinline__ void copy_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int Pending>
__device__ __forceinline__ void copy_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending));
#endif
}

__device__ __forceinline__ void load_matrix_x4(unsigned int (&registers)[4], const void* shared) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto address = static_cast<unsigned int>(__cvta_generic_to_shared(shared));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(registers[0]), "=r"(registers[1]), "=r"(registers[2]), "=r"(registers[3])
                 : "r"(address));
#endif
}

// d += a (16 x 32, s8) * b (32 x 8, s8), exact int32.
__device__ __forceinline__ void mma_s8(int (&d)[4], const unsigned int (&a)[4], unsigned int b0, unsigned int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

template <int WeightLimbs, int ActivationLimbs>
__device__ __forceinline__ void load_stage(
    const IntegerOperands& op, unsigned char* stage, int row0, int column0, int k0) {
    using Shape = IntegerShape<WeightLimbs, ActivationLimbs>;
    constexpr int parts = Shape::k / 16;
    const int thread = static_cast<int>(threadIdx.x);
    unsigned char* a_high = stage;
    unsigned char* a_low = a_high + Shape::a_bytes;
    unsigned char* b_high = stage + WeightLimbs * Shape::a_bytes;
    unsigned char* b_low = b_high + Shape::b_bytes;
#pragma unroll
    for (int chunk = thread; chunk < block_rows * parts; chunk += integer_threads) {
        const int line = chunk / parts;
        const int part = (chunk % parts) * 16;
        const auto source = static_cast<std::size_t>(row0 + line) * op.k_padded + k0 + part;
        copy16(a_high + line * Shape::stride + part, op.weight_high + source);
        if constexpr (WeightLimbs == 2) copy16(a_low + line * Shape::stride + part, op.weight_low + source);
    }
#pragma unroll
    for (int chunk = thread; chunk < block_columns * parts; chunk += integer_threads) {
        const int line = chunk / parts;
        const int part = (chunk % parts) * 16;
        const auto source = static_cast<std::size_t>(column0 + line) * op.k_padded + k0 + part;
        copy16(b_high + line * Shape::stride + part, op.input_high + source);
        if constexpr (ActivationLimbs == 2) copy16(b_low + line * Shape::stride + part, op.input_low + source);
    }
}

constexpr int warp_m_tiles = 2;  // m16 tiles per warp
constexpr int warp_n_tiles = 4;  // n8 tiles per warp

template <int WeightLimbs, int ActivationLimbs, bool MeanLow>
__global__ void __launch_bounds__(integer_threads, (IntegerShape<WeightLimbs, ActivationLimbs>::blocks_per_sm))
integer_pointwise_kernel(
    IntegerOperands op, IntegerEpilogue epilogue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool weight_low = WeightLimbs == 2;
    constexpr bool input_low = ActivationLimbs == 2;
    constexpr bool middle = weight_low || input_low;
    constexpr bool both_low = weight_low && input_low && !MeanLow;
    // Accumulators: hh = sum Wh Xh, middle = sum Wh Xl' and/or sum Wl' Xh (both
    // carry the same factor, so one int32 holds their sum exactly), and
    // low = sum Wl' Xl'.
    constexpr long long high_factor = (weight_low ? 256LL : 1LL) * (input_low ? 256LL : 1LL);
    constexpr long long middle_factor = weight_low && input_low ? 256LL : 1LL;
    using Shape = IntegerShape<WeightLimbs, ActivationLimbs>;
    extern __shared__ __align__(128) unsigned char shared[];
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp_row = (warp / 4) * warp_m_tiles * tile;
    const int warp_column = (warp % 4) * warp_n_tiles * 8;
    // Row blocks vary fastest, so the blocks resident together share one
    // activation tile in L2 instead of streaming every position per row block.
    const int row0 = static_cast<int>(blockIdx.x) * block_rows;
    const int column0 = static_cast<int>(blockIdx.y) * block_columns;

    int high_high[warp_m_tiles][warp_n_tiles][4] = {};
    int mixed[warp_m_tiles][warp_n_tiles][4] = {};
    int low_low[warp_m_tiles][warp_n_tiles][4] = {};

    // ldmatrix addresses: A matrices are (rows 0-7 | 8-15) x (k 0-15 | 16-31),
    // B matrices (k 0-15 | 16-31) x (columns 0-7 | 8-15).
    const int a_line = (lane % 8) + ((lane / 8) % 2) * 8;
    const int a_part = (lane / 16) * 16;
    const int b_line = (lane % 8) + (lane / 16) * 8;
    const int b_part = ((lane / 8) % 2) * 16;

    // Ring of Shape::stages buffers: one wait and one barrier per stage. The
    // barrier also guarantees the buffer refilled below was finished by every
    // warp in the previous iteration.
    const int stages = op.k_padded / Shape::k;
#pragma unroll
    for (int stage = 0; stage < Shape::stages - 1; ++stage) {
        if (stage < stages) {
            load_stage<WeightLimbs, ActivationLimbs>(
                op, shared + stage * Shape::stage_bytes, row0, column0, stage * Shape::k);
        }
        copy_commit();
    }
    for (int stage = 0; stage < stages; ++stage) {
        const int next = stage + Shape::stages - 1;
        if constexpr (Shape::stages == 2) {
            // Double buffer: issue the next copy before waiting on this one;
            // the barrier at the end of the iteration protects its buffer.
            if (next < stages) {
                load_stage<WeightLimbs, ActivationLimbs>(
                    op, shared + (next % 2) * Shape::stage_bytes, row0, column0, next * Shape::k);
            }
            copy_commit();
            copy_wait<1>();
            __syncthreads();
        } else {
            copy_wait<Shape::stages - 2>();
            __syncthreads();
            if (next < stages) {
                load_stage<WeightLimbs, ActivationLimbs>(
                    op, shared + (next % Shape::stages) * Shape::stage_bytes, row0, column0, next * Shape::k);
            }
            copy_commit();
        }
        const unsigned char* a_high_stage = shared + (stage % Shape::stages) * Shape::stage_bytes;
        const unsigned char* a_low_stage = a_high_stage + Shape::a_bytes;
        const unsigned char* b_high_stage = a_high_stage + WeightLimbs * Shape::a_bytes;
        const unsigned char* b_low_stage = b_high_stage + Shape::b_bytes;
#pragma unroll
        for (int step = 0; step < Shape::k / 32; ++step) {
        const unsigned char* a_high = a_high_stage + step * 32;
        const unsigned char* a_low = a_low_stage + step * 32;
        const unsigned char* b_high = b_high_stage + step * 32;
        const unsigned char* b_low = b_low_stage + step * 32;
        unsigned int a[warp_m_tiles][4], a2[warp_m_tiles][4];
#pragma unroll
        for (int m = 0; m < warp_m_tiles; ++m) {
            const int offset = (warp_row + m * tile + a_line) * Shape::stride + a_part;
            load_matrix_x4(a[m], a_high + offset);
            if constexpr (weight_low) load_matrix_x4(a2[m], a_low + offset);
        }
        // B fragments one column pair at a time keeps register pressure low.
#pragma unroll
        for (int pair = 0; pair < warp_n_tiles / 2; ++pair) {
            unsigned int b[4], b2[4];
            const int offset = (warp_column + pair * 16 + b_line) * Shape::stride + b_part;
            load_matrix_x4(b, b_high + offset);
            if constexpr (input_low) load_matrix_x4(b2, b_low + offset);
#pragma unroll
            for (int m = 0; m < warp_m_tiles; ++m) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int n = pair * 2 + half;
                    mma_s8(high_high[m][n], a[m], b[2 * half], b[2 * half + 1]);
                    if constexpr (input_low) mma_s8(mixed[m][n], a[m], b2[2 * half], b2[2 * half + 1]);
                    if constexpr (weight_low) mma_s8(mixed[m][n], a2[m], b[2 * half], b[2 * half + 1]);
                    if constexpr (both_low) mma_s8(low_low[m][n], a2[m], b2[2 * half], b2[2 * half + 1]);
                }
            }
        }
        }
        if constexpr (Shape::stages == 2) __syncthreads();
    }

    // Accumulator element e of tile (m, n) is row lane/4 (+8 for e >= 2) and
    // column 2 (lane % 4) + e % 2.
    const int group = lane / 4;
    int exponents[warp_n_tiles][2];
    long long column_constants[warp_n_tiles][2];
#pragma unroll
    for (int n = 0; n < warp_n_tiles; ++n) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int column = column0 + warp_column + n * 8 + (lane % 4) * 2 + e;
            const bool inside = column < op.positions;
            exponents[n][e] = inside ? op.exponents[column] : INT_MIN;
            column_constants[n][e] = inside ? op.column_constants[column] : 0;
        }
    }
#pragma unroll
    for (int m = 0; m < warp_m_tiles; ++m) {
        float values[2][warp_n_tiles][2];
        const int tile_row = row0 + warp_row + m * tile;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = tile_row + group + h * 8;
            const bool inside = row < op.out_channels;
            const long long row_constant = inside ? op.row_constants[row] : 0;
            const float scale = inside ? op.scales[row] : 0.0F;
            const float bias = inside ? op.bias[row] : 0.0F;
#pragma unroll
            for (int n = 0; n < warp_n_tiles; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int element = h * 2 + e;
                    float value;
                    if (exponents[n][e] == INT_MIN) {
                        value = __int_as_float(0x7fc00000);
                    } else {
                        // Limb products plus the precomputed row and column
                        // terms of the exact decomposition.
                        long long accumulator = high_factor * high_high[m][n][element] +
                            row_constant + column_constants[n][e];
                        if constexpr (middle) accumulator += middle_factor * mixed[m][n][element];
                        if constexpr (both_low) accumulator += low_low[m][n][element];
                        value = det_int_output(accumulator, scale, exponents[n][e], bias);
                    }
                    // Same per-element sequence as the separate mul_scalar,
                    // sigmoid, and mul nodes of the reference.
                    if (epilogue.wsilu) {
                        value = det_mul(det_sigmoid(det_mul(value, epilogue.wsilu_scale)), value);
                    }
                    values[h][n][e] = value;
                }
            }
        }
        // Chunk add: rows k and k + 8 of a tile are node rows r and r + C.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (epilogue.chunk_add && h == 1) break;
            const int row = epilogue.chunk_add ? tile_row / 2 + group : tile_row + group + h * 8;
            const int limit = epilogue.chunk_add ? op.out_channels / 2 : op.out_channels;
            if (row >= limit) continue;
#pragma unroll
            for (int n = 0; n < warp_n_tiles; ++n) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int column = column0 + warp_column + n * 8 + (lane % 4) * 2 + e;
                    if (column >= op.positions) continue;
                    float value = epilogue.chunk_add ? det_add(values[0][n][e], values[1][n][e])
                                                     : values[h][n][e];
                    const auto index = static_cast<std::size_t>(row) * op.positions + column;
                    if (epilogue.residual != nullptr) value = det_add(value, epilogue.residual[index]);
                    op.output[index] = value;
                }
            }
        }
    }
#endif
}

// Dynamic shared memory above 48 KiB must be enabled per kernel.
template <int WeightLimbs, int ActivationLimbs, bool MeanLow>
cudaError_t launch_integer_kernel(dim3 grid, const IntegerOperands& operands,
                                  const IntegerEpilogue& epilogue, cudaStream_t stream) {
    using Shape = IntegerShape<WeightLimbs, ActivationLimbs>;
    static const cudaError_t configured = cudaFuncSetAttribute(
        integer_pointwise_kernel<WeightLimbs, ActivationLimbs, MeanLow>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::shared_bytes);
    if (configured != cudaSuccess) return configured;
    integer_pointwise_kernel<WeightLimbs, ActivationLimbs, MeanLow>
        <<<grid, integer_threads, Shape::shared_bytes, stream>>>(operands, epilogue);
    return cudaPeekAtLastError();
}

__global__ void half_to_float_kernel(const std::uint16_t* input, float* output, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = det_half_bits_to_float(input[index]);
}

__global__ void float_to_half_kernel(const float* input, std::uint16_t* output, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __half_as_ushort(__float2half_rn(input[index]));
}

// slice(0, C) + slice(C, 2C) of one 2C-channel tensor.
__global__ void chunk_add_kernel(const float* input, float* output, std::size_t half_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= half_count) return;
    output[index] = det_add(input[index], input[index + half_count]);
}

int round_up_to(int value, int multiple) { return (value + multiple - 1) / multiple * multiple; }

unsigned int blocks_for(std::size_t count) {
    return static_cast<unsigned int>((count + threads_per_block - 1) / threads_per_block);
}

Error cuda_failure(std::string_view operation, cudaError_t status) {
    return Error(ErrorCode::backend_error,
                 std::string(operation) + ": " + cudaGetErrorString(status), "deterministic.cuda");
}

struct DeviceBuffer {
    float* data{nullptr};
    std::size_t capacity{0};
    bool in_use{false};
};

struct DeviceTensor {
    Extent extent;
    std::size_t buffer{0};
};

// Device copy of one det-int-v1 pointwise conv's split weights.
struct IntegerConv {
    signed char* high{nullptr};
    signed char* low{nullptr};
    long long* row_constants{nullptr};
    float* scales{nullptr};
    float* bias{nullptr};
    bool chunk_add{false};  // rows ordered so each 16-row tile holds 8 pairs (r, r + C)
    int row_tiles{0};  // padded to whole warp blocks
    int k_tiles{0};
    int weight_limbs{2};
    int activation_limbs{2};
    bool mean_low{false};
};

// Nodes folded into an integer conv's epilogue.
struct ConvFusion {
    bool wsilu{false};
    float wsilu_scale{0.0F};
    bool chunk_add{false};
    std::optional<std::size_t> residual;  // tensor id
    std::size_t output{0};                // tensor id the fused group produces
};

struct ChunkAdd {
    std::size_t source{0};
    std::size_t output{0};
};

int tiles_for(int count) { return (count + tile - 1) / tile; }

}  // namespace

struct CudaExecutor::Impl final {
    const Program* program{nullptr};
    NumericsProfile profile{NumericsProfile::fp32_v1};
    float* weights{nullptr};
    cudaStream_t stream{nullptr};
    bool owns_stream{true};
    std::vector<DeviceBuffer> pool;
    std::vector<std::optional<IntegerConv>> integer_convs;
    std::vector<std::optional<ConvFusion>> conv_fusions;
    std::vector<std::optional<ChunkAdd>> chunk_adds;
    std::vector<bool> absorbed;
    std::vector<void*> integer_allocations;

    // Folds elementwise chains into integer conv epilogues and slice/slice/add
    // into one kernel. Only tensors with no other reader are folded away.
    void plan_fusions() {
        const auto& nodes = program->nodes;
        const auto tensor_count = program->tensor_names.size();
        std::vector<int> readers(tensor_count, 0);
        std::vector<std::ptrdiff_t> producer(tensor_count, -1);
        std::vector<bool> is_output(tensor_count, false);
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            for (std::size_t input = 0; input < nodes[index].input_count; ++input) {
                ++readers[nodes[index].inputs[input]];
            }
            producer[nodes[index].output] = static_cast<std::ptrdiff_t>(index);
        }
        for (const auto id : program->outputs) is_output[id] = true;
        auto private_tensor = [&](std::size_t id, int expected_readers) {
            return readers[id] == expected_readers && !is_output[id];
        };
        // nodes[at..at+2] are slice(source, 0, C), slice(source, C, 2C), add(first, second).
        auto chunk_add_at = [&](std::size_t at, std::size_t source) {
            if (at + 2 >= nodes.size()) return false;
            const auto& first = nodes[at];
            const auto& second = nodes[at + 1];
            const auto& add = nodes[at + 2];
            const auto channels = program->tensor_channels[source];
            return first.kind == OpKind::slice_channels && first.inputs[0] == source &&
                second.kind == OpKind::slice_channels && second.inputs[0] == source &&
                first.begin == 0 && first.end == channels / 2 && second.begin == channels / 2 &&
                second.end == channels && channels % 2 == 0 && add.kind == OpKind::add &&
                add.inputs[0] == first.output && add.inputs[1] == second.output &&
                private_tensor(first.output, 1) && private_tensor(second.output, 1);
        };
        conv_fusions.assign(nodes.size(), std::nullopt);
        chunk_adds.assign(nodes.size(), std::nullopt);
        absorbed.assign(nodes.size(), false);
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            if (absorbed[index]) continue;
            if (is_integer_pointwise(nodes[index])) {
                ConvFusion fusion;
                fusion.output = nodes[index].output;
                std::size_t last = index;
                if (index + 3 < nodes.size()) {
                    const auto& scale = nodes[index + 1];
                    const auto& gate = nodes[index + 2];
                    const auto& product = nodes[index + 3];
                    const auto conv_out = nodes[index].output;
                    const bool matches = scale.kind == OpKind::mul_scalar && scale.inputs[0] == conv_out &&
                        gate.kind == OpKind::sigmoid && gate.inputs[0] == scale.output &&
                        product.kind == OpKind::mul &&
                        ((product.inputs[0] == gate.output && product.inputs[1] == conv_out) ||
                         (product.inputs[1] == gate.output && product.inputs[0] == conv_out)) &&
                        private_tensor(conv_out, 2) && private_tensor(scale.output, 1) &&
                        private_tensor(gate.output, 1);
                    if (matches) {
                        fusion.wsilu = true;
                        fusion.wsilu_scale = scale.scalar;
                        fusion.output = product.output;
                        last = index + 3;
                    }
                }
                // slice(0, C) + slice(C, 2C), with C a multiple of 8 so each
                // 16-row tile can hold 8 row pairs.
                const auto channels = static_cast<std::size_t>(nodes[index].out_channels);
                if (last + 3 < nodes.size() && channels % 16 == 0 &&
                    chunk_add_at(last + 1, fusion.output) && private_tensor(fusion.output, 2)) {
                    fusion.chunk_add = true;
                    fusion.output = nodes[last + 3].output;
                    last += 3;
                }
                if (last + 1 < nodes.size() && nodes[last + 1].kind == OpKind::add &&
                    private_tensor(fusion.output, 1)) {
                    const auto& add = nodes[last + 1];
                    const auto other = add.inputs[0] == fusion.output ? add.inputs[1]
                        : add.inputs[1] == fusion.output ? add.inputs[0] : fusion.output;
                    if (other != fusion.output && producer[other] < static_cast<std::ptrdiff_t>(index)) {
                        fusion.residual = other;
                        fusion.output = add.output;
                        last += 1;
                    }
                }
                if (last != index) {
                    conv_fusions[index] = fusion;
                    for (auto folded = index + 1; folded <= last; ++folded) absorbed[folded] = true;
                }
                continue;
            }
            if (index + 2 < nodes.size() && nodes[index].kind == OpKind::slice_channels) {
                const auto& first = nodes[index];
                if (chunk_add_at(index, first.inputs[0])) {
                    chunk_adds[index] = ChunkAdd{first.inputs[0], nodes[index + 2].output};
                    absorbed[index + 1] = true;
                    absorbed[index + 2] = true;
                }
            }
        }
    }

    ~Impl() {
        for (auto* allocation : integer_allocations) cudaFree(allocation);
        for (auto& buffer : pool) cudaFree(buffer.data);
        if (weights != nullptr) cudaFree(weights);
        if (stream != nullptr && owns_stream) cudaStreamDestroy(stream);
    }

    Result<std::size_t> acquire(std::size_t elements) {
        std::optional<std::size_t> best;
        for (std::size_t index = 0; index < pool.size(); ++index) {
            if (!pool[index].in_use && pool[index].capacity >= elements &&
                (!best || pool[index].capacity < pool[*best].capacity)) {
                best = index;
            }
        }
        if (!best) {
            DeviceBuffer buffer;
            const auto status = cudaMalloc(reinterpret_cast<void**>(&buffer.data),
                                           std::max<std::size_t>(elements, 1) * sizeof(float));
            if (status != cudaSuccess) return cuda_failure("cudaMalloc", status);
            buffer.capacity = elements;
            pool.push_back(buffer);
            best = pool.size() - 1;
        }
        pool[*best].in_use = true;
        return *best;
    }

    template <typename T>
    Result<T*> upload_array(const std::vector<T>& values) {
        void* device = nullptr;
        auto status = cudaMalloc(&device, std::max<std::size_t>(values.size(), 1) * sizeof(T));
        if (status != cudaSuccess) return cuda_failure("cudaMalloc integer weights", status);
        integer_allocations.push_back(device);
        status = cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return cuda_failure("cudaMemcpy integer weights", status);
        return static_cast<T*>(device);
    }

    Result<void> prepare_integer_weights() {
        integer_convs.assign(program->nodes.size(), std::nullopt);
        for (std::size_t index = 0; index < program->nodes.size(); ++index) {
            const auto& node = program->nodes[index];
            if (!is_integer_pointwise(node)) continue;
            const auto quantized = quantize_weights(*program, node);
            IntegerConv conv;
            conv.weight_limbs = node.weight_bits == 16 ? 2 : 1;
            conv.activation_limbs = node.activation_bits == 16 ? 2 : 1;
            conv.mean_low = node.mean_low_product;
            conv.chunk_add = conv_fusions[index] && conv_fusions[index]->chunk_add;
            // Physical row -> node row. A chunk-add conv puts rows 8t..8t+7 and
            // C+8t..C+8t+7 in tile t, so one lane holds both rows of a pair.
            const int half = node.out_channels / 2;
            auto source_row = [&](int row) {
                if (!conv.chunk_add) return row;
                const int pair_tile = row / tile;
                const int within = row % tile;
                return within < tile / 2 ? pair_tile * (tile / 2) + within
                                         : half + pair_tile * (tile / 2) + within - tile / 2;
            };
            conv.row_tiles = round_up_to(tiles_for(node.out_channels), block_row_tiles);
            conv.k_tiles = round_up_to(tiles_for(node.in_channels), block_k_tiles);
            const auto padded = static_cast<std::size_t>(conv.row_tiles) * conv.k_tiles * tile * tile;
            std::vector<signed char> high(padded, 0);
            std::vector<signed char> low(padded, 0);
            std::vector<long long> high_sums(static_cast<std::size_t>(node.out_channels), 0);
            std::vector<long long> low_sums(static_cast<std::size_t>(node.out_channels), 0);
            std::vector<float> scales(static_cast<std::size_t>(node.out_channels));
            std::vector<float> bias(static_cast<std::size_t>(node.out_channels));
            for (int row = 0; row < node.out_channels; ++row) {
                const int source = source_row(row);
                scales[static_cast<std::size_t>(row)] = quantized.scales[static_cast<std::size_t>(source)];
                bias[static_cast<std::size_t>(row)] = program->weights[node.bias_offset + static_cast<std::size_t>(source)];
                for (int column = 0; column < node.in_channels; ++column) {
                    const int q = quantized.values[static_cast<std::size_t>(source) * node.in_channels + column];
                    const int h = conv.weight_limbs == 2 ? q >> 8 : q;
                    const int l = conv.weight_limbs == 2 ? (q & 255) - 128 : 0;
                    const auto offset = static_cast<std::size_t>(row) * conv.k_tiles * tile + column;
                    high[offset] = static_cast<signed char>(h);
                    low[offset] = static_cast<signed char>(l);
                    high_sums[static_cast<std::size_t>(row)] += h;
                    low_sums[static_cast<std::size_t>(row)] += l;
                }
            }
            auto high_device = upload_array(high);
            auto low_device = upload_array(low);
            // [activations two limbs] Aw * 128 * sum Wh
            //   + [both two limbs] (128 * sum Wl' + 16384 K), or the mean-low constant.
            std::vector<long long> row_constants(static_cast<std::size_t>(node.out_channels), 0);
            const long long weight_factor = conv.weight_limbs == 2 ? 256 : 1;
            for (std::size_t row = 0; row < row_constants.size(); ++row) {
                if (conv.activation_limbs == 2) row_constants[row] = weight_factor * 128LL * high_sums[row];
                if (conv.weight_limbs == 2 && conv.activation_limbs == 2) {
                    row_constants[row] += conv.mean_low
                        ? det_int_mean_low_product(node.in_channels)
                        : 128LL * low_sums[row] + 16384LL * node.in_channels;
                }
            }
            auto row_constants_device = upload_array(row_constants);
            auto scales_device = upload_array(scales);
            auto bias_device = upload_array(bias);
            if (!high_device) return high_device.error();
            if (!low_device) return low_device.error();
            if (!row_constants_device) return row_constants_device.error();
            if (!scales_device) return scales_device.error();
            if (!bias_device) return bias_device.error();
            conv.high = high_device.value();
            conv.low = low_device.value();
            conv.row_constants = row_constants_device.value();
            conv.scales = scales_device.value();
            conv.bias = bias_device.value();
            integer_convs[index] = conv;
        }
        return {};
    }

    // Scratch bytes come from the float pool, rounded up to whole floats.
    Result<std::size_t> acquire_bytes(std::size_t bytes) {
        return acquire((bytes + sizeof(float) - 1) / sizeof(float));
    }

    Result<void> launch_integer(const Node& node, const IntegerConv& conv, const DeviceTensor& in,
                                const DeviceTensor& output, IntegerEpilogue epilogue) {
        const int positions = in.extent.height * in.extent.width;
        const int position_tiles = round_up_to(tiles_for(positions), block_column_tiles);
        const auto padded = static_cast<std::size_t>(conv.k_tiles) * position_tiles * tile * tile;
        auto high = acquire_bytes(padded);
        auto low = acquire_bytes(padded);
        auto sums = acquire_bytes(static_cast<std::size_t>(positions) * (sizeof(long long) + sizeof(int)));
        if (!high) return high.error();
        if (!low) return low.error();
        if (!sums) return sums.error();
        auto* high_data = reinterpret_cast<signed char*>(pool[high.value()].data);
        auto* low_data = reinterpret_cast<signed char*>(pool[low.value()].data);
        auto* column_constants = reinterpret_cast<long long*>(pool[sums.value()].data);
        auto* exponents = reinterpret_cast<int*>(column_constants + positions);
        // Column terms of the exact decomposition (see the kernel).
        const long long high_multiplier = conv.weight_limbs == 2 ? (conv.activation_limbs == 2 ? 256LL : 1LL) * 128LL : 0;
        const long long low_multiplier =
            conv.weight_limbs == 2 && conv.activation_limbs == 2 && !conv.mean_low ? 128LL : 0;
        // The quantizer writes every padded tap as zero, so padding adds
        // nothing to the exact sums.
        const int k_padded = conv.k_tiles * tile;
        const dim3 quantize_block(quantize_positions, quantize_lanes);
        const auto quantize_grid = static_cast<unsigned int>(position_tiles * tile / quantize_positions);
        if (conv.activation_limbs == 2) {
            quantize_activations_kernel<2><<<quantize_grid, quantize_block, 0, stream>>>(
                pool[in.buffer].data, node.in_channels, positions, k_padded, high_data, low_data,
                high_multiplier, low_multiplier, column_constants, exponents);
        } else {
            quantize_activations_kernel<1><<<quantize_grid, quantize_block, 0, stream>>>(
                pool[in.buffer].data, node.in_channels, positions, k_padded, high_data, low_data,
                high_multiplier, low_multiplier, column_constants, exponents);
        }
        const dim3 grid(conv.row_tiles / block_row_tiles, position_tiles / block_column_tiles);
        const IntegerOperands operands{
            conv.high, conv.low, conv.row_constants, conv.scales,
            conv.bias, high_data, low_data, column_constants,
            exponents, pool[output.buffer].data, node.out_channels, positions, node.in_channels,
            k_padded};
        cudaError_t launched;
        if (conv.weight_limbs == 2 && conv.activation_limbs == 2) {
            launched = conv.mean_low ? launch_integer_kernel<2, 2, true>(grid, operands, epilogue, stream)
                                     : launch_integer_kernel<2, 2, false>(grid, operands, epilogue, stream);
        } else if (conv.weight_limbs == 2) {
            launched = launch_integer_kernel<2, 1, false>(grid, operands, epilogue, stream);
        } else if (conv.activation_limbs == 2) {
            launched = launch_integer_kernel<1, 2, false>(grid, operands, epilogue, stream);
        } else {
            launched = launch_integer_kernel<1, 1, false>(grid, operands, epilogue, stream);
        }
        if (launched != cudaSuccess) return cuda_failure("integer pointwise launch", launched);
        // Scratch is reusable once the kernels that read it are enqueued.
        pool[high.value()].in_use = false;
        pool[low.value()].in_use = false;
        pool[sums.value()].in_use = false;
        const auto status = cudaPeekAtLastError();
        if (status != cudaSuccess) return cuda_failure("integer pointwise launch", status);
        return {};
    }

    Result<void> launch(const Node& node, std::span<const DeviceTensor> inputs, const DeviceTensor& output,
                        std::size_t node_index) {
        if (profile == NumericsProfile::integer_v1 && integer_convs[node_index]) {
            return launch_integer(node, *integer_convs[node_index], inputs[0], output,
                                  IntegerEpilogue{0, 0.0F, 0, nullptr});
        }
        const auto& in = inputs[0];
        float* source = pool[in.buffer].data;
        float* target = pool[output.buffer].data;
        const auto count = static_cast<std::size_t>(output.extent.channels) *
            output.extent.height * output.extent.width;
        switch (node.kind) {
        case OpKind::conv: {
            const float* conv_weights = weights + node.weight_offset;
            const float* conv_bias = weights + node.bias_offset;
            const bool pointwise = !node.depthwise && node.kernel == std::array<std::int32_t, 2>{1, 1} &&
                node.stride == std::array<std::int32_t, 2>{1, 1} &&
                node.pads == std::array<std::int32_t, 4>{0, 0, 0, 0};
            if (pointwise) {
                const int positions = in.extent.height * in.extent.width;
                const dim3 grid((positions + tile_n - 1) / tile_n,
                                (node.out_channels + tile_m - 1) / tile_m);
                pointwise_conv_kernel<<<grid, threads_per_block, 0, stream>>>(
                    conv_weights, conv_bias, source, target, node.out_channels, positions,
                    node.in_channels);
            } else {
                const ConvGeometry geometry{
                    node.in_channels, node.out_channels, node.depthwise ? 1 : node.in_channels,
                    node.depthwise ? 1 : 0, in.extent.height, in.extent.width,
                    output.extent.height, output.extent.width, node.kernel[0], node.kernel[1],
                    node.stride[0], node.stride[1], node.pads[0], node.pads[1]};
                conv_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                    conv_weights, conv_bias, source, target, geometry);
            }
            break;
        }
        case OpKind::mul:
        case OpKind::add:
            binary_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, pool[inputs[1].buffer].data, target, count, node.kind == OpKind::mul ? 1 : 0);
            break;
        case OpKind::mul_scalar:
            mul_scalar_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, count, node.scalar);
            break;
        case OpKind::sigmoid:
            sigmoid_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(source, target, count);
            break;
        case OpKind::slice_channels: {
            const auto plane = static_cast<std::size_t>(in.extent.height) * in.extent.width;
            const auto status = cudaMemcpyAsync(
                target, source + static_cast<std::size_t>(node.begin) * plane, count * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) return cuda_failure("cudaMemcpyAsync slice", status);
            break;
        }
        case OpKind::depth_to_space:
            depth_to_space_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, output.extent.channels, output.extent.height, output.extent.width,
                in.extent.height, in.extent.width, node.block);
            break;
        case OpKind::space_to_depth:
            space_to_depth_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, output.extent.channels, output.extent.height, output.extent.width,
                in.extent.height, in.extent.width, node.block);
            break;
        case OpKind::im2col:
            im2col_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, output.extent.channels, output.extent.height, output.extent.width,
                in.extent.height, in.extent.width, node.kernel[0], node.kernel[1], node.stride[0],
                node.stride[1], node.pads[0], node.pads[1]);
            break;
        case OpKind::scale_channels:
            scale_channels_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, pool[inputs[1].buffer].data, target, count,
                static_cast<std::size_t>(in.extent.height) * in.extent.width);
            break;
        case OpKind::clamp:
            clamp_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, count, node.minimum, node.maximum);
            break;
        case OpKind::crop:
            crop_kernel<<<blocks_for(count), threads_per_block, 0, stream>>>(
                source, target, output.extent.channels, output.extent.height, output.extent.width,
                in.extent.height, in.extent.width);
            break;
        case OpKind::concat: {
            const auto first = static_cast<std::size_t>(in.extent.channels) * in.extent.height * in.extent.width;
            auto status = cudaMemcpyAsync(target, source, first * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            if (status == cudaSuccess) {
                status = cudaMemcpyAsync(target + first, pool[inputs[1].buffer].data,
                                         (count - first) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            if (status != cudaSuccess) return cuda_failure("cudaMemcpyAsync concat", status);
            break;
        }
        }
        const auto status = cudaPeekAtLastError();
        if (status != cudaSuccess) return cuda_failure("deterministic kernel launch", status);
        return {};
    }

    // Runs the program on device-resident inputs; returns the output tensors.
    Result<std::vector<DeviceTensor>> execute(std::vector<std::optional<DeviceTensor>> tensors) {
        auto release_inputs = [&](std::size_t index) {
            const auto& node = program->nodes[index];
            for (std::size_t input = 0; input < node.input_count; ++input) {
                const auto id = node.inputs[input];
                // Stream order makes reuse safe once the last reader is enqueued.
                if (program->last_use[id] == index && tensors[id]) {
                    pool[tensors[id]->buffer].in_use = false;
                    tensors[id].reset();
                }
            }
        };
        for (std::size_t index = 0; index < program->nodes.size(); ++index) {
            const auto& node = program->nodes[index];
            if (!absorbed.empty() && absorbed[index]) {
                release_inputs(index);
                continue;
            }
            if (!conv_fusions.empty() && conv_fusions[index]) {
                const auto& fusion = *conv_fusions[index];
                const auto& in = tensors[node.inputs[0]];
                if (!in) return Error(ErrorCode::internal_error, "released tensor read", "deterministic.cuda");
                const Extent in_extent = in->extent;
                auto extent = output_extent(*program, node, std::span(&in_extent, 1));
                if (!extent) return extent.error();
                if (fusion.chunk_add) extent.value().channels /= 2;
                const float* residual = nullptr;
                if (fusion.residual) {
                    const auto& tensor = tensors[*fusion.residual];
                    if (!tensor || tensor->extent.channels != extent.value().channels ||
                        tensor->extent.height != extent.value().height ||
                        tensor->extent.width != extent.value().width) {
                        return Error(ErrorCode::internal_error, "residual tensor mismatch", "deterministic.cuda");
                    }
                    residual = pool[tensor->buffer].data;
                }
                const auto elements = static_cast<std::size_t>(extent.value().channels) *
                    extent.value().height * extent.value().width;
                auto buffer = acquire(elements);
                if (!buffer) return buffer.error();
                const DeviceTensor output{extent.value(), buffer.value()};
                auto launched = launch_integer(
                    node, *integer_convs[index], *in, output,
                    IntegerEpilogue{fusion.wsilu ? 1 : 0, fusion.wsilu_scale, fusion.chunk_add ? 1 : 0, residual});
                if (!launched) return launched.error();
                tensors[fusion.output] = output;
                release_inputs(index);
                continue;
            }
            if (!chunk_adds.empty() && chunk_adds[index]) {
                const auto& chunk = *chunk_adds[index];
                const auto& in = tensors[chunk.source];
                if (!in) return Error(ErrorCode::internal_error, "released tensor read", "deterministic.cuda");
                const Extent extent{in->extent.channels / 2, in->extent.height, in->extent.width};
                const auto half = static_cast<std::size_t>(extent.channels) * extent.height * extent.width;
                auto buffer = acquire(half);
                if (!buffer) return buffer.error();
                chunk_add_kernel<<<blocks_for(half), threads_per_block, 0, stream>>>(
                    pool[in->buffer].data, pool[buffer.value()].data, half);
                const auto status = cudaPeekAtLastError();
                if (status != cudaSuccess) return cuda_failure("chunk add launch", status);
                tensors[chunk.output] = DeviceTensor{extent, buffer.value()};
                release_inputs(index);
                continue;
            }
            std::array<DeviceTensor, 2> inputs{};
            std::array<Extent, 2> extents{};
            for (std::size_t input = 0; input < node.input_count; ++input) {
                const auto& tensor = tensors[node.inputs[input]];
                if (!tensor) return Error(ErrorCode::internal_error, "released tensor read", "deterministic.cuda");
                inputs[input] = *tensor;
                extents[input] = tensor->extent;
            }
            auto extent = output_extent(*program, node, std::span(extents).first(node.input_count));
            if (!extent) return extent.error();
            const auto elements = static_cast<std::size_t>(extent.value().channels) *
                extent.value().height * extent.value().width;
            auto buffer = acquire(elements);
            if (!buffer) return buffer.error();
            const DeviceTensor output{extent.value(), buffer.value()};
            auto launched = launch(node, std::span(inputs).first(node.input_count), output, index);
            if (!launched) return launched.error();
            tensors[node.output] = output;
            release_inputs(index);
        }
        std::vector<DeviceTensor> outputs;
        for (const auto id : program->outputs) {
            if (!tensors[id]) return Error(ErrorCode::internal_error, "output not produced", "deterministic.cuda");
            outputs.push_back(*tensors[id]);
        }
        return outputs;
    }

    Result<std::vector<std::optional<DeviceTensor>>> upload(std::span<const Tensor> inputs) {
        if (inputs.size() != program->inputs.size()) {
            return Error(ErrorCode::invalid_argument, "wrong number of program inputs", "deterministic.cuda");
        }
        std::vector<std::optional<DeviceTensor>> tensors(program->tensor_names.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto& input = inputs[index];
            if (input.channels != program->tensor_channels[program->inputs[index]] ||
                input.values.size() != input.size()) {
                return Error(ErrorCode::invalid_argument, "input has the wrong shape", "deterministic.cuda");
            }
            auto buffer = acquire(input.values.size());
            if (!buffer) return buffer.error();
            const auto status = cudaMemcpyAsync(
                pool[buffer.value()].data, input.values.data(), input.values.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess) return cuda_failure("cudaMemcpyAsync input", status);
            tensors[program->inputs[index]] =
                DeviceTensor{Extent{input.channels, input.height, input.width}, buffer.value()};
        }
        return tensors;
    }

    void release_all() {
        for (auto& buffer : pool) buffer.in_use = false;
    }
};

CudaExecutor::CudaExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaExecutor::CudaExecutor(CudaExecutor&&) noexcept = default;
CudaExecutor& CudaExecutor::operator=(CudaExecutor&&) noexcept = default;
CudaExecutor::~CudaExecutor() = default;

Result<CudaExecutor> CudaExecutor::create(const Program& program, NumericsProfile profile,
                                          CUstream_st* external_stream) {
    auto impl = std::make_unique<Impl>();
    impl->program = &program;
    impl->profile = profile;
    if (profile == NumericsProfile::integer_v1) {
        // The det-int-v1 kernels use mma.m16n8k32 (s8), which needs SM 8.0.
        int device = 0;
        int major = 0;
        auto queried = cudaGetDevice(&device);
        if (queried == cudaSuccess) {
            queried = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        }
        if (queried != cudaSuccess) return cuda_failure("cudaDeviceGetAttribute", queried);
        if (major < 8) {
            return Error(ErrorCode::incompatible_target,
                         "nvcr.det-int.v1 on CUDA needs compute capability 8.0 or newer",
                         "deterministic.cuda");
        }
    }
    cudaError_t status = cudaSuccess;
    if (external_stream != nullptr) {
        impl->stream = external_stream;
        impl->owns_stream = false;
    } else {
        status = cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) return cuda_failure("cudaStreamCreate", status);
    }
    const auto bytes = std::max<std::size_t>(program.weights.size(), 1) * sizeof(float);
    status = cudaMalloc(reinterpret_cast<void**>(&impl->weights), bytes);
    if (status != cudaSuccess) return cuda_failure("cudaMalloc weights", status);
    status = cudaMemcpy(impl->weights, program.weights.data(),
                        program.weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return cuda_failure("cudaMemcpy weights", status);
    if (profile == NumericsProfile::integer_v1) {
        impl->plan_fusions();
        auto prepared = impl->prepare_integer_weights();
        if (!prepared) return prepared.error();
    }
    return CudaExecutor(std::move(impl));
}

Result<void> CudaExecutor::run_half(
    std::span<const HalfDeviceTensor> inputs, std::span<const HalfDeviceTensor> outputs) {
    auto& impl = *impl_;
    const auto& program = *impl.program;
    auto find = [](std::span<const HalfDeviceTensor> tensors, std::string_view name) -> const HalfDeviceTensor* {
        for (const auto& tensor : tensors) {
            if (tensor.name == name) return &tensor;
        }
        return nullptr;
    };
    auto count_of = [](Extent extent) {
        return static_cast<std::size_t>(extent.channels) * extent.height * extent.width;
    };
    std::vector<std::optional<DeviceTensor>> tensors(program.tensor_names.size());
    for (const auto id : program.inputs) {
        const auto* input = find(inputs, program.tensor_names[id]);
        if (input == nullptr || input->data == nullptr || input->extent.channels != program.tensor_channels[id] ||
            input->extent.height <= 0 || input->extent.width <= 0) {
            impl.release_all();
            return Error(ErrorCode::invalid_argument,
                         program.name + ": missing or misshapen input " + program.tensor_names[id],
                         "deterministic.cuda");
        }
        const auto count = count_of(input->extent);
        auto buffer = impl.acquire(count);
        if (!buffer) {
            impl.release_all();
            return buffer.error();
        }
        half_to_float_kernel<<<blocks_for(count), threads_per_block, 0, impl.stream>>>(
            static_cast<const std::uint16_t*>(input->data), impl.pool[buffer.value()].data, count);
        tensors[id] = DeviceTensor{input->extent, buffer.value()};
    }
    auto results = impl.execute(std::move(tensors));
    if (!results) {
        impl.release_all();
        return results.error();
    }
    for (std::size_t slot = 0; slot < program.outputs.size(); ++slot) {
        const auto& result = results.value()[slot];
        const auto* output = find(outputs, program.tensor_names[program.outputs[slot]]);
        if (output == nullptr || output->data == nullptr || output->extent.channels != result.extent.channels ||
            output->extent.height != result.extent.height || output->extent.width != result.extent.width) {
            impl.release_all();
            return Error(ErrorCode::invalid_argument,
                         program.name + ": missing or misshapen output " +
                             program.tensor_names[program.outputs[slot]],
                         "deterministic.cuda");
        }
        const auto count = count_of(result.extent);
        float_to_half_kernel<<<blocks_for(count), threads_per_block, 0, impl.stream>>>(
            impl.pool[result.buffer].data, static_cast<std::uint16_t*>(output->data), count);
    }
    // Stream order makes the pool reusable once the conversions are enqueued.
    impl.release_all();
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) return cuda_failure("half conversion launch", status);
    return {};
}

Result<std::vector<Tensor>> CudaExecutor::run(std::span<const Tensor> inputs) {
    auto uploaded = impl_->upload(inputs);
    if (!uploaded) return uploaded.error();
    auto outputs = impl_->execute(std::move(uploaded.value()));
    if (!outputs) {
        impl_->release_all();
        return outputs.error();
    }
    std::vector<Tensor> results;
    for (const auto& output : outputs.value()) {
        Tensor tensor{output.extent.channels, output.extent.height, output.extent.width, {}};
        tensor.values.resize(tensor.size());
        const auto status = cudaMemcpyAsync(
            tensor.values.data(), impl_->pool[output.buffer].data,
            tensor.values.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream);
        if (status != cudaSuccess) return cuda_failure("cudaMemcpyAsync output", status);
        results.push_back(std::move(tensor));
    }
    const auto status = cudaStreamSynchronize(impl_->stream);
    impl_->release_all();
    if (status != cudaSuccess) return cuda_failure("cudaStreamSynchronize", status);
    return results;
}

Result<double> CudaExecutor::time(std::span<const Tensor> inputs, int iterations) {
    if (iterations <= 0) {
        return Error(ErrorCode::invalid_argument, "iterations must be positive", "deterministic.cuda");
    }
    auto warm = run(inputs);
    if (!warm) return warm.error();
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto uploaded = impl_->upload(inputs);
        if (!uploaded) return uploaded.error();
        cudaEventRecord(start, impl_->stream);
        auto outputs = impl_->execute(std::move(uploaded.value()));
        cudaEventRecord(stop, impl_->stream);
        if (!outputs) return outputs.error();
        const auto status = cudaEventSynchronize(stop);
        if (status != cudaSuccess) return cuda_failure("cudaEventSynchronize", status);
        float milliseconds = 0.0F;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;
        impl_->release_all();
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / iterations;
}

}  // namespace nvcr::deterministic
