#pragma once

// Device-side types for the DCVC-UF glue kernels generated from pinned
// upstream sources (glue_kernels.inc). They reproduce what those kernels rely
// on from libtorch and upstream's common_cu.h/common_cutlass.h/def_const.h:
//
// - Half has c10::Half's device semantics: conversions round to nearest via
//   __float2half/__half2float, Half op Half computes in float and rounds back,
//   Half op float yields float, and only negation uses a half intrinsic.
// - GPUTensorND indexes like at::PackedTensorAccessor32 (any strides).
//
// Compile with upstream's flags (-O3 --use_fast_math
// --extra-device-vectorization) and its __CUDA_NO_HALF*_ defines, which make
// every Half comparison and operation go through float as upstream's do.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace nvcr::dcvcuf::glue {

struct alignas(2) Half {
    unsigned short x;

    Half() = default;
    __host__ __device__ Half(float value) : x(bits(__float2half(value))) {}
    __host__ __device__ Half(const __half& value) : x(bits(value)) {}
    __host__ __device__ operator float() const { return __half2float(as_half()); }
    __host__ __device__ operator __half() const { return as_half(); }

private:
    __host__ __device__ static unsigned short bits(__half value) {
        return *reinterpret_cast<const unsigned short*>(&value);
    }
    __host__ __device__ __half as_half() const {
        return *reinterpret_cast<const __half*>(&x);
    }
};

__host__ __device__ inline Half operator+(const Half& a, const Half& b) {
    return static_cast<float>(a) + static_cast<float>(b);
}
__host__ __device__ inline Half operator-(const Half& a, const Half& b) {
    return static_cast<float>(a) - static_cast<float>(b);
}
__host__ __device__ inline Half operator*(const Half& a, const Half& b) {
    return static_cast<float>(a) * static_cast<float>(b);
}
__host__ __device__ inline Half operator/(const Half& a, const Half& b) {
    return static_cast<float>(a) / static_cast<float>(b);
}
__device__ inline Half operator-(const Half& a) { return __hneg(a); }
__host__ __device__ inline Half& operator+=(Half& a, const Half& b) { return a = a + b; }
__host__ __device__ inline Half& operator-=(Half& a, const Half& b) { return a = a - b; }
__host__ __device__ inline Half& operator*=(Half& a, const Half& b) { return a = a * b; }
__host__ __device__ inline Half& operator/=(Half& a, const Half& b) { return a = a / b; }
__host__ __device__ inline float operator+(Half a, float b) { return static_cast<float>(a) + b; }
__host__ __device__ inline float operator-(Half a, float b) { return static_cast<float>(a) - b; }
__host__ __device__ inline float operator*(Half a, float b) { return static_cast<float>(a) * b; }
__host__ __device__ inline float operator/(Half a, float b) { return static_cast<float>(a) / b; }
__host__ __device__ inline float operator+(float a, Half b) { return a + static_cast<float>(b); }
__host__ __device__ inline float operator-(float a, Half b) { return a - static_cast<float>(b); }
__host__ __device__ inline float operator*(float a, Half b) { return a * static_cast<float>(b); }
__host__ __device__ inline float operator/(float a, Half b) { return a / static_cast<float>(b); }
__host__ __device__ inline Half operator+(Half a, int b) { return a + static_cast<Half>(b); }
__host__ __device__ inline Half operator-(Half a, int b) { return a - static_cast<Half>(b); }
__host__ __device__ inline Half operator*(Half a, int b) { return a * static_cast<Half>(b); }
__host__ __device__ inline Half operator/(Half a, int b) { return a / static_cast<Half>(b); }
__host__ __device__ inline Half operator+(int a, Half b) { return static_cast<Half>(a) + b; }
__host__ __device__ inline Half operator-(int a, Half b) { return static_cast<Half>(a) - b; }
__host__ __device__ inline Half operator*(int a, Half b) { return static_cast<Half>(a) * b; }
__host__ __device__ inline Half operator/(int a, Half b) { return static_cast<Half>(a) / b; }

// at::PackedTensorAccessor32 / at::TensorAccessor indexing.
template <typename T, std::size_t N>
struct TensorAccessor {
    T* data;
    const std::int32_t* sizes;
    const std::int32_t* strides;

    __device__ TensorAccessor<T, N - 1> operator[](std::int32_t index) const {
        return {data + strides[0] * index, sizes + 1, strides + 1};
    }
};

template <typename T>
struct TensorAccessor<T, 1> {
    T* data;
    const std::int32_t* sizes;
    const std::int32_t* strides;

    __device__ T& operator[](std::int32_t index) const { return data[strides[0] * index]; }
};

template <typename T, std::size_t N>
struct GPUTensorND {
    T* __restrict__ data;
    std::int32_t sizes[N];
    std::int32_t strides[N];

    __device__ decltype(auto) operator[](std::int32_t index) const {
        if constexpr (N == 1) {
            return static_cast<T&>(data[strides[0] * index]);
        } else {
            return TensorAccessor<T, N - 1>{data + strides[0] * index, sizes + 1, strides + 1};
        }
    }
};

template <typename T>
struct GPUTensor1D {
    T* __restrict__ const ptr;

    __device__ T& operator[](int idx) { return ptr[idx]; }
    __device__ T& operator[](int idx) const { return ptr[idx]; }
};

template <typename T>
using GPUTensor2D = GPUTensorND<T, 2>;

template <typename T>
using GPUTensor4D = GPUTensorND<T, 4>;

struct CudaCHW {
    int C;
    int H;
    int W;
};

// ---- upstream def_const.h
constexpr float SCALE_MIN = 0.11f;
constexpr float SCALE_MAX = 16.f;
constexpr float LOG_SCALE_MIN = -2.2073f;  // std::log(SCALE_MIN);
constexpr float LOG_SCALE_MAX = 2.7726f;   // std::log(SCALE_MAX)
constexpr int SCALE_LEVEL = 128;
constexpr float LOG_SCALE_STEP = (LOG_SCALE_MAX - LOG_SCALE_MIN) / (SCALE_LEVEL - 1);
constexpr float LOG_SCALE_STEP_RECIP = 1.f / LOG_SCALE_STEP;

constexpr int COND_KERNEL_THREAD_NUM1 = 1024;
constexpr int COND_KERNEL_THREAD_NUM2 = 1024;
constexpr int COND_KERNEL_PER_THREAD_NUM = 8;

// ---- upstream common_cutlass.h
// Upstream keys this on a single build-wide CURRENT_DEVICE_SM; __CUDA_ARCH__
// selects per target, so multi-architecture builds get the right branch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
    #define cudaGDC() cudaGridDependencySynchronize()
#else
    #define cudaGDC() ((void)0)
#endif

__device__ __forceinline__ std::tuple<int, int, int> global_idx_chw()
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = blockIdx.z * blockDim.z + threadIdx.z;
    return { c, h, w };
}

// ---- upstream common_cu.h device helpers (at::Half renamed to Half)
struct __align__(2) bool2
{
    bool x;
    bool y;
};

template <typename T>
__forceinline__ __device__ void load2(const T* src, T& dst0, T& dst1)
{
    if constexpr (std::is_same_v<T, Half>) {
        half2 v = *reinterpret_cast<const half2*>(src);
        dst0 = __low2half(v);
        dst1 = __high2half(v);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        short2 v = *reinterpret_cast<const short2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        char2 v = *reinterpret_cast<const char2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        uchar2 v = *reinterpret_cast<const uchar2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, bool>) {
        bool2 v = *reinterpret_cast<const bool2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else {
        dst0 = src[0];
        dst1 = src[1];
    }
}

__forceinline__ __device__ Half log(const Half& a)
{
    return hlog(a);
}

__forceinline__ __device__ bool2 make_bool2(bool x, bool y)
{
    bool2 t;
    t.x = x;
    t.y = y;
    return t;
}

__forceinline__ __device__ Half max(const Half& a, const Half& b)
{
    return __hmax(a, b);
}

__forceinline__ __device__ Half min(const Half& a, const Half& b)
{
    return __hmin(a, b);
}

__forceinline__ __device__ Half operator*(const Half& a, const bool b)
{
    return b ? a : static_cast<Half>(0.f);
}

__forceinline__ __device__ Half reciprocal(const Half& a)
{
    return hrcp(a);
}

template <typename T>
__forceinline__ __device__ void store2(T* dst, T src0, T src1)
{
    if constexpr (std::is_same_v<T, Half>) {
        *reinterpret_cast<half2*>(dst) = __halves2half2(src0, src1);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        *reinterpret_cast<short2*>(dst) = make_short2(src0, src1);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        *reinterpret_cast<char2*>(dst) = make_char2(src0, src1);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        *reinterpret_cast<uchar2*>(dst) = make_uchar2(src0, src1);
    } else if constexpr (std::is_same_v<T, bool>) {
        *reinterpret_cast<bool2*>(dst) = make_bool2(src0, src1);
    } else {
        dst[0] = src0;
        dst[1] = src1;
    }
}

__forceinline__ __device__ int16_t to_int16(const Half& a)
{
    return static_cast<int16_t>(__half2int_rd(a));
}

__forceinline__ __device__ uint8_t to_uint8(const Half& a)
{
    return static_cast<uint8_t>(__half2uint_rd(a));
}

}  // namespace nvcr::dcvcuf::glue
