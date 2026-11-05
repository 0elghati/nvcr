#pragma once

// det-fp32-v1 arithmetic, shared verbatim by the CPU reference and the CUDA
// kernels (issue #216).
//
// Every operation here is a single IEEE-754 binary32 operation rounded to
// nearest-even (fma, mul, add, div, rint) or an exact bit manipulation. Those
// are specified to the last bit, so any target that performs the same sequence
// produces the same bits. What is deliberately avoided:
//   - compiler contraction or reassociation: device code uses the _rn
//     intrinsics, and host code is compiled with -ffp-contract=off;
//   - hardware approximations and libm (expf, logf, __expf): exp is built here;
//   - denormal-dependent results: exp never returns a denormal, and the CPU
//     executor clears flush-to-zero so convolutions match the GPU (which keeps
//     denormals because device code is not built with -ftz).

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define NVCR_DET_HD __host__ __device__ __forceinline__
#else
#include <cmath>
#define NVCR_DET_HD inline
#endif

namespace nvcr::deterministic {

NVCR_DET_HD float det_fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

NVCR_DET_HD float det_mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

NVCR_DET_HD float det_add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

NVCR_DET_HD float det_div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

// Round to nearest integer, ties to even.
NVCR_DET_HD float det_rint(float value) {
#if defined(__CUDA_ARCH__)
    return rintf(value);
#else
    return std::nearbyint(value);
#endif
}

NVCR_DET_HD float det_bits_to_float(std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

NVCR_DET_HD std::uint32_t det_float_to_bits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

// exp(x) to about 2 ulp, identical on every target. Inputs below -87 return 0
// (the true result would be denormal) and inputs at or above 88 return +inf.
NVCR_DET_HD float det_exp(float x) {
    if (!(x == x)) return x;
    if (x >= 88.0F) return det_bits_to_float(0x7f800000U);
    if (x < -87.0F) return 0.0F;
    // x = k ln2 + r with |r| <= ln2 / 2, computed with Cody-Waite constants.
    const float k = det_rint(det_mul(x, 1.44269502162933349609375F));
    float r = det_fma(-k, 0.693359375F, x);
    r = det_fma(-k, -2.12194440e-4F, r);
    // Taylor series of exp(r) through r^7; truncation error < 6e-9 on |r| <= 0.35.
    float p = 1.98412698e-4F;  // 1/5040
    p = det_fma(p, r, 1.38888889e-3F);  // 1/720
    p = det_fma(p, r, 8.33333377e-3F);  // 1/120
    p = det_fma(p, r, 4.16666679e-2F);  // 1/24
    p = det_fma(p, r, 1.66666672e-1F);  // 1/6
    p = det_fma(p, r, 0.5F);
    p = det_fma(p, r, 1.0F);
    p = det_fma(p, r, 1.0F);
    // k is in [-126, 127], so 2^k is a normal float built exactly from bits.
    const auto exponent = static_cast<std::int32_t>(k) + 127;
    return det_mul(p, det_bits_to_float(static_cast<std::uint32_t>(exponent) << 23U));
}

NVCR_DET_HD float det_sigmoid(float x) {
    return det_div(1.0F, det_add(1.0F, det_exp(-x)));
}

// ldexp: exact when the result is representable, rounded to nearest otherwise.
NVCR_DET_HD float det_ldexp(float value, int exponent) {
    // Multiplying by a normal power of two is exact unless the result is
    // subnormal or overflows, and then rounds once to nearest, exactly as
    // ldexp does. It is much cheaper than the library call.
    if (exponent >= -126 && exponent <= 127) {
        return det_mul(value, det_bits_to_float(static_cast<std::uint32_t>(exponent + 127) << 23U));
    }
#if defined(__CUDA_ARCH__)
    return ldexpf(value, exponent);
#else
    return std::ldexp(value, exponent);
#endif
}

// Binary exponent E with value = f * 2^E, 0.5 <= |f| < 1 (frexp).
NVCR_DET_HD int det_frexp_exponent(float value) {
    int exponent = 0;
#if defined(__CUDA_ARCH__)
    (void)frexpf(value, &exponent);
#else
    (void)std::frexp(value, &exponent);
#endif
    return exponent;
}

// Exact integer to binary32, rounded to nearest-even.
NVCR_DET_HD float det_int64_to_float(long long value) {
#if defined(__CUDA_ARCH__)
    // Both conversions round the same integer to nearest-even.
    if (value >= -2147483648LL && value <= 2147483647LL) return __int2float_rn(static_cast<int>(value));
    return __ll2float_rn(value);
#else
    return static_cast<float>(value);
#endif
}

NVCR_DET_HD bool det_isfinite(float value) {
    return value == value && value - value == 0.0F;
}

// det-int-v1 activation exponent for a block whose largest magnitude is
// `maximum`, at 8 or 16 bits (see program.hpp).
NVCR_DET_HD int det_int_exponent(float maximum, int bits) {
    return maximum > 0.0F ? det_frexp_exponent(maximum) - (bits - 1) : 0;
}

NVCR_DET_HD int det_int_quantize(float value, int exponent, int bits) {
    const float limit = bits == 16 ? 32767.0F : 127.0F;
    const float scaled = det_rint(det_ldexp(value, -exponent));
    return scaled > limit ? static_cast<int>(limit)
                          : (scaled < -limit ? -static_cast<int>(limit) : static_cast<int>(scaled));
}

NVCR_DET_HD float det_int_output(long long accumulator, float scale, int exponent, float bias) {
    return det_add(det_ldexp(det_mul(det_int64_to_float(accumulator), scale), exponent), bias);
}

// Mean-mode replacement for the low x low limb products of a 16x16-bit sum:
// floor(K * 127.5^2).
NVCR_DET_HD long long det_int_mean_low_product(int channels) {
    return (65025LL * channels) / 4;
}

// Exact IEEE binary16 to binary32 conversion (weights and FP16 inputs).
NVCR_DET_HD float det_half_bits_to_float(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16U;
    std::uint32_t exponent = (half >> 10U) & 0x1fU;
    std::uint32_t mantissa = half & 0x3ffU;
    if (exponent == 0x1fU) {
        return det_bits_to_float(sign | 0x7f800000U | (mantissa << 13U));
    }
    if (exponent == 0) {
        if (mantissa == 0) return det_bits_to_float(sign);
        // Subnormal half: normalize into a binary32 normal.
        std::uint32_t shift = 0;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1U;
            ++shift;
        }
        mantissa &= 0x3ffU;
        exponent = 1;
        return det_bits_to_float(
            sign | ((exponent + 112U - shift) << 23U) | (mantissa << 13U));
    }
    return det_bits_to_float(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

// IEEE binary32 to binary16, round to nearest even (as __float2half_rn), and
// the storage rounding of det-int-v1 tensors built on it. NaN payloads are
// unspecified.
NVCR_DET_HD std::uint16_t det_float_to_half_bits(float value) {
    const std::uint32_t bits = det_float_to_bits(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0U ? 0x200U : 0U));
    }
    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (unbiased >= -14) {
        // Normal half: keep 10 mantissa bits, round the 13 dropped bits.
        std::uint32_t half = (static_cast<std::uint32_t>(unbiased + 15) << 10U) | (mantissa >> 13U);
        const std::uint32_t dropped = mantissa & 0x1fffU;
        if (dropped > 0x1000U || (dropped == 0x1000U && (half & 1U) != 0U)) ++half;  // may carry to inf
        return static_cast<std::uint16_t>(sign | half);
    }
    if (unbiased < -25) return static_cast<std::uint16_t>(sign);  // below half the smallest subnormal
    // Subnormal half: value = m * 2^-24 with the implicit bit restored.
    mantissa |= 0x800000U;
    const auto shift = static_cast<std::uint32_t>(-unbiased - 14 + 13);  // 14..24
    std::uint32_t half = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U) != 0U)) ++half;
    return static_cast<std::uint16_t>(sign | half);
}

NVCR_DET_HD float det_round_half(float value) {
    return det_half_bits_to_float(det_float_to_half_bits(value));
}

}  // namespace nvcr::deterministic
