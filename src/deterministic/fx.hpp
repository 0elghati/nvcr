#pragma once

// nvcr.fx.v1 reference arithmetic (docs/spec/numerics-fx-v1.md). Plain loops on
// int64: every provider implementation must produce these integers and fail on
// exactly the inputs these functions reject.

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nvcr::deterministic::fx {

inline constexpr std::int64_t digit2_max = 32639;                    // two balanced int8 digits
inline constexpr std::int64_t stored_max = std::int64_t{1} << 30;    // |stored value| bound
inline constexpr std::int64_t gemm_limit = (std::int64_t{1} << 29) - 1;  // |r| must stay below

// Batch-1 NCHW int32 tensor.
struct Tensor final {
    std::int32_t channels{0};
    std::int32_t height{0};
    std::int32_t width{0};
    std::vector<std::int32_t> values;

    [[nodiscard]] std::size_t plane() const noexcept {
        return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    }
};

// floor((a + 2^(s-1)) / 2^s) for s >= 1; a for s = 0.
[[nodiscard]] std::int64_t rsr(std::int64_t a, int s) noexcept;

// act(r) = max(r, 0) + dc[(r + h) >> shift] on [-h, h], max(r, 0) outside.
struct DeltaTable final {
    std::int64_t h{0};
    int shift{0};
    std::vector<std::int16_t> dc;
};
[[nodiscard]] std::int64_t delta_act(std::int64_t r, const DeltaTable& table) noexcept;

struct Gemm final {
    std::int32_t in_channels{0};
    std::int32_t out_channels{0};       // before grouping
    std::vector<std::int32_t> weights;  // [out][in], |w| <= digit2_max
    std::vector<std::int64_t> bias;     // [out]
    std::vector<std::int32_t> shift;    // [out], 1..31
    int input_shift{0};
    const DeltaTable* activation{nullptr};
    int group{1};                       // 1, 2 or 4 adjacent rows summed
};

[[nodiscard]] Result<Tensor> gemm(const Gemm& op, const Tensor& x, std::span<const Tensor* const> residuals);

struct Depthwise final {
    std::vector<std::int32_t> weights;  // [channels][3][3]
    std::vector<std::int64_t> bias;
    std::vector<std::int32_t> shift;
};

[[nodiscard]] Result<Tensor> depthwise3x3(const Depthwise& op, const Tensor& x);
// rsr(a * b, shift); b has a's channels or one, and a's size or 1x1 (broadcast).
[[nodiscard]] Result<Tensor> mul(const Tensor& a, const Tensor& b, int shift);
[[nodiscard]] Result<Tensor> affine(const Tensor& x, std::int64_t alpha, std::int64_t beta, int shift);
[[nodiscard]] Result<Tensor> add(const Tensor& a, const Tensor& b);
// T[x + n] on [-n, n), top above, 0 below; table holds 2n entries.
[[nodiscard]] Tensor sat_table(const Tensor& x, std::span<const std::int32_t> table, std::int32_t top);
[[nodiscard]] Tensor delta(const Tensor& x, const DeltaTable& table);
[[nodiscard]] Tensor clamp(const Tensor& x, std::int32_t lo, std::int32_t hi);

}  // namespace nvcr::deterministic::fx
