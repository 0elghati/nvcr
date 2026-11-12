#include "fx.hpp"

#include <algorithm>
#include <string>

namespace nvcr::deterministic::fx {
namespace {

Error failure(std::string what) {
    return Error(ErrorCode::malformed_bitstream, "nvcr.fx.v1: " + std::move(what), "deterministic.fx");
}

Tensor like(const Tensor& x, std::int32_t channels) {
    Tensor y{channels, x.height, x.width, {}};
    y.values.resize(static_cast<std::size_t>(channels) * x.plane());
    return y;
}

// Stores v, or reports a stored value outside the profile's bound.
bool store(std::int64_t v, std::int32_t& out) {
    if (v > stored_max || v < -stored_max) return false;
    out = static_cast<std::int32_t>(v);
    return true;
}

}  // namespace

std::int64_t rsr(std::int64_t a, int s) noexcept {
    return s == 0 ? a : (a + (std::int64_t{1} << (s - 1))) >> s;
}

std::int64_t delta_act(std::int64_t r, const DeltaTable& table) noexcept {
    const std::int64_t base = std::max<std::int64_t>(r, 0);
    if (r < -table.h || r > table.h) return base;
    return base + table.dc[static_cast<std::size_t>((r + table.h) >> table.shift)];
}

Result<Tensor> gemm(const Gemm& op, const Tensor& x, std::span<const Tensor* const> residuals) {
    const std::size_t n = x.plane();
    Tensor y = like(x, op.out_channels / op.group);
    for (std::size_t p = 0; p < n; ++p) {
        for (std::int32_t g = 0; g < y.channels; ++g) {
            std::int64_t sum = 0;
            for (int j = 0; j < op.group; ++j) {
                const auto o = static_cast<std::size_t>(g * op.group + j);
                std::int64_t acc = op.bias[o];
                for (std::int32_t k = 0; k < op.in_channels; ++k) {
                    const auto xk = static_cast<std::size_t>(k) * n + p;
                    const auto xi = std::clamp(rsr(x.values[xk], op.input_shift), -digit2_max, digit2_max);
                    acc += op.weights[o * static_cast<std::size_t>(op.in_channels) + static_cast<std::size_t>(k)] * xi;
                }
                const std::int64_t r = rsr(acc, op.shift[o]);
                if (r >= gemm_limit || r <= -gemm_limit) return failure("gemm pre-activation out of range");
                sum += op.activation ? delta_act(r, *op.activation) : r;
            }
            const auto at = static_cast<std::size_t>(g) * n + p;
            for (const Tensor* residual : residuals) sum += residual->values[at];
            if (!store(sum, y.values[at])) return failure("gemm output out of range");
        }
    }
    return y;
}

Result<Tensor> depthwise3x3(const Depthwise& op, const Tensor& x) {
    Tensor y = like(x, x.channels);
    for (std::int32_t c = 0; c < x.channels; ++c) {
        const auto ci = static_cast<std::size_t>(c);
        for (std::int32_t row = 0; row < x.height; ++row) {
            for (std::int32_t col = 0; col < x.width; ++col) {
                std::int64_t acc = op.bias[ci];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        const int yy = row + i - 1;
                        const int xx = col + j - 1;
                        if (yy < 0 || yy >= x.height || xx < 0 || xx >= x.width) continue;
                        acc += std::int64_t{op.weights[ci * 9 + static_cast<std::size_t>(i * 3 + j)]} *
                            x.values[ci * x.plane() + static_cast<std::size_t>(yy * x.width + xx)];
                    }
                }
                const auto at = ci * x.plane() + static_cast<std::size_t>(row * x.width + col);
                if (!store(rsr(acc, op.shift[ci]), y.values[at])) return failure("depthwise output out of range");
            }
        }
    }
    return y;
}

Result<Tensor> mul(const Tensor& a, const Tensor& b, int shift) {
    Tensor y = like(a, a.channels);
    const std::size_t plane = a.plane();
    const std::size_t b_plane = b.plane();
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        const std::size_t c = b.channels == 1 ? 0 : i / plane;
        const std::size_t p = b_plane == 1 ? 0 : i % plane;
        const std::int64_t product = std::int64_t{a.values[i]} * b.values[c * b_plane + p];
        if (!store(rsr(product, shift), y.values[i])) return failure("mul output out of range");
    }
    return y;
}

Result<Tensor> affine(const Tensor& x, std::int64_t alpha, std::int64_t beta, int shift) {
    Tensor y = like(x, x.channels);
    for (std::size_t i = 0; i < x.values.size(); ++i) {
        if (!store(rsr(alpha * x.values[i] + beta, shift), y.values[i])) return failure("affine output out of range");
    }
    return y;
}

Result<Tensor> add(const Tensor& a, const Tensor& b) {
    Tensor y = like(a, a.channels);
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        if (!store(std::int64_t{a.values[i]} + b.values[i], y.values[i])) return failure("add output out of range");
    }
    return y;
}

Tensor sat_table(const Tensor& x, std::span<const std::int32_t> table, std::int32_t top) {
    Tensor y = like(x, x.channels);
    const auto n = static_cast<std::int64_t>(table.size() / 2);
    for (std::size_t i = 0; i < x.values.size(); ++i) {
        const std::int64_t v = x.values[i];
        y.values[i] = v >= n ? top : v < -n ? 0 : table[static_cast<std::size_t>(v + n)];
    }
    return y;
}

Tensor delta(const Tensor& x, const DeltaTable& table) {
    Tensor y = like(x, x.channels);
    for (std::size_t i = 0; i < x.values.size(); ++i) {
        y.values[i] = static_cast<std::int32_t>(delta_act(x.values[i], table));
    }
    return y;
}

Tensor clamp(const Tensor& x, std::int32_t lo, std::int32_t hi) {
    Tensor y = like(x, x.channels);
    for (std::size_t i = 0; i < x.values.size(); ++i) y.values[i] = std::clamp(x.values[i], lo, hi);
    return y;
}

}  // namespace nvcr::deterministic::fx
