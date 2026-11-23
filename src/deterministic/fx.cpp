#include "fx.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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

Result<Tensor> gemm_tiled(const Gemm& op, const Tensor& x, std::span<const Tensor* const> residuals) {
    // |w|, |x_i| <= digit2_max, so one product fits int16 x int16 -> int32 and two
    // products still fit int32: 2 * 32639^2 < 2^31. The int64 sums are exact
    // (spec section 5), so the order they are formed in cannot change a result.
    static_assert(2 * digit2_max * digit2_max < (std::int64_t{1} << 31));
    constexpr std::size_t tile = 128;
    const std::size_t n = x.plane();
    const auto k_count = static_cast<std::size_t>(op.in_channels);
    Tensor y = like(x, op.out_channels / op.group);
    std::vector<std::int16_t> weights(op.weights.size());
    std::transform(op.weights.begin(), op.weights.end(), weights.begin(),
                   [](std::int32_t w) { return static_cast<std::int16_t>(w); });
    // Work items are pixel tiles times ranges of output groups, so small planes
    // (a latent of a few hundred pixels) still occupy every worker.
    const auto tiles = static_cast<std::int32_t>((n + tile - 1) / tile);
    const std::int32_t chunks = std::clamp<std::int32_t>(64 / tiles, 1, y.channels);
    std::atomic<bool> failed{false};
    parallel_for(tiles * chunks, [&](std::int32_t item) {
        const std::size_t p0 = static_cast<std::size_t>(item / chunks) * tile;
        const std::size_t len = std::min(tile, n - p0);
        const std::int32_t g_begin = y.channels * (item % chunks) / chunks;
        const std::int32_t g_end = y.channels * (item % chunks + 1) / chunks;
        std::vector<std::int16_t> xi(k_count * tile, 0);  // quantized once, zero past the plane's end
        for (std::size_t k = 0; k < k_count; ++k) {
            for (std::size_t i = 0; i < len; ++i) {
                xi[k * tile + i] = static_cast<std::int16_t>(
                    std::clamp(rsr(x.values[k * n + p0 + i], op.input_shift), -digit2_max, digit2_max));
            }
        }
        std::array<std::int64_t, tile> acc{};
        std::array<std::int64_t, tile> sum{};
        for (std::int32_t g = g_begin; g < g_end && !failed.load(std::memory_order_relaxed); ++g) {
            sum.fill(0);
            for (int j = 0; j < op.group; ++j) {
                const auto o = static_cast<std::size_t>(g * op.group + j);
                const std::int16_t* w = weights.data() + o * k_count;
                acc.fill(op.bias[o]);
                std::size_t k = 0;
                for (; k + 1 < k_count; k += 2) {
                    const std::int32_t w0 = w[k], w1 = w[k + 1];
                    const std::int16_t* x0 = xi.data() + k * tile;
                    const std::int16_t* x1 = x0 + tile;
                    for (std::size_t i = 0; i < tile; ++i) acc[i] += w0 * x0[i] + w1 * x1[i];
                }
                if (k < k_count) {
                    const std::int32_t w0 = w[k];
                    const std::int16_t* x0 = xi.data() + k * tile;
                    for (std::size_t i = 0; i < tile; ++i) acc[i] += w0 * x0[i];
                }
                for (std::size_t i = 0; i < len; ++i) {
                    const std::int64_t r = rsr(acc[i], op.shift[o]);
                    if (r >= gemm_limit || r <= -gemm_limit) {
                        failed = true;
                        return;
                    }
                    sum[i] += op.activation ? delta_act(r, *op.activation) : r;
                }
            }
            for (std::size_t i = 0; i < len; ++i) {
                const auto at = static_cast<std::size_t>(g) * n + p0 + i;
                std::int64_t v = sum[i];
                for (const Tensor* residual : residuals) v += residual->values[at];
                if (!store(v, y.values[at])) {
                    failed = true;
                    return;
                }
            }
        }
    });
    if (failed) return gemm(op, x, residuals);  // the reference reports the failure it meets first
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

Result<Tensor> depthwise3x3_tiled(const Depthwise& op, const Tensor& x) {
    Tensor y = like(x, x.channels);
    std::atomic<bool> failed{false};
    parallel_for(x.channels, [&](std::int32_t c) {
        const auto ci = static_cast<std::size_t>(c);
        const std::int32_t* in = x.values.data() + ci * x.plane();
        for (std::int32_t row = 0; row < x.height && !failed.load(std::memory_order_relaxed); ++row) {
            for (std::int32_t col = 0; col < x.width; ++col) {
                std::int64_t acc = op.bias[ci];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        const int yy = row + i - 1;
                        const int xx = col + j - 1;
                        if (yy < 0 || yy >= x.height || xx < 0 || xx >= x.width) continue;
                        acc += std::int64_t{op.weights[ci * 9 + static_cast<std::size_t>(i * 3 + j)]} *
                            in[static_cast<std::size_t>(yy * x.width + xx)];
                    }
                }
                const auto at = ci * x.plane() + static_cast<std::size_t>(row * x.width + col);
                if (!store(rsr(acc, op.shift[ci]), y.values[at])) {
                    failed = true;
                    return;
                }
            }
        }
    });
    if (failed) return depthwise3x3(op, x);  // the reference reports the failure it meets first
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
