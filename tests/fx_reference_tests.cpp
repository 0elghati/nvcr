// nvcr.fx.v1 reference arithmetic: rounding, saturation, tables and the
// failure rules (docs/spec/numerics-fx-v1.md), on hand-computed values.
#include "deterministic/fx.hpp"

#include <iostream>
#include <vector>

namespace {

namespace fx = nvcr::deterministic::fx;

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

fx::Tensor pixel(std::vector<std::int32_t> channels) {
    const auto count = static_cast<std::int32_t>(channels.size());
    return {count, 1, 1, std::move(channels)};
}

std::int32_t first(const nvcr::Result<fx::Tensor>& y) {
    return y ? y.value().values[0] : -1;
}

void rounding() {
    expect(fx::rsr(5, 1) == 3, "rsr rounds a positive tie up");
    expect(fx::rsr(-5, 1) == -2, "rsr rounds a negative tie up");
    expect(fx::rsr(-6, 2) == -1, "rsr floors after adding half");
    expect(fx::rsr(7, 0) == 7, "rsr with shift 0 is the identity");
}

void tables() {
    const fx::DeltaTable table{4, 1, {10, 20, 30, 40, 50}};
    expect(fx::delta_act(-5, table) == 0, "delta below -h is max(r, 0)");
    expect(fx::delta_act(-4, table) == 10, "delta at -h uses the first bin");
    expect(fx::delta_act(-3, table) == 10, "delta bins span 2^shift steps");
    expect(fx::delta_act(-2, table) == 20, "delta next bin");
    expect(fx::delta_act(0, table) == 30, "delta at zero");
    expect(fx::delta_act(4, table) == 54, "delta at h adds max(r, 0)");
    expect(fx::delta_act(5, table) == 5, "delta above h is r");

    const std::vector<std::int32_t> sigmoid{1, 2, 3, 4};
    const auto y = fx::sat_table(pixel({-3, -2, 1, 2}), sigmoid, 99);
    expect(y.values == std::vector<std::int32_t>{0, 1, 4, 99}, "saturating table edges");
}

void gemm() {
    fx::Gemm op;
    op.in_channels = 1;
    op.out_channels = 1;
    op.weights = {3};
    op.bias = {0};
    op.shift = {1};
    op.input_shift = 1;
    expect(first(fx::gemm(op, pixel({5}), {})) == 5, "gemm rounds its input, then its output");

    op.weights = {1};
    op.input_shift = 0;
    expect(first(fx::gemm(op, pixel({1 << 20}), {})) == 16320, "gemm saturates its input to two digits");

    op.in_channels = 2;
    op.weights = {32639, 32639};
    expect(!fx::gemm(op, pixel({32639, 32639}), {}), "gemm fails at |r| >= 2^29 - 1");

    op.in_channels = 1;
    op.out_channels = 2;
    op.weights = {1, 2};
    op.bias = {0, 0};
    op.shift = {1, 1};
    op.group = 2;
    const auto residual = pixel({10});
    const fx::Tensor* residuals[] = {&residual};
    expect(first(fx::gemm(op, pixel({4}), residuals)) == 16, "gemm sums adjacent rows, then residuals");
}

void elementwise() {
    fx::Depthwise dw{{0, 0, 0, 0, 4, 0, 0, 0, 0}, {2}, {1}};
    expect(first(fx::depthwise3x3(dw, pixel({3}))) == 7, "depthwise centre tap with bias");

    const auto product = fx::mul(pixel({3, 5}), pixel({2}), 1);
    expect(product && product.value().values == std::vector<std::int32_t>{3, 5}, "mul broadcasts one channel");
    const fx::Tensor plane{1, 1, 2, {3, 5}};
    const auto scaled = fx::mul(plane, pixel({4}), 2);
    expect(scaled && scaled.value().values == std::vector<std::int32_t>{3, 5}, "mul broadcasts a 1x1 factor");

    expect(first(fx::affine(pixel({1 << 14}), 3, 1 << 14, 1)) == 1 << 15, "affine quant step at sigmoid 1.0");
    expect(first(fx::add(pixel({1 << 30}), pixel({0}))) == 1 << 30, "a stored value of 2^30 is allowed");
    expect(!fx::add(pixel({1 << 30}), pixel({1})), "a stored value above 2^30 fails");
}

}  // namespace

int main() {
    rounding();
    tables();
    gemm();
    elementwise();
    if (failures == 0) std::cout << "nvcr.fx.v1 reference tests passed\n";
    return failures == 0 ? 0 : 1;
}
