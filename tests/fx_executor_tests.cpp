// nvcr.fx.v1 CPU executor: the tiled kernels must compute exactly the reference
// kernels' integers and fail on exactly the same inputs with the same error.
//
// usage: nvcr_fx_executor_tests [PROGRAM_DIR [WIDTHxHEIGHT...]]
//   Without arguments, seeded synthetic gemms at the profile's bounds are checked.
//   PROGRAM_DIR: fx programs (a bundle's normative/ directory); every stage then
//   also runs on seeded inputs at the given sizes (default 176x144 and 72x40).

#include "deterministic/fx_program.hpp"
#include "fx_inputs.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace det = nvcr::deterministic::fx;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename T>
void agree(const nvcr::Result<T>& reference, const nvcr::Result<T>& tiled, const std::string& what) {
    expect(reference.has_value() == tiled.has_value(), what + ": both succeed or both fail");
    if (reference && tiled) {
        if constexpr (std::is_same_v<T, det::Tensor>) {
            expect(reference.value().values == tiled.value().values, what + ": bit-identical");
        } else {
            for (std::size_t i = 0; i < reference.value().size(); ++i) {
                expect(reference.value()[i].values == tiled.value()[i].values, what + ": output bit-identical");
            }
        }
    } else if (!reference && !tiled) {
        expect(reference.error().describe() == tiled.error().describe(), what + ": same error");
    }
}

det::Tensor tensor(int channels, int height, int width, std::int64_t bound, std::mt19937& rng) {
    std::uniform_int_distribution<std::int64_t> value(-bound, bound);
    det::Tensor t{channels, height, width, {}};
    t.values.resize(static_cast<std::size_t>(channels) * t.plane());
    for (auto& v : t.values) v = static_cast<std::int32_t>(value(rng));
    return t;
}

// Weights at +-digit2_max and random interior values; odd and even K; one to
// four rows per group; zero to two residuals; planes smaller, equal to and
// larger than a tile and not a multiple of it.
void synthetic(std::mt19937& rng) {
    const det::DeltaTable table{4096, 6, std::vector<std::int16_t>(((2 * 4096) >> 6) + 1, 3)};
    int checked = 0;
    for (const int k : {1, 2, 9, 37, 64}) {
        for (const int group : {1, 2, 4}) {
            for (const auto& [h, w] : std::vector<std::pair<int, int>>{{1, 1}, {5, 7}, {8, 16}, {13, 21}}) {
                for (const int residuals : {0, 1, 2}) {
                    det::Gemm op;
                    op.in_channels = k;
                    op.out_channels = 3 * group;
                    op.group = group;
                    op.input_shift = static_cast<int>(rng() % 4);
                    op.activation = rng() % 2 == 0 ? &table : nullptr;
                    std::uniform_int_distribution<std::int32_t> weight(-det::digit2_max, det::digit2_max);
                    op.weights.resize(static_cast<std::size_t>(op.out_channels) * static_cast<std::size_t>(k));
                    for (auto& v : op.weights) {
                        const auto pick = rng() % 8;
                        v = pick == 0 ? det::digit2_max : pick == 1 ? -det::digit2_max : weight(rng);
                    }
                    std::uniform_int_distribution<std::int64_t> bias(-(1LL << 34), 1LL << 34);
                    for (int o = 0; o < op.out_channels; ++o) {
                        op.bias.push_back(bias(rng));
                        op.shift.push_back(static_cast<std::int32_t>(18 + rng() % 14));
                    }
                    const auto x = tensor(k, h, w, det::stored_max, rng);
                    std::vector<det::Tensor> res;
                    for (int r = 0; r < residuals; ++r) res.push_back(tensor(3, h, w, 1 << 20, rng));
                    std::vector<const det::Tensor*> pointers;
                    for (const auto& r : res) pointers.push_back(&r);
                    const std::string what = "gemm k=" + std::to_string(k) + " group=" + std::to_string(group) +
                                             " " + std::to_string(h) + "x" + std::to_string(w);
                    agree(det::gemm(op, x, pointers), det::gemm_tiled(op, x, pointers), what);

                    // Failures: a pre-activation at the limit, then a stored value above 2^30.
                    auto large = op;
                    large.bias.assign(large.bias.size(), 0);
                    large.bias[large.bias.size() - 1] = (det::gemm_limit + (1LL << 20)) << large.shift.back();
                    agree(det::gemm(large, x, pointers), det::gemm_tiled(large, x, pointers), what + " pre-activation");
                    if (!res.empty()) {
                        auto over = res;
                        over.back().values.back() = static_cast<std::int32_t>(det::stored_max);
                        over.front().values.back() = static_cast<std::int32_t>(det::stored_max);
                        std::vector<const det::Tensor*> over_pointers;
                        for (const auto& r : over) over_pointers.push_back(&r);
                        agree(det::gemm(op, x, over_pointers), det::gemm_tiled(op, x, over_pointers), what + " store");
                    }
                    ++checked;
                }
            }
        }
    }
    std::cout << checked << " synthetic gemms checked\n";

    // Depthwise: weights up to the profile's 2^24, a border-only 1x1 plane, and a
    // stored value above 2^30.
    for (const auto& [h, w] : std::vector<std::pair<int, int>>{{1, 1}, {3, 5}, {17, 9}}) {
        det::Depthwise op;
        std::uniform_int_distribution<std::int32_t> weight(-(1 << 24), 1 << 24);
        std::uniform_int_distribution<std::int64_t> bias(-(1LL << 40), 1LL << 40);
        for (int i = 0; i < 6 * 9; ++i) op.weights.push_back(weight(rng));
        for (int c = 0; c < 6; ++c) {
            op.bias.push_back(bias(rng));
            op.shift.push_back(static_cast<std::int32_t>(24 + rng() % 8));
        }
        const auto x = tensor(6, h, w, 1 << 22, rng);
        const std::string what = "depthwise " + std::to_string(h) + "x" + std::to_string(w);
        agree(det::depthwise3x3(op, x), det::depthwise3x3_tiled(op, x), what);
        auto over = op;
        over.bias.back() = det::stored_max << (over.shift.back() + 1);
        agree(det::depthwise3x3(over, x), det::depthwise3x3_tiled(over, x), what + " store");
    }
}

void programs(const std::filesystem::path& directory, const std::vector<std::pair<int, int>>& sizes) {
    std::vector<std::string> stages;
    const std::string suffix = ".fxprog.json";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.size() > suffix.size() && name.ends_with(suffix)) stages.push_back(name.substr(0, name.size() - suffix.size()));
    }
    std::sort(stages.begin(), stages.end());
    expect(!stages.empty(), directory.string() + ": has fx programs");
    for (const auto& stage : stages) {
        auto program = det::load_program(directory, stage);
        expect(program.has_value(), stage + ": program loads");
        if (!program) continue;
        for (const auto& [width, height] : sizes) {
            std::mt19937 rng(7);
            const auto inputs = nvcr::test::fx::stage_inputs(program.value(), width, height, rng);
            const auto size = std::to_string(width) + "x" + std::to_string(height);
            agree(det::run(program.value(), inputs, det::Kernels::reference),
                  det::run(program.value(), inputs, det::Kernels::tiled), stage + " at " + size);
            std::cout << stage << " at " << size << ": checked\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(11);
    synthetic(rng);
    if (argc > 1) {
        std::vector<std::pair<int, int>> sizes;
        for (int a = 2; a < argc; ++a) {
            int width = 0, height = 0;
            if (std::sscanf(argv[a], "%dx%d", &width, &height) != 2) return 2;
            sizes.emplace_back(width, height);
        }
        if (sizes.empty()) sizes = {{176, 144}, {72, 40}};
        programs(argv[1], sizes);
    }
    if (failures == 0) std::cout << "nvcr.fx.v1 executor tests passed\n";
    return failures == 0 ? 0 : 1;
}
