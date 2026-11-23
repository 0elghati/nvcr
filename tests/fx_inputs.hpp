#pragma once

// Seeded nvcr.fx.v1 stage inputs at the scale the codec feeds, shared by the
// TensorRT conformance test, the CPU executor test and the CPU benchmark.

#include "deterministic/fx_program.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace nvcr::test::fx {

namespace det = nvcr::deterministic::fx;

// Small integers at 2^-14, pictures in [0, 2^14], QP 37, the spatial steps'
// parity masks and entropy symbols.
inline det::Tensor input(const std::string& name, int channels, int height, int width, std::mt19937& rng) {
    det::Tensor t{channels, height, width, {}};
    std::normal_distribution<double> normal(0.0, name == "feature" || name.find("context") != std::string::npos ? 4.0 : 2.0);
    std::uniform_int_distribution<std::int32_t> pixel(0, 1 << 14);
    for (std::size_t i = 0; i < static_cast<std::size_t>(channels) * t.plane(); ++i) {
        const auto at = static_cast<int>(i % t.plane());
        if (name == "parity") {
            t.values.push_back(static_cast<std::size_t>(2 * (at / width % 2) + at % width % 2) == i / t.plane() ? 1 : 0);
        } else if (name == "symbols") {
            t.values.push_back(static_cast<std::int32_t>(std::lround(normal(rng))));
        } else if (name == "qp") {
            t.values.push_back(37);
        } else if (name == "frame") {
            t.values.push_back(pixel(rng));
        } else if (name == "latent_ref") {
            t.values.push_back(0);
        } else {
            t.values.push_back(static_cast<std::int32_t>(std::lround(normal(rng))) * (1 << 14));
        }
    }
    return t;
}

// Input shapes of a stage at a frame size (padded to 16); channels come from the program.
inline std::vector<std::array<int, 3>> stage_shapes(const det::Program& program, int width, int height) {
    const int h = (height + 15) / 16 * 16, w = (width + 15) / 16 * 16, yh = h / 16, yw = w / 16;
    const int zh = (yh + 3) / 4, zw = (yw + 3) / 4;
    const std::map<std::string, std::array<int, 3>> sizes{
        {"z_fixed", {128, zh, zw}}, {"latent_ref", {1, yh, yw}}, {"common", {256, yh, yw}},
        {"frame", {3, h, w}},       {"qp", {1, 1, 1}},           {"feature", {256, h / 8, w / 8}},
        {"context", {256, h / 8, w / 8}}, {"temporal_context", {256, h / 8, w / 8}}, {"memory", {512, h / 8, w / 8}}};
    std::vector<std::array<int, 3>> shapes;
    for (const auto id : program.inputs) {
        const auto found = sizes.find(program.tensor_names[id]);
        shapes.push_back({program.tensor_channels[id], found != sizes.end() ? found->second[1] : yh,
                          found != sizes.end() ? found->second[2] : yw});
    }
    return shapes;
}

// All inputs of a stage at a frame size, in program.inputs order.
inline std::vector<det::Tensor> stage_inputs(const det::Program& program, int width, int height, std::mt19937& rng) {
    std::vector<det::Tensor> inputs;
    const auto shapes = stage_shapes(program, width, height);
    for (std::size_t slot = 0; slot < shapes.size(); ++slot) {
        const auto& name = program.tensor_names[program.inputs[slot]];
        inputs.push_back(input(name, shapes[slot][0], shapes[slot][1], shapes[slot][2], rng));
    }
    return inputs;
}

}  // namespace nvcr::test::fx
