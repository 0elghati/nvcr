#pragma once

// The TensorRT provider's implementation of nvcr.fx.v1: plugins, the kernels
// compiled ahead of time for one GPU architecture, and the compiler that turns
// an fx program into a TensorRT plan for fixed or ranged input shapes.

#include "../../../deterministic/fx_program.hpp"

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace nvcr::tensorrt::fx {

// Registers NvcrFxGemm, NvcrFxDepthwise and NvcrFxOp with TensorRT's registry.
[[nodiscard]] bool register_plugins();

// True when kernels were compiled for this compute capability.
[[nodiscard]] bool kernels_available(int major, int minor) noexcept;

// height x width is the optimum; a range from min to max (equal when unset)
// builds a plan whose height and width are dynamic.
struct InputShape final {
    std::string name;
    int channels{0};
    int height{0};
    int width{0};
    int min_height{0};
    int min_width{0};
    int max_height{0};
    int max_width{0};
    [[nodiscard]] bool ranged() const noexcept { return min_height != max_height || min_width != max_width; }
};

struct Plan final {
    std::vector<std::byte> bytes;
    std::vector<InputShape> outputs;  // output shapes, program.outputs order; -1 for a dynamic axis
};

// Builds a serialized plan computing the program for these input shapes.
[[nodiscard]] Result<Plan> build_plan(const deterministic::fx::Program& program, std::span<const InputShape> inputs);

}  // namespace nvcr::tensorrt::fx
