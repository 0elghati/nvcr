#pragma once

// nvcr.fx.v1 stage programs (format nvcr.fx-program.v1, docs/spec/numerics-fx-v1.md
// section 6): <stage>.fxprog.json plus <stage>.fxw. The JSON pins the SHA-256 of
// the data file, so the JSON's own SHA-256 identifies the whole program.

#include "fx.hpp"

#include "nvcr/common/error.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::deterministic::fx {

enum class Op : std::uint8_t {
    gemm,
    depthwise3x3,
    mul,
    affine,
    add,
    sat_table,
    delta,
    clamp,
    concat,
    slice,
    pixel_shuffle,
    pixel_unshuffle,
    im2col,
    param_row,
    crop_like,
};

struct Node final {
    Op op{Op::add};
    std::vector<std::size_t> inputs;
    std::size_t output{0};
    Gemm gemm;                       // gemm (activation set from delta_table)
    Depthwise depthwise;             // depthwise3x3
    std::vector<std::int64_t> rows;  // param_row: [qp][channel]
    std::vector<std::int32_t> table; // sat_table
    std::int64_t alpha{0};           // affine
    std::int64_t beta{0};            // affine
    std::int64_t lo{0};              // clamp
    std::int64_t hi{0};              // clamp
    int shift{0};                    // mul, affine
    int delta_table{-1};             // gemm (optional), delta
    int begin{0};                    // slice
    int end{0};                      // slice
    int factor{1};                   // pixel_shuffle, pixel_unshuffle
    int kernel{3};                   // im2col: 3 or 2
    int stride{1};                   // im2col: 1 or 2
};

struct Program final {
    std::string name;
    int fa{14};
    std::vector<std::string> tensor_names;
    std::vector<std::int32_t> tensor_channels;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    std::vector<DeltaTable> delta_tables;
    std::vector<Node> nodes;
};

// Loads <directory>/<stage>.fxprog.json and its data file, and applies the
// spec's load-time checks (weight, bias and shift bounds; valid references).
[[nodiscard]] Result<Program> load_program(const std::filesystem::path& directory, std::string_view stage);

// SHA-256 (lowercase hex) of <directory>/<stage>.fxprog.json: the program's identity.
[[nodiscard]] Result<std::string> program_digest(const std::filesystem::path& directory, std::string_view stage);

// Runs the program on its inputs (program.inputs order) and returns its outputs
// (program.outputs order), or the stage failure the spec defines.
[[nodiscard]] Result<std::vector<Tensor>> run(const Program& program, std::span<const Tensor> inputs);

}  // namespace nvcr::deterministic::fx
