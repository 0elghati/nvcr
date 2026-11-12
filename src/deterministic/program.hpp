#pragma once

// Stage programs of the nvcr.intconv.v1 numerics profile, lowered from exported
// ONNX stages by scripts/backends/dcvcrt/export_deterministic_program.py.
//
// Arithmetic order is part of the format, so every executor must match it:
//   conv            out = bias; then out = fma(w, x, out) over input channel,
//                   kernel row and kernel column in that order. Padding taps
//                   contribute fma(w, 0, out). Depthwise convs have one input
//                   channel per output channel.
//   mul, add        one IEEE operation per element; mul_scalar likewise.
//   sigmoid         det_sigmoid (det_math.hpp).
//   scale_channels  x[c, y, x] * s[c] for a [C, 1, 1] tensor s, one IEEE mul.
//   clamp           x < lo ? lo : (x > hi ? hi : x); NaN passes through.
//   slice_channels, concat (channels of the first input, then the second),
//   crop (to the second input's height or width, from the origin),
//   depth_to_space (CRD), space_to_depth (out[c b^2 + i b + j, y, x] =
//   in[c, y b + i, x b + j], as PixelUnshuffle), im2col (out[(c kh + ky) kw +
//   kx, oy, ox] = in[c, oy sy - pad_top + ky, ox sx - pad_left + kx], zero
//   outside)  exact data movement. Exporters lower every dense spatial conv to
//   im2col and a pointwise conv; the fma order (channel, row, column) is the
//   conv's, and the integer conv below then quantizes each output position's
//   receptive field with one scale.
//
// Every pointwise conv (dense, 1x1, stride 1, no padding) runs in exact
// integer arithmetic at the node's bit widths (8 or 16 for weights and for
// activations; default 16). Integer sums are exact in any
// order, so executors may tile, split, or use integer tensor cores freely.
// With L = 2^(bits - 1) - 1:
//   weights      per output channel c: m = max|w|; s = m / L in double;
//                q = clamp(rint(w / s), +-L); scale = float(s). m = 0 gives
//                q = 0 and scale 1.
//   activations  per position p over the input channels: a = max|x|; with
//                frexp(a) = (f, E), e = E - (bits - 1) (a = 0 gives e = 0);
//                q = clamp(rint(ldexp(x, -e)), +-L). A non-finite a makes
//                every output at p NaN.
//   output       acc = sum q_w * q_x (exact); out = float(acc) * scale, then
//                ldexp(., e), then + bias, each rounded to nearest-even.
// Tensors are batch-1 NCHW float32.

#include "nvcr/common/error.hpp"
#include "nvcr/numerics/profiles.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::deterministic {

inline constexpr std::string_view intconv_profile = nvcr::numerics::intconv_v1;

enum class OpKind : std::uint8_t {
    conv,
    mul,
    add,
    mul_scalar,
    sigmoid,
    slice_channels,
    depth_to_space,
    scale_channels,
    clamp,
    concat,
    crop,
    space_to_depth,
    im2col,
};

struct Node final {
    OpKind kind{OpKind::conv};
    std::array<std::size_t, 2> inputs{};
    std::size_t input_count{1};
    std::size_t output{0};
    // conv
    std::int32_t in_channels{0};
    std::int32_t out_channels{0};
    bool depthwise{false};
    std::array<std::int32_t, 2> kernel{1, 1};
    std::array<std::int32_t, 2> stride{1, 1};
    // top, left, bottom, right
    std::array<std::int32_t, 4> pads{0, 0, 0, 0};
    std::size_t weight_offset{0};
    std::size_t bias_offset{0};
    // mul_scalar
    float scalar{0.0F};
    // slice_channels
    std::int32_t begin{0};
    std::int32_t end{0};
    // depth_to_space, space_to_depth
    std::int32_t block{1};
    // crop: 2 (height) or 3 (width)
    std::int32_t axis{2};
    // clamp
    float minimum{0.0F};
    float maximum{0.0F};
    // Integer widths of a pointwise conv
    std::int32_t weight_bits{16};
    std::int32_t activation_bits{16};
};

struct Program final {
    std::string name;
    std::string numerics;
    std::string weights_sha256;
    std::vector<std::string> tensor_names;
    std::vector<std::int32_t> tensor_channels;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    std::vector<Node> nodes;
    // FP16 weights widened exactly to float32.
    std::vector<float> weights;
    // Index of the last node that reads each tensor (outputs: nodes.size()).
    std::vector<std::size_t> last_use;
};

struct Tensor final {
    std::int32_t channels{0};
    std::int32_t height{0};
    std::int32_t width{0};
    std::vector<float> values;

    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(channels) * static_cast<std::size_t>(height) *
            static_cast<std::size_t>(width);
    }
};

struct Extent final {
    std::int32_t channels{0};
    std::int32_t height{0};
    std::int32_t width{0};
};

// True when this node runs in integer arithmetic.
[[nodiscard]] bool is_integer_pointwise(const Node& node) noexcept;

// Integer weights of one pointwise conv, row-major [out][in].
struct IntegerWeights final {
    std::vector<std::int16_t> values;
    std::vector<float> scales;
};

[[nodiscard]] IntegerWeights quantize_weights(const Program& program, const Node& node);

[[nodiscard]] Result<Program> load_program(const std::filesystem::path& program_json);

// Validates a node's inputs against the program and returns its output extent.
[[nodiscard]] Result<Extent> output_extent(
    const Program& program, const Node& node, std::span<const Extent> inputs);

// Reference executor. Every conforming executor reproduces its bits.
// Outputs follow program.outputs order.
[[nodiscard]] Result<std::vector<Tensor>> run_cpu(const Program& program, std::span<const Tensor> inputs);

}  // namespace nvcr::deterministic
