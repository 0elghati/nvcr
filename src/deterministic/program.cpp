#include "program.hpp"

#include "../common/json.hpp"
#include "../common/sha256.hpp"
#include "det_math.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace nvcr::deterministic {
namespace {

constexpr std::string_view subsystem = "deterministic";
constexpr std::string_view program_format = "nvcr.deterministic-program";
constexpr std::string_view program_numerics = "det-fp32-v1";

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), std::string(subsystem));
}

Result<std::int64_t> integer(const detail::json::Value& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr || !value->integer_value.has_value()) {
        return invalid("program field '" + std::string(key) + "' must be an integer");
    }
    return value->integer_value.value();
}

Result<std::string> text(const detail::json::Value& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr || value->kind != detail::json::Value::Kind::string) {
        return invalid("program field '" + std::string(key) + "' must be a string");
    }
    return value->string_value;
}

Result<std::int32_t> integers_value(const detail::json::Value& object, std::string_view key) {
    auto value = integer(object, key);
    if (!value) return value.error();
    return static_cast<std::int32_t>(value.value());
}

template <std::size_t Size>
Result<std::array<std::int32_t, Size>> integers(
    const detail::json::Value& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr || value->kind != detail::json::Value::Kind::array ||
        value->array_value.size() != Size) {
        return invalid("program field '" + std::string(key) + "' has the wrong length");
    }
    std::array<std::int32_t, Size> output{};
    for (std::size_t index = 0; index < Size; ++index) {
        if (!value->array_value[index].integer_value.has_value()) {
            return invalid("program field '" + std::string(key) + "' must hold integers");
        }
        output[index] = static_cast<std::int32_t>(value->array_value[index].integer_value.value());
    }
    return output;
}

class TensorTable final {
public:
    explicit TensorTable(Program& program) : program_(program) {}

    std::size_t define(const std::string& name, std::int32_t channels) {
        const auto id = program_.tensor_names.size();
        program_.tensor_names.push_back(name);
        program_.tensor_channels.push_back(channels);
        ids_.emplace(name, id);
        return id;
    }

    [[nodiscard]] Result<std::size_t> find(const std::string& name) const {
        const auto found = ids_.find(name);
        if (found == ids_.end()) return invalid("program reads undefined tensor '" + name + "'");
        return found->second;
    }

    [[nodiscard]] bool contains(const std::string& name) const { return ids_.contains(name); }

    [[nodiscard]] std::int32_t channels(std::size_t id) const { return program_.tensor_channels[id]; }

private:
    Program& program_;
    std::map<std::string, std::size_t, std::less<>> ids_;
};

Result<OpKind> op_kind(std::string_view op) {
    if (op == "conv") return OpKind::conv;
    if (op == "mul") return OpKind::mul;
    if (op == "add") return OpKind::add;
    if (op == "mul_scalar") return OpKind::mul_scalar;
    if (op == "sigmoid") return OpKind::sigmoid;
    if (op == "slice_channels") return OpKind::slice_channels;
    if (op == "depth_to_space") return OpKind::depth_to_space;
    if (op == "scale_channels") return OpKind::scale_channels;
    if (op == "clamp") return OpKind::clamp;
    if (op == "concat") return OpKind::concat;
    if (op == "crop") return OpKind::crop;
    if (op == "space_to_depth") return OpKind::space_to_depth;
    if (op == "im2col") return OpKind::im2col;
    return invalid("unsupported deterministic op '" + std::string(op) + "'");
}

Result<std::vector<float>> read_weights(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return Error(ErrorCode::io_error, "cannot open weights " + path.string(),
                     std::string(subsystem));
    }
    const auto bytes = static_cast<std::size_t>(stream.tellg());
    if (bytes != count * sizeof(std::uint16_t)) {
        return invalid("weights file size does not match the declared count");
    }
    std::vector<std::uint16_t> halves(count);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(halves.data()), static_cast<std::streamsize>(bytes));
    if (!stream) {
        return Error(ErrorCode::io_error, "cannot read weights " + path.string(),
                     std::string(subsystem));
    }
    std::vector<float> weights(count);
    std::transform(halves.begin(), halves.end(), weights.begin(), det_half_bits_to_float);
    return weights;
}

Result<Node> parse_node(
    const detail::json::Value& value, TensorTable& tensors, std::size_t weight_count) {
    auto op = text(value, "op");
    if (!op) return op.error();
    auto kind = op_kind(op.value());
    if (!kind) return kind.error();
    Node node;
    node.kind = kind.value();

    if (node.kind == OpKind::mul || node.kind == OpKind::add || node.kind == OpKind::scale_channels ||
        node.kind == OpKind::concat || node.kind == OpKind::crop) {
        const auto* inputs = value.find("inputs");
        if (inputs == nullptr || inputs->array_value.size() != 2) {
            return invalid("binary op needs two inputs");
        }
        for (std::size_t index = 0; index < 2; ++index) {
            auto id = tensors.find(inputs->array_value[index].string_value);
            if (!id) return id.error();
            node.inputs[index] = id.value();
        }
        node.input_count = 2;
    } else {
        auto input = text(value, "input");
        if (!input) return input.error();
        auto id = tensors.find(input.value());
        if (!id) return id.error();
        node.inputs[0] = id.value();
    }

    std::int32_t channels = 0;
    switch (node.kind) {
    case OpKind::conv: {
        auto in_channels = integer(value, "in_channels");
        auto out_channels = integer(value, "out_channels");
        auto kernel = integers<2>(value, "kernel");
        auto stride = integers<2>(value, "stride");
        auto pads = integers<4>(value, "pads");
        auto weight_offset = integer(value, "weight_offset");
        auto bias_offset = integer(value, "bias_offset");
        const auto* depthwise = value.find("depthwise");
        if (!in_channels) return in_channels.error();
        if (!out_channels) return out_channels.error();
        if (!kernel) return kernel.error();
        if (!stride) return stride.error();
        if (!pads) return pads.error();
        if (!weight_offset) return weight_offset.error();
        if (!bias_offset) return bias_offset.error();
        if (depthwise == nullptr || depthwise->kind != detail::json::Value::Kind::boolean) {
            return invalid("conv needs a boolean 'depthwise'");
        }
        node.in_channels = static_cast<std::int32_t>(in_channels.value());
        node.out_channels = static_cast<std::int32_t>(out_channels.value());
        node.depthwise = depthwise->boolean_value;
        node.kernel = kernel.value();
        node.stride = stride.value();
        // ONNX pads are [top, left, bottom, right].
        node.pads = pads.value();
        if (node.in_channels <= 0 || node.out_channels <= 0 || node.kernel[0] <= 0 ||
            node.kernel[1] <= 0 || node.stride[0] <= 0 || node.stride[1] <= 0 ||
            std::ranges::any_of(node.pads, [](std::int32_t pad) { return pad < 0; }) ||
            (node.depthwise && node.in_channels != node.out_channels) ||
            weight_offset.value() < 0 || bias_offset.value() < 0) {
            return invalid("conv has an invalid geometry");
        }
        node.weight_offset = static_cast<std::size_t>(weight_offset.value());
        node.bias_offset = static_cast<std::size_t>(bias_offset.value());
        const auto per_output = static_cast<std::size_t>(node.depthwise ? 1 : node.in_channels) *
            static_cast<std::size_t>(node.kernel[0]) * static_cast<std::size_t>(node.kernel[1]);
        const auto weights = static_cast<std::size_t>(node.out_channels) * per_output;
        if (node.weight_offset > weight_count || weights > weight_count - node.weight_offset ||
            node.bias_offset > weight_count ||
            static_cast<std::size_t>(node.out_channels) > weight_count - node.bias_offset) {
            return invalid("conv weights exceed the weights file");
        }
        channels = node.out_channels;
        break;
    }
    case OpKind::mul_scalar: {
        const auto* scalar = value.find("value");
        if (scalar == nullptr || scalar->kind != detail::json::Value::Kind::number) {
            return invalid("mul_scalar needs a numeric 'value'");
        }
        node.scalar = static_cast<float>(scalar->number_value);
        if (static_cast<double>(node.scalar) != scalar->number_value) {
            return invalid("mul_scalar value is not exactly representable in float32");
        }
        break;
    }
    case OpKind::slice_channels: {
        auto begin = integer(value, "begin");
        auto end = integer(value, "end");
        if (!begin) return begin.error();
        if (!end) return end.error();
        node.begin = static_cast<std::int32_t>(begin.value());
        node.end = static_cast<std::int32_t>(end.value());
        channels = node.end - node.begin;
        break;
    }
    case OpKind::depth_to_space:
    case OpKind::space_to_depth: {
        auto block = integer(value, "block");
        if (!block) return block.error();
        node.block = static_cast<std::int32_t>(block.value());
        if (node.block <= 0 || node.block > 64) return invalid("block size must be in [1, 64]");
        if (node.kind == OpKind::space_to_depth) {
            channels = tensors.channels(node.inputs[0]) * node.block * node.block;
        }
        break;
    }
    case OpKind::clamp: {
        const auto* minimum = value.find("min");
        const auto* maximum = value.find("max");
        if (minimum == nullptr || maximum == nullptr ||
            minimum->kind != detail::json::Value::Kind::number ||
            maximum->kind != detail::json::Value::Kind::number) {
            return invalid("clamp needs numeric 'min' and 'max'");
        }
        node.minimum = static_cast<float>(minimum->number_value);
        node.maximum = static_cast<float>(maximum->number_value);
        if (static_cast<double>(node.minimum) != minimum->number_value ||
            static_cast<double>(node.maximum) != maximum->number_value ||
            !(node.minimum <= node.maximum)) {
            return invalid("clamp bounds must be ordered float32 values");
        }
        break;
    }
    case OpKind::concat:
        channels = tensors.channels(node.inputs[0]) + tensors.channels(node.inputs[1]);
        break;
    case OpKind::im2col: {
        auto kernel = integers<2>(value, "kernel");
        auto stride = integers<2>(value, "stride");
        auto pads = integers<4>(value, "pads");
        if (!kernel) return kernel.error();
        if (!stride) return stride.error();
        if (!pads) return pads.error();
        node.kernel = kernel.value();
        node.stride = stride.value();
        node.pads = pads.value();
        if (node.kernel[0] <= 0 || node.kernel[1] <= 0 || node.kernel[0] * node.kernel[1] > 64 ||
            node.stride[0] <= 0 || node.stride[1] <= 0 ||
            std::ranges::any_of(node.pads, [](std::int32_t pad) { return pad < 0; })) {
            return invalid("im2col has an invalid geometry");
        }
        channels = tensors.channels(node.inputs[0]) * node.kernel[0] * node.kernel[1];
        break;
    }
    case OpKind::crop: {
        auto axis = integer(value, "axis");
        if (!axis) return axis.error();
        if (axis.value() != 2 && axis.value() != 3) return invalid("crop axis must be 2 or 3");
        node.axis = static_cast<std::int32_t>(axis.value());
        break;
    }
    case OpKind::mul:
    case OpKind::add:
    case OpKind::scale_channels:
    case OpKind::sigmoid:
        break;
    }

    if (const auto* integer = value.find("integer"); integer != nullptr) {
        auto weight_bits = integers_value(*integer, "weight_bits");
        auto activation_bits = integers_value(*integer, "activation_bits");
        if (!weight_bits) return weight_bits.error();
        if (!activation_bits) return activation_bits.error();
        node.weight_bits = weight_bits.value();
        node.activation_bits = activation_bits.value();
        const auto* low = integer->find("low_product");
        node.mean_low_product = low != nullptr && low->string_value == "mean";
        const auto valid_bits = [](std::int32_t bits) { return bits == 8 || bits == 16; };
        if (node.kind != OpKind::conv || !valid_bits(node.weight_bits) ||
            !valid_bits(node.activation_bits) ||
            (node.mean_low_product && (node.weight_bits != 16 || node.activation_bits != 16)) ||
            (low != nullptr && low->string_value != "mean" && low->string_value != "exact")) {
            return invalid("invalid integer widths on a node");
        }
    }

    auto output = text(value, "output");
    if (!output) return output.error();
    if (tensors.contains(output.value())) {
        return invalid("tensor '" + output.value() + "' is defined twice");
    }
    node.output = tensors.define(output.value(), channels);
    return node;
}

// Clears flush-to-zero/denormals-are-zero for the calling thread so CPU results
// keep denormals exactly like the GPU kernels.
class DenormalGuard final {
public:
    DenormalGuard() {
#if defined(__x86_64__) || defined(_M_X64)
        saved_ = _mm_getcsr();
        _mm_setcsr(saved_ & ~0x8040U);
#endif
    }
    ~DenormalGuard() {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_setcsr(saved_);
#endif
    }
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;

private:
    [[maybe_unused]] unsigned int saved_{0};
};

template <typename Function>
void parallel_for(std::int32_t count, Function function) {
    const auto workers = static_cast<std::int32_t>(std::clamp<unsigned>(
        std::thread::hardware_concurrency(), 1U, 16U));
    if (workers <= 1 || count < 4) {
        for (std::int32_t index = 0; index < count; ++index) function(index);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));
    for (std::int32_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            const DenormalGuard guard;
            for (std::int32_t index = worker; index < count; index += workers) function(index);
        });
    }
    for (auto& thread : threads) thread.join();
}

void conv(const Program& program, const Node& node, const Tensor& input, Tensor& output) {
    const auto* weights = program.weights.data() + node.weight_offset;
    const auto* bias = program.weights.data() + node.bias_offset;
    const auto input_plane = static_cast<std::size_t>(input.height) * input.width;
    const auto output_plane = static_cast<std::size_t>(output.height) * output.width;
    const auto group_inputs = node.depthwise ? 1 : node.in_channels;
    const bool pointwise = node.kernel[0] == 1 && node.kernel[1] == 1 && node.stride[0] == 1 &&
        node.stride[1] == 1 && node.pads == std::array<std::int32_t, 4>{0, 0, 0, 0};
    parallel_for(node.out_channels, [&](std::int32_t out_channel) {
        float* out = output.values.data() + static_cast<std::size_t>(out_channel) * output_plane;
        std::fill(out, out + output_plane, bias[out_channel]);
        for (std::int32_t local = 0; local < group_inputs; ++local) {
            const auto in_channel = node.depthwise ? out_channel : local;
            const float* in = input.values.data() + static_cast<std::size_t>(in_channel) * input_plane;
            const float* kernel = weights +
                ((static_cast<std::size_t>(out_channel) * group_inputs + local) *
                 node.kernel[0] * node.kernel[1]);
            if (pointwise) {
                const float weight = kernel[0];
                for (std::size_t position = 0; position < output_plane; ++position) {
                    out[position] = det_fma(weight, in[position], out[position]);
                }
                continue;
            }
            for (std::int32_t ky = 0; ky < node.kernel[0]; ++ky) {
                for (std::int32_t kx = 0; kx < node.kernel[1]; ++kx) {
                    const float weight = kernel[ky * node.kernel[1] + kx];
                    for (std::int32_t oy = 0; oy < output.height; ++oy) {
                        const auto iy = oy * node.stride[0] - node.pads[0] + ky;
                        for (std::int32_t ox = 0; ox < output.width; ++ox) {
                            const auto ix = ox * node.stride[1] - node.pads[1] + kx;
                            const float value =
                                iy >= 0 && iy < input.height && ix >= 0 && ix < input.width
                                ? in[static_cast<std::size_t>(iy) * input.width + ix]
                                : 0.0F;
                            float& target = out[static_cast<std::size_t>(oy) * output.width + ox];
                            target = det_fma(weight, value, target);
                        }
                    }
                }
            }
        }
    });
}

// det-int-v1 pointwise conv: exact integer sums, specified epilogue.
void integer_pointwise(const Program& program, const Node& node, const Tensor& input,
                       Tensor& output) {
    const auto weights = quantize_weights(program, node);
    const auto* bias = program.weights.data() + node.bias_offset;
    const auto positions = static_cast<std::size_t>(input.height) * input.width;
    const auto in_channels = static_cast<std::size_t>(node.in_channels);
    std::vector<std::int32_t> activations(in_channels * positions);
    std::vector<int> exponents(positions);
    std::vector<unsigned char> finite(positions, 1);
    for (std::size_t position = 0; position < positions; ++position) {
        float maximum = 0.0F;
        for (std::size_t channel = 0; channel < in_channels; ++channel) {
            const float value = input.values[channel * positions + position];
            if (!det_isfinite(value)) finite[position] = 0;
            maximum = std::max(maximum, std::abs(value));
        }
        exponents[position] = det_int_exponent(maximum, node.activation_bits);
        for (std::size_t channel = 0; channel < in_channels; ++channel) {
            activations[channel * positions + position] = finite[position]
                ? det_int_quantize(input.values[channel * positions + position], exponents[position],
                                   node.activation_bits)
                : 0;
        }
    }
    parallel_for(node.out_channels, [&](std::int32_t out_channel) {
        const auto row = static_cast<std::size_t>(out_channel);
        std::vector<long long> accumulators(positions, 0);
        for (std::size_t channel = 0; channel < in_channels; ++channel) {
            const long long weight = weights.values[row * in_channels + channel];
            const auto* column = activations.data() + channel * positions;
            if (node.mean_low_product) {
                // Drop the low x low limb product; its mean is added below.
                const long long weight_low = weight & 255;
                for (std::size_t position = 0; position < positions; ++position) {
                    accumulators[position] +=
                        weight * column[position] - weight_low * (column[position] & 255);
                }
            } else {
                for (std::size_t position = 0; position < positions; ++position) {
                    accumulators[position] += weight * column[position];
                }
            }
        }
        if (node.mean_low_product) {
            const auto mean = det_int_mean_low_product(node.in_channels);
            for (auto& accumulator : accumulators) accumulator += mean;
        }
        float* out = output.values.data() + row * positions;
        for (std::size_t position = 0; position < positions; ++position) {
            out[position] = finite[position]
                ? det_int_output(accumulators[position], weights.scales[row],
                                   exponents[position], bias[out_channel])
                : std::numeric_limits<float>::quiet_NaN();
        }
    });
}

void depth_to_space(const Node& node, const Tensor& input, Tensor& output) {
    const auto block = node.block;
    for (std::int32_t channel = 0; channel < output.channels; ++channel) {
        for (std::int32_t y = 0; y < output.height; ++y) {
            for (std::int32_t x = 0; x < output.width; ++x) {
                const auto source_channel = channel * block * block + (y % block) * block + x % block;
                output.values[(static_cast<std::size_t>(channel) * output.height + y) * output.width + x] =
                    input.values[(static_cast<std::size_t>(source_channel) * input.height + y / block) *
                                     input.width + x / block];
            }
        }
    }
}

void space_to_depth(const Node& node, const Tensor& input, Tensor& output) {
    const auto block = node.block;
    for (std::int32_t channel = 0; channel < output.channels; ++channel) {
        const auto source_channel = channel / (block * block);
        const auto offset_y = (channel / block) % block;
        const auto offset_x = channel % block;
        for (std::int32_t y = 0; y < output.height; ++y) {
            for (std::int32_t x = 0; x < output.width; ++x) {
                output.values[(static_cast<std::size_t>(channel) * output.height + y) * output.width + x] =
                    input.values[(static_cast<std::size_t>(source_channel) * input.height + y * block + offset_y) *
                                     input.width + x * block + offset_x];
            }
        }
    }
}

void im2col(const Node& node, const Tensor& input, Tensor& output) {
    const auto taps = node.kernel[0] * node.kernel[1];
    for (std::int32_t channel = 0; channel < output.channels; ++channel) {
        const auto source = channel / taps;
        const auto ky = (channel % taps) / node.kernel[1];
        const auto kx = channel % node.kernel[1];
        for (std::int32_t y = 0; y < output.height; ++y) {
            const auto iy = y * node.stride[0] - node.pads[0] + ky;
            for (std::int32_t x = 0; x < output.width; ++x) {
                const auto ix = x * node.stride[1] - node.pads[1] + kx;
                output.values[(static_cast<std::size_t>(channel) * output.height + y) * output.width + x] =
                    iy >= 0 && iy < input.height && ix >= 0 && ix < input.width
                    ? input.values[(static_cast<std::size_t>(source) * input.height + iy) * input.width + ix]
                    : 0.0F;
            }
        }
    }
}

}  // namespace

std::string_view profile_id(NumericsProfile profile) noexcept {
    return profile == NumericsProfile::integer_v1 ? "nvcr.det-int.v1" : "nvcr.det-fp32.v1";
}

bool is_integer_pointwise(const Node& node) noexcept {
    return node.kind == OpKind::conv && !node.depthwise &&
        node.kernel == std::array<std::int32_t, 2>{1, 1} &&
        node.stride == std::array<std::int32_t, 2>{1, 1} &&
        node.pads == std::array<std::int32_t, 4>{0, 0, 0, 0};
}

IntegerWeights quantize_weights(const Program& program, const Node& node) {
    const auto rows = static_cast<std::size_t>(node.out_channels);
    const auto columns = static_cast<std::size_t>(node.in_channels);
    IntegerWeights output;
    output.values.resize(rows * columns);
    output.scales.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* weights = program.weights.data() + node.weight_offset + row * columns;
        float maximum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column) {
            maximum = std::max(maximum, std::abs(weights[column]));
        }
        if (maximum == 0.0F) {
            output.scales[row] = 1.0F;
            continue;
        }
        const double limit = node.weight_bits == 16 ? 32767.0 : 127.0;
        const double scale = static_cast<double>(maximum) / limit;
        output.scales[row] = static_cast<float>(scale);
        for (std::size_t column = 0; column < columns; ++column) {
            const double quantized = std::nearbyint(static_cast<double>(weights[column]) / scale);
            output.values[row * columns + column] =
                static_cast<std::int16_t>(std::clamp(quantized, -limit, limit));
        }
    }
    return output;
}

Result<Program> load_program(const std::filesystem::path& program_json) {
    auto document = detail::json::parse_file(program_json, subsystem);
    if (!document) return document.error();
    const auto& root = document.value();
    auto format = text(root, "format");
    auto numerics = text(root, "numerics");
    auto version = integer(root, "version");
    if (!format || format.value() != program_format || !version || version.value() != 1) {
        return invalid("not an NVCR deterministic program v1: " + program_json.string());
    }
    if (!numerics || numerics.value() != program_numerics) {
        return invalid("unsupported deterministic numerics in " + program_json.string());
    }

    Program program;
    program.numerics = numerics.value();
    auto name = text(root, "name");
    if (!name) return name.error();
    program.name = name.value();

    const auto* weights = root.find("weights");
    if (weights == nullptr) return invalid("program has no weights");
    auto weights_file = text(*weights, "file");
    auto weights_count = integer(*weights, "count");
    auto weights_sha = text(*weights, "sha256");
    if (!weights_file) return weights_file.error();
    if (!weights_count || weights_count.value() < 0) return invalid("invalid weights count");
    if (!weights_sha) return weights_sha.error();
    if (weights_file.value().find('/') != std::string::npos ||
        weights_file.value().find('\\') != std::string::npos) {
        return invalid("weights file must sit next to the program");
    }
    const auto weights_path = program_json.parent_path() / weights_file.value();
    auto digest = detail::sha256_file(weights_path, subsystem);
    if (!digest) return digest.error();
    if (digest.value() != weights_sha.value()) {
        return Error(ErrorCode::digest_mismatch, "deterministic weights digest mismatch: " +
                     weights_path.string(), std::string(subsystem));
    }
    auto loaded = read_weights(weights_path, static_cast<std::size_t>(weights_count.value()));
    if (!loaded) return loaded.error();
    program.weights = std::move(loaded.value());
    program.weights_sha256 = weights_sha.value();

    TensorTable tensors(program);
    const auto* inputs = root.find("inputs");
    const auto* outputs = root.find("outputs");
    const auto* nodes = root.find("nodes");
    if (inputs == nullptr || outputs == nullptr || nodes == nullptr) {
        return invalid("program needs inputs, outputs, and nodes");
    }
    for (const auto& input : inputs->array_value) {
        auto input_name = text(input, "name");
        auto channels = integer(input, "channels");
        if (!input_name) return input_name.error();
        if (!channels || channels.value() <= 0) return invalid("input channels must be positive");
        program.inputs.push_back(
            tensors.define(input_name.value(), static_cast<std::int32_t>(channels.value())));
    }
    for (const auto& value : nodes->array_value) {
        auto node = parse_node(value, tensors, program.weights.size());
        if (!node) return node.error();
        // Channel-preserving ops inherit their input's channel count.
        auto& channels = program.tensor_channels[node.value().output];
        if (channels == 0) {
            const auto source = program.tensor_channels[node.value().inputs[0]];
            channels = node.value().kind == OpKind::depth_to_space
                ? source / (node.value().block * node.value().block)
                : source;
        }
        program.nodes.push_back(node.value());
    }
    for (const auto& output : outputs->array_value) {
        auto output_name = text(output, "name");
        if (!output_name) return output_name.error();
        auto id = tensors.find(output_name.value());
        if (!id) return id.error();
        program.outputs.push_back(id.value());
    }

    program.last_use.assign(program.tensor_names.size(), 0);
    for (std::size_t index = 0; index < program.nodes.size(); ++index) {
        const auto& node = program.nodes[index];
        for (std::size_t input = 0; input < node.input_count; ++input) {
            program.last_use[node.inputs[input]] = index;
        }
    }
    for (const auto output : program.outputs) program.last_use[output] = program.nodes.size();
    return program;
}

Result<Extent> output_extent(
    const Program& program, const Node& node, std::span<const Extent> inputs) {
    const auto& input = inputs[0];
    const auto expected = program.tensor_channels[node.inputs[0]];
    if (input.channels != expected || input.height <= 0 || input.width <= 0) {
        return invalid("tensor '" + program.tensor_names[node.inputs[0]] +
                       "' has an unexpected shape");
    }
    switch (node.kind) {
    case OpKind::conv: {
        if (input.channels != node.in_channels) return invalid("conv input channel mismatch");
        const auto height = input.height + node.pads[0] + node.pads[2] - node.kernel[0];
        const auto width = input.width + node.pads[1] + node.pads[3] - node.kernel[1];
        if (height < 0 || width < 0) return invalid("conv input is smaller than its kernel");
        return Extent{node.out_channels, height / node.stride[0] + 1, width / node.stride[1] + 1};
    }
    case OpKind::mul:
    case OpKind::add:
        if (inputs.size() != 2 || inputs[1].channels != input.channels ||
            inputs[1].height != input.height || inputs[1].width != input.width) {
            return invalid("elementwise inputs differ in shape");
        }
        return input;
    case OpKind::scale_channels:
        if (inputs.size() != 2 || inputs[1].channels != input.channels || inputs[1].height != 1 ||
            inputs[1].width != 1) {
            return invalid("scale_channels needs a [C, 1, 1] scale");
        }
        return input;
    case OpKind::concat:
        if (inputs.size() != 2 || inputs[1].height != input.height || inputs[1].width != input.width ||
            inputs[1].channels != program.tensor_channels[node.inputs[1]]) {
            return invalid("concat inputs differ in spatial shape");
        }
        return Extent{input.channels + inputs[1].channels, input.height, input.width};
    case OpKind::crop:
        if (inputs.size() != 2 ||
            (node.axis == 2 ? inputs[1].height > input.height : inputs[1].width > input.width)) {
            return invalid("crop reference is larger than its input");
        }
        return node.axis == 2 ? Extent{input.channels, inputs[1].height, input.width}
                              : Extent{input.channels, input.height, inputs[1].width};
    case OpKind::space_to_depth:
        if (input.height % node.block != 0 || input.width % node.block != 0) {
            return invalid("space_to_depth input is not a multiple of its block");
        }
        return Extent{input.channels * node.block * node.block, input.height / node.block,
                      input.width / node.block};
    case OpKind::im2col: {
        const auto height = input.height + node.pads[0] + node.pads[2] - node.kernel[0];
        const auto width = input.width + node.pads[1] + node.pads[3] - node.kernel[1];
        if (height < 0 || width < 0) return invalid("im2col input is smaller than its kernel");
        return Extent{input.channels * node.kernel[0] * node.kernel[1], height / node.stride[0] + 1,
                      width / node.stride[1] + 1};
    }
    case OpKind::mul_scalar:
    case OpKind::sigmoid:
    case OpKind::clamp:
        return input;
    case OpKind::slice_channels:
        if (node.begin < 0 || node.end <= node.begin || node.end > input.channels) {
            return invalid("channel slice is out of range");
        }
        return Extent{node.end - node.begin, input.height, input.width};
    case OpKind::depth_to_space:
        if (input.channels % (node.block * node.block) != 0) {
            return invalid("depth_to_space channels are not divisible");
        }
        return Extent{input.channels / (node.block * node.block),
                      input.height * node.block, input.width * node.block};
    }
    return invalid("unknown op");
}

Result<std::vector<Tensor>> run_cpu(
    const Program& program, std::span<const Tensor> inputs, NumericsProfile profile) {
    if (inputs.size() != program.inputs.size()) return invalid("wrong number of program inputs");
    const DenormalGuard guard;
    std::vector<std::optional<Tensor>> tensors(program.tensor_names.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto id = program.inputs[index];
        if (inputs[index].channels != program.tensor_channels[id] ||
            inputs[index].values.size() != inputs[index].size()) {
            return invalid("input '" + program.tensor_names[id] + "' has the wrong shape");
        }
        tensors[id] = inputs[index];
    }

    for (std::size_t index = 0; index < program.nodes.size(); ++index) {
        const auto& node = program.nodes[index];
        std::array<Extent, 2> extents{};
        for (std::size_t input = 0; input < node.input_count; ++input) {
            const auto& tensor = tensors[node.inputs[input]];
            if (!tensor.has_value()) return invalid("program reads a released tensor");
            extents[input] = Extent{tensor->channels, tensor->height, tensor->width};
        }
        auto extent = output_extent(program, node, std::span(extents).first(node.input_count));
        if (!extent) return extent.error();
        Tensor output{extent.value().channels, extent.value().height, extent.value().width, {}};
        output.values.resize(output.size());
        const auto& first = *tensors[node.inputs[0]];
        switch (node.kind) {
        case OpKind::conv:
            if (profile == NumericsProfile::integer_v1 && is_integer_pointwise(node)) {
                integer_pointwise(program, node, first, output);
            } else {
                conv(program, node, first, output);
            }
            break;
        case OpKind::mul:
        case OpKind::add: {
            const auto& second = *tensors[node.inputs[1]];
            for (std::size_t element = 0; element < output.values.size(); ++element) {
                output.values[element] = node.kind == OpKind::mul
                    ? det_mul(first.values[element], second.values[element])
                    : det_add(first.values[element], second.values[element]);
            }
            break;
        }
        case OpKind::mul_scalar:
            for (std::size_t element = 0; element < output.values.size(); ++element) {
                output.values[element] = det_mul(first.values[element], node.scalar);
            }
            break;
        case OpKind::sigmoid:
            for (std::size_t element = 0; element < output.values.size(); ++element) {
                output.values[element] = det_sigmoid(first.values[element]);
            }
            break;
        case OpKind::slice_channels: {
            const auto plane = static_cast<std::size_t>(first.height) * first.width;
            std::copy_n(first.values.begin() + static_cast<std::ptrdiff_t>(node.begin * plane),
                        output.values.size(), output.values.begin());
            break;
        }
        case OpKind::depth_to_space:
            depth_to_space(node, first, output);
            break;
        case OpKind::scale_channels: {
            const auto& scale = *tensors[node.inputs[1]];
            const auto plane = static_cast<std::size_t>(first.height) * first.width;
            for (std::size_t element = 0; element < output.values.size(); ++element) {
                output.values[element] = det_mul(first.values[element], scale.values[element / plane]);
            }
            break;
        }
        case OpKind::clamp:
            for (std::size_t element = 0; element < output.values.size(); ++element) {
                const float value = first.values[element];
                output.values[element] = value < node.minimum ? node.minimum
                    : value > node.maximum ? node.maximum : value;
            }
            break;
        case OpKind::concat: {
            const auto& second = *tensors[node.inputs[1]];
            std::copy(first.values.begin(), first.values.end(), output.values.begin());
            std::copy(second.values.begin(), second.values.end(),
                      output.values.begin() + static_cast<std::ptrdiff_t>(first.values.size()));
            break;
        }
        case OpKind::crop:
            for (std::int32_t channel = 0; channel < output.channels; ++channel) {
                for (std::int32_t y = 0; y < output.height; ++y) {
                    std::copy_n(first.values.begin() +
                                    static_cast<std::ptrdiff_t>((static_cast<std::size_t>(channel) * first.height + y) *
                                                                first.width),
                                output.width,
                                output.values.begin() +
                                    static_cast<std::ptrdiff_t>((static_cast<std::size_t>(channel) * output.height + y) *
                                                                output.width));
                }
            }
            break;
        case OpKind::space_to_depth:
            space_to_depth(node, first, output);
            break;
        case OpKind::im2col:
            im2col(node, first, output);
            break;
        }
        tensors[node.output] = std::move(output);
        for (std::size_t input = 0; input < node.input_count; ++input) {
            if (program.last_use[node.inputs[input]] == index) tensors[node.inputs[input]].reset();
        }
    }

    std::vector<Tensor> outputs;
    outputs.reserve(program.outputs.size());
    for (const auto id : program.outputs) {
        if (!tensors[id].has_value()) return invalid("program output was not produced");
        outputs.push_back(*tensors[id]);
    }
    return outputs;
}

}  // namespace nvcr::deterministic
