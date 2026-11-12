// nvcr.intconv.v1 conformance (issue #216): the CPU reference is checked
// against independent double-precision math, and every CUDA result must equal
// the CPU reference bit for bit. Because both follow the same exactly rounded operation
// sequence, CPU/GPU bit equality on one machine is the same property that makes
// two different GPUs agree.
//
// Usage: nvcr_deterministic_executor_tests [PROGRAM_DIR]

#include "deterministic/det_math.hpp"
#include "deterministic/program.hpp"
#if defined(NVCR_TEST_HAS_CUDA)
#include "deterministic/cuda_executor.hpp"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace det = nvcr::deterministic;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

float uniform(std::mt19937& random, float low, float high) {
    // Built from raw engine output so the inputs are identical on every platform.
    const auto unit = static_cast<float>(random() >> 8U) / static_cast<float>(1U << 24U);
    return low + (high - low) * unit;
}

det::Tensor random_tensor(std::mt19937& random, std::int32_t channels, std::int32_t height,
                          std::int32_t width, float scale) {
    det::Tensor tensor{channels, height, width, {}};
    tensor.values.resize(tensor.size());
    for (auto& value : tensor.values) value = uniform(random, -scale, scale);
    return tensor;
}

// Bit equality, except that NaN payloads are unspecified by IEEE-754 (x86
// propagates an input payload, CUDA returns a canonical NaN). The profile
// therefore treats every NaN alike: no decision may depend on a payload.
bool same_bits(float left, float right) {
    if (std::isnan(left) || std::isnan(right)) return std::isnan(left) && std::isnan(right);
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

std::size_t differing(const det::Tensor& left, const det::Tensor& right) {
    std::size_t count = left.values.size() == right.values.size() ? 0 : 1;
    for (std::size_t index = 0; index < std::min(left.values.size(), right.values.size()); ++index) {
        if (!same_bits(left.values[index], right.values[index])) ++count;
    }
    return count;
}

bool bit_equal(const det::Tensor& left, const det::Tensor& right) {
    return left.channels == right.channels && left.height == right.height &&
        left.width == right.width && differing(left, right) == 0;
}

void exp_accuracy_test() {
    double worst = 0.0;
    for (float x = -87.0F; x < 88.0F; x += 0.0137F) {
        const double expected = std::exp(static_cast<double>(x));
        const double actual = det::det_exp(x);
        worst = std::max(worst, std::abs(actual - expected) / expected);
    }
    expect(worst < 5e-7, "det_exp stays within a few ulp of exp");
    expect(det::det_exp(-87.5F) == 0.0F, "det_exp returns zero instead of a denormal");
    expect(std::isinf(det::det_exp(88.0F)), "det_exp overflows to infinity");
    expect(std::isnan(det::det_exp(std::numeric_limits<float>::quiet_NaN())), "det_exp keeps NaN");
    expect(det::det_sigmoid(-200.0F) == 0.0F && det::det_sigmoid(200.0F) == 1.0F,
           "det_sigmoid saturates");
    expect(det::det_half_bits_to_float(0x3c00U) == 1.0F &&
               det::det_half_bits_to_float(0xc000U) == -2.0F &&
               det::det_half_bits_to_float(0x0001U) == std::ldexp(1.0F, -24) &&
               det::det_half_bits_to_float(0x7bffU) == 65504.0F,
           "half widening is exact");
}

// A program exercising every op on awkward, non-tile-multiple shapes.
det::Program op_program(std::mt19937& random) {
    det::Program program;
    program.name = "ops";
    program.numerics = std::string(det::intconv_profile);
    auto define = [&](std::string name, std::int32_t channels) {
        program.tensor_names.push_back(std::move(name));
        program.tensor_channels.push_back(channels);
        return program.tensor_names.size() - 1;
    };
    auto add_weights = [&](std::size_t count, float scale) {
        const auto offset = program.weights.size();
        for (std::size_t index = 0; index < count; ++index) {
            program.weights.push_back(uniform(random, -scale, scale));
        }
        return offset;
    };
    auto conv = [&](std::size_t input, std::int32_t in, std::int32_t out, bool depthwise,
                    std::array<std::int32_t, 2> kernel, std::array<std::int32_t, 2> stride,
                    std::array<std::int32_t, 4> pads, std::string name) {
        det::Node node;
        node.kind = det::OpKind::conv;
        node.inputs[0] = input;
        node.in_channels = in;
        node.out_channels = out;
        node.depthwise = depthwise;
        node.kernel = kernel;
        node.stride = stride;
        node.pads = pads;
        const auto per_output = static_cast<std::size_t>(depthwise ? 1 : in) * kernel[0] * kernel[1];
        node.weight_offset = add_weights(static_cast<std::size_t>(out) * per_output, 0.3F);
        node.bias_offset = add_weights(static_cast<std::size_t>(out), 0.1F);
        node.output = define(std::move(name), out);
        program.nodes.push_back(node);
        return node.output;
    };
    auto unary = [&](det::OpKind kind, std::size_t input, std::string name, std::int32_t channels) {
        det::Node node;
        node.kind = kind;
        node.inputs[0] = input;
        node.output = define(std::move(name), channels);
        program.nodes.push_back(node);
        return &program.nodes.back();
    };

    const auto x = define("x", 70);
    program.inputs.push_back(x);
    const auto pointwise = conv(x, 70, 133, false, {1, 1}, {1, 1}, {0, 0, 0, 0}, "pointwise");
    auto* scaled = unary(det::OpKind::mul_scalar, pointwise, "scaled", 133);
    scaled->scalar = 4.0F;
    const auto scaled_id = scaled->output;
    const auto gate = unary(det::OpKind::sigmoid, scaled_id, "gate", 133)->output;
    det::Node product;
    product.kind = det::OpKind::mul;
    product.inputs = {gate, pointwise};
    product.input_count = 2;
    product.output = define("product", 133);
    program.nodes.push_back(product);
    const auto depthwise = conv(product.output, 133, 133, true, {3, 3}, {1, 1}, {1, 1, 1, 1}, "depthwise");
    auto* slice = unary(det::OpKind::slice_channels, depthwise, "slice", 128);
    slice->begin = 3;
    slice->end = 131;
    const auto slice_id = slice->output;
    const auto dense = conv(slice_id, 128, 36, false, {3, 3}, {1, 1}, {1, 1, 1, 1}, "dense");
    const auto strided = conv(dense, 36, 36, false, {2, 2}, {2, 2}, {0, 0, 0, 0}, "strided");
    auto* shuffle = unary(det::OpKind::depth_to_space, strided, "shuffle", 9);
    shuffle->block = 2;
    const auto shuffle_id = shuffle->output;
    det::Node sum;
    sum.kind = det::OpKind::add;
    sum.inputs = {shuffle_id, shuffle_id};
    sum.input_count = 2;
    sum.output = define("sum", 9);
    program.nodes.push_back(sum);
    program.outputs = {sum.output, pointwise};

    program.last_use.assign(program.tensor_names.size(), 0);
    for (std::size_t index = 0; index < program.nodes.size(); ++index) {
        for (std::size_t input = 0; input < program.nodes[index].input_count; ++input) {
            program.last_use[program.nodes[index].inputs[input]] = index;
        }
    }
    for (const auto output : program.outputs) program.last_use[output] = program.nodes.size();
    return program;
}

void cpu_reference_test() {
    std::mt19937 random(7);
    const auto program = op_program(random);
    const auto input = random_tensor(random, 70, 11, 18, 2.0F);
    auto outputs = det::run_cpu(program, std::span(&input, 1));
    expect(outputs.has_value(), "CPU executor runs the op program");
    if (!outputs) return;
    const auto& pointwise = outputs.value()[1];
    expect(pointwise.channels == 133 && pointwise.height == 11 && pointwise.width == 18,
           "pointwise output shape");
    // Independent double-precision check of the first conv.
    const auto& node = program.nodes[0];
    double worst = 0.0;
    for (std::int32_t out = 0; out < 133; out += 17) {
        for (std::int32_t position = 0; position < 11 * 18; position += 13) {
            double expected = program.weights[node.bias_offset + static_cast<std::size_t>(out)];
            for (std::int32_t in = 0; in < 70; ++in) {
                expected += static_cast<double>(program.weights[node.weight_offset +
                                                                static_cast<std::size_t>(out) * 70 + in]) *
                    input.values[static_cast<std::size_t>(in) * 11 * 18 + position];
            }
            worst = std::max(worst, std::abs(expected -
                pointwise.values[static_cast<std::size_t>(out) * 11 * 18 + position]));
        }
    }
    expect(worst < 2e-3, "CPU integer pointwise conv stays within 16-bit quantization error");
    const auto& sum = outputs.value()[0];
    expect(sum.channels == 9 && sum.height == 10 && sum.width == 18, "depth_to_space output shape");

    auto again = det::run_cpu(program, std::span(&input, 1));
    expect(again.has_value() && bit_equal(again.value()[0], sum) &&
               bit_equal(again.value()[1], pointwise),
           "CPU executor is deterministic across runs and thread schedules");
}

#if defined(NVCR_TEST_HAS_CUDA)
void compare_cpu_cuda(const det::Program& program, std::span<const det::Tensor> inputs,
                      std::string_view label);
#endif

// scale_channels, clamp, concat, crop, and space_to_depth: the CPU reference
// against direct indexing, then CUDA against the CPU reference.
det::Program data_movement_program() {
    det::Program program;
    program.name = "data movement";
    program.numerics = std::string(det::intconv_profile);
    // 0 x [5, 12, 20], 1 s [20, 1, 1], 2 r [1, 5, 7], 3 a = s2d(x, 2) [20, 6, 10],
    // 4 b = a * s, 5 c = concat(b, a) [40, 6, 10], 6 d = crop_h(c, r) [40, 5, 10],
    // 7 e = crop_w(d, r) [40, 5, 7], 8 f = clamp(e, -0.5, 0.75)
    program.tensor_names = {"x", "s", "r", "a", "b", "c", "d", "e", "f"};
    program.tensor_channels = {5, 20, 1, 20, 20, 40, 40, 40, 40};
    program.inputs = {0, 1, 2};
    auto node = [](det::OpKind kind, std::size_t first, std::size_t second, std::size_t output) {
        det::Node value;
        value.kind = kind;
        value.inputs = {first, second};
        value.input_count = second == static_cast<std::size_t>(-1) ? 1 : 2;
        if (value.input_count == 1) value.inputs[1] = 0;
        value.output = output;
        return value;
    };
    constexpr auto none = static_cast<std::size_t>(-1);
    auto s2d = node(det::OpKind::space_to_depth, 0, none, 3);
    s2d.block = 2;
    auto crop_h = node(det::OpKind::crop, 5, 2, 6);
    crop_h.axis = 2;
    auto crop_w = node(det::OpKind::crop, 6, 2, 7);
    crop_w.axis = 3;
    auto clamp = node(det::OpKind::clamp, 7, none, 8);
    clamp.minimum = -0.5F;
    clamp.maximum = 0.75F;
    program.nodes = {s2d, node(det::OpKind::scale_channels, 3, 1, 4), node(det::OpKind::concat, 4, 3, 5),
                     crop_h, crop_w, clamp};
    program.outputs = {8, 3};
    program.last_use.assign(program.tensor_names.size(), 0);
    for (std::size_t index = 0; index < program.nodes.size(); ++index) {
        for (std::size_t input = 0; input < program.nodes[index].input_count; ++input) {
            program.last_use[program.nodes[index].inputs[input]] = index;
        }
    }
    for (const auto output : program.outputs) program.last_use[output] = program.nodes.size();
    return program;
}

void data_movement_test() {
    std::mt19937 random(41);
    const auto program = data_movement_program();
    std::vector<det::Tensor> inputs{random_tensor(random, 5, 12, 20, 1.5F),
                                    random_tensor(random, 20, 1, 1, 2.0F),
                                    random_tensor(random, 1, 5, 7, 1.0F)};
    auto outputs = det::run_cpu(program, inputs);
    expect(outputs.has_value(), "CPU runs the data-movement program");
    if (!outputs) {
        std::cerr << outputs.error().message() << '\n';
        return;
    }
    const auto& x = inputs[0];
    const auto& f = outputs.value()[0];
    const auto& a = outputs.value()[1];
    expect(a.channels == 20 && a.height == 6 && a.width == 10, "space_to_depth shape");
    expect(f.channels == 40 && f.height == 5 && f.width == 7, "concat and crop shape");
    bool space_to_depth_ok = true;
    for (int c = 0; c < 20; ++c) {
        for (int y = 0; y < 6; ++y) {
            for (int w = 0; w < 10; ++w) {
                // PixelUnshuffle: out[c * 4 + i * 2 + j, y, x] = in[c, 2y + i, 2x + j]
                const int source = c / 4;
                const int i = (c / 2) % 2;
                const int j = c % 2;
                space_to_depth_ok &= same_bits(
                    a.values[(static_cast<std::size_t>(c) * 6 + y) * 10 + w],
                    x.values[(static_cast<std::size_t>(source) * 12 + 2 * y + i) * 20 + 2 * w + j]);
            }
        }
    }
    expect(space_to_depth_ok, "space_to_depth follows PixelUnshuffle channel order");
    bool composed_ok = true;
    for (int c = 0; c < 40; ++c) {
        for (int y = 0; y < 5; ++y) {
            for (int w = 0; w < 7; ++w) {
                const int channel = c % 20;
                float value = a.values[(static_cast<std::size_t>(channel) * 6 + y) * 10 + w];
                if (c < 20) value = det::det_mul(value, inputs[1].values[static_cast<std::size_t>(channel)]);
                value = value < -0.5F ? -0.5F : (value > 0.75F ? 0.75F : value);
                composed_ok &= same_bits(f.values[(static_cast<std::size_t>(c) * 5 + y) * 7 + w], value);
            }
        }
    }
    expect(composed_ok, "scale_channels, concat, crop and clamp match direct indexing");
#if defined(NVCR_TEST_HAS_CUDA)
    compare_cpu_cuda(program, inputs, "data-movement program");
#endif
}

#if defined(NVCR_TEST_HAS_CUDA)
void compare_cpu_cuda(const det::Program& program, std::span<const det::Tensor> inputs,
                      std::string_view label) {
    const std::string what(label);
    auto cpu = det::run_cpu(program, inputs);
    auto executor = det::CudaExecutor::create(program);
    expect(cpu.has_value(), std::string(what) + ": CPU run");
    expect(executor.has_value(), std::string(what) + ": CUDA executor created");
    if (!cpu || !executor) {
        if (!cpu) std::cerr << cpu.error().message() << '\n';
        if (!executor) std::cerr << executor.error().message() << '\n';
        return;
    }
    auto gpu = executor.value().run(inputs);
    expect(gpu.has_value(), std::string(what) + ": CUDA run");
    if (!gpu) {
        std::cerr << gpu.error().message() << '\n';
        return;
    }
    for (std::size_t index = 0; index < cpu.value().size(); ++index) {
        const auto diff = differing(cpu.value()[index], gpu.value()[index]);
        if (diff != 0) {
            std::cerr << what << " output " << index << ": " << diff << " of "
                      << cpu.value()[index].values.size() << " elements differ\n";
        }
        expect(bit_equal(cpu.value()[index], gpu.value()[index]),
               std::string(what) + ": CUDA output is bit-identical to the CPU reference");
    }
}

// Copies a program with every pointwise conv set to the given integer widths.
det::Program with_widths(det::Program program, int weight_bits, int activation_bits) {
    for (auto& node : program.nodes) {
        if (det::is_integer_pointwise(node)) {
            node.weight_bits = weight_bits;
            node.activation_bits = activation_bits;
        }
    }
    return program;
}

struct Widths {
    int weight_bits;
    int activation_bits;
    const char* label;
};

constexpr Widths width_cases[] = {
    {16, 16, "w16a16"}, {8, 16, "w8a16"}, {16, 8, "w16a8"}, {8, 8, "w8a8"}};

void cuda_op_test() {
    std::mt19937 random(11);
    const auto program = op_program(random);
    for (const auto& [height, width] : {std::pair{11, 18}, std::pair{2, 64}, std::pair{33, 47}}) {
        const auto input = random_tensor(random, 70, height, width, 3.0F);
        for (const auto& widths : width_cases) {
            compare_cpu_cuda(with_widths(program, widths.weight_bits, widths.activation_bits),
                             std::span(&input, 1), std::string("op program ") + widths.label);
        }
    }
}

void cuda_sigmoid_sweep_test() {
    det::Program program;
    program.name = "sigmoid";
    program.numerics = std::string(det::intconv_profile);
    program.tensor_names = {"x", "y"};
    program.tensor_channels = {1, 1};
    program.inputs = {0};
    det::Node node;
    node.kind = det::OpKind::sigmoid;
    node.inputs[0] = 0;
    node.output = 1;
    program.nodes = {node};
    program.outputs = {1};
    program.last_use = {0, 1};
    det::Tensor input{1, 1, 0, {}};
    // Every 1021st bit pattern spans every exponent of both signs; the explicit
    // values add both zeros, denormals, infinities, NaN and the exp cut-offs.
    std::vector<std::uint32_t> patterns{
        0x00000000U, 0x80000000U, 0x00000001U, 0x807fffffU, 0x7f800000U, 0xff800000U,
        0x7fc00000U, 0x7f7fffffU, 0xff7fffffU};
    for (std::uint64_t bits = 0; bits <= 0xffffffffULL; bits += 1021) {
        patterns.push_back(static_cast<std::uint32_t>(bits));
    }
    for (const auto word : patterns) {
        float value;
        std::memcpy(&value, &word, sizeof(value));
        input.values.push_back(value);
    }
    for (const float edge : {-88.0F, -87.0F, -86.99999F, 87.0F, 87.99999F, 88.0F}) {
        input.values.push_back(edge);
    }
    input.width = static_cast<std::int32_t>(input.values.size());
    compare_cpu_cuda(program, std::span(&input, 1), "sigmoid bit-pattern sweep");
}

#if defined(NVCR_TEST_HAS_CUDA)
// Input extents for a stage program at a frame of (64 bh) x (64 bw).
std::optional<std::vector<det::Tensor>> stage_inputs(
    const det::Program& program, std::mt19937& random, int bh, int bw) {
    std::vector<det::Tensor> inputs;
    for (const auto id : program.inputs) {
        const auto& name = program.tensor_names[id];
        const auto channels = program.tensor_channels[id];
        int height = 0;
        int width = 0;
        float scale = 2.0F;
        bool integral = false;
        if (name == "q_feature" || name == "q_decoder" || name == "q_recon" || name == "q_dec") {
            height = width = 1;
            scale = 1.5F;
        } else if (name == "reference_frame") {
            height = 64 * bh;
            width = 64 * bw;
            scale = 1.0F;
        } else if (name == "reference_feature" || name == "temporal_context" ||
                   (name == "context" && channels == 256)) {
            height = 8 * bh;
            width = 8 * bw;
        } else if (name == "y_hat" || (name == "context" && channels == 512)) {
            height = 4 * bh;
            width = 4 * bw;
            integral = name == "y_hat";
        } else if (name == "z_hat") {
            height = bh;
            width = bw;
            integral = true;
        } else {
            return std::nullopt;
        }
        auto tensor = random_tensor(random, channels, height, width, scale);
        if (integral) {
            for (auto& value : tensor.values) value = std::nearbyint(value * 3.0F);
        }
        if (name == "reference_frame") {
            for (auto& value : tensor.values) value = std::abs(value);
        }
        inputs.push_back(std::move(tensor));
    }
    return inputs;
}

// Every decoder-side stage program in the directory: CUDA must equal the CPU
// reference bit for bit at the exported widths.
void stage_programs_test(const std::filesystem::path& directory) {
    std::mt19937 random(53);
    for (const char* name : {"i_hyper_synthesis", "i_spatial_prior_1", "i_spatial_prior_2",
                             "i_spatial_prior_3", "i_synthesis", "p_reference_frame",
                             "p_reference_feature", "p_prior", "p_spatial_prior", "p_synthesis"}) {
        const auto path = directory / (std::string(name) + ".detprog.json");
        if (!std::filesystem::exists(path)) continue;
        auto program = det::load_program(path);
        expect(program.has_value(), std::string(name) + " program loads");
        if (!program) {
            std::cerr << program.error().message() << '\n';
            continue;
        }
        auto inputs = stage_inputs(program.value(), random, 1, 2);
        expect(inputs.has_value(), std::string(name) + " has known inputs");
        if (!inputs) continue;
        compare_cpu_cuda(program.value(), inputs.value(), name);
    }
}
#endif

void real_program_test(const std::filesystem::path& directory) {
    std::mt19937 random(23);
    auto hyper = det::load_program(directory / "i_hyper_synthesis.detprog.json");
    expect(hyper.has_value(), "i_hyper_synthesis program loads");
    if (hyper) {
        det::Tensor z{128, 3, 4, {}};
        z.values.resize(z.size());
        for (auto& value : z.values) value = static_cast<float>(static_cast<int>(random() % 17) - 8);
        compare_cpu_cuda(hyper.value(), std::span(&z, 1), "i_hyper_synthesis");
    } else {
        std::cerr << hyper.error().message() << '\n';
    }
    auto spatial = det::load_program(directory / "i_spatial_prior_1.detprog.json");
    expect(spatial.has_value(), "i_spatial_prior_1 program loads");
    if (spatial) {
        const auto context = random_tensor(random, 512, 9, 11, 4.0F);
        compare_cpu_cuda(spatial.value(), std::span(&context, 1), "i_spatial_prior_1");
        for (const auto& widths : width_cases) {
            compare_cpu_cuda(with_widths(spatial.value(), widths.weight_bits, widths.activation_bits),
                             std::span(&context, 1), std::string("i_spatial_prior_1 ") + widths.label);
        }
    } else {
        std::cerr << spatial.error().message() << '\n';
    }
}

#endif

}  // namespace

int main(int argc, char** argv) {
    exp_accuracy_test();
    cpu_reference_test();
    data_movement_test();
#if defined(NVCR_TEST_HAS_CUDA)
    cuda_op_test();
    cuda_sigmoid_sweep_test();
    if (argc >= 2) {
        real_program_test(argv[1]);
        stage_programs_test(argv[1]);
    }
#else
    (void)argc;
    (void)argv;
#endif
    if (failures == 0) std::cout << "deterministic executor conformance passed (nvcr.intconv.v1)\n";
    return failures == 0 ? 0 : 1;
}
