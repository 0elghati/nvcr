// Compiles an nvcr.fx.v1 program into a TensorRT plan: one plugin layer per
// node (concat and slice as native int32 layers), fixed or ranged input shapes. The plan
// computes exactly the program's integers; the program stays the normative
// artifact.

#include "plugins.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>

namespace nvcr::tensorrt::fx {
namespace {

using namespace nvinfer1;
using deterministic::fx::Op;
using deterministic::fx::Program;

// Kernel tile compiled for every GEMM: BM, BN, BK, warps, stages.
constexpr int tile[5] = {64, 128, 32, 4, 3};
constexpr std::int64_t digit_offset = 128 * (256 * 256 - 1) / 255;

class Logger final : public ILogger {
    void log(Severity severity, AsciiChar const* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

Error failure(std::string what) {
    return Error(ErrorCode::invalid_argument, "fx plan: " + std::move(what), "tensorrt.fx");
}

// Plugin fields that own their storage until the layer is created.
struct FieldList final {
    std::vector<std::vector<unsigned char>> storage;
    std::vector<PluginField> fields;

    template <typename T>
    void add(const char* name, const std::vector<T>& values, PluginFieldType type) {
        storage.emplace_back(reinterpret_cast<const unsigned char*>(values.data()),
                             reinterpret_cast<const unsigned char*>(values.data() + values.size()));
        fields.emplace_back(name, storage.back().data(), type, static_cast<std::int32_t>(values.size()));
    }
    void add(const char* name, std::int64_t value) { add(name, std::vector<std::int64_t>{value}, PluginFieldType::kINT64); }
};

ITensor* plugin(INetworkDefinition& network, const char* name, std::vector<ITensor*> inputs, FieldList& f) {
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(name, "1", ""));
    if (creator == nullptr) return nullptr;
    PluginFieldCollection collection{static_cast<std::int32_t>(f.fields.size()), f.fields.data()};
    auto* layer_plugin = creator->createPlugin(name, &collection, TensorRTPhase::kBUILD);
    if (layer_plugin == nullptr) return nullptr;
    auto* layer = network.addPluginV3(inputs.data(), static_cast<std::int32_t>(inputs.size()), nullptr, 0, *layer_plugin);
    return layer ? layer->getOutput(0) : nullptr;
}

Result<ITensor*> gemm(INetworkDefinition& network, const Program& program, const deterministic::fx::Node& node,
                      std::vector<ITensor*> inputs) {
    const auto& g = node.gemm;
    const auto O = static_cast<std::size_t>(g.out_channels);
    const auto K = static_cast<std::size_t>(g.in_channels);
    // Two balanced int8 digit planes: w = d0 + 256 d1.
    std::vector<std::int8_t> wd(2 * O * K);
    for (std::size_t i = 0; i < O * K; ++i) {
        const auto u = static_cast<std::int64_t>(g.weights[i]) + digit_offset;
        wd[i] = static_cast<std::int8_t>((u & 255) - 128);
        wd[O * K + i] = static_cast<std::int8_t>(((u >> 8) & 255) - 128);
    }
    // Row constants: c = bias + 2^(s-1), split as c mod 2^24 and c >> 24 (the kernel's int32 requantization).
    std::vector<std::int32_t> clo(O), chi(O);
    std::int64_t chi_max = 0;
    for (std::size_t o = 0; o < O; ++o) {
        const std::int64_t c = g.bias[o] + (std::int64_t{1} << (g.shift[o] - 1));
        clo[o] = static_cast<std::int32_t>(c & 0xFFFFFF);
        const std::int64_t hi = c >> 24;
        chi_max = std::max(chi_max, hi < 0 ? -hi : hi);
        chi[o] = static_cast<std::int32_t>(hi);
    }
    // The kernel's int32 accumulators are exact only within these bounds.
    const double acc = 2.0 * static_cast<double>(K) * 16384.0;
    if (chi_max >= (std::int64_t{1} << 26) || acc >= 2147483648.0 - 3.0 * 16777216.0 ||
        acc * (1.0 + 1.0 / 256.0 + 1.0 / 65536.0) + static_cast<double>(chi_max) + 16.0 >= 1073741824.0) {
        return failure(program.name + ": a gemm exceeds the kernel's exact int32 range");
    }
    FieldList f;
    f.add("wd", wd, PluginFieldType::kINT8);
    f.add("clo", clo, PluginFieldType::kINT32);
    f.add("chi", chi, PluginFieldType::kINT32);
    f.add("s", g.shift, PluginFieldType::kINT32);
    f.add("O", g.out_channels);
    f.add("K", g.in_channels);
    f.add("g", g.group);
    f.add("nres", static_cast<std::int64_t>(inputs.size()) - 1);
    f.add("xs", g.input_shift);
    const bool ws = node.delta_table >= 0;
    f.add("wsilu", ws ? 1 : 0);
    if (ws) {
        const auto& table = program.delta_tables[static_cast<std::size_t>(node.delta_table)];
        f.add("dc", table.dc, PluginFieldType::kINT16);
        f.add("h", table.h);
        f.add("wsh", table.shift);
    }
    const char* tile_names[5] = {"bm", "bn", "bk", "warps", "stages"};
    for (int i = 0; i < 5; ++i) f.add(tile_names[i], tile[i]);
    auto* out = plugin(network, "NvcrFxGemm", std::move(inputs), f);
    if (out == nullptr) return failure(program.name + ": cannot create a gemm layer");
    return out;
}

}  // namespace

Result<Plan> build_plan(const Program& program, std::span<const InputShape> inputs) {
    if (!register_plugins()) return failure("cannot register the fx plugins");
    Logger logger;
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) return failure("cannot create a TensorRT builder");
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!network || !config) return failure("cannot create a TensorRT network");

    auto* profile = builder->createOptimizationProfile();
    bool ranged = false;
    std::vector<ITensor*> tensors(program.tensor_names.size(), nullptr);
    for (const auto id : program.inputs) {
        const auto& name = program.tensor_names[id];
        const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& s) { return s.name == name; });
        if (found == inputs.end() || found->channels != program.tensor_channels[id]) {
            return failure(program.name + ": no shape given for input " + name);
        }
        const auto c = found->channels;
        if (!found->ranged()) {
            tensors[id] = network->addInput(name.c_str(), DataType::kINT32, Dims4{1, c, found->height, found->width});
            continue;
        }
        tensors[id] = network->addInput(name.c_str(), DataType::kINT32, Dims4{1, c, -1, -1});
        profile->setDimensions(name.c_str(), OptProfileSelector::kMIN, Dims4{1, c, found->min_height, found->min_width});
        profile->setDimensions(name.c_str(), OptProfileSelector::kOPT, Dims4{1, c, found->height, found->width});
        profile->setDimensions(name.c_str(), OptProfileSelector::kMAX, Dims4{1, c, found->max_height, found->max_width});
        ranged = true;
    }
    if (ranged) config->addOptimizationProfile(profile);
    // Slice sizes of dynamic tensors are computed as shape * keep + fixed.
    static const std::int64_t keep[4] = {0, 0, 1, 1};
    std::vector<std::array<std::int64_t, 4>> fixed;
    fixed.reserve(program.nodes.size());
    for (const auto& node : program.nodes) {
        std::vector<ITensor*> in;
        for (const auto id : node.inputs) {
            if (tensors[id] == nullptr) return failure(program.name + ": tensor read before it is written");
            in.push_back(tensors[id]);
        }
        ITensor* out = nullptr;
        FieldList f;
        switch (node.op) {
        case Op::gemm: {
            auto built = gemm(*network, program, node, in);
            if (!built) return built.error();
            out = built.value();
            break;
        }
        case Op::depthwise3x3: {
            const auto& d = node.depthwise;
            f.add("w", std::vector<std::int64_t>(d.weights.begin(), d.weights.end()), PluginFieldType::kINT64);
            f.add("b", d.bias, PluginFieldType::kINT64);
            f.add("s", std::vector<std::int64_t>(d.shift.begin(), d.shift.end()), PluginFieldType::kINT64);
            out = plugin(*network, "NvcrFxDepthwise", in, f);
            break;
        }
        case Op::concat: {
            auto* layer = network->addConcatenation(in.data(), 2);
            layer->setAxis(1);
            out = layer->getOutput(0);
            break;
        }
        case Op::slice: {
            const auto dims = in[0]->getDimensions();
            auto* layer = network->addSlice(*in[0], Dims4{0, node.begin, 0, 0},
                                            Dims4{1, node.end - node.begin, dims.d[2], dims.d[3]}, Dims4{1, 1, 1, 1});
            if (dims.d[2] < 0 || dims.d[3] < 0) {
                fixed.push_back({1, node.end - node.begin, 0, 0});
                const Dims shape4{1, {4}};
                auto* k = network->addConstant(shape4, Weights{DataType::kINT64, keep, 4})->getOutput(0);
                auto* c = network->addConstant(shape4, Weights{DataType::kINT64, fixed.back().data(), 4})->getOutput(0);
                auto* kept = network->addElementWise(*network->addShape(*in[0])->getOutput(0), *k,
                                                     ElementWiseOperation::kPROD)->getOutput(0);
                layer->setInput(2, *network->addElementWise(*kept, *c, ElementWiseOperation::kSUM)->getOutput(0));
            }
            out = layer->getOutput(0);
            break;
        }
        default: {
            f.add("op", static_cast<std::int64_t>(node.op));
            f.add("shift", node.shift);
            f.add("alpha", node.alpha);
            f.add("beta", node.beta);
            f.add("lo", node.lo);
            f.add("hi", node.hi);
            f.add("top", node.hi);
            f.add("factor", node.factor);
            f.add("kernel", node.kernel);
            f.add("stride", node.stride);
            if (node.op == Op::sat_table) f.add("table", node.table, PluginFieldType::kINT32);
            if (node.op == Op::param_row) {
                f.add("rows", node.rows, PluginFieldType::kINT64);
                f.add("channels", program.tensor_channels[node.output]);
            }
            if (node.op == Op::delta) {
                const auto& table = program.delta_tables[static_cast<std::size_t>(node.delta_table)];
                f.add("dc", table.dc, PluginFieldType::kINT16);
                f.add("h", table.h);
                f.add("wsh", table.shift);
            }
            out = plugin(*network, "NvcrFxOp", in, f);
            break;
        }
        }
        if (out == nullptr) return failure(program.name + ": cannot create the layer for a node");
        tensors[node.output] = out;
    }
    Plan result;
    for (const auto id : program.outputs) {
        tensors[id]->setName(program.tensor_names[id].c_str());
        network->markOutput(*tensors[id]);
        tensors[id]->setType(DataType::kINT32);
        const auto d = tensors[id]->getDimensions();
        result.outputs.push_back({program.tensor_names[id], static_cast<int>(d.d[1]), static_cast<int>(d.d[2]),
                                  static_cast<int>(d.d[3])});
    }
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return failure(program.name + ": TensorRT could not build the plan");
    result.bytes.resize(plan->size());
    std::memcpy(result.bytes.data(), plan->data(), plan->size());
    return result;
}

}  // namespace nvcr::tensorrt::fx
