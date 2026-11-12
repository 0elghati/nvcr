// nvcr.fx.v1 on TensorRT: plans compiled from programs must compute exactly the
// CPU reference's integers and fail on exactly the same inputs.
//
// usage: nvcr_fx_tensorrt_tests [PROGRAM_DIR [BUNDLE_DIR WIDTHxHEIGHT...]]
//   PROGRAM_DIR: DCVC-RT fx programs (scripts/backends/dcvcrt/export_fx_programs.py);
//   every stage is then checked on seeded inputs: plans built for 64x64 to 240x240
//   run at 240x240 and 176x144, or a bundle's own plans run at the given sizes.

#include "deterministic/fx_program.hpp"
#include "provider/tensorrt/fx/ops.hpp"
#include "provider/tensorrt/fx/plugins.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>

namespace {

namespace det = nvcr::deterministic::fx;
namespace trt = nvcr::tensorrt::fx;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, nvinfer1::AsciiChar const* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "TensorRT: " << message << '\n';
    }
};

// Runs a plan; returns nullopt when a kernel set the failure word.
std::optional<std::vector<det::Tensor>> run_plan(const std::vector<std::byte>& plan, const det::Program& program,
                                                 const std::vector<det::Tensor>& inputs) {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    expect(engine != nullptr, program.name + ": plan loads");
    if (!engine) return std::nullopt;
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    std::vector<void*> buffers;
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        const auto& t = inputs[slot];
        context->setInputShape(program.tensor_names[program.inputs[slot]].c_str(),
                               nvinfer1::Dims4{1, t.channels, t.height, t.width});
        void* device = nullptr;
        cudaMalloc(&device, inputs[slot].values.size() * 4);
        cudaMemcpy(device, inputs[slot].values.data(), inputs[slot].values.size() * 4, cudaMemcpyHostToDevice);
        context->setTensorAddress(program.tensor_names[program.inputs[slot]].c_str(), device);
        buffers.push_back(device);
    }
    std::vector<det::Tensor> outputs;
    for (const auto id : program.outputs) {
        const auto dims = context->getTensorShape(program.tensor_names[id].c_str());
        det::Tensor t{static_cast<std::int32_t>(dims.d[1]), static_cast<std::int32_t>(dims.d[2]),
                      static_cast<std::int32_t>(dims.d[3]), {}};
        t.values.resize(static_cast<std::size_t>(t.channels) * t.plane());
        void* device = nullptr;
        cudaMalloc(&device, t.values.size() * 4);
        context->setTensorAddress(program.tensor_names[id].c_str(), device);
        buffers.push_back(device);
        outputs.push_back(std::move(t));
    }
    std::uint32_t* word = nullptr;
    std::uint32_t failed = 0;
    cudaMalloc(&word, 4);
    cudaMemset(word, 0, 4);
    const auto previous = trt::install_failure_word(word);
    expect(context->enqueueV3(nullptr), program.name + ": enqueue");
    trt::install_failure_word(previous);
    cudaDeviceSynchronize();
    cudaMemcpy(&failed, word, 4, cudaMemcpyDeviceToHost);
    cudaFree(word);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        cudaMemcpy(outputs[i].values.data(), buffers[inputs.size() + i], outputs[i].values.size() * 4,
                   cudaMemcpyDeviceToHost);
    }
    for (void* b : buffers) cudaFree(b);
    if (failed != 0) return std::nullopt;
    return outputs;
}

void compare(const std::vector<std::byte>& plan, const det::Program& program, const std::vector<det::Tensor>& inputs,
             const std::string& frame) {
    const auto reference = det::run(program, inputs);
    const auto got = run_plan(plan, program, inputs);
    expect(reference.has_value() == got.has_value(), program.name + ": TensorRT and the reference agree on failure");
    if (!reference || !got) return;
    for (std::size_t i = 0; i < got->size(); ++i) {
        expect((*got)[i].values == reference.value()[i].values,
               program.name + "." + program.tensor_names[program.outputs[i]] + " is bit-identical");
    }
    std::cout << program.name << " at " << frame << ": identical\n";
}

// Seeded inputs at the scale the codec feeds: small integers at 2^-14, pictures in [0, 2^14], QP 37,
// the spatial steps' parity masks and entropy symbols.
det::Tensor input(const std::string& name, int channels, int height, int width, std::mt19937& rng) {
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

// Input shapes of a DCVC-RT stage at a frame size (padded to 16).
std::vector<std::array<int, 3>> stage_shapes(const det::Program& program, int width, int height) {
    const int h = (height + 15) / 16 * 16, w = (width + 15) / 16 * 16, yh = h / 16, yw = w / 16;
    const int zh = (yh + 3) / 4, zw = (yw + 3) / 4;
    const std::map<std::string, std::array<int, 3>> sizes{
        {"z_fixed", {128, zh, zw}}, {"latent_ref", {1, yh, yw}}, {"common", {256, yh, yw}},
        {"frame", {3, h, w}},       {"qp", {1, 1, 1}},           {"feature", {256, h / 8, w / 8}},
        {"context", {256, h / 8, w / 8}}, {"temporal_context", {256, h / 8, w / 8}}};
    std::vector<std::array<int, 3>> shapes;
    for (const auto id : program.inputs) {
        const auto found = sizes.find(program.tensor_names[id]);
        shapes.push_back({program.tensor_channels[id], found != sizes.end() ? found->second[1] : yh,
                          found != sizes.end() ? found->second[2] : yw});
    }
    return shapes;
}

// Without a bundle, plans are built for 64x64 to 240x240 and run at two sizes.
void dcvcrt_stages(const std::filesystem::path& directory, const std::filesystem::path& bundle,
                   std::vector<std::pair<int, int>> frames) {
    if (bundle.empty()) frames = {{240, 240}, {176, 144}};
    std::mt19937 rng(7);
    for (const auto* stage : {"i_prior", "i_spatial_prior_1", "i_spatial_prior_2", "i_spatial_prior_3",
                              "i_synthesis", "p_reference_frame", "p_reference_feature", "p_prior",
                              "p_spatial_prior", "p_synthesis", "p_reset_frame"}) {
        auto program = det::load_program(directory, stage);
        expect(program.has_value(), std::string(stage) + ": program loads");
        if (!program) continue;
        std::vector<std::byte> plan;
        if (bundle.empty()) {
            const auto low = stage_shapes(program.value(), 64, 64), high = stage_shapes(program.value(), 240, 240);
            std::vector<trt::InputShape> shapes;
            for (std::size_t slot = 0; slot < high.size(); ++slot) {
                shapes.push_back({program.value().tensor_names[program.value().inputs[slot]], high[slot][0],
                                  high[slot][1], high[slot][2], low[slot][1], low[slot][2], high[slot][1],
                                  high[slot][2]});
            }
            auto built = trt::build_plan(program.value(), shapes);
            expect(built.has_value(), std::string(stage) + ": plan builds");
            if (!built) {
                std::cerr << built.error().describe() << '\n';
                continue;
            }
            plan = std::move(built.value().bytes);
        } else {
            std::ifstream file(bundle / (std::string(stage) + ".fx.plan"), std::ios::binary);
            std::transform(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(),
                           std::back_inserter(plan), [](char c) { return static_cast<std::byte>(c); });
            expect(!plan.empty(), std::string(stage) + ": the bundle has its plan");
            if (plan.empty()) continue;
        }
        for (const auto& [width, height] : frames) {
            std::vector<det::Tensor> inputs;
            const auto shapes = stage_shapes(program.value(), width, height);
            for (std::size_t slot = 0; slot < shapes.size(); ++slot) {
                const auto& name = program.value().tensor_names[program.value().inputs[slot]];
                inputs.push_back(input(name, shapes[slot][0], shapes[slot][1], shapes[slot][2], rng));
            }
            compare(plan, program.value(), inputs, std::to_string(width) + "x" + std::to_string(height));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    if (!trt::kernels_available(major, minor)) {
        std::cout << "nvcr.fx.v1 kernels were not compiled for this GPU; skipping\n";
        return 0;
    }
    expect(trt::register_plugins(), "the fx plugins register");
    std::vector<std::pair<int, int>> frames;
    for (int a = 3; a < argc; ++a) {
        int width = 0, height = 0;
        if (std::sscanf(argv[a], "%dx%d", &width, &height) != 2) return 2;
        frames.emplace_back(width, height);
    }
    if (argc > 1) dcvcrt_stages(argv[1], argc > 2 ? argv[2] : "", frames);
    if (failures == 0) std::cout << "nvcr.fx.v1 TensorRT tests passed\n";
    return failures == 0 ? 0 : 1;
}
