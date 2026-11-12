// nvcr.fx.v1 on TensorRT: plans compiled from programs must compute exactly the
// CPU reference's integers and fail on exactly the same inputs.
//
// usage: nvcr_fx_tensorrt_tests [PROGRAM_DIR]
//   PROGRAM_DIR: DCVC-RT fx programs (scripts/backends/dcvcrt/export_fx_programs.py);
//   every stage is then checked at 240x240 on seeded inputs.

#include "deterministic/fx_program.hpp"
#include "provider/tensorrt/fx/ops.hpp"
#include "provider/tensorrt/fx/plugins.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <iostream>
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
    std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    std::vector<void*> buffers;
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        void* device = nullptr;
        cudaMalloc(&device, inputs[slot].values.size() * 4);
        cudaMemcpy(device, inputs[slot].values.data(), inputs[slot].values.size() * 4, cudaMemcpyHostToDevice);
        context->setTensorAddress(program.tensor_names[program.inputs[slot]].c_str(), device);
        buffers.push_back(device);
    }
    std::vector<det::Tensor> outputs;
    for (const auto id : program.outputs) {
        const auto dims = engine->getTensorShape(program.tensor_names[id].c_str());
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
    trt::failure_word(&word);
    cudaMemset(word, 0, 4);
    expect(context->enqueueV3(nullptr), program.name + ": enqueue");
    cudaDeviceSynchronize();
    cudaMemcpy(&failed, word, 4, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        cudaMemcpy(outputs[i].values.data(), buffers[inputs.size() + i], outputs[i].values.size() * 4,
                   cudaMemcpyDeviceToHost);
    }
    for (void* b : buffers) cudaFree(b);
    if (failed != 0) return std::nullopt;
    return outputs;
}

void compare(const det::Program& program, const std::vector<det::Tensor>& inputs) {
    std::vector<trt::InputShape> shapes;
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        const auto& t = inputs[slot];
        shapes.push_back({program.tensor_names[program.inputs[slot]], t.channels, t.height, t.width});
    }
    auto plan = trt::build_plan(program, shapes);
    expect(plan.has_value(), program.name + ": plan builds");
    if (!plan) {
        std::cerr << plan.error().describe() << '\n';
        return;
    }
    const auto reference = det::run(program, inputs);
    const auto got = run_plan(plan.value().bytes, program, inputs);
    expect(reference.has_value() == got.has_value(), program.name + ": TensorRT and the reference agree on failure");
    if (!reference || !got) return;
    for (std::size_t i = 0; i < got->size(); ++i) {
        expect((*got)[i].values == reference.value()[i].values,
               program.name + "." + program.tensor_names[program.outputs[i]] + " is bit-identical");
    }
    std::cout << program.name << ": identical\n";
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

void dcvcrt_stages(const std::filesystem::path& directory) {
    constexpr int h = 240, w = 240, yh = h / 16, yw = w / 16, zh = (h + 63) / 64, zw = (w + 63) / 64;
    const std::map<std::string, std::array<int, 3>> sizes{
        {"z_fixed", {128, zh, zw}}, {"latent_ref", {1, yh, yw}}, {"common", {256, yh, yw}},
        {"frame", {3, h, w}},       {"qp", {1, 1, 1}},           {"feature", {256, h / 8, w / 8}},
        {"context", {256, h / 8, w / 8}}, {"temporal_context", {256, h / 8, w / 8}}};
    std::mt19937 rng(7);
    for (const auto* stage : {"i_prior", "i_spatial_prior_1", "i_spatial_prior_2", "i_spatial_prior_3",
                              "i_synthesis", "p_reference_frame", "p_reference_feature", "p_prior",
                              "p_spatial_prior", "p_synthesis", "p_reset_frame"}) {
        auto program = det::load_program(directory, stage);
        expect(program.has_value(), std::string(stage) + ": program loads");
        if (!program) continue;
        std::vector<det::Tensor> inputs;
        for (const auto id : program.value().inputs) {
            const auto& name = program.value().tensor_names[id];
            const int channels = program.value().tensor_channels[id];
            const auto found = sizes.find(name);
            const int ih = found != sizes.end() ? found->second[1] : yh;
            const int iw = found != sizes.end() ? found->second[2] : yw;
            inputs.push_back(input(name, channels, ih, iw, rng));
        }
        compare(program.value(), inputs);
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
    if (argc > 1) dcvcrt_stages(argv[1]);
    if (failures == 0) std::cout << "nvcr.fx.v1 TensorRT tests passed\n";
    return failures == 0 ? 0 : 1;
}
