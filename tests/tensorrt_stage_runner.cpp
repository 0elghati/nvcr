// Runs TensorRT stages from a validated NVCR engine bundle through the
// codec-agnostic provider session, for offline per-stage conformance checks.
//
// Job file, one directive per line:
//   bundle DIR
//   model MODEL_ID
//   stage FILE TENSOR_COUNT
//   tensor NAME read|write D0 D1 D2 D3      (-1 marks a dynamic axis)
//   run FILE INPUT_COUNT OUTPUT_COUNT
//   input NAME D0 D1 D2 D3 PATH             (raw little-endian FP16)
//   output NAME PATH
// Each run prints "shape NAME D0 D1 D2 D3" for every output it writes.

#include "provider/tensorrt/session.hpp"

#include <nvcr/configuration/configuration.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace session_api = nvcr::provider::experimental;
using nvcr::tensorrt::DeviceTensor;

struct Input final {
    std::string name;
    nvinfer1::Dims shape{};
    std::string path;
};

struct Run final {
    std::string file;
    std::vector<Input> inputs;
    std::vector<std::pair<std::string, std::string>> outputs;
};

struct Job final {
    std::filesystem::path bundle;
    std::string model;
    std::vector<session_api::ExecutableStageDescriptor> stages;
    std::vector<Run> runs;
};

bool fail(const std::string& message) {
    std::cerr << "nvcr_tensorrt_stage_runner: " << message << '\n';
    return false;
}

nvinfer1::Dims read_dims(std::istream& input) {
    nvinfer1::Dims shape{};
    shape.nbDims = 4;
    for (std::int32_t axis = 0; axis < 4; ++axis) input >> shape.d[axis];
    return shape;
}

bool parse(const std::filesystem::path& path, Job& job) {
    std::ifstream file(path);
    if (!file) return fail("cannot open job file " + path.string());
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        std::string directive;
        if (!(fields >> directive)) continue;
        if (directive == "bundle") {
            std::string value;
            fields >> value;
            job.bundle = std::filesystem::absolute(value);
        } else if (directive == "model") {
            fields >> job.model;
        } else if (directive == "stage") {
            session_api::ExecutableStageDescriptor descriptor;
            std::size_t count = 0;
            fields >> descriptor.stage_id >> count;
            descriptor.artifact.provider_id = "tensorrt";
            descriptor.artifact.codec_id = "stage-runner";
            descriptor.artifact.component_id = descriptor.stage_id;
            descriptor.artifact.path = (job.bundle / descriptor.stage_id).string();
            for (std::size_t index = 0; index < count && std::getline(file, line); ++index) {
                std::istringstream tensor_fields(line);
                std::string keyword;
                std::string access;
                session_api::TensorContract contract;
                tensor_fields >> keyword >> contract.name >> access;
                if (keyword != "tensor") return fail("expected a tensor line: " + line);
                contract.data_type = session_api::TensorDataType::float16;
                contract.access = access == "read" ?
                    session_api::TensorAccess::read : session_api::TensorAccess::write;
                const auto shape = read_dims(tensor_fields);
                for (std::int32_t axis = 0; axis < 4; ++axis) {
                    const auto size = shape.d[axis];
                    contract.dimensions.push_back(size < 0 ?
                        session_api::DimensionBounds{1, std::numeric_limits<std::int32_t>::max()} :
                        session_api::DimensionBounds{size, size});
                }
                descriptor.tensors.push_back(std::move(contract));
            }
            job.stages.push_back(std::move(descriptor));
        } else if (directive == "run") {
            Run run;
            std::size_t inputs = 0;
            std::size_t outputs = 0;
            fields >> run.file >> inputs >> outputs;
            for (std::size_t index = 0; index < inputs + outputs && std::getline(file, line);
                 ++index) {
                std::istringstream tensor_fields(line);
                std::string keyword;
                tensor_fields >> keyword;
                if (keyword == "input") {
                    Input input;
                    tensor_fields >> input.name;
                    input.shape = read_dims(tensor_fields);
                    tensor_fields >> input.path;
                    run.inputs.push_back(std::move(input));
                } else if (keyword == "output") {
                    std::string name;
                    std::string output_path;
                    tensor_fields >> name >> output_path;
                    run.outputs.emplace_back(std::move(name), std::move(output_path));
                } else {
                    return fail("expected an input or output line: " + line);
                }
            }
            job.runs.push_back(std::move(run));
        } else {
            return fail("unknown directive: " + directive);
        }
    }
    return true;
}

nvcr::Result<DeviceTensor> upload(const Input& input, cudaStream_t stream) {
    std::ifstream file(input.path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    auto allocated = nvcr::tensorrt::allocate_device_tensor(input.name, input.shape);
    if (!allocated) return allocated.error();
    if (bytes.size() != allocated.value().storage.bytes) {
        return nvcr::Error(nvcr::ErrorCode::invalid_argument,
                           "input " + input.path + " does not match its shape", "stage-runner");
    }
    auto copied = nvcr::tensorrt::copy_async(
        allocated.value().storage.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice,
        stream, "stage runner upload");
    if (!copied) return copied.error();
    auto synchronized = nvcr::tensorrt::synchronize_stream(stream, "stage runner upload");
    if (!synchronized) return synchronized.error();
    return std::move(allocated.value());
}

bool download(const DeviceTensor& tensor, cudaStream_t stream, const std::string& path) {
    std::vector<char> bytes(tensor.storage.bytes);
    auto copied = nvcr::tensorrt::copy_async(
        bytes.data(), tensor.storage.data, bytes.size(), cudaMemcpyDeviceToHost, stream,
        "stage runner download");
    if (!copied) return fail(copied.error().describe());
    auto synchronized = nvcr::tensorrt::synchronize_stream(stream, "stage runner download");
    if (!synchronized) return fail(synchronized.error().describe());
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file) || fail("cannot write " + path);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: nvcr_tensorrt_stage_runner JOB_FILE\n";
        return 2;
    }
    Job job;
    if (!parse(argv[1], job)) return 1;

    nvcr::RuntimeConfiguration configuration;
    configuration.artifacts.intra_engine_path = job.bundle;
    configuration.artifacts.model_id = job.model;
    configuration.provider.id = "tensorrt";
    auto created = nvcr::tensorrt::make_provider_session(configuration);
    if (!created) return fail(created.error().describe()), 1;
    auto session =
        std::dynamic_pointer_cast<nvcr::tensorrt::TensorRTExecutionSession>(created.value());
    std::map<std::string, session_api::ExecutableStage> stages;
    for (auto& descriptor : job.stages) {
        const auto file = descriptor.stage_id;
        auto loaded = session->load_stage(std::move(descriptor));
        if (!loaded) return fail(file + ": " + loaded.error().describe()), 1;
        stages.emplace(file, loaded.value());
    }
    auto finalized = session->finalize_stages();
    if (!finalized) return fail(finalized.error().describe()), 1;

    const auto scope = session->allocation_scope();
    for (const auto& run : job.runs) {
        const auto stage = stages.find(run.file);
        if (stage == stages.end()) return fail("run names an unloaded stage: " + run.file), 1;
        std::vector<DeviceTensor> inputs;
        inputs.reserve(run.inputs.size());
        for (const auto& input : run.inputs) {
            auto uploaded = upload(input, session->stream());
            if (!uploaded) return fail(input.name + ": " + uploaded.error().describe()), 1;
            inputs.push_back(std::move(uploaded.value()));
        }
        std::vector<const DeviceTensor*> pointers;
        for (const auto& input : inputs) pointers.push_back(&input);
        auto outputs = session->run_stage(stage->second, pointers);
        if (!outputs) return fail(run.file + ": " + outputs.error().describe()), 1;
        for (const auto& [name, path] : run.outputs) {
            const DeviceTensor* found = nullptr;
            for (const auto& output : outputs.value()) {
                if (output.name == name) found = &output;
            }
            if (found == nullptr) return fail(run.file + " has no output " + name), 1;
            if (!download(*found, session->stream(), path)) return 1;
            std::cout << "shape " << name;
            for (std::int32_t axis = 0; axis < found->shape.nbDims; ++axis) {
                std::cout << ' ' << found->shape.d[axis];
            }
            std::cout << '\n';
        }
    }
    return 0;
}
