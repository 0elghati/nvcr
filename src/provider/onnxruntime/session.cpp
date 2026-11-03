#include "session.hpp"

#include "nvcr/provider/experimental/session.hpp"
#include "nvcr/runtime/registry.hpp"

#include "../../common/sha256.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvcr::provider::onnxruntime {
namespace {

namespace fs = std::filesystem;
namespace session_api = provider::experimental;

constexpr std::string_view subsystem = "provider.onnxruntime";

struct ExecutionTarget final {
    std::string_view provider_id;
    std::string_view runtime_name;  // as reported by Ort::GetAvailableProviders()
    std::string_view target_device;
    std::string_view display_name;
    bool dynamic_shapes;
};

constexpr std::array<ExecutionTarget, 3> execution_targets{{
    {"onnxruntime-cpu", "CPUExecutionProvider", "cpu", "ONNX Runtime CPU", true},
    {"onnxruntime-cuda", "CUDAExecutionProvider", "cuda", "ONNX Runtime CUDA", true},
    // QNN HTP compiles fixed shapes, so artifacts must be exported per profile.
    {"onnxruntime-qnn", "QNNExecutionProvider", "qnn-htp", "ONNX Runtime QNN HTP", false},
}};

#if defined(_WIN32)
constexpr const char* qnn_htp_backend = "QnnHtp.dll";
#else
constexpr const char* qnn_htp_backend = "libQnnHtp.so";
#endif

Error session_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), std::string(subsystem));
}

std::string environment_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

// NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION=disable|basic|extended|all (default all).
GraphOptimizationLevel graph_optimization_level() {
    const auto level = environment_value("NVCR_ONNXRUNTIME_GRAPH_OPTIMIZATION");
    if (level == "disable") return GraphOptimizationLevel::ORT_DISABLE_ALL;
    if (level == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
    if (level == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    return GraphOptimizationLevel::ORT_ENABLE_ALL;
}

Ort::Env& environment() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nvcr");
    return env;
}

std::string next_owner_id() {
    static std::atomic<std::uint64_t> next{0};
    return "onnxruntime-session-" + std::to_string(++next);
}

// A plain file stem: codec bundles name parts such as "i_analysis" or
// "MLVCEncoder_320x192"; separators and dots are never accepted.
bool simple_component_id(std::string_view id) {
    return !id.empty() && std::all_of(id.begin(), id.end(), [](char character) {
        return (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_';
    });
}

std::optional<ONNXTensorElementDataType> onnx_type(session_api::TensorDataType type) {
    switch (type) {
    case session_api::TensorDataType::uint8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case session_api::TensorDataType::int8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case session_api::TensorDataType::int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case session_api::TensorDataType::float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case session_api::TensorDataType::float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    return std::nullopt;
}

std::size_t element_bytes(session_api::TensorDataType type) {
    switch (type) {
    case session_api::TensorDataType::uint8:
    case session_api::TensorDataType::int8: return 1;
    case session_api::TensorDataType::float16: return 2;
    case session_api::TensorDataType::int32:
    case session_api::TensorDataType::float32: return 4;
    }
    return 0;
}

std::string hex(std::span<const std::byte> bytes) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string output;
    output.reserve(bytes.size() * 2);
    for (const auto byte : bytes) {
        const auto value = std::to_integer<unsigned>(byte);
        output.push_back(digits[value >> 4U]);
        output.push_back(digits[value & 0xfU]);
    }
    return output;
}

class OrtBuffer final : public session_api::IProviderBuffer {
public:
    OrtBuffer(std::size_t size, session_api::MemoryDomain domain, std::string owner_id)
        : storage_(std::make_unique<std::byte[]>(size)),
          size_(size),
          domain_(domain),
          owner_id_(std::move(owner_id)) {}

    [[nodiscard]] std::size_t size_bytes() const noexcept override { return size_; }
    [[nodiscard]] session_api::MemoryDomain domain() const noexcept override { return domain_; }
    [[nodiscard]] std::string_view owner_id() const noexcept override { return owner_id_; }
    [[nodiscard]] std::span<std::byte> host_bytes() noexcept override {
        return {storage_.get(), size_};
    }

private:
    std::unique_ptr<std::byte[]> storage_;
    std::size_t size_;
    session_api::MemoryDomain domain_;
    std::string owner_id_;
};

class OrtStage final : public session_api::IExecutableStage {
public:
    OrtStage(
        session_api::ExecutableStageDescriptor descriptor,
        std::string owner_id,
        Ort::Session session)
        : descriptor_(std::move(descriptor)),
          owner_id_(std::move(owner_id)),
          session_(std::move(session)) {
        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t index = 0; index < session_.GetInputCount(); ++index) {
            input_names_.emplace_back(session_.GetInputNameAllocated(index, allocator).get());
            input_types_.push_back(
                session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType());
        }
        for (std::size_t index = 0; index < session_.GetOutputCount(); ++index) {
            output_names_.emplace_back(session_.GetOutputNameAllocated(index, allocator).get());
            output_types_.push_back(
                session_.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType());
        }
    }

    [[nodiscard]] const session_api::ExecutableStageDescriptor& descriptor() const noexcept override {
        return descriptor_;
    }
    [[nodiscard]] std::string_view owner_id() const noexcept override { return owner_id_; }

    // Every floating-point input and output uses the declared precision;
    // integer tensors (for example a quality index) are unaffected.
    [[nodiscard]] bool float_tensors_are(ONNXTensorElementDataType precision) const noexcept {
        const auto matches = [precision](ONNXTensorElementDataType type) {
            const bool floating = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
                type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
            return !floating || type == precision;
        };
        return std::all_of(input_types_.begin(), input_types_.end(), matches) &&
            std::all_of(output_types_.begin(), output_types_.end(), matches);
    }

    [[nodiscard]] std::optional<ONNXTensorElementDataType> type_of(
        std::string_view name, bool output) const noexcept {
        const auto& names = output ? output_names_ : input_names_;
        const auto& types = output ? output_types_ : input_types_;
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (names[index] == name) return types[index];
        }
        return std::nullopt;
    }

    const std::vector<std::string>& input_names() const noexcept { return input_names_; }
    const std::vector<std::string>& output_names() const noexcept { return output_names_; }
    Ort::Session& session() const noexcept { return session_; }

private:
    session_api::ExecutableStageDescriptor descriptor_;
    std::string owner_id_;
    mutable Ort::Session session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<ONNXTensorElementDataType> input_types_;
    std::vector<ONNXTensorElementDataType> output_types_;
};

// Run() returns after the outputs are written, so every completion is ready.
class OrtCompletion final : public session_api::ICompletion {
public:
    explicit OrtCompletion(std::string_view owner_id) : owner_id_(owner_id) {}

    [[nodiscard]] std::string_view owner_id() const noexcept override { return owner_id_; }
    [[nodiscard]] bool ready() const noexcept override { return true; }
    [[nodiscard]] Result<void> wait() override { return {}; }

private:
    std::string owner_id_;
};

class OrtExecutionSession final : public session_api::IProviderSession {
public:
    OrtExecutionSession(const ExecutionTarget& target, std::int32_t device_id)
        : target_(target), device_id_(device_id), owner_id_(next_owner_id()) {}

    [[nodiscard]] std::string_view owner_id() const noexcept override { return owner_id_; }

    [[nodiscard]] Result<session_api::BufferHandle> allocate(
        std::size_t size_bytes, session_api::MemoryDomain domain) override {
        if (size_bytes == 0) {
            return session_error(ErrorCode::invalid_argument, "buffer size must be non-zero");
        }
        if (domain == session_api::MemoryDomain::provider_device) {
            return session_error(
                ErrorCode::not_implemented,
                "ONNX Runtime sessions allocate host buffers only");
        }
        try {
            return session_api::BufferHandle(
                std::make_shared<OrtBuffer>(size_bytes, domain, owner_id_));
        } catch (const std::bad_alloc&) {
            return session_error(ErrorCode::resource_exhausted, "unable to allocate host buffer");
        }
    }

    [[nodiscard]] Result<session_api::ExecutableStage> load_stage(
        session_api::ExecutableStageDescriptor descriptor) override {
        const auto& artifact = descriptor.artifact;
        if (!simple_component_id(artifact.component_id)) {
            return session_error(
                ErrorCode::invalid_argument,
                "stage component id must be a lowercase identifier: " + artifact.component_id);
        }
        const auto path = fs::path(artifact.path) / (artifact.component_id + ".onnx");
        std::error_code error;
        if (!fs::is_regular_file(path, error)) {
            return session_error(
                ErrorCode::missing_artifact, "ONNX stage is missing: " + path.string());
        }
        if (!artifact.expected_digest.empty()) {
            auto digest = detail::sha256_file(path, subsystem);
            if (!digest) return digest.error();
            if (digest.value() != hex(artifact.expected_digest)) {
                return session_error(
                    ErrorCode::digest_mismatch, "ONNX stage digest mismatch: " + path.string());
            }
        }
        try {
            // Execution providers load shared libraries that log through the
            // default logger, which exists only once an Env does.
            (void)environment();
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(graph_optimization_level());
            // Diagnostic switches; unset, the defaults below are unchanged.
            if (const auto prefix = environment_value("NVCR_ONNXRUNTIME_PROFILE"); !prefix.empty()) {
                options.EnableProfiling((prefix + "-" + artifact.component_id).c_str());
            }
            if (environment_value("NVCR_ONNXRUNTIME_DISABLE_CPU_FALLBACK") == "1") {
                options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            }
            if (target_.runtime_name == "CUDAExecutionProvider") {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = device_id_;
                options.AppendExecutionProvider_CUDA(cuda);
            } else if (target_.runtime_name == "QNNExecutionProvider") {
                // Every node must run on the HTP; a silent CPU fallback would
                // misreport where the codec executes.
                options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
                std::unordered_map<std::string, std::string> qnn{{"backend_path", qnn_htp_backend}};
                if (const auto mode = environment_value("NVCR_QNN_HTP_PERFORMANCE_MODE");
                    !mode.empty()) {
                    qnn["htp_performance_mode"] = mode;
                }
                if (const auto mode = environment_value("NVCR_QNN_HTP_FINALIZATION_MODE");
                    !mode.empty()) {
                    qnn["htp_graph_finalization_optimization_mode"] = mode;
                }
                options.AppendExecutionProvider("QNN", qnn);
            }
            const auto precision = declared_precision(descriptor.artifact.precision);
            if (!precision) {
                return session_error(
                    ErrorCode::incompatible_precision,
                    "ONNX stage precision must be fp16 or fp32: " + path.string());
            }
            auto stage = std::make_shared<const OrtStage>(
                std::move(descriptor), owner_id_,
                Ort::Session(environment(), path.c_str(), options));
            if (!stage->float_tensors_are(*precision)) {
                return session_error(
                    ErrorCode::incompatible_precision,
                    std::string("ONNX stage must use ") +
                        (*precision == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? "fp32" : "fp16") +
                        " floating-point inputs and outputs: " + path.string());
            }
            for (const auto& contract : stage->descriptor().tensors) {
                const auto expected = onnx_type(contract.data_type);
                const auto actual = stage->type_of(
                    contract.name, contract.access != session_api::TensorAccess::read);
                if (!expected || actual != expected) {
                    return session_error(
                        ErrorCode::invalid_argument,
                        "ONNX stage tensor " + contract.name +
                            " is missing or has another type: " + path.string());
                }
            }
            return session_api::ExecutableStage(std::move(stage));
        } catch (const Ort::Exception& exception) {
            return session_error(
                ErrorCode::backend_error,
                "ONNX Runtime could not load " + path.string() + ": " + exception.what());
        } catch (const std::bad_alloc&) {
            return session_error(ErrorCode::resource_exhausted, "unable to load ONNX stage");
        }
    }

    [[nodiscard]] Result<session_api::Completion> submit(
        const session_api::ExecutableStage& stage,
        std::span<const session_api::TensorView> tensors,
        std::span<const session_api::ExecutionDependency> dependencies) override {
        const auto concrete = std::dynamic_pointer_cast<const OrtStage>(stage);
        if (!concrete || concrete->owner_id() != owner_id_) {
            return session_error(
                ErrorCode::invalid_argument, "stage belongs to another provider session");
        }
        for (const auto& dependency : dependencies) {
            if (!dependency.completion || dependency.completion->owner_id() != owner_id_) {
                return session_error(
                    ErrorCode::invalid_argument, "dependency belongs to another provider session");
            }
            auto waited = dependency.completion->wait();
            if (!waited) return waited.error();
        }

        try {
            const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::size_t bound = 0;
            const auto bind = [&](const std::vector<std::string>& names, bool writable,
                                  std::vector<Ort::Value>& values) -> Result<void> {
                for (const auto& name : names) {
                    const auto tensor = std::find_if(
                        tensors.begin(), tensors.end(),
                        [&](const session_api::TensorView& view) { return view.name == name; });
                    if (tensor == tensors.end()) {
                        return session_error(
                            ErrorCode::invalid_argument, "missing stage tensor " + name);
                    }
                    const auto expected = concrete->type_of(name, writable);
                    auto checked = validate(*tensor, writable, expected);
                    if (!checked) return checked.error();
                    auto bytes = tensor->storage.buffer->host_bytes().subspan(
                        tensor->storage.offset_bytes, tensor->storage.size_bytes);
                    values.push_back(Ort::Value::CreateTensor(
                        memory, bytes.data(), bytes.size(), tensor->shape.data(),
                        tensor->shape.size(), *expected));
                    ++bound;
                }
                return {};
            };
            std::vector<Ort::Value> inputs;
            std::vector<Ort::Value> outputs;
            auto inputs_bound = bind(concrete->input_names(), false, inputs);
            if (!inputs_bound) return inputs_bound.error();
            auto outputs_bound = bind(concrete->output_names(), true, outputs);
            if (!outputs_bound) return outputs_bound.error();
            if (bound != tensors.size()) {
                return session_error(
                    ErrorCode::invalid_argument, "submission names tensors the stage does not use");
            }

            std::vector<const char*> input_names;
            std::vector<const char*> output_names;
            for (const auto& name : concrete->input_names()) input_names.push_back(name.c_str());
            for (const auto& name : concrete->output_names()) output_names.push_back(name.c_str());
            concrete->session().Run(
                Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), outputs.data(), outputs.size());
            return session_api::Completion(std::make_shared<OrtCompletion>(owner_id_));
        } catch (const Ort::Exception& exception) {
            return session_error(
                ErrorCode::backend_error,
                concrete->descriptor().stage_id + " failed in ONNX Runtime: " + exception.what());
        } catch (const std::bad_alloc&) {
            return session_error(ErrorCode::resource_exhausted, "unable to submit ONNX stage");
        }
    }

    // Execution is synchronous, so no work is outstanding.
    [[nodiscard]] Result<void> reset() override { return {}; }

private:
    [[nodiscard]] static std::optional<ONNXTensorElementDataType> declared_precision(
        std::string_view precision) noexcept {
        // An undeclared precision keeps the provider's original fp16-only rule.
        if (precision.empty() || precision == "fp16") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        if (precision == "fp32") return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        return std::nullopt;
    }

    [[nodiscard]] Result<void> validate(
        const session_api::TensorView& tensor, bool writable,
        std::optional<ONNXTensorElementDataType> model_type) const {
        const bool access_ok = writable ?
            tensor.access != session_api::TensorAccess::read :
            tensor.access != session_api::TensorAccess::write;
        if (!access_ok || !model_type || onnx_type(tensor.data_type) != model_type) {
            return session_error(
                ErrorCode::invalid_argument,
                "tensor " + std::string(tensor.name) + " has the wrong access or data type");
        }
        const auto& buffer = tensor.storage.buffer;
        if (!buffer || buffer->owner_id() != owner_id_) {
            return session_error(
                ErrorCode::invalid_argument,
                "tensor " + std::string(tensor.name) + " uses another session's buffer");
        }
        std::size_t elements = 1;
        for (const auto dimension : tensor.shape) {
            if (dimension <= 0) {
                return session_error(
                    ErrorCode::invalid_argument,
                    "tensor " + std::string(tensor.name) + " has a non-positive dimension");
            }
            elements *= static_cast<std::size_t>(dimension);
        }
        const auto end = tensor.storage.offset_bytes + tensor.storage.size_bytes;
        if (tensor.storage.size_bytes != elements * element_bytes(tensor.data_type) ||
            end < tensor.storage.offset_bytes || end > buffer->size_bytes()) {
            return session_error(
                ErrorCode::invalid_argument,
                "tensor " + std::string(tensor.name) + " does not fit its buffer slice");
        }
        return {};
    }

    const ExecutionTarget& target_;
    std::int32_t device_id_;
    std::string owner_id_;
};

}  // namespace

void register_providers() {
    std::vector<std::string> available;
    std::string version;
    try {
        available = Ort::GetAvailableProviders();
        version = OrtGetApiBase()->GetVersionString();
    } catch (const Ort::Exception&) {
        return;
    }
    for (const auto& target : execution_targets) {
        if (std::find(available.begin(), available.end(), target.runtime_name) ==
            available.end()) {
            continue;
        }
        runtime::Registry::instance().register_provider({
            ProviderDescriptor{
                ProviderKind::onnx_runtime,
                std::string(target.provider_id),
                version,
                std::string(target.display_name),
            },
            ProviderCapabilities{
                .supports_fp16 = true,
                .supports_int8 = false,
                .supports_dynamic_shapes = target.dynamic_shapes,
                .supports_cuda_graphs = false,
                .target_device = std::string(target.target_device),
            },
            {},
            {},
            [&target](const RuntimeConfiguration& configuration)
                -> Result<std::shared_ptr<session_api::IProviderSession>> {
                try {
                    return std::shared_ptr<session_api::IProviderSession>(
                        std::make_shared<OrtExecutionSession>(
                            target, configuration.provider.device_id));
                } catch (const std::bad_alloc&) {
                    return session_error(
                        ErrorCode::resource_exhausted,
                        "unable to allocate ONNX Runtime provider session");
                }
            },
        });
    }
}

}  // namespace nvcr::provider::onnxruntime
