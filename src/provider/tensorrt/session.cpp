#include "session.hpp"

#include "../../common/json.hpp"
#include "../../common/sha256.hpp"
#include "fx/ops.hpp"
#include "fx/plugins.hpp"
#include "nvcr/numerics/profiles.hpp"

#include <NvInferVersion.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace nvcr::tensorrt {

thread_local cudaStream_t allocation_stream = nullptr;
thread_local BackendProfiler* active_profiler = nullptr;
thread_local bool cuda_graphs_allowed = false;
thread_local std::string_view allocation_owner_id;
thread_local std::shared_ptr<CudaStreamOwner> allocation_stream_owner;
thread_local DeviceScratchArena* active_scratch_arena = nullptr;

Result<ContextPolicy> determine_context_policy(const RuntimeConfiguration& configuration) {
    if (configuration.provider.tensorrt_execution_mode ==
        TensorRTExecutionMode::low_memory) {
        return ContextPolicy::per_engine;
    }
    if (configuration.provider.tensorrt_execution_mode ==
        TensorRTExecutionMode::performance) {
        return ContextPolicy::persistent;
    }
    const auto device_id = configuration.provider.device_id;
    if (const char* raw = std::getenv("NVCR_TENSORRT_LOW_MEMORY_MODE"); raw != nullptr) {
        const std::string_view value(raw);
        if (value == "1" || value == "true" || value == "TRUE" ||
            value == "on" || value == "ON") {
            return ContextPolicy::per_engine;
        }
        if (value == "0" || value == "false" || value == "FALSE" ||
            value == "off" || value == "OFF") {
            return ContextPolicy::persistent;
        }
        return Error(
            ErrorCode::invalid_argument,
            "NVCR_TENSORRT_LOW_MEMORY_MODE must be one of 0/1, true/false, or on/off",
            std::string(subsystem));
    }

    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, device_id);
    if (status != cudaSuccess) {
        return Error(
            ErrorCode::backend_error,
            std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(status),
            std::string(subsystem));
    }

    // Discrete GPUs should keep persistent TensorRT contexts by default. A
    // capacity-only cutoff incorrectly classifies nominal 8 GiB devices whose
    // reported usable memory is slightly below 8 GiB, adding context churn and
    // stream waits to every device stage. Explicit low-memory mode remains
    // available for constrained discrete deployments.
    // Integrated/Jetson-class devices keep persistent context metadata while
    // sharing one user-managed activation workspace across engines. All engine
    // invocations use one stream, so their workspace lifetimes do not overlap.
    if (properties.integrated != 0) return ContextPolicy::shared_workspace_persistent;
    return ContextPolicy::persistent;
}

std::string_view context_policy_name(ContextPolicy policy) noexcept {
    switch (policy) {
    case ContextPolicy::per_engine: return "low-memory";
    case ContextPolicy::shared_workspace_persistent: return "shared-workspace";
    case ContextPolicy::persistent: return "performance";
    }
    return "unknown";
}


Result<nvinfer1::IExecutionContext*> acquire_context(EngineInstance& instance) {
    if (!instance.low_memory_mode) {
        if (!instance.context) {
            const auto strategy = instance.shared_context_memory != nullptr ?
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED :
                nvinfer1::ExecutionContextAllocationStrategy::kSTATIC;
            instance.context.reset(instance.engine->createExecutionContext(strategy));
            if (!instance.context) {
                return Error(
                    ErrorCode::backend_error,
                    "failed to create execution context: " + instance.path.string(),
                    std::string(subsystem));
            }
            if (instance.shared_context_memory != nullptr) {
                instance.context->setDeviceMemoryV2(
                    instance.shared_context_memory, instance.shared_context_memory_bytes);
            }
        }
        return instance.context.get();
    }

    // Low-memory mode creates short-lived contexts to minimize steady residency.
    instance.context.reset(instance.engine->createExecutionContext());
    if (!instance.context) {
        return Error(
            ErrorCode::backend_error,
            "failed to create execution context: " + instance.path.string(),
            std::string(subsystem));
    }
    return instance.context.get();
}

Error backend_error(std::string message) {
    return Error(ErrorCode::backend_error, std::move(message), std::string(subsystem));
}

Error cuda_error(std::string_view operation, cudaError_t status) {
    return backend_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}


Result<std::vector<std::byte>> read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return Error(
            ErrorCode::io_error,
            "cannot open file: " + path.string(),
            std::string(subsystem));
    }
    const auto end = input.tellg();
    if (end <= 0) {
        return Error(
            ErrorCode::io_error,
            "file is empty: " + path.string(),
            std::string(subsystem));
    }
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::byte> bytes(size);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!input) {
        return Error(
            ErrorCode::io_error,
            "failed to read file: " + path.string(),
            std::string(subsystem));
    }
    return bytes;
}

Result<std::string> read_text_file(const fs::path& path) {
    auto bytes = read_binary(path);
    if (!bytes) return bytes.error();
    std::string text;
    text.reserve(bytes.value().size());
    for (const auto byte : bytes.value()) {
        text.push_back(static_cast<char>(std::to_integer<unsigned char>(byte)));
    }
    return text;
}

Result<std::size_t> find_json_value(std::string_view text, std::string_view key) {
    const auto key_position = text.find('"' + std::string(key) + '"');
    if (key_position == std::string_view::npos) {
        return backend_error("engine_manifest.json is missing key '" + std::string(key) + "'");
    }
    const auto colon = text.find(':', key_position + key.size() + 2);
    if (colon == std::string_view::npos) {
        return backend_error("engine_manifest.json has invalid key '" + std::string(key) + "'");
    }
    auto value = colon + 1;
    while (value < text.size() &&
           std::isspace(static_cast<unsigned char>(text[value])) != 0) {
        ++value;
    }
    if (value == text.size()) {
        return backend_error("engine_manifest.json has empty key '" + std::string(key) + "'");
    }
    return value;
}

Result<std::string> read_json_string(std::string_view text, std::string_view key) {
    auto value = find_json_value(text, key);
    if (!value) return value.error();
    if (text[value.value()] != '"') {
        return backend_error("engine_manifest.json key '" + std::string(key) + "' is not a string");
    }
    std::string output;
    for (auto index = value.value() + 1; index < text.size(); ++index) {
        const char current = text[index];
        if (current == '"') return output;
        if (current == '\\') {
            return backend_error(
                "engine_manifest.json key '" + std::string(key) + "' must not use escapes");
        }
        output.push_back(current);
    }
    return backend_error("engine_manifest.json has unterminated key '" + std::string(key) + "'");
}

Result<std::optional<std::string>> read_optional_json_string(
    std::string_view text, std::string_view key) {
    if (text.find('"' + std::string(key) + '"') == std::string_view::npos) {
        return std::optional<std::string>{};
    }
    auto value = read_json_string(text, key);
    if (!value) return value.error();
    return std::optional<std::string>(std::move(value.value()));
}

Result<std::int64_t> read_json_integer(std::string_view text, std::string_view key) {
    auto value = find_json_value(text, key);
    if (!value) return value.error();
    auto index = value.value();
    bool negative = false;
    if (text[index] == '-') {
        negative = true;
        ++index;
    }
    if (index == text.size() || std::isdigit(static_cast<unsigned char>(text[index])) == 0) {
        return backend_error("engine_manifest.json key '" + std::string(key) + "' is not an integer");
    }
    std::int64_t output = 0;
    while (index < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[index])) != 0) {
        output = output * 10 + static_cast<std::int64_t>(text[index] - '0');
        ++index;
    }
    return negative ? -output : output;
}

struct EngineManifest final {
    std::string schema;
    std::string kind;
    std::string model_profile_id;
    std::string target_profile_id;
    std::string engine_profile_id;
    std::string model_profile_sha256;
    std::string engine_profile_sha256;
    std::string target_profile_sha256;
    std::string precision;
    std::string optimization_point;
    std::string shape_profile{"dynamic"};
    std::string hardware_compatibility{"exact"};
    std::string device_name;
    std::string checksum_manifest;
    std::string checksum_manifest_sha256;
    // Every engine.sha256 entry, filled once each file's digest is verified.
    std::map<std::string, std::string> files;
    std::int32_t cuda_runtime_version{};
    std::int32_t tensorrt_major{};
    std::int32_t tensorrt_minor{};
    std::int32_t tensorrt_patch{};
    std::int32_t compute_major{};
    std::int32_t compute_minor{};
    std::int32_t multiprocessor_count{};
};

Result<std::int32_t> checked_manifest_i32(
    std::string_view text, std::string_view key, std::int32_t minimum = 0) {
    auto value = read_json_integer(text, key);
    if (!value) return value.error();
    if (value.value() < minimum || value.value() > std::numeric_limits<std::int32_t>::max()) {
        return backend_error("engine_manifest.json key '" + std::string(key) + "' is out of range");
    }
    return static_cast<std::int32_t>(value.value());
}

Result<EngineManifest> load_engine_manifest(const fs::path& root) {
    const auto path = root / "engine_manifest.json";
    if (!fs::is_regular_file(path)) {
        return backend_error(
            "missing engine_manifest.json in " + root.string() +
            "; rebuild TensorRT plans on the target device with scripts/nvcr_artifacts.py build");
    }
    std::error_code filesystem_error;
    if (fs::file_size(path, filesystem_error) > 4U * 1024U * 1024U || filesystem_error) {
        return backend_error("engine_manifest.json is unavailable or exceeds 4 MiB");
    }
    auto text = read_text_file(path);
    if (!text) return text.error();
    auto format = checked_manifest_i32(text.value(), "format", 2);
    if (!format) return format.error();
    if (format.value() != 2) {
        return backend_error("unsupported engine_manifest.json format: " +
                             std::to_string(format.value()));
    }
    EngineManifest manifest;
    auto schema = read_json_string(text.value(), "schema");
    auto kind = read_json_string(text.value(), "kind");
    auto model_profile_id = read_json_string(text.value(), "model_profile_id");
    auto target_profile_id = read_json_string(text.value(), "target_profile_id");
    auto engine_profile_id = read_json_string(text.value(), "engine_profile_id");
    auto model_profile_sha256 = read_json_string(text.value(), "model_profile_sha256");
    auto engine_profile_sha256 = read_json_string(text.value(), "engine_profile_sha256");
    auto target_profile_sha256 = read_json_string(text.value(), "target_profile_sha256");
    auto precision = read_json_string(text.value(), "precision");
    auto profile = read_json_string(text.value(), "optimization_point");
    auto shape_profile = read_optional_json_string(text.value(), "shape_profile");
    auto hardware_compatibility =
        read_optional_json_string(text.value(), "hardware_compatibility");
    auto device_name = read_json_string(text.value(), "device_name");
    auto checksum_manifest = read_json_string(text.value(), "checksum_manifest");
    auto checksum_manifest_sha256 =
        read_json_string(text.value(), "checksum_manifest_sha256");
    auto cuda_version = checked_manifest_i32(text.value(), "cuda_runtime_version", 1);
    auto trt_major = checked_manifest_i32(text.value(), "tensorrt_version_major");
    auto trt_minor = checked_manifest_i32(text.value(), "tensorrt_version_minor");
    auto trt_patch = checked_manifest_i32(text.value(), "tensorrt_version_patch");
    auto cc_major = checked_manifest_i32(text.value(), "compute_capability_major");
    auto cc_minor = checked_manifest_i32(text.value(), "compute_capability_minor");
    auto sm_count = checked_manifest_i32(text.value(), "multiprocessor_count", 1);
    if (!schema) return schema.error();
    if (!kind) return kind.error();
    if (!model_profile_id) return model_profile_id.error();
    if (!target_profile_id) return target_profile_id.error();
    if (!engine_profile_id) return engine_profile_id.error();
    if (!model_profile_sha256) return model_profile_sha256.error();
    if (!engine_profile_sha256) return engine_profile_sha256.error();
    if (!target_profile_sha256) return target_profile_sha256.error();
    if (!precision) return precision.error();
    if (!profile) return profile.error();
    if (!shape_profile) return shape_profile.error();
    if (!hardware_compatibility) return hardware_compatibility.error();
    if (!device_name) return device_name.error();
    if (!checksum_manifest) return checksum_manifest.error();
    if (!checksum_manifest_sha256) return checksum_manifest_sha256.error();
    if (!cuda_version) return cuda_version.error();
    if (!trt_major) return trt_major.error();
    if (!trt_minor) return trt_minor.error();
    if (!trt_patch) return trt_patch.error();
    if (!cc_major) return cc_major.error();
    if (!cc_minor) return cc_minor.error();
    if (!sm_count) return sm_count.error();
    manifest.schema = std::move(schema.value());
    manifest.kind = std::move(kind.value());
    manifest.model_profile_id = std::move(model_profile_id.value());
    manifest.target_profile_id = std::move(target_profile_id.value());
    manifest.engine_profile_id = std::move(engine_profile_id.value());
    manifest.model_profile_sha256 = std::move(model_profile_sha256.value());
    manifest.engine_profile_sha256 = std::move(engine_profile_sha256.value());
    manifest.target_profile_sha256 = std::move(target_profile_sha256.value());
    manifest.precision = std::move(precision.value());
    manifest.optimization_point = std::move(profile.value());
    if (shape_profile.value().has_value()) {
        manifest.shape_profile = std::move(shape_profile.value().value());
    }
    if (hardware_compatibility.value().has_value()) {
        manifest.hardware_compatibility =
            std::move(hardware_compatibility.value().value());
    }
    manifest.device_name = std::move(device_name.value());
    manifest.checksum_manifest = std::move(checksum_manifest.value());
    manifest.checksum_manifest_sha256 = std::move(checksum_manifest_sha256.value());
    manifest.cuda_runtime_version = cuda_version.value();
    manifest.tensorrt_major = trt_major.value();
    manifest.tensorrt_minor = trt_minor.value();
    manifest.tensorrt_patch = trt_patch.value();
    manifest.compute_major = cc_major.value();
    manifest.compute_minor = cc_minor.value();
    manifest.multiprocessor_count = sm_count.value();
    return manifest;
}

bool valid_profile_id(std::string_view value) {
    if (value.empty() || value.size() > 128U) return false;
    return std::ranges::all_of(value, [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '.' ||
            character == '_' || character == '-';
    });
}

bool valid_sha256(std::string_view value) {
    if (value.size() != 64U) return false;
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

std::int32_t cuda_runtime_major(std::int32_t version) {
    return version / 1000;
}

bool cuda_runtime_compatible(
    std::int32_t engine_runtime,
    std::int32_t active_runtime,
    const cudaDeviceProp& properties) {
    if (properties.integrated != 0) {
        return engine_runtime == active_runtime;
    }
    return cuda_runtime_major(engine_runtime) == cuda_runtime_major(active_runtime) &&
        engine_runtime <= active_runtime;
}

Result<std::map<std::string, std::string>> validate_bundle_hashes(
    const fs::path& root, const EngineManifest& manifest) {
    if (manifest.checksum_manifest != "engine.sha256") {
        return backend_error("engine manifest must reference portable engine.sha256");
    }
    const auto checksum_path = root / manifest.checksum_manifest;
    std::error_code filesystem_error;
    if (!fs::is_regular_file(checksum_path, filesystem_error) ||
        fs::file_size(checksum_path, filesystem_error) > 64U * 1024U || filesystem_error) {
        return backend_error("engine.sha256 is missing, unavailable, or oversized");
    }
    auto checksum_hash = detail::sha256_file(checksum_path, subsystem);
    if (!checksum_hash) return checksum_hash.error();
    if (!valid_sha256(manifest.checksum_manifest_sha256) ||
        checksum_hash.value() != manifest.checksum_manifest_sha256) {
        return backend_error("engine.sha256 digest does not match engine_manifest.json");
    }
    auto text = read_text_file(checksum_path);
    if (!text) return text.error();
    std::map<std::string, std::string> checksums;
    std::istringstream input(text.value());
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string digest;
        std::string name;
        std::string extra;
        if (!(fields >> digest >> name) || (fields >> extra) || !valid_sha256(digest) ||
            name.empty() || name == "." || name == ".." || name.find('/') != std::string::npos ||
            name.find('\\') != std::string::npos || !checksums.emplace(name, digest).second) {
            return backend_error("engine.sha256 contains an invalid or duplicate entry");
        }
    }
    if (checksums.empty()) return backend_error("engine.sha256 lists no files");
    for (const auto& [name, digest] : checksums) {
        auto actual = detail::sha256_file(root / name, subsystem);
        if (!actual) return actual.error();
        if (actual.value() != digest) {
            return backend_error("engine bundle SHA-256 mismatch: " + name);
        }
    }
    return checksums;
}

Result<EngineManifest> validate_engine_manifest(
    const fs::path& root,
    std::int32_t device_id,
    std::string_view expected_model_profile) {
    auto manifest = load_engine_manifest(root);
    if (!manifest) return manifest.error();
    if (manifest.value().schema != "nvcr.engine-bundle.v2" ||
        manifest.value().kind != "nvcr-tensorrt-engine-bundle") {
        return backend_error("engine_manifest.json has an unsupported schema or kind");
    }
    if (manifest.value().model_profile_id != expected_model_profile) {
        return backend_error(
            "TensorRT engine bundle model '" + manifest.value().model_profile_id +
            "' does not match configured model '" + std::string(expected_model_profile) + "'");
    }
    if (manifest.value().precision != "fp16") {
        return backend_error(
            "engine_manifest.json precision is not supported by v1: " +
            manifest.value().precision);
    }
    if (manifest.value().shape_profile != "dynamic" &&
        manifest.value().shape_profile != "fixed") {
        return backend_error("engine_manifest.json has an invalid shape_profile");
    }
    if (manifest.value().hardware_compatibility != "exact" &&
        manifest.value().hardware_compatibility != "same_compute_capability" &&
        manifest.value().hardware_compatibility != "ampere_plus") {
        return backend_error("engine_manifest.json has an invalid hardware_compatibility");
    }
    if (!valid_profile_id(manifest.value().model_profile_id) ||
        !valid_profile_id(manifest.value().target_profile_id) ||
        !valid_profile_id(manifest.value().engine_profile_id) ||
        (manifest.value().engine_profile_id != manifest.value().optimization_point &&
         manifest.value().engine_profile_id != manifest.value().optimization_point + "-fp16") ||
        !valid_sha256(manifest.value().model_profile_sha256) ||
        !valid_sha256(manifest.value().engine_profile_sha256) ||
        !valid_sha256(manifest.value().target_profile_sha256)) {
        return backend_error(
            "engine_manifest.json has invalid model/target/engine profile identity or digest");
    }
    auto hashes = validate_bundle_hashes(root, manifest.value());
    if (!hashes) return hashes.error();
    manifest.value().files = std::move(hashes.value());
    int cuda_runtime_version = 0;
    const auto cuda_version_status = cudaRuntimeGetVersion(&cuda_runtime_version);
    if (cuda_version_status != cudaSuccess) {
        return cuda_error("cudaRuntimeGetVersion", cuda_version_status);
    }
    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, device_id);
    if (status != cudaSuccess) return cuda_error("cudaGetDeviceProperties", status);
    if (!cuda_runtime_compatible(
            manifest.value().cuda_runtime_version, cuda_runtime_version, properties)) {
        return backend_error(
            "TensorRT engine bundle CUDA runtime " +
            std::to_string(manifest.value().cuda_runtime_version) +
            " is not compatible with active CUDA runtime " +
            std::to_string(cuda_runtime_version));
    }
    if (manifest.value().tensorrt_major != NV_TENSORRT_MAJOR ||
        manifest.value().tensorrt_minor != NV_TENSORRT_MINOR ||
        manifest.value().tensorrt_patch != NV_TENSORRT_PATCH) {
        return backend_error(
            "TensorRT engine bundle was built with TensorRT " +
            std::to_string(manifest.value().tensorrt_major) + "." +
            std::to_string(manifest.value().tensorrt_minor) + "." +
            std::to_string(manifest.value().tensorrt_patch) +
            " but runtime is TensorRT " + std::to_string(NV_TENSORRT_MAJOR) + "." +
            std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH) +
            "; rebuild the engine directory on this runtime");
    }

    const bool exact_device =
        manifest.value().device_name == properties.name &&
        manifest.value().compute_major == properties.major &&
        manifest.value().compute_minor == properties.minor &&
        manifest.value().multiprocessor_count == properties.multiProcessorCount;
    const bool same_compute_capability =
        !properties.integrated &&
        manifest.value().compute_major == properties.major &&
        manifest.value().compute_minor == properties.minor;
    const bool ampere_plus = !properties.integrated && properties.major >= 8;
    const bool compatible_device =
        (manifest.value().hardware_compatibility == "exact" && exact_device) ||
        (manifest.value().hardware_compatibility == "same_compute_capability" &&
         same_compute_capability) ||
        (manifest.value().hardware_compatibility == "ampere_plus" && ampere_plus);
    if (!compatible_device) {
        return backend_error(
            "TensorRT engine bundle was built for " + manifest.value().device_name +
            " (SM " + std::to_string(manifest.value().compute_major) + "." +
            std::to_string(manifest.value().compute_minor) + ", " +
            std::to_string(manifest.value().multiprocessor_count) +
            " multiprocessors) but selected device " + std::to_string(device_id) +
            " is " + properties.name + " (SM " + std::to_string(properties.major) + "." +
            std::to_string(properties.minor) + ", " +
            std::to_string(properties.multiProcessorCount) +
            " multiprocessors); hardware compatibility class is " +
            manifest.value().hardware_compatibility +
            "; rebuild the engine directory on the selected device");
    }
    return std::move(manifest.value());
}


std::string dimensions_string(const nvinfer1::Dims& dimensions) {
    std::string output;
    for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (!output.empty()) output += 'x';
        output += std::to_string(dimensions.d[index]);
    }
    return output;
}

std::optional<nvinfer1::DataType> tensorrt_data_type(session_api::TensorDataType type) {
    switch (type) {
    case session_api::TensorDataType::int32: return nvinfer1::DataType::kINT32;
    case session_api::TensorDataType::float16: return nvinfer1::DataType::kHALF;
    case session_api::TensorDataType::float32: return nvinfer1::DataType::kFLOAT;
    case session_api::TensorDataType::uint8: return nvinfer1::DataType::kUINT8;
    default: return std::nullopt;
    }
}

std::size_t tensor_element_bytes(session_api::TensorDataType type) {
    switch (type) {
    case session_api::TensorDataType::float16: return sizeof(__half);
    case session_api::TensorDataType::int32:
    case session_api::TensorDataType::float32: return sizeof(std::uint32_t);
    case session_api::TensorDataType::uint8: return 1;
    default: return 0;
    }
}

Result<void> validate_engine(
    const nvinfer1::ICudaEngine& engine,
    const TensorRTStageSpec& specification,
    bool fixed_shape_profile) {
    if (engine.getNbIOTensors() != static_cast<std::int32_t>(specification.tensors.size())) {
        return backend_error(
            std::string(specification.filename) + " has " +
            std::to_string(engine.getNbIOTensors()) + " I/O tensors; expected " +
            std::to_string(specification.tensors.size()));
    }

    for (const auto& tensor : specification.tensors) {
        const std::string name(tensor.name);
        if (engine.getTensorIOMode(name.c_str()) != tensor.mode) {
            return backend_error(
                std::string(specification.filename) + " has missing or incorrect I/O mode for '" +
                name + "'");
        }
        const auto expected_type = tensorrt_data_type(tensor.data_type);
        if (!expected_type || engine.getTensorDataType(name.c_str()) != *expected_type) {
            return backend_error(
                std::string(specification.filename) + " tensor '" + name +
                "' has an incompatible data type");
        }
        const auto actual = engine.getTensorShape(name.c_str());
        if (actual.nbDims != static_cast<std::int32_t>(tensor.rank)) {
            return backend_error(
                std::string(specification.filename) + " tensor '" + name +
                "' has rank " + std::to_string(actual.nbDims) + "; expected rank " +
                std::to_string(tensor.rank));
        }
        for (std::size_t index = 0; index < tensor.rank; ++index) {
            const auto expected = tensor.dimensions[index];
            const auto observed = actual.d[index];
            // Preserve the existing dynamic contract exactly. A bundle that
            // opts into fixed shapes must instead resolve every variable axis
            // to a positive concrete dimension. This rejects partial/hybrid
            // bundles rather than allowing them to masquerade as fixed.
            const bool matches = expected == -1
                ? (fixed_shape_profile ? observed > 0 : observed == -1)
                : observed == expected;
            if (!matches) {
                return backend_error(
                    std::string(specification.filename) + " tensor '" + name +
                    "' has shape " + dimensions_string(actual) + "; contract mismatch");
            }
        }
    }
    return {};
}

Result<EngineInstance> load_engine(
    nvinfer1::IRuntime& runtime,
    const fs::path& root,
    const TensorRTStageSpec& specification,
    bool fixed_shape_profile) {
    const auto path = root / specification.filename;
    auto bytes = read_binary(path);
    if (!bytes) return bytes.error();

    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime.deserializeCudaEngine(bytes.value().data(), bytes.value().size()));
    if (!engine) {
        return backend_error("failed to deserialize file: " + path.string());
    }
    auto valid = validate_engine(*engine, specification, fixed_shape_profile);
    if (!valid) return valid.error();

    return EngineInstance{path, std::move(engine), nullptr, {}};
}


Result<CudaAllocation> allocate_cuda(std::size_t bytes) {
    void* pointer = nullptr;
    const auto stream = allocation_stream;
    if (stream != nullptr) {
        auto status = cudaMallocAsync(&pointer, bytes, stream);
        if (status == cudaErrorMemoryAllocation) {
            // Give stream-ordered frees a chance to retire before retrying.
            const auto synchronized = cudaStreamSynchronize(stream);
            if (synchronized == cudaSuccess) {
                status = cudaMallocAsync(&pointer, bytes, stream);
            }
            if (status == cudaErrorMemoryAllocation) {
                // Fallback for devices where async-pool growth/fragmentation is constrained.
                status = cudaMalloc(&pointer, bytes);
                if (status == cudaSuccess) {
                    if (active_profiler != nullptr) active_profiler->record_allocation(bytes);
                    return CudaAllocation(pointer, bytes, nullptr);
                }
                return cuda_error("cudaMalloc (fallback after cudaMallocAsync)", status);
            }
        }
        if (status != cudaSuccess) return cuda_error("cudaMallocAsync", status);
    } else {
        const auto status = cudaMalloc(&pointer, bytes);
        if (status != cudaSuccess) return cuda_error("cudaMalloc", status);
    }
    if (active_profiler != nullptr) active_profiler->record_allocation(bytes);
    return CudaAllocation(pointer, bytes, stream);
}

Result<CudaAllocation> allocate_cuda_scratch(std::size_t bytes) {
    if (active_scratch_arena != nullptr && active_scratch_arena->initialized()) {
        auto allocation = active_scratch_arena->allocate(bytes);
        if (allocation) return allocation;
    }
    return allocate_cuda(bytes);
}

Result<void> copy_async(
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    std::string_view operation) {
    const auto copied = cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (copied != cudaSuccess) return cuda_error(operation, copied);
    if (active_profiler != nullptr) active_profiler->record_transfer(kind, bytes);
    return {};
}

Result<void> synchronize_stream(cudaStream_t stream, std::string_view operation) {
    if (active_profiler != nullptr) active_profiler->record_synchronization();
    const auto synchronized = cudaStreamSynchronize(stream);
    if (synchronized != cudaSuccess) return cuda_error(operation, synchronized);
    return {};
}

Result<void> synchronize_event(cudaEvent_t event, std::string_view operation) {
    if (active_profiler != nullptr) active_profiler->record_synchronization();
    const auto synchronized = cudaEventSynchronize(event);
    if (synchronized != cudaSuccess) return cuda_error(operation, synchronized);
    return {};
}

Result<void> record_cuda_event(cudaEvent_t event, cudaStream_t stream, std::string_view operation) {
    const auto status = cudaEventRecord(event, stream);
    if (status != cudaSuccess) return cuda_error(operation, status);
    return {};
}

Result<std::pair<cudaEvent_t, cudaEvent_t>> create_profile_events(std::string_view operation) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    auto status = cudaEventCreate(&start);
    if (status != cudaSuccess) return cuda_error(operation, status);
    status = cudaEventCreate(&stop);
    if (status != cudaSuccess) {
        static_cast<void>(cudaEventDestroy(start));
        return cuda_error(operation, status);
    }
    return std::pair<cudaEvent_t, cudaEvent_t>{start, stop};
}

Result<std::size_t> tensor_bytes(
    const nvinfer1::Dims& shape, session_api::TensorDataType data_type) {
    std::size_t elements = 1;
    for (std::int32_t index = 0; index < shape.nbDims; ++index) {
        if (shape.d[index] <= 0) return backend_error("TensorRT produced an unresolved shape");
        const auto dimension = static_cast<std::size_t>(shape.d[index]);
        if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
            return backend_error("TensorRT tensor size overflow");
        }
        elements *= dimension;
    }
    const auto element_bytes = tensor_element_bytes(data_type);
    if (element_bytes == 0U) return backend_error("TensorRT tensor data type is unsupported");
    if (elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
        return backend_error("TensorRT tensor byte size overflow");
    }
    return elements * element_bytes;
}

Result<void> warm_up_engine(
    EngineInstance& instance, const TensorRTStageSpec& specification, cudaStream_t stream) {
    auto context_result = acquire_context(instance);
    if (!context_result) return context_result.error();
    auto* context = context_result.value();
    for (const auto& tensor : specification.tensors) {
        if (tensor.mode != nvinfer1::TensorIOMode::kINPUT) continue;
        const std::string name(tensor.name);
        auto shape = instance.engine->getTensorShape(name.c_str());
        const auto profile_minimum = instance.engine->getProfileShape(
            name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
        for (std::int32_t axis = 0; axis < shape.nbDims; ++axis) {
            if (shape.d[axis] != -1) continue;
            const auto chosen = tensor.warmup_dimensions[static_cast<std::size_t>(axis)];
            shape.d[axis] = chosen > 0 ? chosen : profile_minimum.d[axis];
        }
        if (!context->setInputShape(name.c_str(), shape)) {
            return backend_error(
                std::string(specification.filename) + " rejected warm-up shape for " + name);
        }
    }

    std::vector<CudaAllocation> allocations;
    allocations.reserve(specification.tensors.size());
    for (const auto& tensor : specification.tensors) {
        const std::string name(tensor.name);
        const auto shape = context->getTensorShape(name.c_str());
        auto bytes = tensor_bytes(shape, tensor.data_type);
        if (!bytes) return bytes.error();
        auto allocation = allocate_cuda(bytes.value());
        if (!allocation) return allocation.error();
        const auto cleared = cudaMemsetAsync(
            allocation.value().data, 0, allocation.value().bytes, stream);
        if (cleared != cudaSuccess) return cuda_error("cudaMemsetAsync", cleared);
        if (!context->setTensorAddress(name.c_str(), allocation.value().data)) {
            return backend_error(
                std::string(specification.filename) + " rejected address for " + name);
        }
        allocations.push_back(std::move(allocation.value()));
    }
    if (!context->enqueueV3(stream)) {
        return backend_error(std::string(specification.filename) + " enqueueV3 failed");
    }
    const auto synchronized = cudaStreamSynchronize(stream);
    if (synchronized != cudaSuccess) {
        return cuda_error(
            std::string("cudaStreamSynchronize warm-up for ") +
                std::string(specification.filename),
            synchronized);
    }
    if (instance.low_memory_mode) instance.context.reset();
    return {};
}


const DeviceTensor* find_device_tensor(
    std::span<const DeviceTensor* const> tensors, std::string_view name) {
    for (const auto* tensor : tensors) {
        if (tensor != nullptr && tensor->name == name) return tensor;
    }
    return nullptr;
}

Result<DeviceTensor> allocate_device_tensor(
    std::string name, nvinfer1::Dims shape, session_api::TensorDataType data_type) {
    auto bytes = tensor_bytes(shape, data_type);
    if (!bytes) return bytes.error();
    auto storage = allocate_cuda(bytes.value());
    if (!storage) return storage.error();
    return DeviceTensor{
        std::move(name), shape,
        DeviceTensor::Storage(std::move(storage.value())), data_type};
}

Result<DeviceTensor> allocate_device_tensor_scratch(
    std::string name, nvinfer1::Dims shape, session_api::TensorDataType data_type) {
    auto bytes = tensor_bytes(shape, data_type);
    if (!bytes) return bytes.error();
    auto storage = allocate_cuda_scratch(bytes.value());
    if (!storage) return storage.error();
    return DeviceTensor{
        std::move(name), shape,
        DeviceTensor::Storage(std::move(storage.value())), data_type};
}


Result<void> enqueue_engine(
    EngineInstance& instance, nvinfer1::IExecutionContext& context,
    cudaStream_t stream, std::vector<std::uintptr_t> signature) {
    if (!instance.use_cuda_graphs || !cuda_graphs_allowed) {
        if (!context.enqueueV3(stream)) {
            return backend_error(instance.path.filename().string() + " enqueueV3 failed");
        }
        return {};
    }

    for (const auto& invocation : instance.graph_invocations) {
        if (invocation.signature != signature) continue;
        const auto launched = cudaGraphLaunch(invocation.executable, stream);
        if (launched != cudaSuccess) return cuda_error("cudaGraphLaunch TensorRT engine", launched);
        ++instance.graph_hits;
        return {};
    }

    // Bound address/shape variants prevent an unbounded graph cache if an
    // application alternates many buffers or resolutions in one session.
    if (instance.graph_invocations.size() >=
        max_cuda_graph_invocations_per_engine) {
        if (!context.enqueueV3(stream)) {
            return backend_error(instance.path.filename().string() + " enqueueV3 failed");
        }
        return {};
    }

    auto capture_status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (capture_status != cudaSuccess) {
        return cuda_error("cudaStreamBeginCapture TensorRT engine", capture_status);
    }
    if (!context.enqueueV3(stream)) {
        cudaGraph_t discarded{};
        static_cast<void>(cudaStreamEndCapture(stream, &discarded));
        if (discarded != nullptr) static_cast<void>(cudaGraphDestroy(discarded));
        return backend_error(instance.path.filename().string() + " captured enqueueV3 failed");
    }
    cudaGraph_t graph{};
    capture_status = cudaStreamEndCapture(stream, &graph);
    if (capture_status != cudaSuccess) {
        if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
        return cuda_error("cudaStreamEndCapture TensorRT engine", capture_status);
    }
    cudaGraphExec_t executable{};
    const auto instantiated = cudaGraphInstantiate(&executable, graph, 0);
    if (instantiated != cudaSuccess) {
        static_cast<void>(cudaGraphDestroy(graph));
        return cuda_error("cudaGraphInstantiate TensorRT engine", instantiated);
    }
    CudaGraphInvocation invocation;
    invocation.signature = std::move(signature);
    invocation.graph = graph;
    invocation.executable = executable;
    const auto launched = cudaGraphLaunch(invocation.executable, stream);
    if (launched != cudaSuccess) return cuda_error("cudaGraphLaunch captured TensorRT engine", launched);
    instance.graph_invocations.push_back(std::move(invocation));
    ++instance.graph_captures;
    return {};
}

Result<std::vector<DeviceTensor>> TensorRTExecutionSession::run_stage(
    std::size_t engine_index,
    std::span<const DeviceTensor* const> inputs,
    EngineOutputStorage output_storage) {
    if (!initialized_ || engine_index >= engines_.size() || engine_index >= stages_.size()) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session stage is not loaded",
            std::string(subsystem));
    }
    const auto setup_start = std::chrono::steady_clock::now();
    auto& instance = engines_[engine_index];
    const auto& specification = stage_specs_[engine_index];
    auto context_result = acquire_context(instance);
    if (!context_result) return context_result.error();
    auto* context = context_result.value();
    for (const auto& tensor : specification.tensors) {
        if (tensor.mode != nvinfer1::TensorIOMode::kINPUT) continue;
        const auto* input = find_device_tensor(inputs, tensor.name);
        if (input == nullptr) {
            return backend_error(std::string(specification.filename) +
                                 " missing device input " + std::string(tensor.name));
        }
        const std::string name(tensor.name);
        if (!context->setInputShape(name.c_str(), input->shape)) {
            return backend_error(std::string(specification.filename) +
                                 " rejected device input shape for " + name);
        }
    }

    std::vector<DeviceTensor> outputs;
    outputs.reserve(specification.tensors.size());
    for (const auto& tensor : specification.tensors) {
        const std::string name(tensor.name);
        if (tensor.mode == nvinfer1::TensorIOMode::kINPUT) {
            continue;
        }
        const auto shape = context->getTensorShape(name.c_str());
        auto output = output_storage == EngineOutputStorage::scratch ?
            allocate_device_tensor_scratch(name, shape, tensor.data_type) :
            allocate_device_tensor(name, shape, tensor.data_type);
        if (!output) return output.error();
        outputs.push_back(std::move(output.value()));
    }

    std::vector<session_api::TensorView> tensors;
    tensors.reserve(specification.tensors.size());
    for (const auto& tensor : specification.tensors) {
        const DeviceTensor* device_tensor = nullptr;
        const auto access = tensor.mode == nvinfer1::TensorIOMode::kINPUT ?
            session_api::TensorAccess::read : session_api::TensorAccess::write;
        if (tensor.mode == nvinfer1::TensorIOMode::kINPUT) {
            device_tensor = find_device_tensor(inputs, tensor.name);
        } else {
            for (const auto& output : outputs) {
                if (output.name == tensor.name) {
                    device_tensor = &output;
                    break;
                }
            }
        }
        if (device_tensor == nullptr) {
            return backend_error(
                std::string(specification.filename) + " missing device tensor " +
                std::string(tensor.name));
        }
        std::vector<std::int64_t> shape;
        shape.reserve(static_cast<std::size_t>(device_tensor->shape.nbDims));
        for (std::int32_t index = 0; index < device_tensor->shape.nbDims; ++index) {
            shape.push_back(device_tensor->shape.d[index]);
        }
        tensors.push_back(session_api::TensorView{
            tensor.name,
            device_tensor->data_type,
            std::move(shape),
            access,
            session_api::BufferSlice{
                device_tensor->storage.buffer,
                0,
                device_tensor->storage.bytes}});
    }

    auto completion = submit_impl(
        stages_[engine_index], tensors, {}, context, setup_start);
    if (!completion) return completion.error();
    return outputs;
}

Result<std::vector<DeviceTensor>> TensorRTExecutionSession::run_stage(
    const session_api::ExecutableStage& stage,
    std::span<const DeviceTensor* const> inputs,
    EngineOutputStorage output_storage) {
    if (const auto exact = std::dynamic_pointer_cast<const TensorRTExactStage>(stage)) {
        if (exact->owner_id() != owner_id_) {
            return Error(ErrorCode::invalid_argument, "TensorRT stage belongs to another provider session",
                         std::string(subsystem));
        }
        return run_exact_stage(*exact, inputs, output_storage);
    }
    const auto concrete = std::dynamic_pointer_cast<const TensorRTStage>(stage);
    if (!concrete || concrete->owner_id() != owner_id_) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT stage belongs to another provider session",
            std::string(subsystem));
    }
    auto outputs = run_stage(concrete->engine_index(), inputs, output_storage);
    if (!outputs || concrete->descriptor().numerics_profile != numerics::fx_v1) return outputs;
    auto checked = check_fx_failure(concrete->descriptor().stage_id);
    if (!checked) return checked.error();
    return outputs;
}

// A kernel that met one of nvcr.fx.v1's failure conditions set the word: the stage fails.
Result<void> TensorRTExecutionSession::check_fx_failure(const std::string& stage_id) {
    std::uint32_t* word = nullptr;
    std::uint32_t failed = 0;
    if (const auto status = fx::failure_word(&word); status != cudaSuccess) return cuda_error("fx failure word", status);
    auto copied = copy_async(&failed, word, sizeof(failed), cudaMemcpyDeviceToHost, stream_, "fx failure word");
    if (!copied) return copied.error();
    auto synchronized = synchronize_stream(stream_, "fx failure word");
    if (!synchronized) return synchronized.error();
    if (failed == 0) return {};
    cudaMemsetAsync(word, 0, sizeof(*word), stream_);
    return Error(ErrorCode::malformed_bitstream, stage_id + ": a value left the nvcr.fx.v1 range",
                 std::string(subsystem));
}

Result<std::vector<DeviceTensor>> run_device_stage(
    TensorRTExecutionSession& session,
    const session_api::ExecutableStage& stage,
    std::span<const DeviceTensor* const> inputs,
    EngineOutputStorage output_storage) {
    return session.run_stage(stage, inputs, output_storage);
}

Result<session_api::Completion> TensorRTExecutionSession::submit(
    const session_api::ExecutableStage& stage,
    std::span<const session_api::TensorView> tensors,
    std::span<const session_api::ExecutionDependency> dependencies) {
    if (!stage) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT session received a null stage",
            std::string(subsystem));
    }
    if (std::dynamic_pointer_cast<const TensorRTExactStage>(stage)) {
        return Error(
            ErrorCode::not_implemented,
            "normative stages on TensorRT run through device tensors, not TensorView submission",
            std::string(subsystem));
    }
    auto concrete_stage = std::dynamic_pointer_cast<const TensorRTStage>(stage);
    if (!concrete_stage || concrete_stage->owner_id() != owner_id_) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT stage belongs to another provider session",
            std::string(subsystem));
    }
    // Codecs that use only this generic contract never call finalize_stages().
    if (!initialized_) {
        auto finalized = finalize_stages();
        if (!finalized) return finalized.error();
    }
    auto completion = submit_impl(concrete_stage, tensors, dependencies);
    if (!completion || concrete_stage->descriptor().numerics_profile != numerics::fx_v1) return completion;
    auto checked = check_fx_failure(concrete_stage->descriptor().stage_id);
    if (!checked) return checked.error();
    return completion;
}

Result<session_api::Completion> TensorRTExecutionSession::submit_impl(
    const std::shared_ptr<const TensorRTStage>& stage,
    std::span<const session_api::TensorView> tensors,
    std::span<const session_api::ExecutionDependency> dependencies,
    nvinfer1::IExecutionContext* prepared_context,
    std::chrono::steady_clock::time_point setup_start) {
    const auto engine_index = stage->engine_index();
    if (!initialized_ || engine_index >= engines_.size() || engine_index >= stages_.size() ||
        stages_[engine_index].get() != stage.get()) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session stage is not loaded",
            std::string(subsystem));
    }

    if (setup_start == std::chrono::steady_clock::time_point{}) {
        setup_start = std::chrono::steady_clock::now();
    }
    auto& instance = engines_[engine_index];
    const auto& specification = stage_specs_[engine_index];
    if (tensors.size() != specification.tensors.size()) {
        return Error(
            ErrorCode::invalid_argument,
            std::string(specification.filename) + " tensor count mismatch",
            std::string(subsystem));
    }

    for (const auto& dependency : dependencies) {
        if (!dependency.completion || dependency.completion->owner_id() != owner_id_) {
            return Error(
                ErrorCode::invalid_argument,
                "TensorRT dependency belongs to another provider session",
                std::string(subsystem));
        }
        auto concrete_completion =
            std::dynamic_pointer_cast<TensorRTCompletion>(dependency.completion);
        if (!concrete_completion) {
            return Error(
                ErrorCode::invalid_argument,
                "TensorRT dependency has an incompatible completion handle",
                std::string(subsystem));
        }
        const auto waited = cudaStreamWaitEvent(stream_, concrete_completion->event(), 0);
        if (waited != cudaSuccess) {
            return cuda_error("cudaStreamWaitEvent dependency", waited);
        }
    }

    auto context_result = prepared_context != nullptr ?
        Result<nvinfer1::IExecutionContext*>(prepared_context) : acquire_context(instance);
    if (!context_result) return context_result.error();
    auto* context = context_result.value();

    auto find_tensor = [&tensors](std::string_view name) -> const session_api::TensorView* {
        const session_api::TensorView* found = nullptr;
        for (const auto& tensor : tensors) {
            if (tensor.name != name) continue;
            if (found != nullptr) return nullptr;
            found = &tensor;
        }
        return found;
    };

    for (std::size_t tensor_index = 0;
         tensor_index < specification.tensors.size();
         ++tensor_index) {
        const auto& tensor_specification = specification.tensors[tensor_index];
        const auto& contract = stage->descriptor().tensors[tensor_index];
        const auto* tensor = find_tensor(tensor_specification.name);
        if (tensor == nullptr) {
            return Error(
                ErrorCode::invalid_argument,
                std::string(specification.filename) + " missing or duplicate tensor " +
                    std::string(tensor_specification.name),
                std::string(subsystem));
        }
        const auto expected_access =
            tensor_specification.mode == nvinfer1::TensorIOMode::kINPUT ?
                session_api::TensorAccess::read : session_api::TensorAccess::write;
        if (tensor->data_type != tensor_specification.data_type ||
            tensor->access != expected_access ||
            tensor->shape.size() != tensor_specification.rank) {
            return Error(
                ErrorCode::invalid_argument,
                std::string(specification.filename) + " tensor contract mismatch for " +
                    std::string(tensor_specification.name),
                std::string(subsystem));
        }
        // Pinned host buffers are staged through device copies below.
        if (!tensor->storage.buffer || tensor->storage.buffer->owner_id() != owner_id_ ||
            (tensor->storage.buffer->domain() != session_api::MemoryDomain::provider_device &&
             tensor->storage.buffer->domain() != session_api::MemoryDomain::pinned_host)) {
            return Error(
                ErrorCode::invalid_argument,
                std::string(specification.filename) +
                    " requires a device or pinned buffer owned by this session for " +
                    std::string(tensor_specification.name),
                std::string(subsystem));
        }
        if (tensor->storage.offset_bytes > tensor->storage.buffer->size_bytes() ||
            tensor->storage.size_bytes >
                tensor->storage.buffer->size_bytes() - tensor->storage.offset_bytes) {
            return Error(
                ErrorCode::invalid_argument,
                std::string(specification.filename) + " buffer slice is out of bounds for " +
                    std::string(tensor_specification.name),
                std::string(subsystem));
        }

        std::size_t elements = 1;
        nvinfer1::Dims shape{};
        shape.nbDims = static_cast<std::int32_t>(tensor->shape.size());
        for (std::size_t dimension = 0; dimension < tensor->shape.size(); ++dimension) {
            const auto value = tensor->shape[dimension];
            if (value <= 0 || value > std::numeric_limits<std::int32_t>::max() ||
                value < contract.dimensions[dimension].minimum ||
                value > contract.dimensions[dimension].maximum ||
                elements > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(value)) {
                return Error(
                    ErrorCode::invalid_argument,
                    std::string(specification.filename) + " invalid tensor shape for " +
                        std::string(tensor_specification.name),
                    std::string(subsystem));
            }
            elements *= static_cast<std::size_t>(value);
            shape.d[dimension] = static_cast<std::int32_t>(value);
        }
        const auto element_bytes = tensor_element_bytes(tensor->data_type);
        if (element_bytes == 0U ||
            elements > std::numeric_limits<std::size_t>::max() / element_bytes ||
            tensor->storage.size_bytes < elements * element_bytes) {
            return Error(
                ErrorCode::invalid_argument,
                std::string(specification.filename) + " buffer is too small for " +
                    std::string(tensor_specification.name),
                std::string(subsystem));
        }
        if (tensor_specification.mode == nvinfer1::TensorIOMode::kINPUT &&
            !context->setInputShape(
                std::string(tensor_specification.name).c_str(), shape)) {
            return backend_error(
                std::string(specification.filename) + " rejected input shape for " +
                std::string(tensor_specification.name));
        }
    }

    std::vector<std::uintptr_t> graph_signature;
    graph_signature.reserve(specification.tensors.size() * 6U);
    std::vector<session_api::BufferHandle> retained_buffers;
    retained_buffers.reserve(tensors.size());
    struct StagedOutput final { void* host; const void* device; std::size_t bytes; };
    std::vector<StagedOutput> staged_outputs;
    for (const auto& tensor_specification : specification.tensors) {
        const auto* tensor = find_tensor(tensor_specification.name);
        auto concrete_buffer = std::dynamic_pointer_cast<TensorRTBuffer>(tensor->storage.buffer);
        if (!concrete_buffer) {
            return Error(
                ErrorCode::invalid_argument,
                "TensorRT session received an incompatible buffer handle",
                std::string(subsystem));
        }
        auto* address = static_cast<std::byte*>(concrete_buffer->data()) +
            tensor->storage.offset_bytes;
        const std::string name(tensor_specification.name);
        // A pinned host tensor goes through a device copy, so engines never read or write it over the bus.
        if (tensor->storage.buffer->domain() == session_api::MemoryDomain::pinned_host) {
            std::size_t bytes = tensor_element_bytes(tensor->data_type);
            for (const auto dimension : tensor->shape) bytes *= static_cast<std::size_t>(dimension);
            auto& device = staging_[std::string(specification.filename) + "/" + name];
            if (device.bytes < bytes) {
                auto allocated = allocate_cuda(bytes);
                if (!allocated) return allocated.error();
                device = std::move(allocated.value());
            }
            auto* host = address;
            address = static_cast<std::byte*>(device.data);
            if (tensor_specification.mode == nvinfer1::TensorIOMode::kINPUT) {
                auto copied = copy_async(address, host, bytes, cudaMemcpyHostToDevice, stream_, "stage input");
                if (!copied) return copied.error();
            } else {
                staged_outputs.push_back({host, address, bytes});
            }
        }
        if (!context->setTensorAddress(name.c_str(), address)) {
            return backend_error(
                std::string(specification.filename) + " rejected device address for " + name);
        }
        if (tensor_specification.mode == nvinfer1::TensorIOMode::kOUTPUT) {
            const auto resolved = context->getTensorShape(name.c_str());
            if (resolved.nbDims != static_cast<std::int32_t>(tensor->shape.size())) {
                return backend_error(
                    std::string(specification.filename) + " produced invalid shape for " + name);
            }
            for (std::int32_t dimension = 0; dimension < resolved.nbDims; ++dimension) {
                if (resolved.d[dimension] != tensor->shape[static_cast<std::size_t>(dimension)]) {
                    return backend_error(
                        std::string(specification.filename) + " output shape changed for " + name);
                }
            }
        }
        graph_signature.push_back(reinterpret_cast<std::uintptr_t>(address));
        graph_signature.push_back(static_cast<std::uintptr_t>(tensor->shape.size()));
        for (const auto dimension : tensor->shape) {
            graph_signature.push_back(static_cast<std::uintptr_t>(dimension));
        }
        retained_buffers.push_back(tensor->storage.buffer);
    }

    std::pair<cudaEvent_t, cudaEvent_t> profile_events{};
    const bool profile_cuda = active_profiler != nullptr && active_profiler->enabled();
    if (profile_cuda) {
        auto created = create_profile_events("cudaEventCreate engine profile");
        if (!created) return created.error();
        profile_events = created.value();
        auto recorded = record_cuda_event(
            profile_events.first, stream_, "cudaEventRecord engine start");
        if (!recorded) {
            static_cast<void>(cudaEventDestroy(profile_events.first));
            static_cast<void>(cudaEventDestroy(profile_events.second));
            return recorded.error();
        }
    }

    auto enqueued = enqueue_engine(instance, *context, stream_, std::move(graph_signature));
    if (!enqueued) {
        if (profile_cuda) {
            static_cast<void>(cudaEventDestroy(profile_events.first));
            static_cast<void>(cudaEventDestroy(profile_events.second));
        }
        return enqueued.error();
    }
    for (const auto& output : staged_outputs) {
        auto copied = copy_async(output.host, output.device, output.bytes, cudaMemcpyDeviceToHost, stream_,
                                 "stage output");
        if (!copied) return copied.error();
    }
    if (profile_cuda) {
        auto recorded = record_cuda_event(
            profile_events.second, stream_, "cudaEventRecord engine stop");
        if (!recorded) {
            static_cast<void>(cudaEventDestroy(profile_events.first));
            static_cast<void>(cudaEventDestroy(profile_events.second));
            return recorded.error();
        }
        active_profiler->record_cuda_stage(
            std::string(specification.filename) + " enqueue",
            profile_events.first,
            profile_events.second);
    }

    cudaEvent_t completion_event{};
    auto completion_status = cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming);
    if (completion_status != cudaSuccess) {
        return cuda_error("cudaEventCreateWithFlags completion", completion_status);
    }
    completion_status = cudaEventRecord(completion_event, stream_);
    if (completion_status != cudaSuccess) {
        static_cast<void>(cudaEventDestroy(completion_event));
        return cuda_error("cudaEventRecord completion", completion_status);
    }

    if (instance.low_memory_mode) {
        auto synchronized = synchronize_stream(
            stream_, "cudaStreamSynchronize device engine");
        if (!synchronized) {
            static_cast<void>(cudaEventDestroy(completion_event));
            return synchronized.error();
        }
        instance.context.reset();
    }
    if (active_profiler != nullptr && active_profiler->enabled()) {
        const auto cpu_elapsed = std::chrono::steady_clock::now() - setup_start;
        active_profiler->record_cpu_stage(
            std::string(specification.filename) + " setup",
            std::chrono::duration<double, std::milli>(cpu_elapsed).count());
    }

    return std::static_pointer_cast<session_api::ICompletion>(
        std::make_shared<TensorRTCompletion>(
            owner_id_, completion_event, std::move(retained_buffers)));
}

TensorRTExecutionSession::TensorRTExecutionSession() {
    static std::atomic<std::uint64_t> next_session_id{1U};
    owner_id_ = "tensorrt-session-" +
        std::to_string(next_session_id.fetch_add(1U, std::memory_order_relaxed));
}

TensorRTExecutionSession::~TensorRTExecutionSession() {
    if (stream_ != nullptr) {
        static_cast<void>(cudaStreamSynchronize(stream_));
    }
    std::size_t graph_captures = 0;
    std::size_t graph_hits = 0;
    std::size_t graph_entries = 0;
    for (const auto& engine : engines_) {
        graph_captures += engine.graph_captures;
        graph_hits += engine.graph_hits;
        graph_entries += engine.graph_invocations.size();
    }
    if (profiling_enabled_) {
        std::clog << "[nvcr.profile] cuda_graph captures=" << graph_captures
                  << " hits=" << graph_hits
                  << " entries=" << graph_entries
                  << " limit_per_engine=" << max_cuda_graph_invocations_per_engine
                  << '\n';
    } else if (graph_captures != 0 || graph_hits != 0) {
        std::clog << "[nvcr.tensorrt] [info] CUDA Graph TensorRT captures="
                  << graph_captures << " hits=" << graph_hits << '\n';
    }
    stages_.clear();
    engines_.clear();
    runtime_.reset();
    stream_ = nullptr;
    stream_owner_.reset();
}

Result<session_api::BufferHandle> TensorRTExecutionSession::allocate(
    std::size_t size_bytes,
    session_api::MemoryDomain domain) {
    if (!ready_) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session is not initialized",
            std::string(subsystem));
    }
    if (size_bytes == 0U) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT provider buffer size must be non-zero",
            std::string(subsystem));
    }
    try {
        if (domain == session_api::MemoryDomain::host) {
            return std::static_pointer_cast<session_api::IProviderBuffer>(
                std::make_shared<TensorRTBuffer>(
                    std::make_unique<std::byte[]>(size_bytes), size_bytes, owner_id_));
        }
        if (domain == session_api::MemoryDomain::pinned_host) {
            void* storage = nullptr;
            const auto status = cudaMallocHost(&storage, size_bytes);
            if (status != cudaSuccess) return cuda_error("cudaMallocHost", status);
            return std::static_pointer_cast<session_api::IProviderBuffer>(
                std::make_shared<TensorRTBuffer>(storage, size_bytes, owner_id_));
        }
        CudaAllocationScope allocation_scope(
            stream_, &profiler_, &scratch_arena_, false, owner_id_, stream_owner_);
        auto allocation = allocate_cuda(size_bytes);
        if (!allocation) return allocation.error();
        return std::static_pointer_cast<session_api::IProviderBuffer>(
            std::make_shared<TensorRTBuffer>(
                std::move(allocation.value()), owner_id_));
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "TensorRT provider buffer allocation failed",
            std::string(subsystem));
    }
}

bool TensorRTExecutionSession::supports_numerics_profile(std::string_view profile) const noexcept {
    if (profile == numerics::fx_v1) return fx_ready_;
    return profile == deterministic::intconv_profile && compute_major_ >= 8;
}

// An nvcr.fx.v1 stage runs the plan <component>.fx.plan, which the bundle lists in
// engine.sha256 and whose sidecar names the program digest it was compiled from.
Result<session_api::ExecutableStage> TensorRTExecutionSession::load_fx_stage(
    session_api::ExecutableStageDescriptor descriptor) {
    if (!supports_numerics_profile(descriptor.numerics_profile)) {
        return Error(ErrorCode::incompatible_precision,
                     descriptor.stage_id + " requires nvcr.fx.v1, which this build or device does not implement",
                     std::string(subsystem));
    }
    const auto plan = descriptor.artifact.component_id + ".fx.plan";
    const auto sidecar = plan + ".json";
    if (!bundle_files_.contains(plan) || !bundle_files_.contains(sidecar)) {
        return Error(ErrorCode::missing_artifact, "the bundle has no nvcr.fx.v1 plan for " + descriptor.stage_id,
                     std::string(subsystem));
    }
    auto json = detail::json::parse_file(bundle_root_ / sidecar, subsystem);
    if (!json) return json.error();
    const auto* program = json.value().find("program_sha256");
    std::string expected;
    for (const auto byte : descriptor.artifact.expected_digest) {
        constexpr char digits[] = "0123456789abcdef";
        expected += digits[static_cast<unsigned>(byte) >> 4U];
        expected += digits[static_cast<unsigned>(byte) & 15U];
    }
    if (program == nullptr || program->string_value != expected) {
        return Error(ErrorCode::digest_mismatch,
                     plan + " was not compiled from the program the stream requires", std::string(subsystem));
    }
    // The plan's sidecar records its shapes; -1 marks a dynamic axis.
    TensorRTStageSpec specification;
    specification.filename = plan;
    const auto* shapes = json.value().find("tensors");
    for (const auto& contract : descriptor.tensors) {
        const auto* shape = shapes ? shapes->find(contract.name) : nullptr;
        if (shape == nullptr || shape->array_value.size() != 4) {
            return Error(ErrorCode::invalid_argument, plan + " records no shape for " + contract.name,
                         std::string(subsystem));
        }
        TensorRTTensorSpec tensor;
        tensor.name = contract.name;
        tensor.data_type = session_api::TensorDataType::int32;
        tensor.rank = 4;
        tensor.mode = contract.access == session_api::TensorAccess::read ? nvinfer1::TensorIOMode::kINPUT
                                                                          : nvinfer1::TensorIOMode::kOUTPUT;
        for (std::size_t axis = 0; axis < 4; ++axis) {
            tensor.dimensions[axis] = shape->array_value[axis].integer_value.value_or(0);
            tensor.warmup_dimensions[axis] = tensor.dimensions[axis];
        }
        specification.tensors.push_back(std::move(tensor));
    }
    descriptor.stage_id = plan;
    descriptor.artifact.component_id = plan;
    descriptor.artifact.path = (bundle_root_ / plan).string();
    descriptor.artifact.provider_id = "tensorrt";
    return load_validated_stage(std::move(descriptor), std::move(specification));
}

Result<session_api::ExecutableStage> TensorRTExecutionSession::load_exact_stage(
    session_api::ExecutableStageDescriptor descriptor) {
    if (!ready_) {
        return Error(ErrorCode::invalid_state, "TensorRT runtime is not initialized",
                     std::string(subsystem));
    }
    if (!supports_numerics_profile(descriptor.numerics_profile)) {
        return Error(ErrorCode::incompatible_precision,
                     descriptor.stage_id + " requires " + descriptor.numerics_profile +
                         ", which this CUDA device does not implement",
                     std::string(subsystem));
    }
    auto program = deterministic::load_normative_program(descriptor);
    if (!program) return program.error();
    auto executor = deterministic::CudaExecutor::create(*program.value().program, stream_);
    if (!executor) return executor.error();
    auto stage = std::make_shared<const TensorRTExactStage>(
        std::move(descriptor), owner_id_, std::move(program.value()),
        std::make_shared<deterministic::CudaExecutor>(std::move(executor.value())));
    exact_stages_.push_back(stage);
    return session_api::ExecutableStage(stage);
}

Result<std::vector<DeviceTensor>> TensorRTExecutionSession::run_exact_stage(
    const TensorRTExactStage& stage,
    std::span<const DeviceTensor* const> inputs,
    EngineOutputStorage output_storage) {
    const auto& program = stage.program();
    std::vector<deterministic::HalfDeviceTensor> input_views;
    std::vector<deterministic::Extent> extents;
    for (const auto id : program.inputs) {
        const auto& name = program.tensor_names[id];
        const auto* input = find_device_tensor(inputs, name);
        if (input == nullptr || input->shape.nbDims != 4 || input->shape.d[0] != 1) {
            return backend_error(program.name + " missing rank-4 batch-1 device input " + name);
        }
        const deterministic::Extent extent{
            static_cast<std::int32_t>(input->shape.d[1]), static_cast<std::int32_t>(input->shape.d[2]),
            static_cast<std::int32_t>(input->shape.d[3])};
        extents.push_back(extent);
        input_views.push_back(deterministic::HalfDeviceTensor{name, extent, input->storage.data});
    }
    auto output_extents = deterministic::infer_output_extents(program, extents);
    if (!output_extents) return output_extents.error();
    std::vector<DeviceTensor> outputs;
    outputs.reserve(program.outputs.size());
    for (std::size_t slot = 0; slot < program.outputs.size(); ++slot) {
        const auto& extent = output_extents.value()[slot];
        nvinfer1::Dims shape{};
        shape.nbDims = 4;
        shape.d[0] = 1;
        shape.d[1] = extent.channels;
        shape.d[2] = extent.height;
        shape.d[3] = extent.width;
        const auto& name = program.tensor_names[program.outputs[slot]];
        auto output = output_storage == EngineOutputStorage::scratch ?
            allocate_device_tensor_scratch(name, shape) : allocate_device_tensor(name, shape);
        if (!output) return output.error();
        outputs.push_back(std::move(output.value()));
    }
    std::vector<deterministic::HalfDeviceTensor> output_views;
    for (std::size_t slot = 0; slot < outputs.size(); ++slot) {
        output_views.push_back(deterministic::HalfDeviceTensor{
            outputs[slot].name, output_extents.value()[slot], outputs[slot].storage.data});
    }
    auto ran = stage.executor().run_half(input_views, output_views);
    if (!ran) return ran.error();
    return outputs;
}

Result<session_api::ExecutableStage> TensorRTExecutionSession::load_stage(
    session_api::ExecutableStageDescriptor descriptor) {
    if (descriptor.numerics_profile == numerics::fx_v1) return load_fx_stage(std::move(descriptor));
    auto normative = deterministic::is_normative(descriptor);
    if (!normative) return normative.error();
    if (normative.value()) return load_exact_stage(std::move(descriptor));
    // Generic callers describe a stage only by its contract: dynamic axes warm
    // up at the engine's optimization-profile minimum.
    // A component the bundle holds as <component>.plan resolves to that plan.
    if (!descriptor.artifact.component_id.ends_with(".plan") &&
        bundle_files_.contains(descriptor.artifact.component_id + ".plan")) {
        descriptor.stage_id = descriptor.artifact.component_id + ".plan";
        descriptor.artifact.component_id = descriptor.stage_id;
        descriptor.artifact.path = (bundle_root_ / descriptor.stage_id).string();
    }
    TensorRTStageSpec specification;
    specification.filename = descriptor.stage_id;
    specification.tensors.reserve(descriptor.tensors.size());
    for (const auto& contract : descriptor.tensors) {
        if (!tensorrt_data_type(contract.data_type) || contract.dimensions.empty() ||
            contract.dimensions.size() > 4U) {
            return Error(
                ErrorCode::invalid_argument,
                "TensorRT stages support rank-1 to rank-4 INT32, FP16, and FP32 tensors",
                std::string(subsystem));
        }
        TensorRTTensorSpec tensor;
        tensor.name = contract.name;
        tensor.data_type = contract.data_type;
        tensor.rank = contract.dimensions.size();
        tensor.mode = contract.access == session_api::TensorAccess::read ?
            nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
        for (std::size_t axis = 0; axis < tensor.rank; ++axis) {
            const auto& bounds = contract.dimensions[axis];
            const bool dynamic = bounds.minimum != bounds.maximum;
            tensor.dimensions[axis] = dynamic ? -1 : bounds.minimum;
            tensor.warmup_dimensions[axis] = dynamic ? 0 : bounds.minimum;
        }
        specification.tensors.push_back(std::move(tensor));
    }
    return load_validated_stage(std::move(descriptor), std::move(specification));
}

Result<session_api::ExecutableStage> TensorRTExecutionSession::load_stage(
    TensorRTStageSpec specification, std::string codec_id) {
    session_api::ExecutableStageDescriptor descriptor;
    descriptor.stage_id = specification.filename;
    descriptor.artifact.provider_id = "tensorrt";
    descriptor.artifact.codec_id = std::move(codec_id);
    descriptor.artifact.component_id = specification.filename;
    descriptor.artifact.path = (bundle_root_ / specification.filename).string();
    descriptor.tensors.reserve(specification.tensors.size());
    for (const auto& tensor : specification.tensors) {
        session_api::TensorContract contract;
        contract.name = tensor.name;
        contract.data_type = tensor.data_type;
        contract.access = tensor.mode == nvinfer1::TensorIOMode::kINPUT ?
            session_api::TensorAccess::read : session_api::TensorAccess::write;
        contract.dimensions.reserve(tensor.rank);
        for (std::size_t axis = 0; axis < tensor.rank; ++axis) {
            const auto dimension = tensor.dimensions[axis];
            contract.dimensions.push_back(session_api::DimensionBounds{
                dimension < 0 ? 1 : dimension,
                dimension < 0 ? std::numeric_limits<std::int32_t>::max() : dimension});
        }
        descriptor.tensors.push_back(std::move(contract));
    }
    return load_validated_stage(std::move(descriptor), std::move(specification));
}

Result<session_api::ExecutableStage> TensorRTExecutionSession::load_validated_stage(
    session_api::ExecutableStageDescriptor descriptor, TensorRTStageSpec specification) {
    if (!ready_ || !runtime_) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT runtime is not initialized",
            std::string(subsystem));
    }
    if (initialized_) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT stages cannot be loaded after the session is finalized",
            std::string(subsystem));
    }
    if (descriptor.stage_id.empty() || descriptor.artifact.component_id.empty() ||
        descriptor.artifact.path.empty() || descriptor.artifact.provider_id != "tensorrt") {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT stage identity is incomplete or names another provider",
            std::string(subsystem));
    }
    // A stage is one plan file that engine.sha256 lists and that has been
    // digest-verified, loaded at most once.
    const bool duplicate = std::ranges::any_of(stage_specs_, [&](const auto& loaded) {
        return loaded.filename == specification.filename;
    });
    if (descriptor.stage_id != specification.filename ||
        descriptor.artifact.component_id != specification.filename ||
        !bundle_files_.contains(specification.filename) || duplicate) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT stage is unknown to the validated bundle or already loaded",
            std::string(subsystem));
    }
    if (fs::path(descriptor.artifact.path) != bundle_root_ / specification.filename ||
        descriptor.tensors.size() != specification.tensors.size()) {
        return Error(
            ErrorCode::invalid_argument,
            "TensorRT stage artifact or tensor contract does not match the validated bundle",
            std::string(subsystem));
    }

    auto loaded = load_engine(
        *runtime_, bundle_root_, specification, fixed_shape_profile_);
    if (!loaded) return loaded.error();
    const auto engine_index = engines_.size();
    engines_.push_back(std::move(loaded.value()));
    stage_specs_.push_back(std::move(specification));
    auto stage = std::make_shared<const TensorRTStage>(
        std::move(descriptor), owner_id_, engine_index);
    stages_.push_back(stage);
    return std::static_pointer_cast<const session_api::IExecutableStage>(stage);
}

Result<void> TensorRTExecutionSession::reset() {
    if (!initialized_) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session is not initialized",
            std::string(subsystem));
    }
    auto synchronized = synchronize_stream(stream_, "cudaStreamSynchronize session reset");
    if (!synchronized) return synchronized.error();
    scratch_arena_.reset();
    return {};
}

Result<void> TensorRTExecutionSession::initialize(
    const RuntimeConfiguration& configuration) {
    if (ready_ || runtime_ || stream_ != nullptr) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session is already initialized",
            std::string(subsystem));
    }
    if (configuration.artifacts.intra_engine_path.empty()) {
        return Error(
            ErrorCode::invalid_argument,
            "intra_engine_path must name the I-frame plan directory",
            std::string(subsystem));
    }
    std::error_code filesystem_error;
    if (!fs::is_directory(
            configuration.artifacts.intra_engine_path, filesystem_error)) {
        return Error(
            ErrorCode::dependency_unavailable,
            "I-frame plan directory is unavailable: " +
                configuration.artifacts.intra_engine_path.string(),
            std::string(subsystem));
    }

    const auto device_status = cudaSetDevice(configuration.provider.device_id);
    if (device_status != cudaSuccess) return cuda_error("cudaSetDevice", device_status);
    const auto major_status = cudaDeviceGetAttribute(
        &compute_major_, cudaDevAttrComputeCapabilityMajor, configuration.provider.device_id);
    if (major_status != cudaSuccess) return cuda_error("cudaDeviceGetAttribute", major_status);
    int compute_minor = 0;
    const auto minor_status = cudaDeviceGetAttribute(
        &compute_minor, cudaDevAttrComputeCapabilityMinor, configuration.provider.device_id);
    if (minor_status != cudaSuccess) return cuda_error("cudaDeviceGetAttribute", minor_status);
    fx_ready_ = fx::kernels_available(compute_major_, compute_minor) && fx::register_plugins();
    auto selected_policy = determine_context_policy(configuration);
    if (!selected_policy) return selected_policy.error();
    auto manifest = validate_engine_manifest(
        configuration.artifacts.intra_engine_path,
        configuration.provider.device_id,
        configuration.artifacts.model_id);
    if (!manifest) return manifest.error();

    bundle_root_ = configuration.artifacts.intra_engine_path;
    bundle_files_ = std::move(manifest.value().files);
    fixed_shape_profile_ = manifest.value().shape_profile == "fixed";
    context_policy_ = selected_policy.value();
    profiling_enabled_ = configuration.provider.enable_profiling;
    profiler_ = BackendProfiler(profiling_enabled_);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return backend_error("failed to create TensorRT runtime");
    stream_owner_ = std::make_shared<CudaStreamOwner>();
    const auto stream_status = cudaStreamCreateWithFlags(
        &stream_owner_->stream, cudaStreamNonBlocking);
    if (stream_status != cudaSuccess) {
        stream_owner_.reset();
        return cuda_error("cudaStreamCreateWithFlags", stream_status);
    }
    stream_ = stream_owner_->stream;
    auto scratch_ready = scratch_arena_.initialize(
        configuration.provider.device_arena_bytes);
    if (!scratch_ready) return scratch_ready.error();

    ready_ = true;
    return {};
}

Result<void> TensorRTExecutionSession::finalize_stages() {
    if (!ready_ || initialized_ || engines_.empty()) {
        return Error(
            ErrorCode::invalid_state,
            "TensorRT session has no loaded stages to finalize",
            std::string(subsystem));
    }
    if (context_policy_ == ContextPolicy::shared_workspace_persistent) {
        std::int64_t workspace_bytes = 0;
        for (const auto& engine : engines_) {
            workspace_bytes = std::max(
                workspace_bytes, engine.engine->getDeviceMemorySizeV2());
        }
        if (workspace_bytes < 0 ||
            static_cast<std::uint64_t>(workspace_bytes) >
                std::numeric_limits<std::size_t>::max()) {
            return backend_error("invalid TensorRT context workspace size");
        }
        auto workspace = scratch_arena_.reserve(static_cast<std::size_t>(workspace_bytes));
        if (!workspace) return workspace.error();
        for (auto& engine : engines_) {
            engine.shared_context_memory = workspace.value();
            engine.shared_context_memory_bytes = workspace_bytes;
        }
    }

    const auto scope = allocation_scope();
    for (std::size_t index = 0; index < engines_.size(); ++index) {
        engines_[index].low_memory_mode = context_policy_ == ContextPolicy::per_engine;
        auto warmed = warm_up_engine(engines_[index], stage_specs_[index], stream_);
        if (!warmed) return warmed.error();
    }
    if (context_policy_ == ContextPolicy::shared_workspace_persistent) {
        for (auto& engine : engines_) engine.use_cuda_graphs = true;
    }
    initialized_ = true;
    return {};
}


provider::ProviderDescriptor provider_descriptor() {
    return {
        provider::ProviderKind::tensorrt,
        "tensorrt",
        std::to_string(NV_TENSORRT_MAJOR) + "." +
            std::to_string(NV_TENSORRT_MINOR) + "." +
            std::to_string(NV_TENSORRT_PATCH),
        "NVIDIA TensorRT",
    };
}

Result<std::shared_ptr<session_api::IProviderSession>> make_provider_session(
    const RuntimeConfiguration& configuration) {
    try {
        auto session = std::make_shared<TensorRTExecutionSession>();
        auto initialized = session->initialize(configuration);
        if (!initialized) return initialized.error();
        return std::static_pointer_cast<session_api::IProviderSession>(std::move(session));
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate TensorRT provider session",
            std::string(subsystem));
    } catch (const std::exception& exception) {
        return backend_error(
            std::string("failed to create TensorRT provider session: ") + exception.what());
    } catch (...) {
        return backend_error("failed to create TensorRT provider session");
    }
}

}  // namespace nvcr::tensorrt
