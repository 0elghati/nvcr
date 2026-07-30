#include "nvcr/dcvcrt/tensorrt_backend.hpp"

#include "../common/sha256.hpp"
#include "cuda_ops.hpp"

#include "nvcr/dcvcrt/rans_codec.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::dcvcrt {
namespace {

namespace fs = std::filesystem;
using SequenceStateView = codec::SequenceStateView;

constexpr std::string_view subsystem = "dcvcrt.tensorrt";
constexpr std::size_t image_qp_count = 64U;
constexpr std::size_t video_qp_count = 72U;
constexpr std::uint32_t minimum_frame_dimension = 64U;
constexpr std::uint32_t maximum_frame_width = 1920U;
constexpr std::uint32_t maximum_frame_height = 1080U;

bool supported_frame_dimensions(std::uint32_t width, std::uint32_t height) {
    return width >= minimum_frame_dimension && height >= minimum_frame_dimension &&
        width <= maximum_frame_width && height <= maximum_frame_height &&
        (width & 1U) == 0U && (height & 1U) == 0U;
}

struct TensorSpec final {
    std::string_view name;
    nvinfer1::TensorIOMode mode;
    std::array<std::int64_t, 4> dimensions;
};

struct EngineSpec final {
    std::string_view filename;
    std::span<const TensorSpec> tensors;
};

constexpr std::array<TensorSpec, 3> analysis_tensors{{
    {"frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"q_enc", nvinfer1::TensorIOMode::kINPUT, {1, 368, 1, 1}},
    {"y", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> hyper_analysis_tensors{{
    {"y_padded", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"z", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> hyper_synthesis_tensors{{
    {"z_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"params_padded", nvinfer1::TensorIOMode::kOUTPUT, {1, 514, -1, -1}},
    {"common_padded", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> spatial_prior_tensors{{
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 512, -1, -1}},
    {"scales_means", nvinfer1::TensorIOMode::kOUTPUT, {1, 512, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> synthesis_tensors{{
    {"y_hat", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_dec", nvinfer1::TensorIOMode::kINPUT, {1, 368, 1, 1}},
    {"frame_hat", nvinfer1::TensorIOMode::kOUTPUT, {1, 3, -1, -1}},
}};

constexpr std::array<TensorSpec, 4> p_reference_frame_tensors{{
    {"reference_frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"q_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 4> p_reference_feature_tensors{{
    {"reference_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_feature", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 4> p_analysis_tensors{{
    {"frame", nvinfer1::TensorIOMode::kINPUT, {1, 3, -1, -1}},
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_encoder", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"y", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> p_hyper_analysis_tensors{{
    {"y_padded", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"z", nvinfer1::TensorIOMode::kOUTPUT, {1, 128, -1, -1}},
}};
constexpr std::array<TensorSpec, 3> p_prior_tensors{{
    {"z_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"temporal_context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"params", nvinfer1::TensorIOMode::kOUTPUT, {1, 384, -1, -1}},
}};
constexpr std::array<TensorSpec, 2> p_spatial_prior_tensors{{
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 512, -1, -1}},
    {"scales_means", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};
constexpr std::array<TensorSpec, 6> p_synthesis_tensors{{
    {"y_hat", nvinfer1::TensorIOMode::kINPUT, {1, 128, -1, -1}},
    {"context", nvinfer1::TensorIOMode::kINPUT, {1, 256, -1, -1}},
    {"q_decoder", nvinfer1::TensorIOMode::kINPUT, {1, 256, 1, 1}},
    {"q_recon", nvinfer1::TensorIOMode::kINPUT, {1, 320, 1, 1}},
    {"frame_hat", nvinfer1::TensorIOMode::kOUTPUT, {1, 3, -1, -1}},
    {"feature", nvinfer1::TensorIOMode::kOUTPUT, {1, 256, -1, -1}},
}};

const std::array<EngineSpec, 14> engine_specs{{
    {"i_analysis.plan", analysis_tensors},
    {"i_hyper_analysis.plan", hyper_analysis_tensors},
    {"i_hyper_synthesis.plan", hyper_synthesis_tensors},
    {"i_spatial_prior_1.plan", spatial_prior_tensors},
    {"i_spatial_prior_2.plan", spatial_prior_tensors},
    {"i_spatial_prior_3.plan", spatial_prior_tensors},
    {"i_synthesis.plan", synthesis_tensors},
    {"p_reference_frame.plan", p_reference_frame_tensors},
    {"p_reference_feature.plan", p_reference_feature_tensors},
    {"p_analysis.plan", p_analysis_tensors},
    {"p_hyper_analysis.plan", p_hyper_analysis_tensors},
    {"p_prior.plan", p_prior_tensors},
    {"p_spatial_prior.plan", p_spatial_prior_tensors},
    {"p_synthesis.plan", p_synthesis_tensors},
}};

class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::clog << "[TensorRT] " << (message != nullptr ? message : "") << '\n';
        }
    }
};

struct EngineInstance final {
    fs::path path;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    // Execution mode is owned by this backend session, never process-global.
    bool low_memory_mode{true};
};

Result<bool> determine_low_memory_mode(const RuntimeConfiguration& configuration) {
    if (configuration.tensorrt_execution_mode == TensorRTExecutionMode::low_memory) {
        return true;
    }
    if (configuration.tensorrt_execution_mode == TensorRTExecutionMode::performance) {
        return false;
    }
    const auto device_id = configuration.device_id;
    if (const char* raw = std::getenv("NVCR_TENSORRT_LOW_MEMORY_MODE"); raw != nullptr) {
        const std::string_view value(raw);
        if (value == "1" || value == "true" || value == "TRUE" ||
            value == "on" || value == "ON") {
            return true;
        }
        if (value == "0" || value == "false" || value == "FALSE" ||
            value == "off" || value == "OFF") {
            return false;
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
    // capacity-only cutoff incorrectly classified 12 GiB RTX 4070 cards as
    // low-memory, adding context churn and stream waits to every device stage.
    // Integrated/Jetson-class devices remain conservative unless explicitly
    // overridden through configuration or NVCR_TENSORRT_LOW_MEMORY_MODE.
    if (properties.integrated != 0) return true;
    constexpr std::size_t constrained_discrete_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    return static_cast<std::size_t>(properties.totalGlobalMem) <= constrained_discrete_bytes;
}


Result<nvinfer1::IExecutionContext*> acquire_context(EngineInstance& instance) {
    if (!instance.low_memory_mode) {
        if (!instance.context) {
            instance.context.reset(instance.engine->createExecutionContext());
            if (!instance.context) {
                return Error(
                    ErrorCode::backend_error,
                    "failed to create execution context: " + instance.path.string(),
                    std::string(subsystem));
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
    std::string device_name;
    std::string checksum_manifest;
    std::string checksum_manifest_sha256;
    std::string i_model_manifest_sha256;
    std::string p_model_manifest_sha256;
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
    auto device_name = read_json_string(text.value(), "device_name");
    auto checksum_manifest = read_json_string(text.value(), "checksum_manifest");
    auto checksum_manifest_sha256 =
        read_json_string(text.value(), "checksum_manifest_sha256");
    auto i_manifest_sha256 = read_json_string(text.value(), "i_model_manifest_sha256");
    auto p_manifest_sha256 = read_json_string(text.value(), "p_model_manifest_sha256");
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
    if (!device_name) return device_name.error();
    if (!checksum_manifest) return checksum_manifest.error();
    if (!checksum_manifest_sha256) return checksum_manifest_sha256.error();
    if (!i_manifest_sha256) return i_manifest_sha256.error();
    if (!p_manifest_sha256) return p_manifest_sha256.error();
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
    manifest.device_name = std::move(device_name.value());
    manifest.checksum_manifest = std::move(checksum_manifest.value());
    manifest.checksum_manifest_sha256 = std::move(checksum_manifest_sha256.value());
    manifest.i_model_manifest_sha256 = std::move(i_manifest_sha256.value());
    manifest.p_model_manifest_sha256 = std::move(p_manifest_sha256.value());
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

Result<void> validate_bundle_hashes(const fs::path& root, const EngineManifest& manifest) {
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
    constexpr std::array<std::string_view, 20> required_files{
        "i_analysis.plan", "i_hyper_analysis.plan", "i_hyper_synthesis.plan",
        "i_spatial_prior_1.plan", "i_spatial_prior_2.plan", "i_spatial_prior_3.plan",
        "i_synthesis.plan", "p_reference_frame.plan", "p_reference_feature.plan",
        "p_analysis.plan", "p_hyper_analysis.plan", "p_prior.plan",
        "p_spatial_prior.plan", "p_synthesis.plan", "i_entropy.bin", "i_quant.bin",
        "i_frame_manifest.json", "p_entropy.bin", "p_quant.bin", "p_frame_manifest.json",
    };
    if (checksums.size() != required_files.size()) {
        return backend_error("engine.sha256 does not describe the exact runtime bundle");
    }
    for (const auto required : required_files) {
        const auto entry = checksums.find(std::string(required));
        if (entry == checksums.end()) {
            return backend_error("engine.sha256 is missing " + std::string(required));
        }
        auto actual = detail::sha256_file(root / required, subsystem);
        if (!actual) return actual.error();
        if (actual.value() != entry->second) {
            return backend_error("engine bundle SHA-256 mismatch: " + std::string(required));
        }
    }
    if (!valid_sha256(manifest.i_model_manifest_sha256) ||
        !valid_sha256(manifest.p_model_manifest_sha256) ||
        checksums.at("i_frame_manifest.json") != manifest.i_model_manifest_sha256 ||
        checksums.at("p_frame_manifest.json") != manifest.p_model_manifest_sha256) {
        return backend_error("engine bundle does not match its model manifest identities");
    }
    return {};
}

Result<void> validate_engine_manifest(
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
    if (!valid_profile_id(manifest.value().model_profile_id) ||
        !valid_profile_id(manifest.value().target_profile_id) ||
        !valid_profile_id(manifest.value().engine_profile_id) ||
        manifest.value().engine_profile_id != manifest.value().optimization_point + "-fp16" ||
        !valid_sha256(manifest.value().model_profile_sha256) ||
        !valid_sha256(manifest.value().engine_profile_sha256) ||
        !valid_sha256(manifest.value().target_profile_sha256)) {
        return backend_error(
            "engine_manifest.json has invalid model/target/engine profile identity or digest");
    }
    auto hashes = validate_bundle_hashes(root, manifest.value());
    if (!hashes) return hashes.error();
    int cuda_runtime_version = 0;
    const auto cuda_version_status = cudaRuntimeGetVersion(&cuda_runtime_version);
    if (cuda_version_status != cudaSuccess) {
        return cuda_error("cudaRuntimeGetVersion", cuda_version_status);
    }
    if (manifest.value().cuda_runtime_version != cuda_runtime_version) {
        return backend_error(
            "TensorRT engine bundle CUDA runtime " +
            std::to_string(manifest.value().cuda_runtime_version) +
            " does not match active CUDA runtime " + std::to_string(cuda_runtime_version));
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

    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, device_id);
    if (status != cudaSuccess) return cuda_error("cudaGetDeviceProperties", status);
    if (manifest.value().device_name != properties.name ||
        manifest.value().compute_major != properties.major ||
        manifest.value().compute_minor != properties.minor ||
        manifest.value().multiprocessor_count != properties.multiProcessorCount) {
        return backend_error(
            "TensorRT engine bundle was built for " + manifest.value().device_name +
            " (SM " + std::to_string(manifest.value().compute_major) + "." +
            std::to_string(manifest.value().compute_minor) + ", " +
            std::to_string(manifest.value().multiprocessor_count) +
            " multiprocessors) but selected device " + std::to_string(device_id) +
            " is " + properties.name + " (SM " + std::to_string(properties.major) + "." +
            std::to_string(properties.minor) + ", " +
            std::to_string(properties.multiProcessorCount) +
            " multiprocessors); rebuild the engine directory on the selected device");
    }
    return {};
}

std::string dimensions_string(const nvinfer1::Dims& dimensions) {
    std::string output;
    for (std::int32_t index = 0; index < dimensions.nbDims; ++index) {
        if (!output.empty()) output += 'x';
        output += std::to_string(dimensions.d[index]);
    }
    return output;
}

Result<void> validate_engine(
    const nvinfer1::ICudaEngine& engine,
    const EngineSpec& specification) {
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
        if (engine.getTensorDataType(name.c_str()) != nvinfer1::DataType::kHALF) {
            return backend_error(
                std::string(specification.filename) + " tensor '" + name + "' is not FP16");
        }
        const auto actual = engine.getTensorShape(name.c_str());
        if (actual.nbDims != static_cast<std::int32_t>(tensor.dimensions.size())) {
            return backend_error(
                std::string(specification.filename) + " tensor '" + name +
                "' has rank " + std::to_string(actual.nbDims) + "; expected rank 4");
        }
        for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
            if (actual.d[index] != tensor.dimensions[index]) {
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
    const EngineSpec& specification) {
    const auto path = root / specification.filename;
    auto bytes = read_binary(path);
    if (!bytes) return bytes.error();

    std::unique_ptr<nvinfer1::ICudaEngine> engine(
        runtime.deserializeCudaEngine(bytes.value().data(), bytes.value().size()));
    if (!engine) {
        return backend_error("failed to deserialize file: " + path.string());
    }
    auto valid = validate_engine(*engine, specification);
    if (!valid) return valid.error();

    return EngineInstance{path, std::move(engine), nullptr};
}

struct ProfileStage final {
    std::string name;
    double cpu_ms{};
    double cuda_ms{};
    bool has_cuda{};
};

struct ProfileCounters final {
    std::size_t allocations{};
    std::size_t allocation_bytes{};
    std::size_t h2d_copies{};
    std::size_t h2d_bytes{};
    std::size_t d2h_copies{};
    std::size_t d2h_bytes{};
    std::size_t d2d_copies{};
    std::size_t d2d_bytes{};
    std::size_t synchronizations{};
};

struct PendingCudaStage final {
    std::string name;
    cudaEvent_t start{};
    cudaEvent_t stop{};
};

class BackendProfiler final {
public:
    explicit BackendProfiler(bool enabled) : enabled_(enabled) {}

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    void begin(std::string operation, FrameType frame_type, std::uint64_t frame_index) {
        operation_ = std::move(operation);
        frame_type_ = frame_type;
        frame_index_ = frame_index;
        counters_ = {};
        stages_.clear();
        pending_cuda_stages_.clear();
    }

    void record_allocation(std::size_t bytes) noexcept {
        if (!enabled_) return;
        ++counters_.allocations;
        counters_.allocation_bytes += bytes;
    }

    void record_transfer(cudaMemcpyKind kind, std::size_t bytes) noexcept {
        if (!enabled_) return;
        switch (kind) {
        case cudaMemcpyHostToDevice:
            ++counters_.h2d_copies;
            counters_.h2d_bytes += bytes;
            break;
        case cudaMemcpyDeviceToHost:
            ++counters_.d2h_copies;
            counters_.d2h_bytes += bytes;
            break;
        case cudaMemcpyDeviceToDevice:
            ++counters_.d2d_copies;
            counters_.d2d_bytes += bytes;
            break;
        default:
            break;
        }
    }

    void record_synchronization() noexcept {
        if (enabled_) ++counters_.synchronizations;
    }

    void record_cpu_stage(std::string name, double cpu_ms) {
        if (!enabled_) return;
        stages_.push_back(ProfileStage{std::move(name), cpu_ms, 0.0, false});
    }

    void record_cuda_stage(std::string name, cudaEvent_t start, cudaEvent_t stop) {
        if (!enabled_) return;
        pending_cuda_stages_.push_back(PendingCudaStage{std::move(name), start, stop});
    }

    void finish() noexcept {
        if (!enabled_) return;
        for (auto& pending : pending_cuda_stages_) {
            const auto synchronized = cudaEventSynchronize(pending.stop);
            float elapsed_ms = 0.0F;
            if (synchronized == cudaSuccess) {
                static_cast<void>(cudaEventElapsedTime(&elapsed_ms, pending.start, pending.stop));
            }
            static_cast<void>(cudaEventDestroy(pending.start));
            static_cast<void>(cudaEventDestroy(pending.stop));
            stages_.push_back(ProfileStage{
                std::move(pending.name), 0.0, static_cast<double>(elapsed_ms),
                synchronized == cudaSuccess});
        }
        pending_cuda_stages_.clear();

        const char type = frame_type_ == FrameType::intra ? 'I' : 'P';
        std::clog << "[nvcr.profile] " << operation_ << " frame=" << frame_index_
                  << " type=" << type
                  << " allocs=" << counters_.allocations
                  << " alloc_bytes=" << counters_.allocation_bytes
                  << " h2d=" << counters_.h2d_copies << '/' << counters_.h2d_bytes
                  << " d2h=" << counters_.d2h_copies << '/' << counters_.d2h_bytes
                  << " d2d=" << counters_.d2d_copies << '/' << counters_.d2d_bytes
                  << " syncs=" << counters_.synchronizations << '\n';
        for (const auto& stage : stages_) {
            std::clog << "[nvcr.profile] stage " << stage.name;
            if (stage.cpu_ms > 0.0) {
                std::clog << " cpu_ms=" << std::fixed << std::setprecision(3)
                          << stage.cpu_ms;
            }
            if (stage.has_cuda) {
                std::clog << " cuda_ms=" << std::fixed << std::setprecision(3)
                          << stage.cuda_ms;
            }
            std::clog << '\n';
        }
    }

private:
    bool enabled_{};
    std::string operation_;
    FrameType frame_type_{FrameType::intra};
    std::uint64_t frame_index_{};
    ProfileCounters counters_;
    std::vector<ProfileStage> stages_;
    std::vector<PendingCudaStage> pending_cuda_stages_;
};

thread_local cudaStream_t allocation_stream = nullptr;
thread_local BackendProfiler* active_profiler = nullptr;
class DeviceScratchArena;
thread_local DeviceScratchArena* active_scratch_arena = nullptr;

class CpuProfileScope final {
public:
    explicit CpuProfileScope(std::string name)
        : profiler_(active_profiler),
          name_(std::move(name)),
          start_(std::chrono::steady_clock::now()) {}
    ~CpuProfileScope() {
        if (profiler_ == nullptr || !profiler_->enabled()) return;
        const auto elapsed = std::chrono::steady_clock::now() - start_;
        profiler_->record_cpu_stage(
            std::move(name_), std::chrono::duration<double, std::milli>(elapsed).count());
    }
    CpuProfileScope(const CpuProfileScope&) = delete;
    CpuProfileScope& operator=(const CpuProfileScope&) = delete;

private:
    BackendProfiler* profiler_{};
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

class CudaAllocationScope final {
public:
    CudaAllocationScope(cudaStream_t stream, BackendProfiler* profiler, DeviceScratchArena* arena)
        : previous_stream_(std::exchange(allocation_stream, stream)),
          previous_profiler_(std::exchange(active_profiler, profiler)),
          previous_arena_(std::exchange(active_scratch_arena, arena)) {}
    ~CudaAllocationScope() {
        allocation_stream = previous_stream_;
        active_profiler = previous_profiler_;
        active_scratch_arena = previous_arena_;
    }
    CudaAllocationScope(const CudaAllocationScope&) = delete;
    CudaAllocationScope& operator=(const CudaAllocationScope&) = delete;

private:
    cudaStream_t previous_stream_{};
    BackendProfiler* previous_profiler_{};
    DeviceScratchArena* previous_arena_{};
};

void release_cuda(void* pointer, cudaStream_t stream) noexcept {
    if (pointer == nullptr) return;
    if (stream != nullptr) {
        static_cast<void>(cudaFreeAsync(pointer, stream));
    } else {
        static_cast<void>(cudaFree(pointer));
    }
}

struct CudaAllocation final {
    CudaAllocation() = default;
    CudaAllocation(void* pointer, std::size_t byte_count, cudaStream_t owner_stream, bool owned_allocation = true)
        : data(pointer), bytes(byte_count), stream(owner_stream), owned(owned_allocation) {}
    ~CudaAllocation() { if (owned) release_cuda(data, stream); }
    CudaAllocation(CudaAllocation&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          stream(std::exchange(other.stream, nullptr)),
          owned(std::exchange(other.owned, true)) {}
    CudaAllocation& operator=(CudaAllocation&& other) noexcept {
        if (this != &other) {
            if (owned) release_cuda(data, stream);
            data = std::exchange(other.data, nullptr);
            bytes = std::exchange(other.bytes, 0);
            stream = std::exchange(other.stream, nullptr);
            owned = std::exchange(other.owned, true);
        }
        return *this;
    }
    CudaAllocation(const CudaAllocation&) = delete;
    CudaAllocation& operator=(const CudaAllocation&) = delete;

    void* data{};
    std::size_t bytes{};
    cudaStream_t stream{};
    bool owned{true};
};

class DeviceScratchArena final {
public:
    ~DeviceScratchArena() {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }
    DeviceScratchArena() = default;
    DeviceScratchArena(DeviceScratchArena&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)),
          offset_(std::exchange(other.offset_, 0)) {}
    DeviceScratchArena& operator=(DeviceScratchArena&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) static_cast<void>(cudaFree(data_));
            data_ = std::exchange(other.data_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
            offset_ = std::exchange(other.offset_, 0);
        }
        return *this;
    }
    DeviceScratchArena(const DeviceScratchArena&) = delete;
    DeviceScratchArena& operator=(const DeviceScratchArena&) = delete;

    Result<void> initialize(std::size_t byte_count) {
        if (byte_count == 0) return {};
        void* pointer = nullptr;
        const auto status = cudaMalloc(&pointer, byte_count);
        if (status != cudaSuccess) return cuda_error("cudaMalloc device scratch arena", status);
        data_ = static_cast<std::byte*>(pointer);
        capacity_ = byte_count;
        offset_ = 0;
        return {};
    }

    void reset() noexcept { offset_ = 0; }

    Result<CudaAllocation> allocate(std::size_t bytes) noexcept {
        constexpr std::size_t alignment = 256;
        const auto aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (data_ == nullptr || bytes > capacity_ || aligned > capacity_ - bytes) {
            return Error(
                ErrorCode::resource_exhausted,
                "device scratch arena capacity exceeded",
                std::string(subsystem));
        }
        offset_ = aligned + bytes;
        return CudaAllocation(data_ + aligned, bytes, nullptr, false);
    }

    [[nodiscard]] bool initialized() const noexcept { return data_ != nullptr; }

private:
    std::byte* data_{};
    std::size_t capacity_{};
    std::size_t offset_{};
};

struct CudaEvent final {
    CudaEvent() = default;
    ~CudaEvent() {
        if (data != nullptr) static_cast<void>(cudaEventDestroy(data));
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t data{};
};

template <typename T>
struct PinnedHostBuffer final {
    ~PinnedHostBuffer() {
        if (data != nullptr) static_cast<void>(cudaFreeHost(data));
    }
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer() = default;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          capacity(std::exchange(other.capacity, 0)) {}
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (data != nullptr) static_cast<void>(cudaFreeHost(data));
            data = std::exchange(other.data, nullptr);
            capacity = std::exchange(other.capacity, 0);
        }
        return *this;
    }

    Result<void> ensure(std::size_t required_count, std::string_view name) {
        if (required_count <= capacity) return {};
        if (data != nullptr) {
            static_cast<void>(cudaFreeHost(data));
            data = nullptr;
            capacity = 0;
        }
        void* pointer = nullptr;
        const auto status = cudaMallocHost(&pointer, required_count * sizeof(T));
        if (status != cudaSuccess) {
            return backend_error(
                std::string("cudaMallocHost failed for ") + std::string(name) +
                ": " + cudaGetErrorString(status));
        }
        data = static_cast<T*>(pointer);
        capacity = required_count;
        return {};
    }

    T* data{};
    std::size_t capacity{};
};

struct ImageDecodeStaging final {
    Result<void> ensure(std::size_t z_count, std::size_t reduced_count) {
        auto ready = z_symbols.ensure(z_count, "image_decode_z_symbols");
        if (!ready) return ready.error();
        for (std::size_t stage = 0; stage < indexes.size(); ++stage) {
            ready = indexes[stage].ensure(reduced_count, "image_decode_indexes");
            if (!ready) return ready.error();
            ready = symbols[stage].ensure(reduced_count, "image_decode_symbols");
            if (!ready) return ready.error();
            if (indexes_ready[stage].data == nullptr) {
                const auto status = cudaEventCreateWithFlags(
                    &indexes_ready[stage].data, cudaEventDisableTiming);
                if (status != cudaSuccess) {
                    return cuda_error("cudaEventCreateWithFlags I decode indexes", status);
                }
            }
        }
        return {};
    }

    PinnedHostBuffer<std::int8_t> z_symbols;
    std::array<PinnedHostBuffer<std::uint8_t>, 4> indexes;
    std::array<PinnedHostBuffer<std::int8_t>, 4> symbols;
    std::array<CudaEvent, 4> indexes_ready;
};

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

Result<std::size_t> tensor_bytes(const nvinfer1::Dims& shape) {
    std::size_t elements = 1;
    for (std::int32_t index = 0; index < shape.nbDims; ++index) {
        if (shape.d[index] <= 0) return backend_error("TensorRT produced an unresolved shape");
        const auto dimension = static_cast<std::size_t>(shape.d[index]);
        if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
            return backend_error("TensorRT tensor size overflow");
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return backend_error("TensorRT tensor byte size overflow");
    }
    return elements * sizeof(std::uint16_t);
}

nvinfer1::Dims warmup_shape(
    std::string_view filename, std::string_view name, nvinfer1::Dims shape) {
    if (name == "frame" || name == "reference_frame") {
        shape.d[2] = filename.starts_with("p_") ? 192 : 144;
        shape.d[3] = filename.starts_with("p_") ? 192 : 176;
    } else if (name == "reference_feature" || name == "temporal_context") {
        shape.d[2] = 24;
        shape.d[3] = 24;
    } else if (name == "y_padded" || name == "y_hat") {
        shape.d[2] = 12;
        shape.d[3] = 12;
    } else if (name == "z_hat") {
        shape.d[2] = 3;
        shape.d[3] = 3;
    } else if (name == "context") {
        shape.d[2] = filename == "p_spatial_prior.plan" ? 12 :
            (filename.starts_with("p_") ? 24 : 9);
        shape.d[3] = filename == "p_spatial_prior.plan" ? 12 :
            (filename.starts_with("p_") ? 24 : 11);
    }
    return shape;
}

Result<void> warm_up_engine(
    EngineInstance& instance, const EngineSpec& specification, cudaStream_t stream) {
    auto context_result = acquire_context(instance);
    if (!context_result) return context_result.error();
    auto* context = context_result.value();
    for (const auto& tensor : specification.tensors) {
        if (tensor.mode != nvinfer1::TensorIOMode::kINPUT) continue;
        const std::string name(tensor.name);
        const auto shape = warmup_shape(
            specification.filename, tensor.name,
            instance.engine->getTensorShape(name.c_str()));
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
        auto bytes = tensor_bytes(shape);
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
    return {};
}

struct HostTensor final {
    std::string name;
    nvinfer1::Dims shape;
    std::vector<__half> values;
};

struct RuntimeAssets;
HostTensor make_quant_tensor(
    const RuntimeAssets& assets, bool encoder, std::uint32_t qp, std::string name);
HostTensor make_video_quant_tensor(
    const RuntimeAssets& assets, std::string_view kind, std::uint32_t qp);

nvinfer1::Dims make_dims(
    std::int32_t channels, std::int32_t height, std::int32_t width) {
    nvinfer1::Dims shape{};
    shape.nbDims = 4;
    shape.d[0] = 1;
    shape.d[1] = channels;
    shape.d[2] = height;
    shape.d[3] = width;
    return shape;
}

std::size_t element_count(const nvinfer1::Dims& shape) {
    std::size_t count = 1;
    for (std::int32_t index = 0; index < shape.nbDims; ++index) {
        count *= static_cast<std::size_t>(shape.d[index]);
    }
    return count;
}

struct DeviceTensor final {
    std::string name;
    nvinfer1::Dims shape;
    CudaAllocation storage;
};

enum class EngineOutputStorage {
    owned,
    scratch,
};

struct ImageQuantDeviceCache final {
    std::array<std::optional<DeviceTensor>, image_qp_count> q_enc;
    std::array<std::optional<DeviceTensor>, image_qp_count> q_dec;
};

struct VideoQuantDeviceCache final {
    std::array<std::optional<DeviceTensor>, video_qp_count> q_encoder;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_decoder;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_feature;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_recon;
};

struct DeviceDpb final {
    std::optional<DeviceTensor> frame;
    std::optional<DeviceTensor> feature;
    std::uint64_t next_frame_index{};
    std::uint64_t generation{};

    [[nodiscard]] bool matches(const SequenceStateView& state) const noexcept {
        return frame.has_value() && next_frame_index == state.frame_index &&
            generation == state.generation;
    }

    void clear() noexcept {
        frame.reset();
        feature.reset();
        next_frame_index = 0;
    }
};

const DeviceTensor* find_device_tensor(
    std::span<const DeviceTensor* const> tensors, std::string_view name) {
    for (const auto* tensor : tensors) {
        if (tensor != nullptr && tensor->name == name) return tensor;
    }
    return nullptr;
}

Result<DeviceTensor> upload_tensor(
    const HostTensor& input, std::string name, cudaStream_t stream) {
    const auto bytes = input.values.size() * sizeof(__half);
    auto storage = allocate_cuda(bytes);
    if (!storage) return storage.error();
    auto copied = copy_async(
        storage.value().data, input.values.data(), bytes, cudaMemcpyHostToDevice, stream,
        "cudaMemcpyAsync upload");
    if (!copied) return copied.error();
    return DeviceTensor{std::move(name), input.shape, std::move(storage.value())};
}

Result<DeviceTensor> upload_int8_tensor_scratch(
    std::span<const std::int8_t> values, nvinfer1::Dims shape, std::string name,
    cudaStream_t stream) {
    auto packed = allocate_cuda_scratch(values.size() * sizeof(std::int8_t));
    if (!packed) return packed.error();
    auto storage = allocate_cuda_scratch(values.size() * sizeof(__half));
    if (!storage) return storage.error();
    auto copied = copy_async(
        packed.value().data, values.data(), values.size() * sizeof(std::int8_t),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync packed symbol upload");
    if (!copied) return copied.error();
    const auto converted = cuda_ops::int8_to_half(
        static_cast<const std::int8_t*>(packed.value().data), storage.value().data,
        values.size(), stream);
    if (converted != cudaSuccess) return cuda_error("cuda_ops::int8_to_half", converted);
    return DeviceTensor{std::move(name), shape, std::move(storage.value())};
}

Result<DeviceTensor> allocate_device_tensor(
    std::string name, nvinfer1::Dims shape) {
    auto bytes = tensor_bytes(shape);
    if (!bytes) return bytes.error();
    auto storage = allocate_cuda(bytes.value());
    if (!storage) return storage.error();
    return DeviceTensor{std::move(name), shape, std::move(storage.value())};
}

Result<DeviceTensor> allocate_device_tensor_scratch(
    std::string name, nvinfer1::Dims shape) {
    auto bytes = tensor_bytes(shape);
    if (!bytes) return bytes.error();
    auto storage = allocate_cuda_scratch(bytes.value());
    if (!storage) return storage.error();
    return DeviceTensor{std::move(name), shape, std::move(storage.value())};
}

Result<void> populate_image_quant_cache(
    const RuntimeAssets& assets, ImageQuantDeviceCache& cache, cudaStream_t stream) {
    for (std::uint32_t qp = 0; qp < image_qp_count; ++qp) {
        auto q_enc = upload_tensor(
            make_quant_tensor(assets, true, qp, "q_enc"), "q_enc", stream);
        if (!q_enc) return q_enc.error();
        cache.q_enc[qp] = std::move(q_enc.value());

        auto q_dec = upload_tensor(
            make_quant_tensor(assets, false, qp, "q_dec"), "q_dec", stream);
        if (!q_dec) return q_dec.error();
        cache.q_dec[qp] = std::move(q_dec.value());
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize image quant cache");
    if (!synchronized) return synchronized.error();
    return {};
}

Result<void> populate_video_quant_cache(
    const RuntimeAssets& assets, VideoQuantDeviceCache& cache, cudaStream_t stream) {
    for (std::uint32_t qp = 0; qp < video_qp_count; ++qp) {
        auto q_encoder = upload_tensor(
            make_video_quant_tensor(assets, "q_encoder", qp), "q_encoder", stream);
        if (!q_encoder) return q_encoder.error();
        cache.q_encoder[qp] = std::move(q_encoder.value());

        auto q_decoder = upload_tensor(
            make_video_quant_tensor(assets, "q_decoder", qp), "q_decoder", stream);
        if (!q_decoder) return q_decoder.error();
        cache.q_decoder[qp] = std::move(q_decoder.value());

        auto q_feature = upload_tensor(
            make_video_quant_tensor(assets, "q_feature", qp), "q_feature", stream);
        if (!q_feature) return q_feature.error();
        cache.q_feature[qp] = std::move(q_feature.value());

        auto q_recon = upload_tensor(
            make_video_quant_tensor(assets, "q_recon", qp), "q_recon", stream);
        if (!q_recon) return q_recon.error();
        cache.q_recon[qp] = std::move(q_recon.value());
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize quant cache");
    if (!synchronized) return synchronized.error();
    return {};
}

Result<std::vector<DeviceTensor>> run_device_engine(
    EngineInstance& instance,
    const EngineSpec& specification,
    std::span<const DeviceTensor* const> inputs,
    cudaStream_t stream,
    EngineOutputStorage output_storage = EngineOutputStorage::owned) {
    auto context_result = acquire_context(instance);
    if (!context_result) return context_result.error();
    auto* context = context_result.value();
    const auto cpu_start = std::chrono::steady_clock::now();
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
            const auto* input = find_device_tensor(inputs, tensor.name);
            if (!context->setTensorAddress(name.c_str(), input->storage.data)) {
                return backend_error(std::string(specification.filename) +
                                     " rejected device address for " + name);
            }
            continue;
        }
        const auto shape = context->getTensorShape(name.c_str());
        auto output = output_storage == EngineOutputStorage::scratch ?
            allocate_device_tensor_scratch(name, shape) : allocate_device_tensor(name, shape);
        if (!output) return output.error();
        if (!context->setTensorAddress(name.c_str(), output.value().storage.data)) {
            return backend_error(std::string(specification.filename) +
                                 " rejected device output address for " + name);
        }
        outputs.push_back(std::move(output.value()));
    }
    std::pair<cudaEvent_t, cudaEvent_t> events{};
    bool profile_cuda = active_profiler != nullptr && active_profiler->enabled();
    if (profile_cuda) {
        auto created = create_profile_events("cudaEventCreate engine profile");
        if (!created) return created.error();
        events = created.value();
        auto recorded = record_cuda_event(events.first, stream, "cudaEventRecord engine start");
        if (!recorded) {
            static_cast<void>(cudaEventDestroy(events.first));
            static_cast<void>(cudaEventDestroy(events.second));
            return recorded.error();
        }
    }
    if (!context->enqueueV3(stream)) {
        if (profile_cuda) {
            static_cast<void>(cudaEventDestroy(events.first));
            static_cast<void>(cudaEventDestroy(events.second));
        }
        return backend_error(std::string(specification.filename) + " enqueueV3 failed");
    }
    if (profile_cuda) {
        auto recorded = record_cuda_event(events.second, stream, "cudaEventRecord engine stop");
        if (!recorded) {
            static_cast<void>(cudaEventDestroy(events.first));
            static_cast<void>(cudaEventDestroy(events.second));
            return recorded.error();
        }
        active_profiler->record_cuda_stage(
            std::string(specification.filename) + " enqueue", events.first, events.second);
    }
    if (instance.low_memory_mode) {
        auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize device engine");
        if (!synchronized) return synchronized.error();
        instance.context.reset();
    }
    if (active_profiler != nullptr && active_profiler->enabled()) {
        const auto cpu_elapsed = std::chrono::steady_clock::now() - cpu_start;
        active_profiler->record_cpu_stage(
            std::string(specification.filename) + " setup",
            std::chrono::duration<double, std::milli>(cpu_elapsed).count());
    }
    return outputs;
}

Result<DeviceTensor> take_device_tensor(
    std::vector<DeviceTensor>& tensors, std::string_view name) {
    for (auto& tensor : tensors) {
        if (tensor.name == name) return std::move(tensor);
    }
    return backend_error("TensorRT device output is missing " + std::string(name));
}

Result<std::vector<HostTensor>> download_tensors(
    std::span<const DeviceTensor* const> inputs, cudaStream_t stream) {
    std::vector<HostTensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto* input : inputs) {
        if (input == nullptr) return backend_error("null device tensor download");
        outputs.push_back(HostTensor{input->name, input->shape,
                                     std::vector<__half>(element_count(input->shape))});
        auto copied = copy_async(
            outputs.back().values.data(), input->storage.data,
            outputs.back().values.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream,
            "cudaMemcpyAsync download");
        if (!copied) return copied.error();
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize");
    if (!synchronized) return synchronized.error();
    return outputs;
}

__half to_half(float value) { return __float2half_rn(value); }
float to_float(__half value) { return __half2float(value); }

std::int32_t round_up(std::int32_t value, std::int32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

std::size_t tensor_offset(
    const HostTensor& tensor, std::int32_t channel, std::int32_t y, std::int32_t x) {
    return (static_cast<std::size_t>(channel) * tensor.shape.d[2] + y) *
        tensor.shape.d[3] + x;
}

Result<Frame> ycbcr_to_yuv420p8(
    const HostTensor& tensor, std::uint32_t width, std::uint32_t height, Timestamp timestamp) {
    if ((width & 1U) != 0U || (height & 1U) != 0U) {
        return Error(
            ErrorCode::invalid_argument,
            "YUV420 reconstruction requires even dimensions",
            std::string(subsystem));
    }

    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto uv_width = width / 2U;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (height / 2U);
    std::vector<std::byte> bytes(y_size + 2U * uv_size);
    auto* y_plane = bytes.data();
    auto* u_plane = bytes.data() + y_size;
    auto* v_plane = bytes.data() + y_size + uv_size;

    const auto quantize_y = [](float value) {
        return static_cast<std::byte>(static_cast<unsigned char>(
            std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F)));
    };
    const auto quantize_uv = [](float value) {
        return static_cast<std::byte>(static_cast<unsigned char>(
            std::clamp(value * 255.0F, 0.0F, 255.0F)));
    };

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            y_plane[static_cast<std::size_t>(y) * width + x] =
                quantize_y(to_float(tensor.values[tensor_offset(tensor, 0, y, x)]));
        }
    }
    for (std::uint32_t y = 0; y < height; y += 2U) {
        for (std::uint32_t x = 0; x < width; x += 2U) {
            float u_sum = 0.0F;
            float v_sum = 0.0F;
            for (std::uint32_t dy = 0; dy < 2U; ++dy) {
                for (std::uint32_t dx = 0; dx < 2U; ++dx) {
                    u_sum += to_float(tensor.values[tensor_offset(tensor, 1, y + dy, x + dx)]);
                    v_sum += to_float(tensor.values[tensor_offset(tensor, 2, y + dy, x + dx)]);
                }
            }
            const auto chroma = static_cast<std::size_t>(y / 2U) * uv_width + x / 2U;
            u_plane[chroma] = quantize_uv(u_sum * 0.25F);
            v_plane[chroma] = quantize_uv(v_sum * 0.25F);
        }
    }
    return Frame::copy_from(width, height, PixelFormat::yuv420p8, bytes, timestamp);
}

class AssetReader final {
public:
    AssetReader(fs::path path, std::span<const std::byte> bytes)
        : path_(std::move(path)), bytes_(bytes) {}

    Result<void> expect_magic(std::string_view magic) {
        if (remaining() < magic.size()) return truncated();
        for (std::size_t index = 0; index < magic.size(); ++index) {
            if (std::to_integer<unsigned char>(bytes_[offset_ + index]) !=
                static_cast<unsigned char>(magic[index])) {
                return backend_error("invalid asset magic in " + path_.string());
            }
        }
        offset_ += magic.size();
        return {};
    }

    Result<std::uint32_t> read_u32() {
        if (remaining() < 4) return truncated();
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < 4; ++index) {
            value |= static_cast<std::uint32_t>(
                std::to_integer<unsigned char>(bytes_[offset_ + index])) << (8U * index);
        }
        offset_ += 4;
        return value;
    }

    Result<std::int32_t> read_i32() {
        auto value = read_u32();
        if (!value) return value.error();
        return std::bit_cast<std::int32_t>(value.value());
    }

    Result<float> read_f32() {
        auto value = read_u32();
        if (!value) return value.error();
        return std::bit_cast<float>(value.value());
    }

    Result<std::string> read_string(std::size_t size) {
        if (remaining() < size) return truncated();
        std::string output(size, char{});
        for (std::size_t index = 0; index < size; ++index) {
            output[index] = static_cast<char>(
                std::to_integer<unsigned char>(bytes_[offset_ + index]));
        }
        offset_ += size;
        return output;
    }

    [[nodiscard]] bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
    Error truncated() const {
        return backend_error("truncated asset file: " + path_.string());
    }

    fs::path path_;
    std::span<const std::byte> bytes_;
    std::size_t offset_{};
};

struct RuntimeAssets final {
    std::size_t image_z_group{std::numeric_limits<std::size_t>::max()};
    std::size_t gaussian_y_group{std::numeric_limits<std::size_t>::max()};
    std::size_t video_z_group{std::numeric_limits<std::size_t>::max()};
    std::size_t video_gaussian_y_group{std::numeric_limits<std::size_t>::max()};
    std::uint32_t qp_count{};
    std::uint32_t quant_channels{};
    std::vector<float> q_enc;
    std::vector<float> q_dec;
    std::vector<float> p_q_encoder;
    std::vector<float> p_q_decoder;
    std::vector<float> p_q_feature;
    std::vector<float> p_q_recon;
};

Result<RuntimeAssets> load_runtime_assets(const fs::path& root, RansCodec& rans) {
    const auto entropy_path = root / "i_entropy.bin";
    auto entropy_bytes = read_binary(entropy_path);
    if (!entropy_bytes) return entropy_bytes.error();
    AssetReader entropy_reader(entropy_path, entropy_bytes.value());
    auto entropy_magic = entropy_reader.expect_magic("NVCRENT1");
    if (!entropy_magic) return entropy_magic.error();
    auto group_count = entropy_reader.read_u32();
    if (!group_count) return group_count.error();
    if (group_count.value() != 2) {
        return backend_error("I-frame entropy asset must contain exactly two CDF groups");
    }

    RuntimeAssets assets;
    for (std::uint32_t group_index = 0; group_index < group_count.value(); ++group_index) {
        auto name_size = entropy_reader.read_u32();
        auto row_count = entropy_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!row_count) return row_count.error();
        if (name_size.value() == 0 || name_size.value() > 64 ||
            row_count.value() == 0 || row_count.value() > 16384) {
            return backend_error("invalid CDF group dimensions in " + entropy_path.string());
        }
        auto name = entropy_reader.read_string(name_size.value());
        if (!name) return name.error();

        RansCdfTable table;
        table.values.reserve(row_count.value());
        table.sizes.reserve(row_count.value());
        table.offsets.reserve(row_count.value());
        for (std::uint32_t row_index = 0; row_index < row_count.value(); ++row_index) {
            auto size = entropy_reader.read_i32();
            auto offset = entropy_reader.read_i32();
            if (!size) return size.error();
            if (!offset) return offset.error();
            if (size.value() < 2 || size.value() > 1024) {
                return backend_error("invalid CDF row size in " + entropy_path.string());
            }
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(size.value()));
            for (std::int32_t value_index = 0; value_index < size.value(); ++value_index) {
                auto value = entropy_reader.read_i32();
                if (!value) return value.error();
                values.push_back(value.value());
            }
            table.values.push_back(std::move(values));
            table.sizes.push_back(size.value());
            table.offsets.push_back(offset.value());
        }
        auto registered = rans.add_cdf(std::move(table));
        if (!registered) return registered.error();
        if (name.value() == "image_z") {
            assets.image_z_group = registered.value();
        } else if (name.value() == "gaussian_y") {
            assets.gaussian_y_group = registered.value();
        } else {
            return backend_error("unknown CDF group " + name.value());
        }
    }
    if (!entropy_reader.done() ||
        assets.image_z_group == std::numeric_limits<std::size_t>::max() ||
        assets.gaussian_y_group == std::numeric_limits<std::size_t>::max()) {
        return backend_error("incomplete I-frame entropy asset");
    }

    const auto quant_path = root / "i_quant.bin";
    auto quant_bytes = read_binary(quant_path);
    if (!quant_bytes) return quant_bytes.error();
    AssetReader quant_reader(quant_path, quant_bytes.value());
    auto quant_magic = quant_reader.expect_magic("NVCRQNT1");
    if (!quant_magic) return quant_magic.error();
    auto qp_count = quant_reader.read_u32();
    auto channels = quant_reader.read_u32();
    if (!qp_count) return qp_count.error();
    if (!channels) return channels.error();
    if (qp_count.value() != 64 || channels.value() != 368) {
        return backend_error("unexpected I-frame quantization tensor shape");
    }
    assets.qp_count = qp_count.value();
    assets.quant_channels = channels.value();
    const auto value_count = static_cast<std::size_t>(qp_count.value()) * channels.value();
    assets.q_enc.reserve(value_count);
    assets.q_dec.reserve(value_count);
    for (auto* destination : {&assets.q_enc, &assets.q_dec}) {
        for (std::size_t index = 0; index < value_count; ++index) {
            auto value = quant_reader.read_f32();
            if (!value) return value.error();
            if (!std::isfinite(value.value())) {
                return backend_error("invalid quantization value in " + quant_path.string());
            }
            destination->push_back(value.value());
        }
    }
    if (!quant_reader.done()) return backend_error("trailing data in " + quant_path.string());

    const auto p_entropy_path = root / "p_entropy.bin";
    auto p_entropy_bytes = read_binary(p_entropy_path);
    if (!p_entropy_bytes) return p_entropy_bytes.error();
    AssetReader p_entropy_reader(p_entropy_path, p_entropy_bytes.value());
    auto p_entropy_magic = p_entropy_reader.expect_magic("NVCRPEN1");
    if (!p_entropy_magic) return p_entropy_magic.error();
    auto p_group_count = p_entropy_reader.read_u32();
    if (!p_group_count) return p_group_count.error();
    if (p_group_count.value() != 2) {
        return backend_error("P-frame entropy asset must contain exactly two CDF groups");
    }
    for (std::uint32_t group_index = 0; group_index < p_group_count.value(); ++group_index) {
        auto name_size = p_entropy_reader.read_u32();
        auto row_count = p_entropy_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!row_count) return row_count.error();
        if (name_size.value() == 0 || name_size.value() > 64 ||
            row_count.value() == 0 || row_count.value() > 16384) {
            return backend_error("invalid P-frame CDF group dimensions");
        }
        auto name = p_entropy_reader.read_string(name_size.value());
        if (!name) return name.error();
        RansCdfTable table;
        table.values.reserve(row_count.value());
        table.sizes.reserve(row_count.value());
        table.offsets.reserve(row_count.value());
        for (std::uint32_t row = 0; row < row_count.value(); ++row) {
            auto size = p_entropy_reader.read_i32();
            auto offset = p_entropy_reader.read_i32();
            if (!size) return size.error();
            if (!offset) return offset.error();
            if (size.value() < 2 || size.value() > 1024) {
                return backend_error("invalid P-frame CDF row size");
            }
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(size.value()));
            for (std::int32_t value_index = 0; value_index < size.value(); ++value_index) {
                auto value = p_entropy_reader.read_i32();
                if (!value) return value.error();
                values.push_back(value.value());
            }
            table.values.push_back(std::move(values));
            table.sizes.push_back(size.value());
            table.offsets.push_back(offset.value());
        }
        auto registered = rans.add_cdf(std::move(table));
        if (!registered) return registered.error();
        if (name.value() == "video_z") {
            assets.video_z_group = registered.value();
        } else if (name.value() == "gaussian_y") {
            assets.video_gaussian_y_group = registered.value();
        } else {
            return backend_error("unknown P-frame CDF group " + name.value());
        }
    }
    if (!p_entropy_reader.done() ||
        assets.video_z_group == std::numeric_limits<std::size_t>::max() ||
        assets.video_gaussian_y_group == std::numeric_limits<std::size_t>::max()) {
        return backend_error("incomplete P-frame entropy asset");
    }

    const auto p_quant_path = root / "p_quant.bin";
    auto p_quant_bytes = read_binary(p_quant_path);
    if (!p_quant_bytes) return p_quant_bytes.error();
    AssetReader p_quant_reader(p_quant_path, p_quant_bytes.value());
    auto p_quant_magic = p_quant_reader.expect_magic("NVCRPQN1");
    if (!p_quant_magic) return p_quant_magic.error();
    auto array_count = p_quant_reader.read_u32();
    if (!array_count) return array_count.error();
    if (array_count.value() != 4) return backend_error("P-frame quant asset must contain four arrays");
    for (std::uint32_t array_index = 0; array_index < array_count.value(); ++array_index) {
        auto name_size = p_quant_reader.read_u32();
        auto qps = p_quant_reader.read_u32();
        auto p_channels = p_quant_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!qps) return qps.error();
        if (!p_channels) return p_channels.error();
        auto name = p_quant_reader.read_string(name_size.value());
        if (!name) return name.error();
        const std::uint32_t expected_channels = name.value() == "q_recon" ? 320U : 256U;
        if (qps.value() != 72 || p_channels.value() != expected_channels) {
            return backend_error("unexpected P-frame quantization tensor shape for " + name.value());
        }
        std::vector<float>* destination = nullptr;
        if (name.value() == "q_encoder") destination = &assets.p_q_encoder;
        else if (name.value() == "q_decoder") destination = &assets.p_q_decoder;
        else if (name.value() == "q_feature") destination = &assets.p_q_feature;
        else if (name.value() == "q_recon") destination = &assets.p_q_recon;
        else return backend_error("unknown P-frame quant array " + name.value());
        destination->reserve(static_cast<std::size_t>(qps.value()) * p_channels.value());
        for (std::size_t index = 0; index < static_cast<std::size_t>(qps.value()) * p_channels.value(); ++index) {
            auto value = p_quant_reader.read_f32();
            if (!value) return value.error();
            if (!std::isfinite(value.value())) return backend_error("invalid P-frame quantization value");
            destination->push_back(value.value());
        }
    }
    if (!p_quant_reader.done() || assets.p_q_encoder.empty() || assets.p_q_decoder.empty() ||
        assets.p_q_feature.empty() || assets.p_q_recon.empty()) {
        return backend_error("incomplete P-frame quantization asset");
    }
    return assets;
}

HostTensor make_quant_tensor(
    const RuntimeAssets& assets, bool encoder, std::uint32_t qp, std::string name) {
    HostTensor output{std::move(name), make_dims(368, 1, 1), std::vector<__half>(368)};
    const auto& source = encoder ? assets.q_enc : assets.q_dec;
    const auto first = static_cast<std::size_t>(qp) * 368;
    for (std::size_t index = 0; index < 368; ++index) {
        output.values[index] = to_half(source[first + index]);
    }
    return output;
}

HostTensor make_video_quant_tensor(
    const RuntimeAssets& assets, std::string_view kind, std::uint32_t qp) {
    const std::vector<float>* source = nullptr;
    std::int32_t channels = 256;
    if (kind == "q_encoder") source = &assets.p_q_encoder;
    else if (kind == "q_decoder") source = &assets.p_q_decoder;
    else if (kind == "q_feature") source = &assets.p_q_feature;
    else { source = &assets.p_q_recon; channels = 320; }
    HostTensor output{std::string(kind), make_dims(channels, 1, 1),
                      std::vector<__half>(static_cast<std::size_t>(channels))};
    const auto first = static_cast<std::size_t>(qp) * channels;
    for (std::int32_t index = 0; index < channels; ++index) {
        output.values[static_cast<std::size_t>(index)] =
            to_half((*source)[first + static_cast<std::size_t>(index)]);
    }
    return output;
}

Result<std::uint32_t> video_qp(std::uint32_t base_qp, std::uint64_t frame_index) {
    constexpr std::array<std::uint32_t, 8> index_map{0, 1, 0, 2, 0, 2, 0, 2};
    constexpr std::array<std::uint32_t, 3> shifts{0, 8, 4};
    if (base_qp >= 64U) return backend_error("base P-frame QP must be in [0, 63]");
    const auto effective_qp = base_qp + shifts[index_map[frame_index % index_map.size()]];
    if (effective_qp >= video_qp_count) {
        return backend_error("effective P-frame QP exceeds the 72-entry model table");
    }
    return effective_qp;
}

Result<HostTensor> take_tensor(std::vector<HostTensor>& tensors, std::string_view name) {
    for (auto& tensor : tensors) {
        if (tensor.name == name) return std::move(tensor);
    }
    return backend_error("TensorRT output is missing " + std::string(name));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

Result<std::uint32_t> read_u32(
    std::span<const std::byte> input, std::size_t& offset) {
    if (offset > input.size() || input.size() - offset < 4) {
        return Error(ErrorCode::malformed_bitstream, "truncated I-frame header",
                     std::string(subsystem));
    }
    std::uint32_t value = 0;
    for (std::uint32_t index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(
            std::to_integer<unsigned char>(input[offset + index])) << (index * 8U);
    }
    offset += 4;
    return value;
}

struct DpbState final {
    HostTensor frame;
    std::optional<HostTensor> feature;
};

std::vector<std::byte> serialize_dpb(
    const HostTensor& frame, const HostTensor* feature = nullptr) {
    std::vector<std::byte> output;
    const auto frame_bytes = frame.values.size() * sizeof(__half);
    const auto feature_bytes = feature == nullptr ? 0 : feature->values.size() * sizeof(__half);
    output.reserve(24 + frame_bytes + feature_bytes);
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x44}, std::byte{0x31}});
    append_u32(output, static_cast<std::uint32_t>(frame.shape.d[2]));
    append_u32(output, static_cast<std::uint32_t>(frame.shape.d[3]));
    append_u32(output, feature == nullptr ? 0U : static_cast<std::uint32_t>(feature->shape.d[2]));
    append_u32(output, feature == nullptr ? 0U : static_cast<std::uint32_t>(feature->shape.d[3]));
    const auto* frame_data = reinterpret_cast<const std::byte*>(frame.values.data());
    output.insert(output.end(), frame_data, frame_data + frame_bytes);
    if (feature != nullptr) {
        const auto* feature_data = reinterpret_cast<const std::byte*>(feature->values.data());
        output.insert(output.end(), feature_data, feature_data + feature_bytes);
    }
    return output;
}

struct IntraPayload final {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t qp;
    bool two_coders;
    std::span<const std::byte> rans;
};

std::vector<std::byte> make_intra_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, std::span<const std::byte> rans) {
    std::vector<std::byte> output;
    output.reserve(20 + rans.size());
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x49}, std::byte{0x31}});
    append_u32(output, width);
    append_u32(output, height);
    append_u32(output, qp);
    append_u32(output, two_coders ? 1U : 0U);
    output.insert(output.end(), rans.begin(), rans.end());
    return output;
}

Result<IntraPayload> parse_intra_payload(std::span<const std::byte> payload) {
    constexpr std::array<std::byte, 4> magic{
        std::byte{0x4e}, std::byte{0x56}, std::byte{0x49}, std::byte{0x31}};
    if (payload.size() < 24 || !std::equal(magic.begin(), magic.end(), payload.begin())) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR I-frame payload",
                     std::string(subsystem));
    }
    std::size_t offset = 4;
    auto width = read_u32(payload, offset);
    auto height = read_u32(payload, offset);
    auto qp = read_u32(payload, offset);
    auto flags = read_u32(payload, offset);
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!flags) return flags.error();
    if (!supported_frame_dimensions(width.value(), height.value()) ||
        qp.value() >= 64 || flags.value() > 1) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR I-frame fields",
                     std::string(subsystem));
    }
    return IntraPayload{width.value(), height.value(), qp.value(), flags.value() == 1,
                        payload.subspan(offset)};
}

struct PredictedPayload final {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t qp;
    bool two_coders;
    bool use_frame_reference;
    std::span<const std::byte> rans;
};

std::vector<std::byte> make_predicted_payload(
    std::uint32_t width, std::uint32_t height, std::uint32_t qp,
    bool two_coders, bool use_frame_reference, std::span<const std::byte> rans) {
    std::vector<std::byte> output;
    output.reserve(20 + rans.size());
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x50}, std::byte{0x31}});
    append_u32(output, width);
    append_u32(output, height);
    append_u32(output, qp);
    append_u32(output, (two_coders ? 1U : 0U) | (use_frame_reference ? 2U : 0U));
    output.insert(output.end(), rans.begin(), rans.end());
    return output;
}

Result<PredictedPayload> parse_predicted_payload(std::span<const std::byte> payload) {
    constexpr std::array<std::byte, 4> magic{
        std::byte{0x4e}, std::byte{0x56}, std::byte{0x50}, std::byte{0x31}};
    if (payload.size() < 24 || !std::equal(magic.begin(), magic.end(), payload.begin())) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR P-frame payload",
                     std::string(subsystem));
    }
    std::size_t offset = 4;
    auto width = read_u32(payload, offset);
    auto height = read_u32(payload, offset);
    auto qp = read_u32(payload, offset);
    auto flags = read_u32(payload, offset);
    if (!width) return width.error();
    if (!height) return height.error();
    if (!qp) return qp.error();
    if (!flags) return flags.error();
    if (!supported_frame_dimensions(width.value(), height.value()) ||
        qp.value() >= video_qp_count || flags.value() > 3) {
        return Error(ErrorCode::malformed_bitstream, "invalid NVCR P-frame fields",
                     std::string(subsystem));
    }
    return PredictedPayload{width.value(), height.value(), qp.value(),
                            (flags.value() & 1U) != 0, (flags.value() & 2U) != 0,
                            payload.subspan(offset)};
}

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    std::vector<EngineInstance>& engines,
    RansCodec& rans,
    const RuntimeAssets& assets,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    cudaStream_t stream);

Result<CodecEncodeResult> encode_intra(
    const Frame& frame,
    std::uint32_t qp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    std::vector<EngineInstance>& engines,
    RansCodec& rans,
    const RuntimeAssets& assets,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    PinnedHostBuffer<std::int8_t>& image_z_symbols_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes0_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes1_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes2_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes3_buffer,
    bool verify_reconstruction,
    cudaStream_t stream) {
    if (frame.pixel_format() != PixelFormat::rgb24 &&
        frame.pixel_format() != PixelFormat::yuv420p8) {
        return Error(ErrorCode::invalid_argument,
                     "native I-frame encoding requires RGB24 or YUV420P8",
                     std::string(subsystem));
    }
    if (!supported_frame_dimensions(frame.width(), frame.height())) {
        return Error(ErrorCode::invalid_argument,
                     "native I-frame dimensions must be within 64x64 and 1920x1080",
                     std::string(subsystem));
    }

    const auto padded_height = round_up(static_cast<std::int32_t>(frame.height()), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(frame.width()), 16);
    auto input_frame = allocate_device_tensor_scratch("frame", make_dims(3, padded_height, padded_width));
    if (!input_frame) return input_frame.error();
    auto input_bytes_device = allocate_cuda_scratch(frame.data().size());
    if (!input_bytes_device) return input_bytes_device.error();
    auto copied = copy_async(
        input_bytes_device.value().data, frame.data().data(), frame.data().size(),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync I-frame input bytes");
    if (!copied) return copied.error();
    {
        CpuProfileScope stage("i_input_color_convert_gpu");
        cudaError_t converted = cudaSuccess;
        if (frame.pixel_format() == PixelFormat::yuv420p8) {
            converted = cuda_ops::yuv420p8_to_ycbcr_padded(
                input_bytes_device.value().data, static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()), input_frame.value().storage.data,
                padded_height, padded_width, stream);
        } else {
            converted = cuda_ops::rgb24_to_ycbcr_padded(
                input_bytes_device.value().data, static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()), input_frame.value().storage.data,
                padded_height, padded_width, stream);
        }
        if (converted != cudaSuccess) return cuda_error("cuda_ops I-frame input color convert", converted);
    }

    auto& q_enc_opt = image_quant_cache.q_enc[qp];
    if (!q_enc_opt.has_value()) return backend_error("missing cached image q_enc tensor");
    auto* q_enc = &q_enc_opt.value();
    std::array<DeviceTensor*, 2> analysis_inputs{&input_frame.value(), q_enc};
    auto analysis_outputs = run_device_engine(
        engines[0], engine_specs[0], analysis_inputs, stream, EngineOutputStorage::scratch);
    if (!analysis_outputs) return analysis_outputs.error();
    auto y_result = take_device_tensor(analysis_outputs.value(), "y");
    if (!y_result) return y_result.error();
    auto y_device = std::move(y_result.value());
    const auto y_height = static_cast<std::int32_t>(y_device.shape.d[2]);
    const auto y_width = static_cast<std::int32_t>(y_device.shape.d[3]);

    auto y_padded = allocate_device_tensor_scratch(
        "y_padded", make_dims(256, round_up(y_height, 4), round_up(y_width, 4)));
    if (!y_padded) return y_padded.error();
    const cuda_ops::Shape4D y_shape{1, 256, y_height, y_width};
    auto status = cuda_ops::replicate_pad(
        y_device.storage.data, y_padded.value().storage.data, y_shape,
        static_cast<std::int32_t>(y_padded.value().shape.d[2]),
        static_cast<std::int32_t>(y_padded.value().shape.d[3]), stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::replicate_pad", status);

    std::array<DeviceTensor*, 1> hyper_analysis_inputs{&y_padded.value()};
    auto hyper_analysis_outputs = run_device_engine(
        engines[1], engine_specs[1], hyper_analysis_inputs, stream, EngineOutputStorage::scratch);
    if (!hyper_analysis_outputs) return hyper_analysis_outputs.error();
    auto z_result = take_device_tensor(hyper_analysis_outputs.value(), "z");
    if (!z_result) return z_result.error();
    auto z_device = std::move(z_result.value());
    z_device.name = "z_hat";
    const auto z_count = element_count(z_device.shape);
    auto z_symbol_device = allocate_cuda_scratch(z_count * sizeof(std::int8_t));
    if (!z_symbol_device) return z_symbol_device.error();
    status = cuda_ops::round_to_int8(
        z_device.storage.data, static_cast<std::int8_t*>(z_symbol_device.value().data),
        z_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::round_to_int8", status);

    std::array<DeviceTensor*, 1> hyper_synthesis_inputs{&z_device};
    auto hyper_synthesis_outputs = run_device_engine(
        engines[2], engine_specs[2], hyper_synthesis_inputs, stream, EngineOutputStorage::scratch);
    if (!hyper_synthesis_outputs) return hyper_synthesis_outputs.error();
    auto params_padded_result = take_device_tensor(hyper_synthesis_outputs.value(), "params_padded");
    auto common_padded_result = take_device_tensor(hyper_synthesis_outputs.value(), "common_padded");
    if (!params_padded_result) return params_padded_result.error();
    if (!common_padded_result) return common_padded_result.error();
    auto params_padded = std::move(params_padded_result.value());
    auto common_padded = std::move(common_padded_result.value());
    auto params = allocate_device_tensor_scratch("params", make_dims(514, y_height, y_width));
    auto common = allocate_device_tensor_scratch("common", make_dims(256, y_height, y_width));
    if (!params) return params.error();
    if (!common) return common.error();
    status = cuda_ops::crop_spatial(
        params_padded.storage.data, params.value().storage.data,
        cuda_ops::Shape4D{1, 514, static_cast<std::int32_t>(params_padded.shape.d[2]),
                          static_cast<std::int32_t>(params_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial params", status);
    status = cuda_ops::crop_spatial(
        common_padded.storage.data, common.value().storage.data,
        cuda_ops::Shape4D{1, 256, static_cast<std::int32_t>(common_padded.shape.d[2]),
                          static_cast<std::int32_t>(common_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial common", status);

    const auto spatial_count = static_cast<std::size_t>(y_height) * y_width;
    const auto prior_count = static_cast<std::size_t>(256) * spatial_count;
    status = cuda_ops::apply_image_prior_and_quantize(
        params.value().storage.data, y_device.storage.data, spatial_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::apply_image_prior_and_quantize", status);
    auto* params_values = static_cast<__half*>(params.value().storage.data);
    auto* q_dec_values = params_values + spatial_count;
    auto* current_scales = params_values + 2 * spatial_count;
    auto* current_means = params_values + (2 + 256) * spatial_count;

    const auto reduced_shape = make_dims(64, y_height, y_width);
    const auto reduced_count = prior_count / 4;
    std::array<std::optional<DeviceTensor>, 4> encoded_symbols;
    std::array<std::optional<DeviceTensor>, 4> encoded_scales;
    std::optional<DeviceTensor> y_hat_device;
    std::optional<DeviceTensor> spatial_params;
    for (std::int32_t stage_index = 0; stage_index < 4; ++stage_index) {
        const auto stage_slot = static_cast<std::size_t>(stage_index);
        auto mask = allocate_device_tensor_scratch("mask", y_device.shape);
        auto symbols_full = allocate_device_tensor_scratch("symbols_full", y_device.shape);
        auto reconstructed = allocate_device_tensor_scratch("reconstructed", y_device.shape);
        auto scales_full = allocate_device_tensor_scratch("scales_full", y_device.shape);
        if (!mask) return mask.error();
        if (!symbols_full) return symbols_full.error();
        if (!reconstructed) return reconstructed.error();
        if (!scales_full) return scales_full.error();
        status = cuda_ops::make_four_way_mask(mask.value().storage.data, y_shape, stage_index, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::make_four_way_mask", status);
        status = cuda_ops::process_with_mask(
            y_device.storage.data, current_scales, current_means, mask.value().storage.data,
            nullptr, symbols_full.value().storage.data, reconstructed.value().storage.data,
            scales_full.value().storage.data,
            prior_count, 0.0F, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);

        auto symbols = allocate_device_tensor_scratch("symbols", reduced_shape);
        auto scales = allocate_device_tensor_scratch("scales", reduced_shape);
        if (!symbols) return symbols.error();
        if (!scales) return scales.error();
        status = cuda_ops::reduce_channel_quarters(
            symbols_full.value().storage.data, symbols.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_quarters symbols", status);
        status = cuda_ops::reduce_channel_quarters(
            scales_full.value().storage.data, scales.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_quarters scales", status);
        encoded_symbols[stage_slot] = std::move(symbols.value());
        encoded_scales[stage_slot] = std::move(scales.value());

        if (!y_hat_device.has_value()) {
            reconstructed.value().name = "y_hat";
            y_hat_device = std::move(reconstructed.value());
        } else {
            status = cuda_ops::add_in_place(
                y_hat_device->storage.data, reconstructed.value().storage.data,
                prior_count, stream);
            if (status != cudaSuccess) return cuda_error("cuda_ops::add_in_place", status);
        }

        if (stage_index < 3) {
            auto spatial_context = allocate_device_tensor_scratch("context", make_dims(512, y_height, y_width));
            if (!spatial_context) return spatial_context.error();
            const auto y_bytes = prior_count * sizeof(__half);
            copied = copy_async(
                spatial_context.value().storage.data, y_hat_device->storage.data, y_bytes,
                cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync I spatial y");
            if (!copied) return copied.error();
            copied = copy_async(
                static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
                common.value().storage.data, y_bytes, cudaMemcpyDeviceToDevice, stream,
                "cudaMemcpyAsync I spatial common");
            if (!copied) return copied.error();
            std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
            auto spatial_outputs = run_device_engine(
                engines[static_cast<std::size_t>(3 + stage_index)],
                engine_specs[static_cast<std::size_t>(3 + stage_index)], spatial_inputs, stream,
                EngineOutputStorage::scratch);
            if (!spatial_outputs) return spatial_outputs.error();
            auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
            if (!spatial_result) return spatial_result.error();
            spatial_params = std::move(spatial_result.value());
            auto* spatial_values = static_cast<__half*>(spatial_params->storage.data);
            current_scales = spatial_values;
            current_means = spatial_values + prior_count;
        }
    }

    status = cuda_ops::multiply_broadcast_in_place(
        y_hat_device->storage.data, q_dec_values, spatial_count, 256, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::multiply_broadcast_in_place", status);
    y_hat_device->name = "y_hat";

    std::array<CudaAllocation, 4> index_devices;
    for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
        auto allocation = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
        if (!allocation) return allocation.error();
        index_devices[stage_index] = std::move(allocation.value());
        status = cuda_ops::build_encode_indexes(
            encoded_symbols[stage_index]->storage.data, encoded_scales[stage_index]->storage.data,
            static_cast<std::int16_t*>(index_devices[stage_index].data), reduced_count,
            0.0F, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    }

    auto z_buffer_ready = image_z_symbols_buffer.ensure(z_count, "image_z_symbols");
    if (!z_buffer_ready) return z_buffer_ready.error();
    std::array<PinnedHostBuffer<std::int16_t>*, 4> index_buffers{
        &image_indexes0_buffer, &image_indexes1_buffer,
        &image_indexes2_buffer, &image_indexes3_buffer};
    for (auto* buffer : index_buffers) {
        auto ready = buffer->ensure(reduced_count, "image_indexes");
        if (!ready) return ready.error();
    }

    std::span<std::int8_t> z_symbols(image_z_symbols_buffer.data, z_count);
    copied = copy_async(
        z_symbols.data(), z_symbol_device.value().data, z_count * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync I z symbols");
    if (!copied) return copied.error();
    for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
        copied = copy_async(
            index_buffers[stage_index]->data, index_devices[stage_index].data,
            reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
            "cudaMemcpyAsync I indexes");
        if (!copied) return copied.error();
    }
    CudaEvent entropy_ready;
    status = cudaEventCreateWithFlags(&entropy_ready.data, cudaEventDisableTiming);
    if (status != cudaSuccess) return cuda_error("cudaEventCreateWithFlags", status);
    status = cudaEventRecord(entropy_ready.data, stream);
    if (status != cudaSuccess) return cuda_error("cudaEventRecord I entropy", status);

    status = cudaEventSynchronize(entropy_ready.data);
    if (status != cudaSuccess) return cuda_error("cudaEventSynchronize I entropy", status);

    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    rans.reset_encoder();
    auto mode = rans.set_use_two_coders(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z_device.shape.d[2] * z_device.shape.d[3]);
    auto encoded_z = rans.encode_z(
        z_symbols, assets.image_z_group, static_cast<std::size_t>(qp) * 128, per_channel);
    if (!encoded_z) return encoded_z.error();
    for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
        std::span<std::int16_t> indexes(index_buffers[stage_index]->data, reduced_count);
        auto encoded_y = rans.encode_y(indexes, assets.gaussian_y_group);
        if (!encoded_y) return encoded_y.error();
    }
    auto stream_bytes = rans.finish_encode();
    if (!stream_bytes) return stream_bytes.error();
    auto payload = make_intra_payload(
        frame.width(), frame.height(), qp, two_coders, stream_bytes.value());

    auto& q_dec_opt = image_quant_cache.q_dec[qp];
    if (!q_dec_opt.has_value()) return backend_error("missing cached image q_dec tensor");
    auto* q_dec = &q_dec_opt.value();
    std::array<DeviceTensor*, 2> synthesis_inputs{&*y_hat_device, q_dec};
    auto synthesis_outputs = run_device_engine(engines[6], engine_specs[6], synthesis_inputs, stream);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    if (!frame_device_result) return frame_device_result.error();
    auto frame_device = std::move(frame_device_result.value());

    std::array<const DeviceTensor*, 1> tensors{&frame_device};
    auto downloaded = download_tensors(tensors, stream);
    if (!downloaded) return downloaded.error();
    auto reconstructed_result = take_tensor(downloaded.value(), "frame_hat");
    if (!reconstructed_result) return reconstructed_result.error();
    auto reconstructed = ycbcr_to_yuv420p8(
        reconstructed_result.value(), frame.width(), frame.height(), frame.timestamp());
    if (!reconstructed) return reconstructed.error();

    frame_device.name = "reference_frame";
    device_dpb.clear();
    device_dpb.frame = std::move(frame_device);
    device_dpb.next_frame_index = state.frame_index + 1;
    device_dpb.generation = state.generation;
    std::vector<std::byte> latent_state;

    if (verify_reconstruction) {
        auto conformant_reconstruction = decode_intra(
            payload, frame.timestamp(), state, device_dpb, engines, rans, assets, image_quant_cache, image_decode_staging, stream);
        if (!conformant_reconstruction) return conformant_reconstruction.error();
        return CodecEncodeResult{
            std::move(payload), std::move(conformant_reconstruction.value().frame),
            std::move(conformant_reconstruction.value().latent_state), qp};
    }

    return CodecEncodeResult{
        std::move(payload), std::move(reconstructed.value()), std::move(latent_state), qp};
}

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    std::vector<EngineInstance>& engines,
    RansCodec& rans,
    const RuntimeAssets& assets,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    cudaStream_t stream) {
    auto parsed = parse_intra_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    const auto padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = padded_height / 16;
    const auto y_width = padded_width / 16;
    const auto z_height = round_up(y_height, 4) / 4;
    const auto z_width = round_up(y_width, 4) / 4;

    rans.reset_encoder();
    auto mode = rans.set_use_two_coders(info.two_coders);
    if (!mode) return mode.error();
    auto stream_set = rans.set_stream(info.rans);
    if (!stream_set) return stream_set.error();
    const auto per_channel = static_cast<std::size_t>(z_height * z_width);
    auto z_values = rans.decode_z(
        128 * per_channel, assets.image_z_group,
        static_cast<std::size_t>(info.qp) * 128, per_channel);
    if (!z_values) return z_values.error();
    const auto staging_reduced_count =
        static_cast<std::size_t>(64) * y_height * y_width;
    auto staging_ready = image_decode_staging.ensure(
        z_values.value().size(), staging_reduced_count);
    if (!staging_ready) return staging_ready.error();
    std::copy(
        z_values.value().begin(), z_values.value().end(),
        image_decode_staging.z_symbols.data);
    auto z_device = upload_int8_tensor_scratch(
        std::span<const std::int8_t>(
            image_decode_staging.z_symbols.data, z_values.value().size()),
        make_dims(128, z_height, z_width), "z_hat", stream);
    if (!z_device) return z_device.error();

    std::array<DeviceTensor*, 1> hyper_inputs{&z_device.value()};
    auto hyper_outputs = run_device_engine(
        engines[2], engine_specs[2], hyper_inputs, stream, EngineOutputStorage::scratch);
    if (!hyper_outputs) return hyper_outputs.error();
    auto params_padded_result = take_device_tensor(hyper_outputs.value(), "params_padded");
    auto common_padded_result = take_device_tensor(hyper_outputs.value(), "common_padded");
    if (!params_padded_result) return params_padded_result.error();
    if (!common_padded_result) return common_padded_result.error();
    auto params_padded = std::move(params_padded_result.value());
    auto common_padded = std::move(common_padded_result.value());
    auto params = allocate_device_tensor_scratch("params", make_dims(514, y_height, y_width));
    auto common = allocate_device_tensor_scratch("common", make_dims(256, y_height, y_width));
    if (!params) return params.error();
    if (!common) return common.error();
    auto status = cuda_ops::crop_spatial(
        params_padded.storage.data, params.value().storage.data,
        cuda_ops::Shape4D{1, 514, static_cast<std::int32_t>(params_padded.shape.d[2]),
                          static_cast<std::int32_t>(params_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial I params", status);
    status = cuda_ops::crop_spatial(
        common_padded.storage.data, common.value().storage.data,
        cuda_ops::Shape4D{1, 256, static_cast<std::int32_t>(common_padded.shape.d[2]),
                          static_cast<std::int32_t>(common_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial I common", status);

    const auto spatial_count = static_cast<std::size_t>(y_height) * y_width;
    const auto prior_count = static_cast<std::size_t>(256) * spatial_count;
    const auto reduced_shape = make_dims(64, y_height, y_width);
    const auto reduced_count = prior_count / 4;
    const cuda_ops::Shape4D y_shape{1, 256, y_height, y_width};
    status = cuda_ops::apply_image_prior(
        params.value().storage.data, spatial_count, stream);
    if (status != cudaSuccess) {
        return cuda_error("cuda_ops::apply_image_prior I decode", status);
    }
    auto* params_values = static_cast<__half*>(params.value().storage.data);
    auto* q_dec_values = params_values + spatial_count;
    auto* current_scales = params_values + 2 * spatial_count;
    auto* current_means = params_values + (2 + 256) * spatial_count;

    std::optional<DeviceTensor> y_hat_device;
    std::optional<DeviceTensor> spatial_params;
    for (std::int32_t stage = 0; stage < 4; ++stage) {
        auto mask = allocate_device_tensor_scratch("mask", make_dims(256, y_height, y_width));
        auto scales = allocate_device_tensor_scratch("scales", reduced_shape);
        if (!mask) return mask.error();
        if (!scales) return scales.error();
        status = cuda_ops::make_four_way_mask(mask.value().storage.data, y_shape, stage, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::make_four_way_mask I decode", status);
        status = cuda_ops::reduce_masked_quarters(
            current_scales, mask.value().storage.data, scales.value().storage.data,
            y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_quarters I decode scales", status);

        auto indexes_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
        if (!indexes_device) return indexes_device.error();
        status = cuda_ops::build_decode_indexes(
            scales.value().storage.data, static_cast<std::uint8_t*>(indexes_device.value().data),
            reduced_count, 0.0F, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes I", status);
        const auto stage_slot = static_cast<std::size_t>(stage);
        auto& indexes = image_decode_staging.indexes[stage_slot];
        auto copied = copy_async(
            indexes.data, indexes_device.value().data, reduced_count,
            cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync I decode indexes");
        if (!copied) return copied.error();
        auto recorded = record_cuda_event(
            image_decode_staging.indexes_ready[stage_slot].data, stream,
            "cudaEventRecord I decode indexes");
        if (!recorded) return recorded.error();
        auto synchronized = synchronize_event(
            image_decode_staging.indexes_ready[stage_slot].data,
            "cudaEventSynchronize I decode indexes");
        if (!synchronized) return synchronized.error();

        auto decoded = rans.decode_y(
            std::span<const std::uint8_t>(indexes.data, reduced_count),
            assets.gaussian_y_group);
        if (!decoded) return decoded.error();
        auto& symbols = image_decode_staging.symbols[stage_slot];
        std::copy(decoded.value().begin(), decoded.value().end(), symbols.data);
        auto symbols_device = upload_int8_tensor_scratch(
            std::span<const std::int8_t>(symbols.data, decoded.value().size()),
            reduced_shape, "symbols", stream);
        if (!symbols_device) return symbols_device.error();
        auto reconstructed = allocate_device_tensor_scratch("reconstructed", make_dims(256, y_height, y_width));
        if (!reconstructed) return reconstructed.error();
        status = cuda_ops::restore_four_way(
            symbols_device.value().storage.data, current_means, mask.value().storage.data,
            reconstructed.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::restore_four_way I", status);
        if (!y_hat_device.has_value()) {
            reconstructed.value().name = "y_hat";
            y_hat_device = std::move(reconstructed.value());
        } else {
            status = cuda_ops::add_in_place(
                y_hat_device->storage.data, reconstructed.value().storage.data,
                prior_count, stream);
            if (status != cudaSuccess) return cuda_error("cuda_ops::add_in_place I decode", status);
        }

        if (stage < 3) {
            auto spatial_context = allocate_device_tensor_scratch("context", make_dims(512, y_height, y_width));
            if (!spatial_context) return spatial_context.error();
            const auto y_bytes = prior_count * sizeof(__half);
            copied = copy_async(
                spatial_context.value().storage.data, y_hat_device->storage.data, y_bytes,
                cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync I decode spatial y");
            if (!copied) return copied.error();
            copied = copy_async(
                static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
                common.value().storage.data, y_bytes, cudaMemcpyDeviceToDevice, stream,
                "cudaMemcpyAsync I decode spatial common");
            if (!copied) return copied.error();
            std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
            const auto engine_index = static_cast<std::size_t>(3 + stage);
            auto spatial_outputs = run_device_engine(
                engines[engine_index], engine_specs[engine_index], spatial_inputs, stream,
                EngineOutputStorage::scratch);
            if (!spatial_outputs) return spatial_outputs.error();
            auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
            if (!spatial_result) return spatial_result.error();
            spatial_params = std::move(spatial_result.value());
            auto* spatial_values = static_cast<__half*>(spatial_params->storage.data);
            current_scales = spatial_values;
            current_means = spatial_values + prior_count;
        }
    }

    status = cuda_ops::multiply_broadcast_in_place(
        y_hat_device->storage.data, q_dec_values, spatial_count, 256, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::multiply_broadcast_in_place I decode", status);
    y_hat_device->name = "y_hat";
    auto& q_dec_opt = image_quant_cache.q_dec[info.qp];
    if (!q_dec_opt.has_value()) return backend_error("missing cached image q_dec tensor");
    auto* q_dec = &q_dec_opt.value();
    std::array<DeviceTensor*, 2> synthesis_inputs{&*y_hat_device, q_dec};
    auto synthesis_outputs = run_device_engine(
        engines[6], engine_specs[6], synthesis_inputs, stream);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    if (!frame_device_result) return frame_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    std::array<const DeviceTensor*, 1> reconstruction_inputs{&frame_device};
    auto reconstruction = download_tensors(reconstruction_inputs, stream);
    if (!reconstruction) return reconstruction.error();
    auto frame_result = take_tensor(reconstruction.value(), "frame_hat");
    if (!frame_result) return frame_result.error();
    auto frame = ycbcr_to_yuv420p8(frame_result.value(), info.width, info.height, timestamp);
    if (!frame) return frame.error();
    frame_device.name = "reference_frame";
    device_dpb.clear();
    device_dpb.frame = std::move(frame_device);
    device_dpb.next_frame_index = state.frame_index + 1;
    device_dpb.generation = state.generation;
    auto latent_state = serialize_dpb(frame_result.value());
    return CodecDecodeResult{std::move(frame.value()), std::move(latent_state)};
}

Result<CodecEncodeResult> encode_predicted(
    const Frame& frame, std::uint32_t base_qp, const SequenceStateView& state,
    DeviceDpb& device_dpb,
    std::vector<EngineInstance>& engines, RansCodec& rans,
    const RuntimeAssets& assets, VideoQuantDeviceCache& quant_cache,
    PinnedHostBuffer<std::int8_t>& z_symbols_buffer,
    PinnedHostBuffer<std::int16_t>& indexes0_buffer,
    PinnedHostBuffer<std::int16_t>& indexes1_buffer,
    bool verify_reconstruction, cudaStream_t stream) {
    if (frame.pixel_format() != PixelFormat::rgb24 &&
        frame.pixel_format() != PixelFormat::yuv420p8) {
        return Error(ErrorCode::invalid_argument, "native P-frame encoding requires RGB24 or YUV420P8",
                     std::string(subsystem));
    }
    if (!supported_frame_dimensions(frame.width(), frame.height())) {
        return Error(ErrorCode::invalid_argument,
                     "native P-frame dimensions must be even and within 64x64 and 1920x1080",
                     std::string(subsystem));
    }
    if (!device_dpb.matches(state)) {
        return Error(ErrorCode::invalid_state, "encoder device DPB is unavailable",
                     std::string(subsystem));
    }
    const bool use_frame_reference = !device_dpb.feature || (state.frame_index % 32U) == 1U;
    auto qp_result = video_qp(base_qp, state.frame_index);
    if (!qp_result) return qp_result.error();
    const auto qp = qp_result.value();

    const std::string reference_name = use_frame_reference ? "reference_frame" : "reference_feature";
    auto* reference_device = use_frame_reference ? &*device_dpb.frame : &*device_dpb.feature;
    reference_device->name = reference_name;
    auto& q_feature_opt = quant_cache.q_feature[qp];
    if (!q_feature_opt.has_value()) return backend_error("missing cached q_feature tensor");
    auto* q_feature_device = &q_feature_opt.value();
    std::array<DeviceTensor*, 2> reference_inputs{
        reference_device, q_feature_device};
    const auto reference_engine = use_frame_reference ? 7U : 8U;
    auto reference_outputs = run_device_engine(
        engines[reference_engine], engine_specs[reference_engine], reference_inputs, stream,
        EngineOutputStorage::scratch);
    if (!reference_outputs) return reference_outputs.error();
    auto context_result = take_device_tensor(reference_outputs.value(), "context");
    auto temporal_result = take_device_tensor(reference_outputs.value(), "temporal_context");
    if (!context_result) return context_result.error();
    if (!temporal_result) return temporal_result.error();
    auto context = std::move(context_result.value());
    auto temporal_context = std::move(temporal_result.value());

    const auto padded_height = round_up(static_cast<std::int32_t>(frame.height()), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(frame.width()), 16);
    auto input_frame = allocate_device_tensor_scratch(
        "frame", make_dims(3, padded_height, padded_width));
    if (!input_frame) return input_frame.error();
    auto input_bytes_device = allocate_cuda_scratch(frame.data().size());
    if (!input_bytes_device) return input_bytes_device.error();
    auto input_bytes_copied = copy_async(
        input_bytes_device.value().data, frame.data().data(), frame.data().size(),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync frame input bytes");
    if (!input_bytes_copied) return input_bytes_copied.error();
    {
        CpuProfileScope stage("p_input_color_convert_gpu");
        cudaError_t converted = cudaSuccess;
        if (frame.pixel_format() == PixelFormat::yuv420p8) {
            converted = cuda_ops::yuv420p8_to_ycbcr_padded(
                input_bytes_device.value().data,
                static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()),
                input_frame.value().storage.data,
                padded_height,
                padded_width,
                stream);
        } else {
            converted = cuda_ops::rgb24_to_ycbcr_padded(
                input_bytes_device.value().data,
                static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()),
                input_frame.value().storage.data,
                padded_height,
                padded_width,
                stream);
        }
        if (converted != cudaSuccess) {
            return cuda_error("cuda_ops input color convert", converted);
        }
    }
    auto& q_encoder_opt = quant_cache.q_encoder[qp];
    if (!q_encoder_opt.has_value()) return backend_error("missing cached q_encoder tensor");
    auto* q_encoder = &q_encoder_opt.value();
    std::array<DeviceTensor*, 3> analysis_inputs{
        &input_frame.value(), &context, q_encoder};
    auto analysis_outputs = run_device_engine(
        engines[9], engine_specs[9], analysis_inputs, stream, EngineOutputStorage::scratch);
    if (!analysis_outputs) return analysis_outputs.error();
    auto y_result = take_device_tensor(analysis_outputs.value(), "y");
    if (!y_result) return y_result.error();
    auto y_device = std::move(y_result.value());
    const auto y_height = static_cast<std::int32_t>(y_device.shape.d[2]);
    const auto y_width = static_cast<std::int32_t>(y_device.shape.d[3]);

    auto y_padded = allocate_device_tensor_scratch(
        "y_padded", make_dims(128, round_up(y_height, 4), round_up(y_width, 4)));
    if (!y_padded) return y_padded.error();
    const cuda_ops::Shape4D y_shape{1, 128, y_height, y_width};
    const auto padded = cuda_ops::replicate_pad(
        y_device.storage.data, y_padded.value().storage.data, y_shape,
        static_cast<std::int32_t>(y_padded.value().shape.d[2]),
        static_cast<std::int32_t>(y_padded.value().shape.d[3]), stream);
    if (padded != cudaSuccess) return cuda_error("cuda_ops::replicate_pad", padded);
    std::array<DeviceTensor*, 1> hyper_inputs{&y_padded.value()};
    auto hyper_outputs = run_device_engine(
        engines[10], engine_specs[10], hyper_inputs, stream, EngineOutputStorage::scratch);
    if (!hyper_outputs) return hyper_outputs.error();
    auto z_result = take_device_tensor(hyper_outputs.value(), "z");
    if (!z_result) return z_result.error();
    auto z_device = std::move(z_result.value());
    z_device.name = "z_hat";
    const auto z_count = element_count(z_device.shape);
    auto z_symbol_device = allocate_cuda_scratch(z_count * sizeof(std::int8_t));
    if (!z_symbol_device) return z_symbol_device.error();
    const auto rounded = cuda_ops::round_to_int8(
        z_device.storage.data, static_cast<std::int8_t*>(z_symbol_device.value().data),
        z_count, stream);
    if (rounded != cudaSuccess) return cuda_error("cuda_ops::round_to_int8", rounded);

    std::array<DeviceTensor*, 2> prior_inputs{&z_device, &temporal_context};
    auto prior_outputs = run_device_engine(
        engines[11], engine_specs[11], prior_inputs, stream, EngineOutputStorage::scratch);
    if (!prior_outputs) return prior_outputs.error();
    auto params_result = take_device_tensor(prior_outputs.value(), "params");
    if (!params_result) return params_result.error();
    auto params_device = std::move(params_result.value());

    const cuda_ops::Shape4D prior_shape{1, 128, y_height, y_width};
    const auto prior_count = static_cast<std::size_t>(128) * y_height * y_width;
    auto* params_values = static_cast<__half*>(params_device.storage.data);
    auto* q_dec_values = params_values;
    auto* scales_values = params_values + prior_count;
    auto* means_values = params_values + 2 * prior_count;
    const auto quantized = cuda_ops::clamp_reciprocal_with_quant(
        q_dec_values, y_device.storage.data, prior_count, stream);
    if (quantized != cudaSuccess) {
        return cuda_error("cuda_ops::clamp_reciprocal_with_quant", quantized);
    }

    auto mask0 = allocate_device_tensor_scratch("mask0", y_device.shape);
    auto symbols_full0 = allocate_device_tensor_scratch("symbols_full0", y_device.shape);
    auto y_hat_device = allocate_device_tensor_scratch("y_hat", y_device.shape);
    auto scales_full0 = allocate_device_tensor_scratch("scales_full0", y_device.shape);
    if (!mask0) return mask0.error();
    if (!symbols_full0) return symbols_full0.error();
    if (!y_hat_device) return y_hat_device.error();
    if (!scales_full0) return scales_full0.error();
    auto status = cuda_ops::make_two_way_mask(mask0.value().storage.data, prior_shape, 0, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::process_with_mask(
        y_device.storage.data, scales_values, means_values, mask0.value().storage.data,
        nullptr, symbols_full0.value().storage.data, y_hat_device.value().storage.data,
        scales_full0.value().storage.data, prior_count,
        0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);

    const auto reduced_shape = make_dims(64, y_height, y_width);
    auto symbols0 = allocate_device_tensor_scratch("symbols0", reduced_shape);
    auto scales0 = allocate_device_tensor_scratch("scales0", reduced_shape);
    if (!symbols0) return symbols0.error();
    if (!scales0) return scales0.error();
    status = cuda_ops::reduce_channel_halves(
        symbols_full0.value().storage.data, symbols0.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::reduce_channel_halves(
        scales_full0.value().storage.data, scales0.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);

    auto spatial_context = allocate_device_tensor_scratch(
        "context", make_dims(512, y_height, y_width));
    if (!spatial_context) return spatial_context.error();
    const auto y_bytes = prior_count * sizeof(__half);
    const auto params_bytes = 3 * y_bytes;
    auto copied = copy_async(
        spatial_context.value().storage.data, y_hat_device.value().storage.data, y_bytes,
        cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync spatial y");
    if (!copied) return copied.error();
    copied = copy_async(
        static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
        params_device.storage.data, params_bytes, cudaMemcpyDeviceToDevice, stream,
        "cudaMemcpyAsync spatial params");
    if (!copied) return copied.error();
    std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
    auto spatial_outputs = run_device_engine(
        engines[12], engine_specs[12], spatial_inputs, stream, EngineOutputStorage::scratch);
    if (!spatial_outputs) return spatial_outputs.error();
    auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
    if (!spatial_result) return spatial_result.error();
    auto spatial_params = std::move(spatial_result.value());
    auto* spatial_values = static_cast<__half*>(spatial_params.storage.data);

    auto mask1 = allocate_device_tensor_scratch("mask1", y_device.shape);
    auto symbols_full1 = allocate_device_tensor_scratch("symbols_full1", y_device.shape);
    auto reconstructed1 = allocate_device_tensor_scratch("reconstructed1", y_device.shape);
    auto scales_full1 = allocate_device_tensor_scratch("scales_full1", y_device.shape);
    if (!mask1) return mask1.error();
    if (!symbols_full1) return symbols_full1.error();
    if (!reconstructed1) return reconstructed1.error();
    if (!scales_full1) return scales_full1.error();
    status = cuda_ops::make_two_way_mask(mask1.value().storage.data, prior_shape, 1, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::process_with_mask(
        y_device.storage.data, spatial_values, spatial_values + prior_count,
        mask1.value().storage.data, nullptr, symbols_full1.value().storage.data,
        reconstructed1.value().storage.data,
        scales_full1.value().storage.data, prior_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);
    auto symbols1 = allocate_device_tensor_scratch("symbols1", reduced_shape);
    auto scales1 = allocate_device_tensor_scratch("scales1", reduced_shape);
    if (!symbols1) return symbols1.error();
    if (!scales1) return scales1.error();
    status = cuda_ops::reduce_channel_halves(
        symbols_full1.value().storage.data, symbols1.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::reduce_channel_halves(
        scales_full1.value().storage.data, scales1.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::add_and_multiply(
        y_hat_device.value().storage.data, reconstructed1.value().storage.data, q_dec_values,
        prior_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::add_and_multiply", status);

    const auto reduced_count = prior_count / 2;
    auto indexes0_device = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
    auto indexes1_device = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
    if (!indexes0_device) return indexes0_device.error();
    if (!indexes1_device) return indexes1_device.error();
    status = cuda_ops::build_encode_indexes(
        symbols0.value().storage.data, scales0.value().storage.data,
        static_cast<std::int16_t*>(indexes0_device.value().data), reduced_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    status = cuda_ops::build_encode_indexes(
        symbols1.value().storage.data, scales1.value().storage.data,
        static_cast<std::int16_t*>(indexes1_device.value().data), reduced_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    auto z_buffer_ready = z_symbols_buffer.ensure(z_count, "z_symbols");
    if (!z_buffer_ready) return z_buffer_ready.error();
    auto indexes0_ready = indexes0_buffer.ensure(reduced_count, "indexes0");
    if (!indexes0_ready) return indexes0_ready.error();
    auto indexes1_ready = indexes1_buffer.ensure(reduced_count, "indexes1");
    if (!indexes1_ready) return indexes1_ready.error();

    std::span<std::int8_t> z_symbols(z_symbols_buffer.data, z_count);
    std::span<std::int16_t> indexes0(indexes0_buffer.data, reduced_count);
    std::span<std::int16_t> indexes1(indexes1_buffer.data, reduced_count);
    copied = copy_async(
        z_symbols.data(), z_symbol_device.value().data, z_count * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync z symbols");
    if (!copied) return copied.error();
    copied = copy_async(
        indexes0.data(), indexes0_device.value().data,
        reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
        "cudaMemcpyAsync indexes0");
    if (!copied) return copied.error();
    copied = copy_async(
        indexes1.data(), indexes1_device.value().data,
        reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
        "cudaMemcpyAsync indexes1");
    if (!copied) return copied.error();

    CudaEvent entropy_ready;
    status = cudaEventCreateWithFlags(&entropy_ready.data, cudaEventDisableTiming);
    if (status != cudaSuccess) return cuda_error("cudaEventCreateWithFlags", status);
    status = cudaEventRecord(entropy_ready.data, stream);
    if (status != cudaSuccess) return cuda_error("cudaEventRecord entropy", status);

    auto& q_decoder_opt = quant_cache.q_decoder[qp];
    if (!q_decoder_opt.has_value()) return backend_error("missing cached q_decoder tensor");
    auto& q_recon_opt = quant_cache.q_recon[qp];
    if (!q_recon_opt.has_value()) return backend_error("missing cached q_recon tensor");
    auto* q_decoder = &q_decoder_opt.value();
    auto* q_recon = &q_recon_opt.value();
    std::array<DeviceTensor*, 4> synthesis_inputs{
        &y_hat_device.value(), &context, q_decoder, q_recon};
    auto synthesis_outputs = run_device_engine(
        engines[13], engine_specs[13], synthesis_inputs, stream);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    auto feature_device_result = take_device_tensor(synthesis_outputs.value(), "feature");
    if (!frame_device_result) return frame_device_result.error();
    if (!feature_device_result) return feature_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    auto feature_device = std::move(feature_device_result.value());
    status = cudaEventSynchronize(entropy_ready.data);
    if (status != cudaSuccess) return cuda_error("cudaEventSynchronize entropy", status);

    const auto entropy_start = std::chrono::steady_clock::now();
    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    rans.reset_encoder();
    auto mode = rans.set_use_two_coders(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z_device.shape.d[2] * z_device.shape.d[3]);
    auto encoded_z = rans.encode_z(
        z_symbols, assets.video_z_group, static_cast<std::size_t>(qp) * 128, per_channel);
    if (!encoded_z) return encoded_z.error();
    auto encoded_y0 = rans.encode_y(indexes0, assets.video_gaussian_y_group);
    if (!encoded_y0) return encoded_y0.error();
    auto encoded_y1 = rans.encode_y(indexes1, assets.video_gaussian_y_group);
    if (!encoded_y1) return encoded_y1.error();
    auto stream_bytes = rans.finish_encode();
    if (!stream_bytes) return stream_bytes.error();
    if (active_profiler != nullptr && active_profiler->enabled()) {
        active_profiler->record_cpu_stage(
            "p_entropy_encode",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - entropy_start).count());
    }
    auto payload = make_predicted_payload(
        frame.width(), frame.height(), qp, two_coders, use_frame_reference, stream_bytes.value());

    Frame reconstructed_frame;
    if (verify_reconstruction) {
        std::array<const DeviceTensor*, 1> tensors{&frame_device};
        auto downloaded = download_tensors(tensors, stream);
        if (!downloaded) return downloaded.error();
        auto reconstructed = ycbcr_to_yuv420p8(
            downloaded.value().front(), frame.width(), frame.height(), frame.timestamp());
        if (!reconstructed) return reconstructed.error();
        reconstructed_frame = std::move(reconstructed.value());
    }
    device_dpb.frame = std::move(frame_device);
    device_dpb.feature = std::move(feature_device);
    device_dpb.next_frame_index = state.frame_index + 1;
    device_dpb.generation = state.generation;
    std::vector<std::byte> latent_state;
    return CodecEncodeResult{
        std::move(payload), std::move(reconstructed_frame), std::move(latent_state), qp};
}

Result<CodecDecodeResult> decode_predicted(
    std::span<const std::byte> payload, Timestamp timestamp, const SequenceStateView& state,
    DeviceDpb& device_dpb,
    std::vector<EngineInstance>& engines, RansCodec& rans,
    const RuntimeAssets& assets, VideoQuantDeviceCache& quant_cache,
    cudaStream_t stream) {
    auto parsed = parse_predicted_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    if (!device_dpb.matches(state)) {
        return Error(ErrorCode::invalid_state, "decoder device DPB is unavailable",
                     std::string(subsystem));
    }
    if (!info.use_frame_reference && !device_dpb.feature) {
        return Error(ErrorCode::invalid_state, "P-frame feature reference is unavailable",
                     std::string(subsystem));
    }
    auto* reference_device = info.use_frame_reference ? &*device_dpb.frame :
                                                        &*device_dpb.feature;
    const std::string reference_name = info.use_frame_reference ?
        "reference_frame" : "reference_feature";
    reference_device->name = reference_name;
    auto& q_feature_opt = quant_cache.q_feature[info.qp];
    if (!q_feature_opt.has_value()) return backend_error("missing cached q_feature tensor");
    auto* q_feature_device = &q_feature_opt.value();
    std::array<DeviceTensor*, 2> reference_inputs{
        reference_device, q_feature_device};
    const auto reference_engine = info.use_frame_reference ? 7U : 8U;
    auto reference_outputs = run_device_engine(
        engines[reference_engine], engine_specs[reference_engine], reference_inputs, stream,
        EngineOutputStorage::scratch);
    if (!reference_outputs) return reference_outputs.error();
    auto context_result = take_device_tensor(reference_outputs.value(), "context");
    auto temporal_result = take_device_tensor(reference_outputs.value(), "temporal_context");
    if (!context_result) return context_result.error();
    if (!temporal_result) return temporal_result.error();
    auto context = std::move(context_result.value());
    auto temporal_context = std::move(temporal_result.value());

    const auto padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = padded_height / 16;
    const auto y_width = padded_width / 16;
    const auto z_height = round_up(y_height, 4) / 4;
    const auto z_width = round_up(y_width, 4) / 4;
    rans.reset_encoder();
    auto mode = rans.set_use_two_coders(info.two_coders);
    if (!mode) return mode.error();
    auto stream_set = rans.set_stream(info.rans);
    if (!stream_set) return stream_set.error();
    const auto per_channel = static_cast<std::size_t>(z_height * z_width);
    auto z_values = rans.decode_z(128 * per_channel, assets.video_z_group,
        static_cast<std::size_t>(info.qp) * 128, per_channel);
    if (!z_values) return z_values.error();
    HostTensor z_host{"z_hat", make_dims(128, z_height, z_width),
                      std::vector<__half>(z_values.value().size())};
    std::transform(z_values.value().begin(), z_values.value().end(), z_host.values.begin(),
                   [](std::int8_t value) { return to_half(static_cast<float>(value)); });
    auto z_device = upload_tensor(z_host, "z_hat", stream);
    if (!z_device) return z_device.error();
    std::array<DeviceTensor*, 2> prior_inputs{&z_device.value(), &temporal_context};
    auto prior_outputs = run_device_engine(
        engines[11], engine_specs[11], prior_inputs, stream, EngineOutputStorage::scratch);
    if (!prior_outputs) return prior_outputs.error();
    auto params_device_result = take_device_tensor(prior_outputs.value(), "params");
    if (!params_device_result) return params_device_result.error();
    auto params_device = std::move(params_device_result.value());
    const cuda_ops::Shape4D prior_shape{1, 128, y_height, y_width};
    const auto prior_count = static_cast<std::size_t>(128) * y_height * y_width;
    const auto reduced_count = prior_count / 2;
    const auto reduced_shape = make_dims(64, y_height, y_width);
    auto* params_values = static_cast<__half*>(params_device.storage.data);
    auto* q_dec_values = params_values;
    auto* scales_values = params_values + prior_count;
    auto* means_values = params_values + 2 * prior_count;
    auto status = cuda_ops::clamp_min(q_dec_values, prior_count, 0.5F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::clamp_min", status);

    auto mask0 = allocate_device_tensor_scratch("mask0", make_dims(128, y_height, y_width));
    auto scales0 = allocate_device_tensor_scratch("scales0", reduced_shape);
    if (!mask0) return mask0.error();
    if (!scales0) return scales0.error();
    status = cuda_ops::make_two_way_mask(mask0.value().storage.data, prior_shape, 0, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::reduce_masked_halves(
        scales_values, mask0.value().storage.data, scales0.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_halves", status);
    auto indexes0_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
    if (!indexes0_device) return indexes0_device.error();
    status = cuda_ops::build_decode_indexes(
        scales0.value().storage.data, static_cast<std::uint8_t*>(indexes0_device.value().data),
        reduced_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes", status);
    std::vector<std::uint8_t> indexes0(reduced_count);
    auto copied = copy_async(
        indexes0.data(), indexes0_device.value().data, reduced_count,
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync indexes0");
    if (!copied) return copied.error();
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize indexes0");
    if (!synchronized) return synchronized.error();
    auto decoded0 = rans.decode_y(indexes0, assets.video_gaussian_y_group);
    if (!decoded0) return decoded0.error();
    HostTensor symbols0_host{"symbols0", reduced_shape,
                             std::vector<__half>(decoded0.value().size())};
    std::transform(decoded0.value().begin(), decoded0.value().end(),
                   symbols0_host.values.begin(),
                   [](std::int8_t value) { return to_half(static_cast<float>(value)); });
    auto symbols0 = upload_tensor(symbols0_host, "symbols0", stream);
    if (!symbols0) return symbols0.error();
    auto y_hat_device = allocate_device_tensor_scratch("y_hat", make_dims(128, y_height, y_width));
    if (!y_hat_device) return y_hat_device.error();
    status = cuda_ops::restore_two_way(
        symbols0.value().storage.data, means_values, mask0.value().storage.data,
        y_hat_device.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::restore_two_way", status);

    auto spatial_context = allocate_device_tensor_scratch(
        "context", make_dims(512, y_height, y_width));
    if (!spatial_context) return spatial_context.error();
    const auto y_bytes = prior_count * sizeof(__half);
    const auto params_bytes = 3 * y_bytes;
    copied = copy_async(
        spatial_context.value().storage.data, y_hat_device.value().storage.data, y_bytes,
        cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync spatial y");
    if (!copied) return copied.error();
    copied = copy_async(
        static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
        params_device.storage.data, params_bytes, cudaMemcpyDeviceToDevice, stream,
        "cudaMemcpyAsync spatial params");
    if (!copied) return copied.error();
    std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
    auto spatial_outputs = run_device_engine(
        engines[12], engine_specs[12], spatial_inputs, stream, EngineOutputStorage::scratch);
    if (!spatial_outputs) return spatial_outputs.error();
    auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
    if (!spatial_result) return spatial_result.error();
    auto spatial_params = std::move(spatial_result.value());
    auto* spatial_values = static_cast<__half*>(spatial_params.storage.data);

    auto mask1 = allocate_device_tensor_scratch("mask1", make_dims(128, y_height, y_width));
    auto scales1 = allocate_device_tensor_scratch("scales1", reduced_shape);
    if (!mask1) return mask1.error();
    if (!scales1) return scales1.error();
    status = cuda_ops::make_two_way_mask(mask1.value().storage.data, prior_shape, 1, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::reduce_masked_halves(
        spatial_values, mask1.value().storage.data, scales1.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_halves", status);
    auto indexes1_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
    if (!indexes1_device) return indexes1_device.error();
    status = cuda_ops::build_decode_indexes(
        scales1.value().storage.data, static_cast<std::uint8_t*>(indexes1_device.value().data),
        reduced_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes", status);
    std::vector<std::uint8_t> indexes1(reduced_count);
    copied = copy_async(
        indexes1.data(), indexes1_device.value().data, reduced_count,
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync indexes1");
    if (!copied) return copied.error();
    synchronized = synchronize_stream(stream, "cudaStreamSynchronize indexes1");
    if (!synchronized) return synchronized.error();
    auto decoded1 = rans.decode_y(indexes1, assets.video_gaussian_y_group);
    if (!decoded1) return decoded1.error();
    HostTensor symbols1_host{"symbols1", reduced_shape,
                             std::vector<__half>(decoded1.value().size())};
    std::transform(decoded1.value().begin(), decoded1.value().end(),
                   symbols1_host.values.begin(),
                   [](std::int8_t value) { return to_half(static_cast<float>(value)); });
    auto symbols1 = upload_tensor(symbols1_host, "symbols1", stream);
    if (!symbols1) return symbols1.error();
    auto reconstructed1 = allocate_device_tensor_scratch(
        "reconstructed1", make_dims(128, y_height, y_width));
    if (!reconstructed1) return reconstructed1.error();
    status = cuda_ops::restore_two_way(
        symbols1.value().storage.data, spatial_values + prior_count,
        mask1.value().storage.data, reconstructed1.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::restore_two_way", status);
    status = cuda_ops::add_and_multiply(
        y_hat_device.value().storage.data, reconstructed1.value().storage.data,
        q_dec_values, prior_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::add_and_multiply", status);

    auto& q_decoder_opt = quant_cache.q_decoder[info.qp];
    if (!q_decoder_opt.has_value()) return backend_error("missing cached q_decoder tensor");
    auto& q_recon_opt = quant_cache.q_recon[info.qp];
    if (!q_recon_opt.has_value()) return backend_error("missing cached q_recon tensor");
    auto* q_decoder = &q_decoder_opt.value();
    auto* q_recon = &q_recon_opt.value();
    std::array<DeviceTensor*, 4> synthesis_inputs{
        &y_hat_device.value(), &context, q_decoder, q_recon};
    auto synthesis_outputs = run_device_engine(
        engines[13], engine_specs[13], synthesis_inputs, stream);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    auto feature_device_result = take_device_tensor(synthesis_outputs.value(), "feature");
    if (!frame_device_result) return frame_device_result.error();
    if (!feature_device_result) return feature_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    auto feature_device = std::move(feature_device_result.value());
    std::array<DeviceTensor*, 1> reconstruction_inputs{&frame_device};
    auto reconstruction = download_tensors(reconstruction_inputs, stream);
    if (!reconstruction) return reconstruction.error();
    auto frame_result = take_tensor(reconstruction.value(), "frame_hat");
    if (!frame_result) return frame_result.error();
    auto frame = ycbcr_to_yuv420p8(frame_result.value(), info.width, info.height, timestamp);
    if (!frame) return frame.error();
    device_dpb.frame = std::move(frame_device);
    device_dpb.feature = std::move(feature_device);
    device_dpb.next_frame_index = state.frame_index + 1;
    device_dpb.generation = state.generation;
    std::vector<std::byte> latent_state;
    return CodecDecodeResult{std::move(frame.value()), std::move(latent_state)};
}

class TensorRTBackend final : public CodecBackend {
public:
    ~TensorRTBackend() override {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream_));
            encoder_dpb_.clear();
            decoder_dpb_.clear();
            static_cast<void>(cudaStreamSynchronize(stream_));
            static_cast<void>(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    Result<void> initialize(const RuntimeConfiguration& configuration) override {
#ifndef NDEBUG
        std::clog << "[nvcr.dcvcrt] warning: Debug build; codec timings are not "
                     "representative. Configure with -DCMAKE_BUILD_TYPE=Release.\n";
#endif
        if (initialized_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is already initialized",
                std::string(subsystem));
        }
        if (configuration.intra_engine_path.empty()) {
            return Error(
                ErrorCode::invalid_argument,
                "intra_engine_path must name the I-frame plan directory",
                std::string(subsystem));
        }

        std::error_code filesystem_error;
        if (!fs::is_directory(configuration.intra_engine_path, filesystem_error)) {
            return Error(
                ErrorCode::dependency_unavailable,
                "I-frame plan directory is unavailable: " +
                    configuration.intra_engine_path.string(),
                std::string(subsystem));
        }

        const auto device_status = cudaSetDevice(configuration.device_id);
        if (device_status != cudaSuccess) return cuda_error("cudaSetDevice", device_status);
        auto low_memory_mode = determine_low_memory_mode(configuration);
        if (!low_memory_mode) return low_memory_mode.error();
        auto manifest = validate_engine_manifest(
            configuration.intra_engine_path, configuration.device_id, configuration.model_id);
        if (!manifest) return manifest.error();

        try {
            std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger_));
            if (!runtime) return backend_error("failed to create TensorRT runtime");

            RansCodec rans;
            auto assets = load_runtime_assets(configuration.intra_engine_path, rans);
            if (!assets) return assets.error();

            std::vector<EngineInstance> engines;
            engines.reserve(engine_specs.size());
            for (const auto& specification : engine_specs) {
                auto loaded = load_engine(
                    *runtime, configuration.intra_engine_path, specification);
                if (!loaded) return loaded.error();
                engines.push_back(std::move(loaded.value()));
            }

            cudaStream_t stream{};
            const auto stream_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (stream_status != cudaSuccess) {
                return cuda_error("cudaStreamCreateWithFlags", stream_status);
            }
            DeviceScratchArena scratch_arena;
            auto scratch_ready = scratch_arena.initialize(configuration.device_arena_bytes);
            if (!scratch_ready) {
                static_cast<void>(cudaStreamDestroy(stream));
                return scratch_ready.error();
            }
            ImageQuantDeviceCache image_quant_cache;
            auto image_quant_cache_loaded = populate_image_quant_cache(
                assets.value(), image_quant_cache, stream);
            if (!image_quant_cache_loaded) {
                static_cast<void>(cudaStreamDestroy(stream));
                return image_quant_cache_loaded.error();
            }
            VideoQuantDeviceCache video_quant_cache;
            auto quant_cache_loaded = populate_video_quant_cache(assets.value(), video_quant_cache, stream);
            if (!quant_cache_loaded) {
                static_cast<void>(cudaStreamDestroy(stream));
                return quant_cache_loaded.error();
            }
            for (std::size_t index = 0; index < engines.size(); ++index) {
                engines[index].low_memory_mode = low_memory_mode.value();
                auto warmed = warm_up_engine(engines[index], engine_specs[index], stream);
                if (!warmed) {
                    static_cast<void>(cudaStreamDestroy(stream));
                    return warmed.error();
                }
            }

            runtime_ = std::move(runtime);
            engines_ = std::move(engines);
            rans_ = std::move(rans);
            assets_ = std::move(assets.value());
            image_quant_cache_ = std::move(image_quant_cache);
            video_quant_cache_ = std::move(video_quant_cache);
            scratch_arena_ = std::move(scratch_arena);
            stream_ = stream;
            intra_qp_ = configuration.intra_qp;
            profiling_enabled_ = configuration.enable_profiling;
            verify_encoder_reconstruction_ = configuration.verify_encoder_reconstruction;
            low_memory_mode_ = low_memory_mode.value();
            if (configuration.log_level <= LogLevel::info) {
                std::clog << "[nvcr.dcvcrt] [info] TensorRT mode: "
                          << (low_memory_mode_ ? "low-memory" : "performance")
                          << '\n';
            }
            initialized_ = true;
            return {};
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("TensorRT initialization failed: ") + exception.what());
        } catch (...) {
            return backend_error("TensorRT initialization failed");
        }
    }

    Result<CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const SequenceStateView& state) override {
        if (!initialized_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        try {
            BackendProfiler profiler(profiling_enabled_);
            profiler.begin("encode", frame_type, state.frame_index);
            scratch_arena_.reset();
            CudaAllocationScope allocation_scope(stream_, &profiler, &scratch_arena_);
            if (frame_type == FrameType::intra) {
                auto result = encode_intra(
                    frame, intra_qp_, state, encoder_dpb_, engines_, rans_, assets_,
                    image_quant_cache_, image_decode_staging_,
                    image_z_symbols_buffer_, image_indexes0_buffer_, image_indexes1_buffer_,
                    image_indexes2_buffer_, image_indexes3_buffer_,
                    verify_encoder_reconstruction_, stream_);
                profiler.finish();
                return result;
            }
            auto result = encode_predicted(
                frame, intra_qp_, state, encoder_dpb_, engines_, rans_, assets_,
                video_quant_cache_, z_symbols_buffer_, indexes0_buffer_,
                indexes1_buffer_, verify_encoder_reconstruction_, stream_);
            profiler.finish();
            return result;
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("native I-frame encoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native I-frame encoding failed");
        }
    }

    Result<CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType frame_type,
        Timestamp timestamp,
        const SequenceStateView& state) override {
        if (!initialized_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        try {
            BackendProfiler profiler(profiling_enabled_);
            profiler.begin("decode", frame_type, state.frame_index);
            scratch_arena_.reset();
            CudaAllocationScope allocation_scope(stream_, &profiler, &scratch_arena_);
            if (frame_type == FrameType::intra) {
                auto result = decode_intra(
                    payload, timestamp, state, decoder_dpb_, engines_, rans_, assets_, image_quant_cache_, image_decode_staging_, stream_);
                profiler.finish();
                return result;
            }
            auto result = decode_predicted(
                payload, timestamp, state, decoder_dpb_, engines_, rans_, assets_,
                video_quant_cache_, stream_);
            profiler.finish();
            return result;
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("native I-frame decoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native I-frame decoding failed");
        }
    }

    Result<void> flush() override {
        if (!initialized_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        const auto status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess) return cuda_error("cudaStreamSynchronize", status);
        return {};
    }

    void reset() noexcept override {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream_));
        }
        encoder_dpb_.clear();
        decoder_dpb_.clear();
    }

private:
    TensorRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::vector<EngineInstance> engines_;
    RansCodec rans_;
    RuntimeAssets assets_;
    ImageQuantDeviceCache image_quant_cache_;
    ImageDecodeStaging image_decode_staging_;
    VideoQuantDeviceCache video_quant_cache_;
    DeviceScratchArena scratch_arena_;
    PinnedHostBuffer<std::int8_t> z_symbols_buffer_;
    PinnedHostBuffer<std::int8_t> image_z_symbols_buffer_;
    PinnedHostBuffer<std::int16_t> image_indexes0_buffer_;
    PinnedHostBuffer<std::int16_t> image_indexes1_buffer_;
    PinnedHostBuffer<std::int16_t> image_indexes2_buffer_;
    PinnedHostBuffer<std::int16_t> image_indexes3_buffer_;
    PinnedHostBuffer<std::int16_t> indexes0_buffer_;
    PinnedHostBuffer<std::int16_t> indexes1_buffer_;
    cudaStream_t stream_{};
    DeviceDpb encoder_dpb_;
    DeviceDpb decoder_dpb_;
    std::uint32_t intra_qp_{};
    bool profiling_enabled_{false};
    bool verify_encoder_reconstruction_{false};
    bool low_memory_mode_{true};
    bool initialized_{false};
};

}  // namespace

Result<std::unique_ptr<CodecBackend>> make_tensorrt_backend() {
    try {
        return std::unique_ptr<CodecBackend>(std::make_unique<TensorRTBackend>());
    } catch (const std::exception& exception) {
        return backend_error(
            std::string("failed to allocate TensorRT backend: ") + exception.what());
    } catch (...) {
        return backend_error("failed to allocate TensorRT backend");
    }
}

}  // namespace nvcr::dcvcrt
