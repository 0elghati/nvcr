#pragma once

// Provider-API public types (Phase 3 of M-EXT).
// IExecutionProvider, IExecutable, and IArtifactCompiler define the boundary
// between the codec adapter and the execution technology (TensorRT, ONNX
// Runtime, etc.).  No TensorRT or CUDA types appear in this header.

#include "nvcr/common/error.hpp"
#include "nvcr/common/versions.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::provider {

// Identifies the execution-provider family; new values are additive.
enum class ProviderKind : std::uint8_t {
    tensorrt = 0,
    cpu_reference,  // deterministic test provider (Phase 5)
    // future: onnx_runtime, openvino, core_ml, directml
};

struct ProviderDescriptor {
    ProviderKind kind{};
    std::string id;           // e.g. "tensorrt"
    std::string version;      // provider runtime version, e.g. "10.9.0"
    std::string display_name;
};

struct ProviderCapabilities {
    bool supports_fp16{false};
    bool supports_int8{false};
    bool supports_dynamic_shapes{false};
    bool supports_cuda_graphs{false};
    std::string target_device;  // e.g. "NVIDIA GeForce RTX 4070"
};

// Describes a single model-component artifact a provider is asked to load.
struct ArtifactDescriptor {
    std::string codec_id;         // e.g. "dcvc-rt"
    std::string model_set_id;     // e.g. "dcvcrt-cvpr2025"
    std::string component_id;     // e.g. "p_frame_encoder"
    std::string provider_id;      // e.g. "tensorrt"
    std::string precision;        // e.g. "fp16"
    std::string path;             // resolved local filesystem path
    std::vector<std::byte> expected_digest; // SHA-256, or empty to skip
    CodecApiVersion codec_api_version{1};
    ProviderApiVersion provider_api_version{1};
    ModelSetVersion model_set_version{1, 0};
    ManifestSchemaVersion manifest_schema_version{2, 0};
};

// Named tensor binding: name -> typed byte span (host memory).
struct TensorBinding {
    std::string_view name;
    std::span<std::byte> data;
    std::vector<std::int64_t> shape;
    std::string dtype;  // "float16", "float32", "int8", "int32"
};

using TensorBindings = std::vector<TensorBinding>;

struct ExecutionOptions {
    bool synchronize_after{true};  // wait for GPU work to complete before returning
};

// A loaded, ready-to-execute model component.
class IExecutable {
public:
    virtual ~IExecutable() = default;

    [[nodiscard]] virtual ArtifactDescriptor descriptor() const = 0;

    // Execute the model component; inputs and outputs must match the component
    // tensor contract; the provider owns tensor memory.
    [[nodiscard]] virtual Result<void> execute(
        const TensorBindings& inputs,
        TensorBindings& outputs,
        const ExecutionOptions& options) = 0;
};

// An execution provider that can load and run model components.
// No TensorRT or CUDA types cross this boundary.
class IExecutionProvider {
public:
    virtual ~IExecutionProvider() = default;

    [[nodiscard]] virtual ProviderDescriptor descriptor() const = 0;
    [[nodiscard]] virtual ProviderCapabilities capabilities() const = 0;

    [[nodiscard]] virtual bool supports(const ArtifactDescriptor& artifact) const = 0;

    [[nodiscard]] virtual Result<std::shared_ptr<IExecutable>>
    load(const ArtifactDescriptor& artifact) = 0;
};

// Offline compilation: source model -> provider-specific executable artifact.
// Separate from IExecutionProvider so deployment systems can consume prebuilt
// artifacts without carrying the full compilation toolchain.
class IArtifactCompiler {
public:
    virtual ~IArtifactCompiler() = default;

    [[nodiscard]] virtual ProviderDescriptor provider() const = 0;

    [[nodiscard]] virtual Result<void> compile(
        const std::string& source_path,
        const std::string& output_path,
        const ArtifactDescriptor& descriptor) = 0;
};

}  // namespace nvcr::provider
