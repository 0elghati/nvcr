#pragma once

// Provider-API public types.  Abstract interfaces IExecutionProvider,
// IExecutable, and IArtifactCompiler are added in Phase 3 of M-EXT; only
// identifying types are declared here to anchor the module boundary target.

#include <cstdint>
#include <string>
#include <string_view>

namespace nvcr::provider {

// Identifies the execution-provider family; new values are additive.
enum class ProviderKind : std::uint8_t {
    tensorrt = 0,
    // future: onnx_runtime, openvino, core_ml, directml, cpu_reference
};

// Minimal human-readable identity; full capability struct comes in Phase 3.
struct ProviderDescriptor {
    ProviderKind kind{};
    std::string id;       // e.g. "tensorrt"
    std::string version;  // e.g. "10.9.0"
    std::string display_name;
};

}  // namespace nvcr::provider
