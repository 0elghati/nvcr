#pragma once

// Codec descriptor, capability, and option-schema types (Phase 3 of M-EXT).
// These allow the runtime to discover and query codec adapters without knowing
// their implementation details.  ICodecAdapter is in codec/adapter.hpp.

#include "nvcr/common/error.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::codec {

// Uniquely identifies a codec family, e.g. "dcvc-rt".
struct CodecDescriptor {
    std::string id;           // stable, machine-readable identifier
    std::string display_name; // human-readable name
    std::uint32_t api_version{1};
};

struct CodecCapabilities {
    bool supports_intra{true};
    bool supports_predicted{false};
    bool supports_bidirectional{false};
    bool supports_hierarchical{false};
    bool supports_delayed_output{false}; // codec may buffer frames before emitting AUs
    std::uint32_t min_qp{0};
    std::uint32_t max_qp{71};
};

// A single namespaced option declaration.  The namespace prefix identifies the
// owning scope: "video.", "runtime.", "dcvc_rt.", "tensorrt.", etc.
struct OptionDeclaration {
    std::string name;        // fully qualified, e.g. "dcvc_rt.quality_index"
    std::string type;        // "int", "uint", "float", "bool", "string", "path"
    std::string description;
    std::optional<std::string> default_value;
    std::optional<std::string> min_value;
    std::optional<std::string> max_value;
    bool required{false};
};

// The schema for all options a codec adapter accepts.
struct OptionSchema {
    std::vector<OptionDeclaration> declarations;
};

}  // namespace nvcr::codec
