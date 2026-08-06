#pragma once

#include <cstdint>

namespace nvcr {

struct SoftwareVersion {
    std::uint16_t major{0};
    std::uint16_t minor{0};
    std::uint16_t patch{0};
};

struct CodecApiVersion {
    std::uint16_t major{1};
};

struct ProviderApiVersion {
    std::uint16_t major{1};
};

struct StreamFormatVersion {
    std::uint16_t major{1};
    std::uint16_t minor{0};
};

struct PayloadSyntaxVersion {
    std::uint16_t major{1};
};

struct ModelSetVersion {
    std::uint16_t major{1};
    std::uint16_t minor{0};
};

struct ManifestSchemaVersion {
    std::uint16_t major{2};
    std::uint16_t minor{0};
};

inline constexpr SoftwareVersion current_software_version{0, 8, 2};

}  // namespace nvcr
