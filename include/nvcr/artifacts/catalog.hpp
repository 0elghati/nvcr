#pragma once

#include "nvcr/artifacts/resolver.hpp"
#include "nvcr/common/error.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::artifacts {

inline constexpr std::string_view catalog_schema = "nvcr.engine-catalog.v1";

struct CatalogEntry final {
    std::string backend;
    std::string model_profile_id;
    std::string target_profile_id;
    std::string profile;
    std::string precision;
    std::string operating_system;
    std::string architecture;
    std::string device_name;
    std::string filename;
    std::uint64_t size_bytes{};
    std::string sha256;
    std::uint32_t compute_capability_major{};
    std::uint32_t compute_capability_minor{};
    std::uint32_t multiprocessor_count{};
    std::uint32_t cuda_runtime_version{};
    std::uint32_t tensorrt_version_major{};
    std::uint32_t tensorrt_version_minor{};
    std::uint32_t tensorrt_version_patch{};
    HardwareCompatibility hardware_compatibility{HardwareCompatibility::exact};
};

struct CatalogLoadOptions final {
    std::string codec_id;
    std::string provider_id{"tensorrt"};
    std::filesystem::path artifact_root;
    bool mark_license_restricted{true};
};

class Catalog final {
public:
    [[nodiscard]] static Result<Catalog> from_json(std::string_view document);
    [[nodiscard]] static Result<Catalog> from_file(const std::filesystem::path& path);

    [[nodiscard]] std::string_view schema() const noexcept { return catalog_schema; }
    [[nodiscard]] const std::vector<CatalogEntry>& entries() const noexcept {
        return entries_;
    }

    // Convert catalog bundle records into resolver candidates. The catalog's
    // profile is used as the resolver component_id because each catalog asset
    // represents one executable bundle profile rather than one model tensor.
    [[nodiscard]] Result<void> append_candidates(
        Resolver& resolver,
        const CatalogLoadOptions& options) const;

private:
    std::vector<CatalogEntry> entries_;
};

}  // namespace nvcr::artifacts
