#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/common/versions.hpp"
#include "nvcr/provider/provider_api.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nvcr::artifacts {

enum class HardwareCompatibility : std::uint8_t {
    exact,
    same_compute_capability,
    ampere_plus,
};

struct ArtifactCandidate final {
    provider::ArtifactDescriptor artifact;
    std::string target_profile_id;
    std::string target_device;
    std::string compute_capability;
    std::string operating_system;
    std::string architecture;
    std::string runtime_version;
    std::uint32_t cuda_runtime_version{};
    HardwareCompatibility hardware_compatibility{HardwareCompatibility::exact};
    std::vector<std::byte> digest;
    bool available{true};
    bool license_restricted{false};
};

struct ArtifactRequest final {
    std::string codec_id;
    std::string model_set_id;
    std::string component_id;
    std::string provider_id;
    std::vector<std::string> provider_preference;
    std::string target_profile_id;
    std::string target_device;
    std::string compute_capability;
    std::string operating_system;
    std::string architecture;
    std::string runtime_version;
    std::uint32_t cuda_runtime_version{};
    std::string precision;
    CodecApiVersion codec_api_version{1};
    ProviderApiVersion provider_api_version{1};
    ModelSetVersion model_set_version{1, 0};
    ManifestSchemaVersion manifest_schema_version{2, 0};
    std::vector<std::byte> expected_digest;
    bool allow_license_restricted{false};
};

struct ResolvedArtifact final {
    ArtifactCandidate candidate;
    std::size_t provider_preference_rank{};
    std::uint8_t target_match_rank{};
};

class Resolver final {
public:
    void add(ArtifactCandidate candidate);
    void clear() noexcept;

    [[nodiscard]] const std::vector<ArtifactCandidate>& candidates() const noexcept {
        return candidates_;
    }

    [[nodiscard]] Result<ResolvedArtifact> resolve(const ArtifactRequest& request) const;

private:
    std::vector<ArtifactCandidate> candidates_;
};

}  // namespace nvcr::artifacts
