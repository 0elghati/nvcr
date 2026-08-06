#include "nvcr/artifacts/resolver.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <string_view>
#include <utility>

namespace nvcr::artifacts {
namespace {

Error resolution_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "artifact-resolver");
}

bool compatible_version(
    CodecApiVersion candidate,
    CodecApiVersion requested) noexcept {
    return candidate.major == requested.major;
}

bool compatible_version(
    ProviderApiVersion candidate,
    ProviderApiVersion requested) noexcept {
    return candidate.major == requested.major;
}

bool compatible_version(
    ModelSetVersion candidate,
    ModelSetVersion requested) noexcept {
    return candidate.major == requested.major && candidate.minor >= requested.minor;
}

bool compatible_version(
    ManifestSchemaVersion candidate,
    ManifestSchemaVersion requested) noexcept {
    return candidate.major == requested.major && candidate.minor >= requested.minor;
}

int compute_major(std::string_view value) noexcept {
    if (value.empty()) return -1;
    const auto separator = value.find('.');
    const auto major_text = value.substr(0, separator);
    int major = 0;
    const auto parsed = std::from_chars(
        major_text.data(), major_text.data() + major_text.size(), major);
    if (parsed.ec != std::errc{} || parsed.ptr != major_text.data() + major_text.size()) {
        return -1;
    }
    return major;
}

std::uint32_t cuda_major(std::uint32_t version) noexcept {
    return version / 1000U;
}

bool cuda_runtime_compatible(
    const ArtifactCandidate& candidate,
    const ArtifactRequest& request) noexcept {
    if (candidate.cuda_runtime_version == 0U || request.cuda_runtime_version == 0U) {
        return true;
    }
    if (candidate.architecture == "x86_64" || request.architecture == "x86_64") {
        return cuda_major(candidate.cuda_runtime_version) == cuda_major(request.cuda_runtime_version) &&
            candidate.cuda_runtime_version <= request.cuda_runtime_version;
    }
    return candidate.cuda_runtime_version == request.cuda_runtime_version;
}

std::size_t preference_rank(
    const ArtifactCandidate& candidate,
    const ArtifactRequest& request) noexcept {
    for (std::size_t index = 0; index < request.provider_preference.size(); ++index) {
        if (request.provider_preference[index] == candidate.artifact.provider_id) {
            return index;
        }
    }
    return request.provider_preference.size();
}

std::uint8_t target_match_rank(
    const ArtifactCandidate& candidate,
    const ArtifactRequest& request) noexcept {
    if (!request.target_profile_id.empty() &&
        candidate.target_profile_id == request.target_profile_id) {
        return 0U;
    }
    if (candidate.hardware_compatibility == HardwareCompatibility::exact) {
        if (!request.target_device.empty() &&
            candidate.target_device == request.target_device &&
            (request.compute_capability.empty() ||
             candidate.compute_capability == request.compute_capability)) {
            return 0U;
        }
        return std::numeric_limits<std::uint8_t>::max();
    }
    if (candidate.hardware_compatibility == HardwareCompatibility::same_compute_capability &&
        !request.compute_capability.empty() &&
        candidate.compute_capability == request.compute_capability) {
        return 1U;
    }
    if (candidate.hardware_compatibility == HardwareCompatibility::ampere_plus) {
        const int candidate_major = compute_major(candidate.compute_capability);
        const int requested_major = compute_major(request.compute_capability);
        if (candidate_major >= 8 && requested_major >= candidate_major) {
            return 2U;
        }
    }
    return std::numeric_limits<std::uint8_t>::max();
}

bool same_identity(
    const ArtifactCandidate& candidate,
    const ArtifactRequest& request) noexcept {
    return candidate.artifact.codec_id == request.codec_id &&
        candidate.artifact.model_set_id == request.model_set_id &&
        candidate.artifact.component_id == request.component_id;
}

}  // namespace

void Resolver::add(ArtifactCandidate candidate) {
    candidates_.push_back(std::move(candidate));
}

void Resolver::clear() noexcept {
    candidates_.clear();
}

Result<ResolvedArtifact> Resolver::resolve(const ArtifactRequest& request) const {
    if (request.codec_id.empty() || request.model_set_id.empty() ||
        request.component_id.empty() || request.precision.empty()) {
        return resolution_error(
            ErrorCode::invalid_argument,
            "codec, model set, component, and precision are required");
    }

    const auto codec_exists = std::any_of(
        candidates_.begin(), candidates_.end(), [&](const ArtifactCandidate& candidate) {
            return candidate.artifact.codec_id == request.codec_id;
        });
    if (!codec_exists) {
        return resolution_error(
            ErrorCode::missing_codec,
            "no artifact candidate is registered for codec '" + request.codec_id + "'");
    }

    const auto model_exists = std::any_of(
        candidates_.begin(), candidates_.end(), [&](const ArtifactCandidate& candidate) {
            return candidate.artifact.codec_id == request.codec_id &&
                candidate.artifact.model_set_id == request.model_set_id;
        });
    if (!model_exists) {
        return resolution_error(
            ErrorCode::missing_model_set,
            "no artifact candidate is registered for model set '" + request.model_set_id + "'");
    }

    const auto component_exists = std::any_of(
        candidates_.begin(), candidates_.end(), [&](const ArtifactCandidate& candidate) {
            return same_identity(candidate, request);
        });
    if (!component_exists) {
        return resolution_error(
            ErrorCode::missing_artifact,
            "no artifact candidate is registered for component '" + request.component_id + "'");
    }

    const auto provider_exists = std::any_of(
        candidates_.begin(), candidates_.end(), [&](const ArtifactCandidate& candidate) {
            return same_identity(candidate, request) &&
                (request.provider_id.empty() ||
                 candidate.artifact.provider_id == request.provider_id);
        });
    if (!provider_exists) {
        return resolution_error(
            ErrorCode::missing_provider,
            "no compatible provider is registered for component '" + request.component_id + "'");
    }

    bool saw_version_mismatch = false;
    bool saw_precision_mismatch = false;
    bool saw_target_mismatch = false;
    bool saw_runtime_mismatch = false;
    bool saw_digest_mismatch = false;
    bool saw_license_restriction = false;
    bool saw_unavailable = false;
    std::vector<ResolvedArtifact> matches;

    for (const auto& candidate : candidates_) {
        if (!same_identity(candidate, request) ||
            (!request.provider_id.empty() &&
             candidate.artifact.provider_id != request.provider_id)) {
            continue;
        }
        if (!compatible_version(candidate.artifact.codec_api_version, request.codec_api_version) ||
            !compatible_version(candidate.artifact.provider_api_version, request.provider_api_version) ||
            !compatible_version(candidate.artifact.model_set_version, request.model_set_version) ||
            !compatible_version(candidate.artifact.manifest_schema_version, request.manifest_schema_version)) {
            saw_version_mismatch = true;
            continue;
        }
        if (candidate.artifact.precision != request.precision) {
            saw_precision_mismatch = true;
            continue;
        }
        const auto target_rank = target_match_rank(candidate, request);
        if (target_rank == std::numeric_limits<std::uint8_t>::max()) {
            saw_target_mismatch = true;
            continue;
        }
        if ((!request.operating_system.empty() &&
             candidate.operating_system != request.operating_system) ||
            (!request.architecture.empty() &&
             candidate.architecture != request.architecture)) {
            saw_target_mismatch = true;
            continue;
        }
        if ((!request.runtime_version.empty() &&
             candidate.runtime_version != request.runtime_version) ||
            !cuda_runtime_compatible(candidate, request)) {
            saw_runtime_mismatch = true;
            continue;
        }
        if (candidate.license_restricted && !request.allow_license_restricted) {
            saw_license_restriction = true;
            continue;
        }
        if (!request.expected_digest.empty() && candidate.digest != request.expected_digest) {
            saw_digest_mismatch = true;
            continue;
        }
        if (!candidate.available || candidate.artifact.path.empty()) {
            saw_unavailable = true;
            continue;
        }
        matches.push_back(ResolvedArtifact{
            candidate,
            preference_rank(candidate, request),
            target_rank,
        });
    }

    if (!matches.empty()) {
        const auto best = std::min_element(
            matches.begin(), matches.end(), [](const ResolvedArtifact& lhs, const ResolvedArtifact& rhs) {
                if (lhs.target_match_rank != rhs.target_match_rank) {
                    return lhs.target_match_rank < rhs.target_match_rank;
                }
                if (lhs.provider_preference_rank != rhs.provider_preference_rank) {
                    return lhs.provider_preference_rank < rhs.provider_preference_rank;
                }
                return lhs.candidate.artifact.path < rhs.candidate.artifact.path;
            });
        return *best;
    }

    if (saw_license_restriction) {
        return resolution_error(
            ErrorCode::license_restricted,
            "all compatible artifacts are license-restricted");
    }
    if (saw_digest_mismatch) {
        return resolution_error(
            ErrorCode::digest_mismatch,
            "compatible artifact digest does not match the requested digest");
    }
    if (saw_target_mismatch) {
        return resolution_error(
            ErrorCode::incompatible_target,
            "no artifact matches the requested target identity");
    }
    if (saw_runtime_mismatch) {
        return resolution_error(
            ErrorCode::incompatible_version,
            "no artifact matches the requested runtime or CUDA version");
    }
    if (saw_precision_mismatch) {
        return resolution_error(
            ErrorCode::incompatible_precision,
            "no artifact matches precision '" + request.precision + "'");
    }
    if (saw_version_mismatch) {
        return resolution_error(
            ErrorCode::incompatible_version,
            "no artifact matches the requested API, model, or manifest versions");
    }
    if (saw_unavailable) {
        return resolution_error(
            ErrorCode::missing_artifact,
            "compatible artifact is not available at its recorded path");
    }
    return resolution_error(
        ErrorCode::missing_artifact,
        "no artifact candidate satisfies the resolution request");
}

}  // namespace nvcr::artifacts
