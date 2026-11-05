#include "numerics.hpp"

#include "../deterministic/normative_stage.hpp"

#include <string>

namespace nvcr::dcvcrt {
namespace {

Error numerics_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "dcvcrt.numerics");
}

std::optional<std::array<std::byte, 32>> parse_digest(std::string_view hex) {
    if (hex.size() != 64) return std::nullopt;
    std::array<std::byte, 32> digest{};
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const int high = nibble(hex[2 * index]);
        const int low = nibble(hex[2 * index + 1]);
        if (high < 0 || low < 0) return std::nullopt;
        digest[index] = static_cast<std::byte>((high << 4) | low);
    }
    return digest;
}

}  // namespace

Result<std::optional<NormativeStages>> normative_stages(const RuntimeConfiguration& configuration) {
    const auto& profile = configuration.codec.numerics_profile;
    if (profile.empty()) return std::optional<NormativeStages>{};
    if (profile != exact_numerics_profile) {
        return numerics_error(ErrorCode::incompatible_precision,
                              "DCVC-RT declares its normative stages for " +
                                  std::string(exact_numerics_profile) + " only, not " + profile);
    }
    // The programs ship in the artifact bundle (<bundle>/normative) unless the
    // configuration names another directory.
    auto directory = configuration.artifacts.normative_program_path;
    if (directory.empty() && !configuration.artifacts.intra_engine_path.empty()) {
        directory = configuration.artifacts.intra_engine_path / normative_program_subdirectory;
    }
    if (directory.empty()) {
        return numerics_error(ErrorCode::missing_artifact,
                              std::string(exact_numerics_profile) +
                                  " requires the artifact bundle's normative programs");
    }
    NormativeStages stages;
    std::vector<NormativeStageIdentity> identities;
    for (const auto component : normative_components) {
        auto digest = deterministic::program_digest(directory, component);
        if (!digest) {
            return numerics_error(ErrorCode::missing_artifact,
                                  "normative program " + std::string(component) + " is unavailable in " +
                                      directory.string() + ": " + std::string(digest.error().message()));
        }
        auto bytes = parse_digest(digest.value());
        if (!bytes) return numerics_error(ErrorCode::internal_error, "program digest is not SHA-256");
        identities.push_back(NormativeStageIdentity{std::string(component), *bytes});

        provider::experimental::ExecutableStageDescriptor descriptor;
        descriptor.stage_id = std::string(component) + ".plan";
        descriptor.artifact.codec_id = "dcvc-rt";
        descriptor.artifact.model_set_id = configuration.artifacts.model_id;
        descriptor.artifact.component_id = std::string(component);
        descriptor.artifact.provider_id = configuration.provider.id;
        descriptor.artifact.precision = "fp16";
        descriptor.artifact.path = directory.string();
        descriptor.artifact.expected_digest.assign(bytes->begin(), bytes->end());
        descriptor.numerics_profile = profile;
        stages.descriptors.push_back(std::move(descriptor));
    }
    auto identity = NumericsIdentityIO::make(profile, identities);
    if (!identity) return identity.error();
    stages.identity = std::move(identity.value());
    return std::optional<NormativeStages>(std::move(stages));
}

}  // namespace nvcr::dcvcrt
