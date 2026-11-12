#include "numerics.hpp"

#include "../common/sha256.hpp"
#include "../deterministic/fx_program.hpp"
#include "../deterministic/normative_stage.hpp"
#include "nvcr/numerics/profiles.hpp"

#include <string>

namespace nvcr::dcvcrt {
namespace {

Error numerics_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "dcvcrt.numerics");
}

}  // namespace

Result<std::optional<NormativeStages>> normative_stages(const RuntimeConfiguration& configuration) {
    const auto& profile = configuration.codec.numerics_profile;
    if (profile.empty() || profile == nvcr::numerics::native) return std::optional<NormativeStages>{};
    // The programs ship in the artifact bundle (<bundle>/normative) unless the
    // configuration names another directory.
    auto directory = configuration.artifacts.normative_program_path;
    if (directory.empty() && !configuration.artifacts.intra_engine_path.empty()) {
        directory = configuration.artifacts.intra_engine_path / normative_program_subdirectory;
    }
    if (directory.empty()) {
        return numerics_error(ErrorCode::missing_artifact,
                              profile + " requires the artifact bundle's normative programs");
    }
    if (profile == nvcr::numerics::fx_v1) {
        auto stages = codec::fx::normative_stages(configuration, "dcvc-rt", directory, fx_stage_tensors, fx_entropy_file);
        if (!stages) return stages.error();
        return std::optional<NormativeStages>(std::move(stages.value()));
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
        auto bytes = detail::sha256_bytes(digest.value());
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
