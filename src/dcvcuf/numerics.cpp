#include "numerics.hpp"

#include "nvcr/numerics/profiles.hpp"

namespace nvcr::dcvcuf {

Result<std::optional<codec::NormativeStages>> normative_stages(const RuntimeConfiguration& configuration) {
    const auto& profile = configuration.codec.numerics_profile;
    if (profile.empty() || profile == numerics::native) return std::optional<codec::NormativeStages>{};
    if (profile != numerics::fx_v1) {
        return Error(ErrorCode::incompatible_precision, "DCVC-UF does not run under " + profile, "dcvcuf.numerics");
    }
    // The programs ship in the artifact bundle (<bundle>/normative) unless the
    // configuration names another directory.
    auto directory = configuration.artifacts.normative_program_path;
    if (directory.empty()) directory = configuration.artifacts.intra_engine_path / normative_program_subdirectory;
    auto stages = codec::fx::normative_stages(configuration, "dcvc-uf", directory, fx_stage_tensors, fx_entropy_file);
    if (!stages) return stages.error();
    return std::optional<codec::NormativeStages>(std::move(stages.value()));
}

}  // namespace nvcr::dcvcuf
