#pragma once

// DCVC-RT normative numerics (issue #216).
//
// DCVC-RT picks an entropy table for every latent symbol from predicted
// scales, so any stage whose output reaches that decision must compute
// identical bits on the encoder and every decoder. Under an exact numerics
// profile the codec declares those stages normative; which provider executes
// them is not the codec's concern.
//
// For P-frames that is nearly the whole decoder: the temporal context comes
// from the previous decoded feature, or after an I-frame and at reference
// resets from the previous decoded frame. So every decoder-side stage is
// normative: the I-frame entropy stages and synthesis, and every P-frame stage
// except the encoder-only analysis and hyper analysis. They run under
// nvcr.intconv.v1, their FP16 stage boundaries round to nearest even, the
// inter-stage transforms are single IEEE operations (products with 0/1 masks
// only, so contraction cannot change them), and scales map to tables through
// the exact threshold table (scale_index.hpp). Encoder and decoder then hold
// identical reference state, so every decoder that implements the profile
// reconstructs identical frames.

#include "nvcr/bitstream/numerics_identity.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace nvcr::dcvcrt {

// Codec profile of DCVC-RT's NVAU v2 units: the NVI1/NVP1 payload syntax.
inline constexpr std::string_view payload_profile = "dcvc-rt.v1";
// Bundle subdirectory holding the normative programs
// (scripts/backends/dcvcrt/export_normative_programs.sh).
inline constexpr std::string_view normative_program_subdirectory = "normative";

// The decoder-side closure, by stage component.
inline constexpr std::array<std::string_view, 10> normative_components{
    "i_hyper_synthesis",
    "i_spatial_prior_1",
    "i_spatial_prior_2",
    "i_spatial_prior_3",
    "i_synthesis",
    "p_reference_frame",
    "p_reference_feature",
    "p_prior",
    "p_spatial_prior",
    "p_synthesis",
};

// The nvcr.fx.v1 stages, with their input and output tensors (empty names pad).
struct FxStageTensors final {
    std::string_view component;
    std::array<std::string_view, 7> inputs;
    std::array<std::string_view, 5> outputs;
};
// A spatial step's stage takes the previous step's symbols and their means and
// returns the next step's entropy indexes and means; the priors also return the
// encoder's quantization channels as q (export_fx_programs.py).
inline constexpr std::array<FxStageTensors, 11> fx_stage_tensors{{
    {"i_prior", {"z_fixed", "latent_ref", "parity"}, {"params", "common", "indexes", "means", "q"}},
    {"i_spatial_prior_1", {"symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_spatial_prior_2", {"y_hat", "symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_spatial_prior_3", {"y_hat", "symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_synthesis", {"y_hat", "symbols", "symbol_means", "params", "qp", "parity"}, {"frame"}},
    {"p_reference_frame", {"frame", "qp"}, {"temporal_context", "context"}},
    {"p_reference_feature", {"feature", "qp"}, {"temporal_context", "context"}},
    {"p_prior", {"z_fixed", "temporal_context", "latent_ref", "parity"}, {"params", "indexes", "means", "q"}},
    {"p_spatial_prior", {"symbols", "symbol_means", "params", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"p_synthesis", {"y_hat", "symbols", "symbol_means", "params", "context", "qp", "parity"}, {"feature"}},
    {"p_reset_frame", {"feature", "qp"}, {"frame"}},
}};
// Index edges and skip threshold of the nvcr.fx.v1 stages, next to the programs.
inline constexpr std::string_view fx_entropy_file = "fx_entropy.json";

struct NormativeStages final {
    NumericsIdentity identity;  // over the components above, in order
    // One descriptor per normative stage, bound by stage id like any other
    // DCVC-RT stage.
    std::vector<provider::experimental::ExecutableStageDescriptor> descriptors;
};

// The normative stages the configuration selects, with program digests, or
// nullopt for native numerics. Fails for a profile DCVC-RT does not declare
// or a missing program.
[[nodiscard]] Result<std::optional<NormativeStages>> normative_stages(
    const RuntimeConfiguration& configuration);

}  // namespace nvcr::dcvcrt
