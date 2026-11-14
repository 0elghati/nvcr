#pragma once

// DCVC-UF normative numerics (issue #216).
//
// DCVC-UF picks an entropy table for every latent symbol from predicted
// scales, and carries the HT-S feature memory and the decoded feature from one
// eight-frame chunk to the next. Under nvcr.fx.v1 every stage that reaches a
// table decision or that carried state runs as an fx program
// (scripts/backends/dcvcuf/export_fx_programs.py): the I-frame's prior,
// spatial steps and synthesis, and each chunk's references, prior, spatial
// steps and decoded feature. Encoder and decoder then hold identical state on
// every provider that implements the profile. The encoder's analysis and the
// pictures shown to the user stay FP16; nothing reads them back.

#include "../codec/fx_stages.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"

#include <array>
#include <optional>
#include <string_view>

namespace nvcr::dcvcuf {

// Codec profile of DCVC-UF's NVAU v2 units: the UFI1/UFP1 payload syntax.
inline constexpr std::string_view payload_profile = "dcvc-uf.v1";
// Bundle subdirectory holding the normative programs.
inline constexpr std::string_view normative_program_subdirectory = "normative";
// Index edges and skip threshold of the fx stages, next to the programs.
inline constexpr std::string_view fx_entropy_file = "fx_entropy.json";
// The entropy index of a skipped element (scale at or below the threshold).
inline constexpr std::int32_t skipped_index = 255;

// An I-frame spatial step takes the previous step's symbols and their means
// (the active element of each channel group) and returns the next step's
// indexes and means. A chunk's prior returns all its indexes; its spatial
// steps take the chunk's symbols and return only means.
inline constexpr std::array<codec::fx::StageTensors, 13> fx_stage_tensors{{
    {"i_prior", {"z_fixed", "latent_ref", "parity"}, {"common", "indexes", "means"}},
    {"i_spatial_prior_1", {"symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_spatial_prior_2", {"y_hat", "symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_spatial_prior_3", {"y_hat", "symbols", "symbol_means", "common", "parity"}, {"y_hat_next", "indexes", "means"}},
    {"i_synthesis", {"y_hat", "symbols", "symbol_means", "qp", "parity"}, {"frame"}},
    {"p_reference_frame", {"frame"}, {"memory", "context"}},
    {"p_reset_reference", {"feature"}, {"memory", "context"}},
    {"p_reference_memory", {"memory", "feature"}, {"memory_next", "context"}},
    {"p_prior", {"z_fixed", "memory", "qp", "latent_ref"}, {"q_dec", "indexes", "means", "reduced"}},
    {"p_spatial_prior_1", {"symbols", "symbol_means", "reduced", "parity"}, {"y_hat_next", "means"}},
    {"p_spatial_prior_2", {"y_hat", "symbols", "symbol_means", "reduced", "parity"}, {"y_hat_next", "means"}},
    {"p_spatial_prior_3", {"y_hat", "symbols", "symbol_means", "reduced", "parity"}, {"y_hat_next", "means"}},
    {"p_synthesis", {"y_hat", "symbols", "symbol_means", "q_dec", "context", "qp", "parity"}, {"feature"}},
}};

// The fx stages the configuration selects, or nullopt for native numerics.
[[nodiscard]] Result<std::optional<codec::NormativeStages>> normative_stages(
    const RuntimeConfiguration& configuration);

}  // namespace nvcr::dcvcuf
