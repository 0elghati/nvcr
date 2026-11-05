#pragma once

// Provider-side helpers for normative stages (issue #216).
//
// A codec marks a stage normative by setting ExecutableStageDescriptor::
// numerics_profile to an NVCR exact numerics profile id. The stage artifact is
// then a portable program (<component>.detprog.json plus its weights) rather
// than a provider format, and artifact.expected_digest is the SHA-256 of the
// program JSON, which itself pins the weights digest. A provider that
// implements the profile loads the program with load_normative_program() and
// must produce exactly the bits of the CPU reference.

#include "program.hpp"

#include "nvcr/common/error.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::deterministic {

// The profile a descriptor names, or nullopt for a free (native) stage.
// Unknown profile ids are an incompatible_precision error.
[[nodiscard]] Result<std::optional<NumericsProfile>> requested_profile(
    const provider::experimental::ExecutableStageDescriptor& descriptor);

[[nodiscard]] bool known_profile(std::string_view profile) noexcept;

struct NormativeProgram final {
    NumericsProfile profile{NumericsProfile::integer_v1};
    std::shared_ptr<const Program> program;
};

// Loads and digest-checks the program a normative descriptor names.
[[nodiscard]] Result<NormativeProgram> load_normative_program(
    const provider::experimental::ExecutableStageDescriptor& descriptor,
    NumericsProfile profile);

// Output extents (program.outputs order) for the given input extents
// (program.inputs order).
[[nodiscard]] Result<std::vector<Extent>> infer_output_extents(
    const Program& program, std::span<const Extent> inputs);

// SHA-256 (lowercase hex) of <directory>/<component>.detprog.json.
[[nodiscard]] Result<std::string> program_digest(
    const std::filesystem::path& directory, std::string_view component);

// IEEE binary32 -> binary16, round to nearest even (as __float2half_rn).
[[nodiscard]] std::uint16_t float_to_half_bits(float value) noexcept;

// Runs a normative program on FP16 host tensors through the CPU reference.
// Tensor names and NCHW shapes (batch 1) must match the program's inputs and
// outputs; FP16 values widen exactly and results round to nearest even.
struct HalfTensorView final {
    std::string_view name;
    std::span<const std::int64_t> shape;
    std::span<std::uint16_t> values;
};

[[nodiscard]] Result<void> run_half_cpu(
    const NormativeProgram& program,
    std::span<const HalfTensorView> inputs,
    std::span<const HalfTensorView> outputs);

}  // namespace nvcr::deterministic
