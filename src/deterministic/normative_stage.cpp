#include "normative_stage.hpp"

#include "det_math.hpp"

#include "../common/sha256.hpp"

#include <cstring>
#include <filesystem>
#include <vector>

namespace nvcr::deterministic {
namespace {

constexpr std::string_view subsystem = "deterministic.stage";

Error error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), std::string(subsystem));
}

std::string hex(std::span<const std::byte> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(bytes.size() * 2);
    for (const auto byte : bytes) {
        const auto value = static_cast<unsigned int>(byte);
        text.push_back(digits[value >> 4U]);
        text.push_back(digits[value & 15U]);
    }
    return text;
}

}  // namespace

Result<bool> is_normative(const provider::experimental::ExecutableStageDescriptor& descriptor) {
    const auto& profile = descriptor.numerics_profile;
    if (profile.empty()) return false;
    if (profile == intconv_profile) return true;
    return error(ErrorCode::incompatible_precision,
                 "unknown numerics profile '" + profile + "' for stage " + descriptor.stage_id);
}

Result<std::vector<Extent>> infer_output_extents(const Program& program, std::span<const Extent> inputs) {
    if (inputs.size() != program.inputs.size()) {
        return error(ErrorCode::invalid_argument, program.name + ": wrong number of inputs");
    }
    std::vector<std::optional<Extent>> extents(program.tensor_names.size());
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) extents[program.inputs[slot]] = inputs[slot];
    for (const auto& node : program.nodes) {
        std::vector<Extent> node_inputs;
        for (std::size_t input = 0; input < node.input_count; ++input) {
            const auto& extent = extents[node.inputs[input]];
            if (!extent) return error(ErrorCode::invalid_argument, program.name + ": tensor read before written");
            node_inputs.push_back(*extent);
        }
        auto extent = output_extent(program, node, node_inputs);
        if (!extent) return extent.error();
        extents[node.output] = extent.value();
    }
    std::vector<Extent> outputs;
    for (const auto id : program.outputs) {
        if (!extents[id]) return error(ErrorCode::invalid_argument, program.name + ": output not produced");
        outputs.push_back(*extents[id]);
    }
    return outputs;
}

Result<std::string> program_digest(const std::filesystem::path& directory, std::string_view component) {
    return detail::sha256_file(directory / (std::string(component) + ".detprog.json"), subsystem);
}

Result<NormativeProgram> load_normative_program(
    const provider::experimental::ExecutableStageDescriptor& descriptor) {
    const auto& artifact = descriptor.artifact;
    if (artifact.expected_digest.empty()) {
        return error(ErrorCode::invalid_argument,
                     "normative stage " + descriptor.stage_id + " has no program digest");
    }
    const auto path = std::filesystem::path(artifact.path) / (artifact.component_id + ".detprog.json");
    std::error_code status;
    if (!std::filesystem::is_regular_file(path, status)) {
        return error(ErrorCode::missing_artifact, "normative program is missing: " + path.string());
    }
    auto digest = detail::sha256_file(path, subsystem);
    if (!digest) return digest.error();
    if (digest.value() != hex(artifact.expected_digest)) {
        return error(ErrorCode::digest_mismatch, "normative program digest mismatch: " + path.string());
    }
    auto program = load_program(path);
    if (!program) return program.error();
    return NormativeProgram{std::make_shared<const Program>(std::move(program.value()))};
}

std::uint16_t float_to_half_bits(float value) noexcept { return det_float_to_half_bits(value); }

Result<void> run_half_cpu(
    const NormativeProgram& normative,
    std::span<const HalfTensorView> inputs,
    std::span<const HalfTensorView> outputs) {
    const auto& program = *normative.program;
    auto find = [](std::span<const HalfTensorView> views, std::string_view name) -> const HalfTensorView* {
        for (const auto& view : views) {
            if (view.name == name) return &view;
        }
        return nullptr;
    };
    std::vector<Tensor> tensors;
    tensors.reserve(program.inputs.size());
    for (const auto id : program.inputs) {
        const auto& name = program.tensor_names[id];
        const auto* view = find(inputs, name);
        if (view == nullptr || view->shape.size() != 4 || view->shape[0] != 1 ||
            view->shape[1] != program.tensor_channels[id]) {
            return error(ErrorCode::invalid_argument,
                         program.name + ": missing or misshapen input " + name);
        }
        Tensor tensor;
        tensor.channels = static_cast<std::int32_t>(view->shape[1]);
        tensor.height = static_cast<std::int32_t>(view->shape[2]);
        tensor.width = static_cast<std::int32_t>(view->shape[3]);
        if (view->values.size() != tensor.size()) {
            return error(ErrorCode::invalid_argument, program.name + ": input size mismatch for " + name);
        }
        tensor.values.resize(tensor.size());
        for (std::size_t index = 0; index < tensor.values.size(); ++index) {
            tensor.values[index] = det_half_bits_to_float(view->values[index]);
        }
        tensors.push_back(std::move(tensor));
    }
    auto results = run_cpu(program, tensors);
    if (!results) return results.error();
    for (std::size_t slot = 0; slot < program.outputs.size(); ++slot) {
        const auto& name = program.tensor_names[program.outputs[slot]];
        const auto* view = find(outputs, name);
        const auto& result = results.value()[slot];
        if (view == nullptr || view->shape.size() != 4 || view->shape[0] != 1 ||
            view->shape[1] != result.channels || view->shape[2] != result.height ||
            view->shape[3] != result.width || view->values.size() != result.values.size()) {
            return error(ErrorCode::invalid_argument,
                         program.name + ": missing or misshapen output " + name);
        }
        for (std::size_t index = 0; index < result.values.size(); ++index) {
            view->values[index] = float_to_half_bits(result.values[index]);
        }
    }
    return {};
}

}  // namespace nvcr::deterministic
