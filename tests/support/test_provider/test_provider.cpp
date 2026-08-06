#include "test_provider.hpp"

#include "nvcr/runtime/registry.hpp"

#include <algorithm>
#include <memory>

namespace nvcr::test_support {
namespace {

class TestExecutable final : public provider::IExecutable {
public:
    explicit TestExecutable(provider::ArtifactDescriptor descriptor)
        : descriptor_(std::move(descriptor)) {}

    [[nodiscard]] provider::ArtifactDescriptor descriptor() const override {
        return descriptor_;
    }

    [[nodiscard]] Result<void> execute(
        const provider::TensorBindings& inputs,
        provider::TensorBindings& outputs,
        const provider::ExecutionOptions&) override {
        if (inputs.empty() || outputs.empty()) {
            return Error(ErrorCode::invalid_argument,
                         "test provider requires at least one input and output",
                         "test-provider");
        }
        auto input = inputs.front();
        auto& output = outputs.front();
        if (input.name.empty() || output.name.empty() || input.data.size() != output.data.size() ||
            input.dtype != output.dtype || input.shape != output.shape) {
            return Error(ErrorCode::invalid_argument,
                         "test provider tensor bindings do not match",
                         "test-provider");
        }
        std::copy(input.data.begin(), input.data.end(), output.data.begin());
        return {};
    }

private:
    provider::ArtifactDescriptor descriptor_;
};

class TestProvider final : public provider::IExecutionProvider {
public:
    [[nodiscard]] provider::ProviderDescriptor descriptor() const override {
        return {
            provider::ProviderKind::cpu_reference,
            "test-cpu",
            "1.0.0",
            "Deterministic Test CPU Provider",
        };
    }

    [[nodiscard]] provider::ProviderCapabilities capabilities() const override {
        return {
            .supports_fp16 = false,
            .supports_int8 = true,
            .supports_dynamic_shapes = true,
            .supports_cuda_graphs = false,
            .target_device = "cpu",
        };
    }

    [[nodiscard]] bool supports(const provider::ArtifactDescriptor& artifact) const override {
        return artifact.provider_id == "test-cpu";
    }

    [[nodiscard]] Result<std::shared_ptr<provider::IExecutable>>
    load(const provider::ArtifactDescriptor& artifact) override {
        if (artifact.component_id.empty() || artifact.path.empty()) {
            return Error(ErrorCode::invalid_argument,
                         "test provider requires component and artifact path",
                         "test-provider");
        }
        if (artifact.precision != "int8") {
            return Error(ErrorCode::incompatible_target,
                         "test provider only supports int8 artifacts",
                         "test-provider");
        }
        return std::static_pointer_cast<provider::IExecutable>(
            std::make_shared<TestExecutable>(artifact));
    }
};

}  // namespace

void register_test_provider() {
    runtime::Registry::instance().register_provider({
        {
            provider::ProviderKind::cpu_reference,
            "test-cpu",
            "1.0.0",
            "Deterministic Test CPU Provider",
        },
        {
            .supports_fp16 = false,
            .supports_int8 = true,
            .supports_dynamic_shapes = true,
            .supports_cuda_graphs = false,
            .target_device = "cpu",
        },
        []() -> std::shared_ptr<provider::IExecutionProvider> {
            return std::make_shared<TestProvider>();
        },
    });
}

}  // namespace nvcr::test_support
