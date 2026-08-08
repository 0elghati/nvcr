#include "test_provider.hpp"

#include "nvcr/codec/backend.hpp"
#include "nvcr/runtime/registry.hpp"

#include <algorithm>
#include <memory>
#include <vector>

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

class TestProviderCodecBackend final : public codec::CodecBackend {
public:
    [[nodiscard]] Result<void> initialize(const RuntimeConfiguration&) override {
        initialized_ = true;
        return {};
    }

    [[nodiscard]] Result<codec::CodecEncodeResult> encode(
        const Frame& frame,
        FrameType,
        const codec::SequenceStateView&) override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "test provider backend is not initialized", "test-provider");
        }
        auto reconstructed = Frame::copy_from(
            frame.width(), frame.height(), frame.pixel_format(), frame.data(), frame.timestamp());
        if (!reconstructed) return reconstructed.error();
        return codec::CodecEncodeResult{
            std::vector<std::byte>(frame.data().begin(), frame.data().end()),
            std::move(reconstructed.value()),
            {},
            0U,
        };
    }

    [[nodiscard]] Result<codec::CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType,
        Timestamp timestamp,
        const codec::SequenceStateView&) override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "test provider backend is not initialized", "test-provider");
        }
        auto frame = Frame::copy_from(
            4U, 2U, PixelFormat::yuv420p8, payload, timestamp);
        if (!frame) return frame.error();
        return codec::CodecDecodeResult{std::move(frame.value()), {}};
    }

    [[nodiscard]] Result<void> flush() override { return {}; }

    void reset() noexcept override {}

private:
    bool initialized_{};
};

Result<codec::Components> make_test_provider_components(const RuntimeConfiguration&) {
    codec::Components components;
    components.codec = std::make_unique<TestProviderCodecBackend>();
    return components;
}

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
        make_test_provider_components,
    });
}

}  // namespace nvcr::test_support
