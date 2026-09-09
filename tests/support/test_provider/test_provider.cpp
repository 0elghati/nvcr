#include "test_provider.hpp"

#include "nvcr/codec/backend.hpp"
#include "nvcr/runtime/registry.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nvcr::test_support {
namespace {

namespace session_api = provider::experimental;

struct TestSessionIdentity final {
    std::string id;
};

class TestProviderBuffer final : public session_api::IProviderBuffer {
public:
    TestProviderBuffer(
        std::shared_ptr<const TestSessionIdentity> owner,
        std::size_t size,
        session_api::MemoryDomain domain)
        : owner_(std::move(owner)), bytes_(size), domain_(domain) {}

    [[nodiscard]] std::size_t size_bytes() const noexcept override {
        return bytes_.size();
    }

    [[nodiscard]] session_api::MemoryDomain domain() const noexcept override {
        return domain_;
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_->id;
    }

    [[nodiscard]] const std::shared_ptr<const TestSessionIdentity>& owner() const noexcept {
        return owner_;
    }

    [[nodiscard]] std::span<std::byte> bytes() noexcept { return bytes_; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }

private:
    std::shared_ptr<const TestSessionIdentity> owner_;
    std::vector<std::byte> bytes_;
    session_api::MemoryDomain domain_;
};

class TestExecutableStage final : public session_api::IExecutableStage {
public:
    TestExecutableStage(
        std::shared_ptr<const TestSessionIdentity> owner,
        session_api::ExecutableStageDescriptor descriptor)
        : owner_(std::move(owner)), descriptor_(std::move(descriptor)) {}

    [[nodiscard]] const session_api::ExecutableStageDescriptor&
    descriptor() const noexcept override {
        return descriptor_;
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_->id;
    }

    [[nodiscard]] const std::shared_ptr<const TestSessionIdentity>& owner() const noexcept {
        return owner_;
    }

private:
    std::shared_ptr<const TestSessionIdentity> owner_;
    session_api::ExecutableStageDescriptor descriptor_;
};

class TestCompletion final : public session_api::ICompletion {
public:
    TestCompletion(
        std::shared_ptr<const TestSessionIdentity> owner,
        std::vector<std::shared_ptr<TestCompletion>> dependencies,
        std::function<Result<void>()> action)
        : owner_(std::move(owner)),
          dependencies_(std::move(dependencies)),
          action_(std::move(action)) {}

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_->id;
    }

    [[nodiscard]] bool ready() const noexcept override {
        return state_ == State::complete;
    }

    [[nodiscard]] Result<void> wait() override {
        if (state_ == State::complete) {
            return error_ ? Result<void>(*error_) : Result<void>();
        }
        if (state_ == State::running) {
            return Error(
                ErrorCode::invalid_state,
                "test provider dependency cycle",
                "test-provider-session");
        }

        state_ = State::running;
        for (const auto& dependency : dependencies_) {
            auto waited = dependency->wait();
            if (!waited) {
                error_ = waited.error();
                state_ = State::complete;
                return *error_;
            }
        }
        auto executed = action_();
        if (!executed) error_ = executed.error();
        state_ = State::complete;
        return error_ ? Result<void>(*error_) : Result<void>();
    }

    [[nodiscard]] const std::shared_ptr<const TestSessionIdentity>& owner() const noexcept {
        return owner_;
    }

private:
    enum class State : std::uint8_t {
        pending,
        running,
        complete,
    };

    std::shared_ptr<const TestSessionIdentity> owner_;
    std::vector<std::shared_ptr<TestCompletion>> dependencies_;
    std::function<Result<void>()> action_;
    State state_{State::pending};
    std::optional<Error> error_;
};

[[nodiscard]] std::size_t data_type_size(session_api::TensorDataType type) {
    switch (type) {
        case session_api::TensorDataType::uint8:
        case session_api::TensorDataType::int8:
            return 1U;
        case session_api::TensorDataType::float16:
            return 2U;
        case session_api::TensorDataType::int32:
        case session_api::TensorDataType::float32:
            return 4U;
    }
    return 0U;
}

[[nodiscard]] Result<std::size_t> tensor_size_bytes(
    const session_api::TensorContract& contract,
    const session_api::TensorView& view) {
    if (view.shape.size() != contract.dimensions.size()) {
        return Error(
            ErrorCode::invalid_argument,
            "test provider tensor rank does not match contract",
            "test-provider-session");
    }

    std::size_t elements = 1U;
    for (std::size_t index = 0; index < view.shape.size(); ++index) {
        const auto dimension = view.shape[index];
        const auto bounds = contract.dimensions[index];
        if (dimension < bounds.minimum || dimension > bounds.maximum || dimension <= 0) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider tensor shape is outside contract bounds",
                "test-provider-session");
        }
        const auto extent = static_cast<std::size_t>(dimension);
        if (elements > std::numeric_limits<std::size_t>::max() / extent) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider tensor shape overflows byte size",
                "test-provider-session");
        }
        elements *= extent;
    }

    const auto element_size = data_type_size(view.data_type);
    if (element_size == 0U ||
        elements > std::numeric_limits<std::size_t>::max() / element_size) {
        return Error(
            ErrorCode::invalid_argument,
            "test provider tensor byte size is invalid",
            "test-provider-session");
    }
    return elements * element_size;
}

class TestProviderSession final : public session_api::IProviderSession {
public:
    TestProviderSession()
        : owner_(std::make_shared<TestSessionIdentity>(
              TestSessionIdentity{"test-session-" + std::to_string(next_id_.fetch_add(1U))})) {}

    ~TestProviderSession() override {
        (void)wait_for_owned_work();
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_->id;
    }

    [[nodiscard]] Result<session_api::BufferHandle> allocate(
        std::size_t size_bytes,
        session_api::MemoryDomain domain) override {
        constexpr std::size_t max_test_allocation = 1024U * 1024U;
        if (size_bytes == 0U) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider buffer size must be non-zero",
                "test-provider-session");
        }
        if (size_bytes > max_test_allocation) {
            return Error(
                ErrorCode::resource_exhausted,
                "test provider allocation exceeds deterministic fixture limit",
                "test-provider-session");
        }
        return std::static_pointer_cast<session_api::IProviderBuffer>(
            std::make_shared<TestProviderBuffer>(owner_, size_bytes, domain));
    }

    [[nodiscard]] Result<session_api::ExecutableStage> load_stage(
        session_api::ExecutableStageDescriptor descriptor) override {
        if (descriptor.stage_id.empty() || descriptor.artifact.component_id.empty() ||
            descriptor.artifact.path.empty()) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider stage identity is incomplete",
                "test-provider-session");
        }
        if (descriptor.artifact.provider_id != "test-cpu") {
            return Error(
                ErrorCode::incompatible_target,
                "test provider stage targets another provider",
                "test-provider-session");
        }
        if (descriptor.tensors.empty()) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider stage has no tensor contract",
                "test-provider-session");
        }
        for (std::size_t index = 0; index < descriptor.tensors.size(); ++index) {
            const auto& tensor = descriptor.tensors[index];
            if (tensor.name.empty()) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider tensor contract has no name",
                    "test-provider-session");
            }
            const auto duplicate = std::find_if(
                descriptor.tensors.begin() + static_cast<std::ptrdiff_t>(index + 1U),
                descriptor.tensors.end(),
                [&](const session_api::TensorContract& candidate) {
                    return candidate.name == tensor.name;
                });
            if (duplicate != descriptor.tensors.end()) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider tensor contract contains duplicate names",
                    "test-provider-session");
            }
            for (const auto bounds : tensor.dimensions) {
                if (bounds.minimum <= 0 || bounds.maximum < bounds.minimum) {
                    return Error(
                        ErrorCode::invalid_argument,
                        "test provider tensor contract has invalid bounds",
                        "test-provider-session");
                }
            }
        }
        return std::static_pointer_cast<const session_api::IExecutableStage>(
            std::make_shared<TestExecutableStage>(owner_, std::move(descriptor)));
    }

    [[nodiscard]] Result<session_api::Completion> submit(
        const session_api::ExecutableStage& stage,
        std::span<const session_api::TensorView> tensors,
        std::span<const session_api::ExecutionDependency> dependencies) override {
        const auto test_stage = std::dynamic_pointer_cast<const TestExecutableStage>(stage);
        if (!test_stage || test_stage->owner() != owner_) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider stage belongs to another session",
                "test-provider-session");
        }

        std::vector<std::shared_ptr<TestCompletion>> test_dependencies;
        test_dependencies.reserve(dependencies.size());
        for (const auto& dependency : dependencies) {
            const auto completion =
                std::dynamic_pointer_cast<TestCompletion>(dependency.completion);
            if (!completion || completion->owner() != owner_) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider dependency belongs to another session",
                    "test-provider-session");
            }
            test_dependencies.push_back(completion);
        }

        const auto& contracts = test_stage->descriptor().tensors;
        if (tensors.size() != contracts.size()) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider tensor count does not match stage contract",
                "test-provider-session");
        }

        struct ValidatedTensor final {
            std::shared_ptr<TestProviderBuffer> buffer;
            std::size_t offset{};
            std::size_t size{};
            session_api::TensorAccess access{};
        };
        std::vector<ValidatedTensor> validated;
        validated.reserve(contracts.size());
        for (const auto& contract : contracts) {
            const auto match = std::find_if(
                tensors.begin(), tensors.end(),
                [&](const session_api::TensorView& view) {
                    return view.name == contract.name;
                });
            if (match == tensors.end() ||
                std::count_if(
                    tensors.begin(), tensors.end(),
                    [&](const session_api::TensorView& view) {
                        return view.name == contract.name;
                    }) != 1) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider tensor names do not match stage contract",
                    "test-provider-session");
            }
            if (match->data_type != contract.data_type || match->access != contract.access) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider tensor type or access does not match contract",
                    "test-provider-session");
            }
            const auto buffer =
                std::dynamic_pointer_cast<TestProviderBuffer>(match->storage.buffer);
            if (!buffer || buffer->owner() != owner_) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider buffer belongs to another session",
                    "test-provider-session");
            }
            if (match->storage.offset_bytes > buffer->size_bytes() ||
                match->storage.size_bytes >
                    buffer->size_bytes() - match->storage.offset_bytes) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider buffer slice is outside allocation bounds",
                    "test-provider-session");
            }
            auto required_size = tensor_size_bytes(contract, *match);
            if (!required_size) return required_size.error();
            if (required_size.value() > match->storage.size_bytes) {
                return Error(
                    ErrorCode::invalid_argument,
                    "test provider tensor exceeds its buffer slice",
                    "test-provider-session");
            }
            validated.push_back({
                std::move(buffer),
                match->storage.offset_bytes,
                required_size.value(),
                contract.access,
            });
        }

        const auto source = std::find_if(
            validated.begin(), validated.end(),
            [](const ValidatedTensor& tensor) {
                return tensor.access == session_api::TensorAccess::read;
            });
        const auto destination = std::find_if(
            validated.begin(), validated.end(),
            [](const ValidatedTensor& tensor) {
                return tensor.access == session_api::TensorAccess::write;
            });
        if (source == validated.end() || destination == validated.end() ||
            source->size != destination->size) {
            return Error(
                ErrorCode::invalid_argument,
                "test provider identity stage requires equal input and output tensors",
                "test-provider-session");
        }

        const auto fail = test_stage->descriptor().stage_id == "fail";
        auto action = [input = *source, output = *destination, fail]() -> Result<void> {
            if (fail) {
                return Error(
                    ErrorCode::backend_error,
                    "deterministic test provider execution failure",
                    "test-provider-session");
            }
            std::copy_n(
                input.buffer->bytes().begin() + static_cast<std::ptrdiff_t>(input.offset),
                static_cast<std::ptrdiff_t>(input.size),
                output.buffer->bytes().begin() + static_cast<std::ptrdiff_t>(output.offset));
            return {};
        };
        auto completion = std::make_shared<TestCompletion>(
            owner_, std::move(test_dependencies), std::move(action));
        owned_completions_.push_back(completion);
        return std::static_pointer_cast<session_api::ICompletion>(completion);
    }

    [[nodiscard]] Result<void> reset() override {
        auto waited = wait_for_owned_work();
        owned_completions_.clear();
        return waited;
    }

private:
    [[nodiscard]] Result<void> wait_for_owned_work() {
        std::optional<Error> first_error;
        for (const auto& completion : owned_completions_) {
            auto waited = completion->wait();
            if (!waited && !first_error) first_error = waited.error();
        }
        return first_error ? Result<void>(*first_error) : Result<void>();
    }

    inline static std::atomic_uint64_t next_id_{1U};
    std::shared_ptr<const TestSessionIdentity> owner_;
    std::vector<std::shared_ptr<TestCompletion>> owned_completions_;
};

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

std::shared_ptr<provider::experimental::IProviderSession>
make_test_provider_session() {
    return std::make_shared<TestProviderSession>();
}

Result<void> write_test_buffer(
    const provider::experimental::BufferHandle& buffer,
    std::size_t offset,
    std::span<const std::byte> bytes) {
    const auto test_buffer = std::dynamic_pointer_cast<TestProviderBuffer>(buffer);
    if (!test_buffer || offset > test_buffer->size_bytes() ||
        bytes.size() > test_buffer->size_bytes() - offset) {
        return Error(
            ErrorCode::invalid_argument,
            "test provider write is outside buffer bounds",
            "test-provider-session");
    }
    std::copy(
        bytes.begin(), bytes.end(),
        test_buffer->bytes().begin() + static_cast<std::ptrdiff_t>(offset));
    return {};
}

Result<std::vector<std::byte>> read_test_buffer(
    const provider::experimental::BufferHandle& buffer,
    std::size_t offset,
    std::size_t size) {
    const auto test_buffer = std::dynamic_pointer_cast<TestProviderBuffer>(buffer);
    if (!test_buffer || offset > test_buffer->size_bytes() ||
        size > test_buffer->size_bytes() - offset) {
        return Error(
            ErrorCode::invalid_argument,
            "test provider read is outside buffer bounds",
            "test-provider-session");
    }
    return std::vector<std::byte>(
        test_buffer->bytes().begin() + static_cast<std::ptrdiff_t>(offset),
        test_buffer->bytes().begin() + static_cast<std::ptrdiff_t>(offset + size));
}

}  // namespace nvcr::test_support
