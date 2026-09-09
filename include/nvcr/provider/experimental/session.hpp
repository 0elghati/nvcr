#pragma once

// Experimental provider-session vocabulary for the NVCR v2 boundary.
//
// This contract is intentionally separate from provider_api.hpp. Production
// construction continues to use the v1 provider facade while this API is
// exercised by deterministic fixtures and the TensorRT extraction is staged.

#include "nvcr/common/error.hpp"
#include "nvcr/provider/provider_api.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::provider::experimental {

enum class MemoryDomain : std::uint8_t {
    host = 0,
    pinned_host,
    provider_device,
};

enum class TensorDataType : std::uint8_t {
    uint8 = 0,
    int8,
    int32,
    float16,
    float32,
};

enum class TensorAccess : std::uint8_t {
    read = 0,
    write,
    read_write,
};

struct DimensionBounds final {
    std::int64_t minimum{};
    std::int64_t maximum{};
};

// A provider buffer is opaque to the codec. The shared handle keeps the
// storage alive through every submission and completion that retains it.
// owner_id is a stable identity for the creating session and must be unique
// among live sessions.
class IProviderBuffer {
public:
    virtual ~IProviderBuffer() = default;

    [[nodiscard]] virtual std::size_t size_bytes() const noexcept = 0;
    [[nodiscard]] virtual MemoryDomain domain() const noexcept = 0;
    [[nodiscard]] virtual std::string_view owner_id() const noexcept = 0;
};

using BufferHandle = std::shared_ptr<IProviderBuffer>;

struct BufferSlice final {
    BufferHandle buffer;
    std::size_t offset_bytes{};
    std::size_t size_bytes{};
};

struct TensorContract final {
    std::string name;
    TensorDataType data_type{};
    TensorAccess access{};
    std::vector<DimensionBounds> dimensions;
};

struct TensorView final {
    // Metadata is borrowed only for submit(); the buffer handle may be retained
    // until the returned Completion finishes.
    std::string_view name;
    TensorDataType data_type{};
    std::vector<std::int64_t> shape;
    TensorAccess access{};
    BufferSlice storage;
};

// The descriptor and tensor contract are immutable after the stage is loaded.
struct ExecutableStageDescriptor final {
    std::string stage_id;
    ArtifactDescriptor artifact;
    std::vector<TensorContract> tensors;
};

class IExecutableStage {
public:
    virtual ~IExecutableStage() = default;

    [[nodiscard]] virtual const ExecutableStageDescriptor& descriptor() const noexcept = 0;
    [[nodiscard]] virtual std::string_view owner_id() const noexcept = 0;
};

using ExecutableStage = std::shared_ptr<const IExecutableStage>;

// Completion represents provider work without exposing a provider event type.
// wait() reports asynchronous failures and may be called more than once.
class ICompletion {
public:
    virtual ~ICompletion() = default;

    [[nodiscard]] virtual std::string_view owner_id() const noexcept = 0;
    [[nodiscard]] virtual bool ready() const noexcept = 0;
    [[nodiscard]] virtual Result<void> wait() = 0;
};

using Completion = std::shared_ptr<ICompletion>;

struct ExecutionDependency final {
    Completion completion;
};

class IProviderSession {
public:
    virtual ~IProviderSession() = default;

    [[nodiscard]] virtual std::string_view owner_id() const noexcept = 0;

    [[nodiscard]] virtual Result<BufferHandle> allocate(
        std::size_t size_bytes,
        MemoryDomain domain) = 0;

    [[nodiscard]] virtual Result<ExecutableStage> load_stage(
        ExecutableStageDescriptor descriptor) = 0;

    // All stages, buffers, and dependencies must belong to this session.
    // Implementations validate names, shapes, access modes, and byte bounds
    // before enqueueing work.
    [[nodiscard]] virtual Result<Completion> submit(
        const ExecutableStage& stage,
        std::span<const TensorView> tensors,
        std::span<const ExecutionDependency> dependencies) = 0;

    // Wait for this session's outstanding work and clear transient execution
    // state. Loaded stages and retained buffers remain valid. Destruction has
    // the same wait-for-owned-work rule but cannot report completion errors.
    [[nodiscard]] virtual Result<void> reset() = 0;
};

}  // namespace nvcr::provider::experimental
