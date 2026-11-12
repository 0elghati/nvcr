#pragma once

// Codec-agnostic TensorRT execution-provider session, internal to NVCR.
// Codecs describe their stages; the session validates each stage against its
// deserialized engine and owns the runtime, execution contexts, CUDA graphs,
// stream, device memory and bundle identity checks. Codec semantics (stage
// order, quantization, entropy, references) stay in the codec.

#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/provider/experimental/session.hpp"
#include "nvcr/provider/provider_api.hpp"
#include "nvcr/runtime/packet.hpp"

#include "../../deterministic/cuda_executor.hpp"
#include "../../deterministic/normative_stage.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::tensorrt {

namespace fs = std::filesystem;
namespace session_api = provider::experimental;

inline constexpr std::string_view subsystem = "tensorrt";

[[nodiscard]] Error backend_error(std::string message);
[[nodiscard]] Error cuda_error(std::string_view operation, cudaError_t status);
[[nodiscard]] Result<void> synchronize_event(cudaEvent_t event, std::string_view operation);

class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::clog << "[TensorRT] " << (message != nullptr ? message : "") << '\n';
        }
    }
};

struct CudaGraphInvocation final {
    ~CudaGraphInvocation() {
        if (executable != nullptr) static_cast<void>(cudaGraphExecDestroy(executable));
        if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
    }
    CudaGraphInvocation() = default;
    CudaGraphInvocation(const CudaGraphInvocation&) = delete;
    CudaGraphInvocation& operator=(const CudaGraphInvocation&) = delete;
    CudaGraphInvocation(CudaGraphInvocation&& other) noexcept
        : signature(std::move(other.signature)),
          graph(std::exchange(other.graph, nullptr)),
          executable(std::exchange(other.executable, nullptr)) {}
    CudaGraphInvocation& operator=(CudaGraphInvocation&& other) noexcept {
        if (this != &other) {
            if (executable != nullptr) static_cast<void>(cudaGraphExecDestroy(executable));
            if (graph != nullptr) static_cast<void>(cudaGraphDestroy(graph));
            signature = std::move(other.signature);
            graph = std::exchange(other.graph, nullptr);
            executable = std::exchange(other.executable, nullptr);
        }
        return *this;
    }

    std::vector<std::uintptr_t> signature;
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
};

struct EngineInstance final {
    fs::path path;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<CudaGraphInvocation> graph_invocations;
    // Execution mode is owned by this backend session, never process-global.
    bool low_memory_mode{true};
    void* shared_context_memory{};
    std::int64_t shared_context_memory_bytes{};
    bool use_cuda_graphs{};
    std::size_t graph_hits{};
    std::size_t graph_captures{};
};

constexpr std::size_t max_cuda_graph_invocations_per_engine = 16;

enum class ContextPolicy : std::uint8_t {
    per_engine,
    shared_workspace_persistent,
    persistent,
};

[[nodiscard]] std::string_view context_policy_name(ContextPolicy policy) noexcept;

// Codec-supplied TensorRT stage contract. The session validates it against the
// deserialized engine. Warm-up dimensions resolve dynamic input axes; zero
// selects the engine's optimization-profile minimum.
struct TensorRTTensorSpec final {
    std::string name;
    nvinfer1::TensorIOMode mode{};
    std::array<std::int64_t, 4> dimensions{};
    std::array<std::int64_t, 4> warmup_dimensions{};
    session_api::TensorDataType data_type{session_api::TensorDataType::float16};
    std::size_t rank{4};
};

struct TensorRTStageSpec final {
    std::string filename;
    std::vector<TensorRTTensorSpec> tensors;
};


struct ProfileStage final {
    std::string name;
    double cpu_ms{};
    double cuda_ms{};
    bool has_cuda{};
};

struct ProfileCounters final {
    std::size_t allocations{};
    std::size_t allocation_bytes{};
    std::size_t h2d_copies{};
    std::size_t h2d_bytes{};
    std::size_t d2h_copies{};
    std::size_t d2h_bytes{};
    std::size_t d2d_copies{};
    std::size_t d2d_bytes{};
    std::size_t synchronizations{};
};

struct PendingCudaStage final {
    std::string name;
    cudaEvent_t start{};
    cudaEvent_t stop{};
};

class BackendProfiler final {
public:
    explicit BackendProfiler(bool enabled) : enabled_(enabled) {}

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    void begin(std::string operation, FrameType frame_type, std::uint64_t frame_index) {
        operation_ = std::move(operation);
        frame_type_ = frame_type;
        frame_index_ = frame_index;
        counters_ = {};
        stages_.clear();
        pending_cuda_stages_.clear();
    }

    void record_allocation(std::size_t bytes) noexcept {
        if (!enabled_) return;
        ++counters_.allocations;
        counters_.allocation_bytes += bytes;
    }

    void record_transfer(cudaMemcpyKind kind, std::size_t bytes) noexcept {
        if (!enabled_) return;
        switch (kind) {
        case cudaMemcpyHostToDevice:
            ++counters_.h2d_copies;
            counters_.h2d_bytes += bytes;
            break;
        case cudaMemcpyDeviceToHost:
            ++counters_.d2h_copies;
            counters_.d2h_bytes += bytes;
            break;
        case cudaMemcpyDeviceToDevice:
            ++counters_.d2d_copies;
            counters_.d2d_bytes += bytes;
            break;
        default:
            break;
        }
    }

    void record_synchronization() noexcept {
        if (enabled_) ++counters_.synchronizations;
    }

    void record_cpu_stage(std::string name, double cpu_ms) {
        if (!enabled_) return;
        stages_.push_back(ProfileStage{std::move(name), cpu_ms, 0.0, false});
    }

    void record_cuda_stage(std::string name, cudaEvent_t start, cudaEvent_t stop) {
        if (!enabled_) return;
        pending_cuda_stages_.push_back(PendingCudaStage{std::move(name), start, stop});
    }

    void finish() noexcept {
        if (!enabled_) return;
        for (auto& pending : pending_cuda_stages_) {
            const auto synchronized = cudaEventSynchronize(pending.stop);
            float elapsed_ms = 0.0F;
            if (synchronized == cudaSuccess) {
                static_cast<void>(cudaEventElapsedTime(&elapsed_ms, pending.start, pending.stop));
            }
            static_cast<void>(cudaEventDestroy(pending.start));
            static_cast<void>(cudaEventDestroy(pending.stop));
            stages_.push_back(ProfileStage{
                std::move(pending.name), 0.0, static_cast<double>(elapsed_ms),
                synchronized == cudaSuccess});
        }
        pending_cuda_stages_.clear();

        const char type = frame_type_ == FrameType::intra ? 'I' : 'P';
        std::clog << "[nvcr.profile] " << operation_ << " frame=" << frame_index_
                  << " type=" << type
                  << " allocs=" << counters_.allocations
                  << " alloc_bytes=" << counters_.allocation_bytes
                  << " h2d=" << counters_.h2d_copies << '/' << counters_.h2d_bytes
                  << " d2h=" << counters_.d2h_copies << '/' << counters_.d2h_bytes
                  << " d2d=" << counters_.d2d_copies << '/' << counters_.d2d_bytes
                  << " syncs=" << counters_.synchronizations << '\n';
        for (const auto& stage : stages_) {
            std::clog << "[nvcr.profile] stage " << stage.name;
            if (stage.cpu_ms > 0.0) {
                std::clog << " cpu_ms=" << std::fixed << std::setprecision(3)
                          << stage.cpu_ms;
            }
            if (stage.has_cuda) {
                std::clog << " cuda_ms=" << std::fixed << std::setprecision(3)
                          << stage.cuda_ms;
            }
            std::clog << '\n';
        }
    }

private:
    bool enabled_{};
    std::string operation_;
    FrameType frame_type_{FrameType::intra};
    std::uint64_t frame_index_{};
    ProfileCounters counters_;
    std::vector<ProfileStage> stages_;
    std::vector<PendingCudaStage> pending_cuda_stages_;
};

struct CudaStreamOwner final {
    ~CudaStreamOwner() {
        if (stream == nullptr) return;
        static_cast<void>(cudaStreamSynchronize(stream));
        static_cast<void>(cudaStreamDestroy(stream));
    }

    cudaStream_t stream{};
};

extern thread_local cudaStream_t allocation_stream;
extern thread_local BackendProfiler* active_profiler;
extern thread_local bool cuda_graphs_allowed;
extern thread_local std::string_view allocation_owner_id;
extern thread_local std::shared_ptr<CudaStreamOwner> allocation_stream_owner;
class DeviceScratchArena;
extern thread_local DeviceScratchArena* active_scratch_arena;

class CpuProfileScope final {
public:
    explicit CpuProfileScope(std::string_view name)
        : profiler_(active_profiler),
          enabled_(profiler_ != nullptr && profiler_->enabled()) {
        if (enabled_) {
            name_ = name;
            start_ = std::chrono::steady_clock::now();
        }
    }
    ~CpuProfileScope() {
        if (!enabled_) return;
        const auto elapsed = std::chrono::steady_clock::now() - start_;
        profiler_->record_cpu_stage(
            std::move(name_), std::chrono::duration<double, std::milli>(elapsed).count());
    }
    CpuProfileScope(const CpuProfileScope&) = delete;
    CpuProfileScope& operator=(const CpuProfileScope&) = delete;

private:
    BackendProfiler* profiler_{};
    std::string name_;
    bool enabled_{};
    std::chrono::steady_clock::time_point start_;
};

class CudaAllocationScope final {
public:
    CudaAllocationScope(
        cudaStream_t stream, BackendProfiler* profiler, DeviceScratchArena* arena,
        bool allow_cuda_graphs = false, std::string_view owner_id = {},
        std::shared_ptr<CudaStreamOwner> stream_owner = {})
        : previous_stream_(std::exchange(allocation_stream, stream)),
          previous_profiler_(std::exchange(active_profiler, profiler)),
          previous_arena_(std::exchange(active_scratch_arena, arena)),
          previous_cuda_graphs_allowed_(
              std::exchange(cuda_graphs_allowed, allow_cuda_graphs)),
          previous_owner_id_(std::exchange(allocation_owner_id, owner_id)),
          previous_stream_owner_(
              std::exchange(allocation_stream_owner, std::move(stream_owner))) {}
    ~CudaAllocationScope() {
        allocation_stream = previous_stream_;
        active_profiler = previous_profiler_;
        active_scratch_arena = previous_arena_;
        cuda_graphs_allowed = previous_cuda_graphs_allowed_;
        allocation_owner_id = previous_owner_id_;
        allocation_stream_owner = std::move(previous_stream_owner_);
    }
    CudaAllocationScope(const CudaAllocationScope&) = delete;
    CudaAllocationScope& operator=(const CudaAllocationScope&) = delete;

private:
    cudaStream_t previous_stream_{};
    BackendProfiler* previous_profiler_{};
    DeviceScratchArena* previous_arena_{};
    bool previous_cuda_graphs_allowed_{};
    std::string_view previous_owner_id_;
    std::shared_ptr<CudaStreamOwner> previous_stream_owner_;
};

inline void release_cuda(void* pointer, cudaStream_t stream) noexcept {
    if (pointer == nullptr) return;
    if (stream != nullptr) {
        static_cast<void>(cudaFreeAsync(pointer, stream));
    } else {
        static_cast<void>(cudaFree(pointer));
    }
}

struct CudaAllocation final {
    CudaAllocation() = default;
    CudaAllocation(void* pointer, std::size_t byte_count, cudaStream_t owner_stream, bool owned_allocation = true)
        : data(pointer), bytes(byte_count), stream(owner_stream), owned(owned_allocation) {}
    ~CudaAllocation() { if (owned) release_cuda(data, stream); }
    CudaAllocation(CudaAllocation&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          bytes(std::exchange(other.bytes, 0)),
          stream(std::exchange(other.stream, nullptr)),
          owned(std::exchange(other.owned, true)) {}
    CudaAllocation& operator=(CudaAllocation&& other) noexcept {
        if (this != &other) {
            if (owned) release_cuda(data, stream);
            data = std::exchange(other.data, nullptr);
            bytes = std::exchange(other.bytes, 0);
            stream = std::exchange(other.stream, nullptr);
            owned = std::exchange(other.owned, true);
        }
        return *this;
    }
    CudaAllocation(const CudaAllocation&) = delete;
    CudaAllocation& operator=(const CudaAllocation&) = delete;

    void* data{};
    std::size_t bytes{};
    cudaStream_t stream{};
    bool owned{true};
};

class TensorRTBuffer final : public session_api::IProviderBuffer {
public:
    TensorRTBuffer(CudaAllocation allocation, std::string owner_id)
        : stream_owner_(allocation_stream_owner),
          device_allocation_(std::move(allocation)),
          size_bytes_(device_allocation_->bytes),
          domain_(session_api::MemoryDomain::provider_device),
          owner_id_(std::move(owner_id)) {}

    TensorRTBuffer(
        std::unique_ptr<std::byte[]> storage,
        std::size_t size_bytes,
        std::string owner_id)
        : host_storage_(std::move(storage)),
          size_bytes_(size_bytes),
          domain_(session_api::MemoryDomain::host),
          owner_id_(std::move(owner_id)) {}

    TensorRTBuffer(void* storage, std::size_t size_bytes, std::string owner_id)
        : pinned_storage_(storage),
          size_bytes_(size_bytes),
          domain_(session_api::MemoryDomain::pinned_host),
          owner_id_(std::move(owner_id)) {}

    ~TensorRTBuffer() override {
        if (pinned_storage_ != nullptr) {
            static_cast<void>(cudaFreeHost(pinned_storage_));
        }
    }

    TensorRTBuffer(const TensorRTBuffer&) = delete;
    TensorRTBuffer& operator=(const TensorRTBuffer&) = delete;

    [[nodiscard]] std::size_t size_bytes() const noexcept override {
        return size_bytes_;
    }

    [[nodiscard]] session_api::MemoryDomain domain() const noexcept override {
        return domain_;
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_id_;
    }

    [[nodiscard]] void* data() noexcept {
        if (device_allocation_.has_value()) return device_allocation_->data;
        if (pinned_storage_ != nullptr) return pinned_storage_;
        return host_storage_.get();
    }

    [[nodiscard]] std::span<std::byte> host_bytes() noexcept override {
        if (device_allocation_.has_value()) return {};
        return {static_cast<std::byte*>(data()), size_bytes_};
    }

private:
    std::shared_ptr<CudaStreamOwner> stream_owner_;
    std::optional<CudaAllocation> device_allocation_;
    std::unique_ptr<std::byte[]> host_storage_;
    void* pinned_storage_{};
    std::size_t size_bytes_{};
    session_api::MemoryDomain domain_{};
    std::string owner_id_;
};

class DeviceScratchArena final {
public:
    ~DeviceScratchArena() {
        if (data_ != nullptr) static_cast<void>(cudaFree(data_));
    }
    DeviceScratchArena() = default;
    DeviceScratchArena(DeviceScratchArena&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)),
          reserved_(std::exchange(other.reserved_, 0)),
          offset_(std::exchange(other.offset_, 0)) {}
    DeviceScratchArena& operator=(DeviceScratchArena&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) static_cast<void>(cudaFree(data_));
            data_ = std::exchange(other.data_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
            reserved_ = std::exchange(other.reserved_, 0);
            offset_ = std::exchange(other.offset_, 0);
        }
        return *this;
    }
    DeviceScratchArena(const DeviceScratchArena&) = delete;
    DeviceScratchArena& operator=(const DeviceScratchArena&) = delete;

    Result<void> initialize(std::size_t byte_count) {
        if (byte_count == 0) return {};
        void* pointer = nullptr;
        const auto status = cudaMalloc(&pointer, byte_count);
        if (status != cudaSuccess) return cuda_error("cudaMalloc device scratch arena", status);
        data_ = static_cast<std::byte*>(pointer);
        capacity_ = byte_count;
        reserved_ = 0;
        offset_ = 0;
        return {};
    }

    Result<void*> reserve(std::size_t bytes) noexcept {
        constexpr std::size_t alignment = 256;
        const auto aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
        if (data_ == nullptr || aligned_bytes > capacity_) {
            return Error(
                ErrorCode::resource_exhausted,
                "device scratch arena cannot reserve TensorRT context workspace",
                std::string(subsystem));
        }
        reserved_ = aligned_bytes;
        offset_ = reserved_;
        return data_;
    }

    void reset() noexcept { offset_ = reserved_; }

    Result<CudaAllocation> allocate(std::size_t bytes) noexcept {
        constexpr std::size_t alignment = 256;
        const auto aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (data_ == nullptr || bytes > capacity_ || aligned > capacity_ - bytes) {
            return Error(
                ErrorCode::resource_exhausted,
                "device scratch arena capacity exceeded",
                std::string(subsystem));
        }
        offset_ = aligned + bytes;
        return CudaAllocation(data_ + aligned, bytes, nullptr, false);
    }

    [[nodiscard]] bool initialized() const noexcept { return data_ != nullptr; }

private:
    std::byte* data_{};
    std::size_t capacity_{};
    std::size_t reserved_{};
    std::size_t offset_{};
};

struct CudaEvent final {
    CudaEvent() = default;
    ~CudaEvent() {
        if (data != nullptr) static_cast<void>(cudaEventDestroy(data));
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t data{};
};

template <typename T>
struct PinnedHostBuffer final {
    ~PinnedHostBuffer() {
        if (data != nullptr) static_cast<void>(cudaFreeHost(data));
    }
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer() = default;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          capacity(std::exchange(other.capacity, 0)) {}
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (data != nullptr) static_cast<void>(cudaFreeHost(data));
            data = std::exchange(other.data, nullptr);
            capacity = std::exchange(other.capacity, 0);
        }
        return *this;
    }

    Result<void> ensure(std::size_t required_count, std::string_view name) {
        if (required_count <= capacity) return {};
        if (data != nullptr) {
            static_cast<void>(cudaFreeHost(data));
            data = nullptr;
            capacity = 0;
        }
        void* pointer = nullptr;
        const auto status = cudaMallocHost(&pointer, required_count * sizeof(T));
        if (status != cudaSuccess) {
            return backend_error(
                std::string("cudaMallocHost failed for ") + std::string(name) +
                ": " + cudaGetErrorString(status));
        }
        data = static_cast<T*>(pointer);
        capacity = required_count;
        return {};
    }

    T* data{};
    std::size_t capacity{};
};


inline nvinfer1::Dims make_dims(
    std::int32_t channels, std::int32_t height, std::int32_t width) {
    nvinfer1::Dims shape{};
    shape.nbDims = 4;
    shape.d[0] = 1;
    shape.d[1] = channels;
    shape.d[2] = height;
    shape.d[3] = width;
    return shape;
}

inline std::size_t element_count(const nvinfer1::Dims& shape) {
    std::size_t count = 1;
    for (std::int32_t index = 0; index < shape.nbDims; ++index) {
        count *= static_cast<std::size_t>(shape.d[index]);
    }
    return count;
}

struct DeviceTensor final {
    std::string name;
    nvinfer1::Dims shape;
    struct Storage final {
        explicit Storage(CudaAllocation allocation)
            : buffer(std::make_shared<TensorRTBuffer>(
                  std::move(allocation), std::string(allocation_owner_id))),
              data(buffer->data()),
              bytes(buffer->size_bytes()) {}

        std::shared_ptr<TensorRTBuffer> buffer;
        void* data{};
        std::size_t bytes{};
    } storage;
    session_api::TensorDataType data_type{session_api::TensorDataType::float16};
};

enum class EngineOutputStorage {
    owned,
    scratch,
};



[[nodiscard]] Result<std::string> read_text_file(const fs::path& path);
[[nodiscard]] Result<std::string> read_json_string(std::string_view text, std::string_view key);
bool valid_sha256(std::string_view value);
[[nodiscard]] Result<CudaAllocation> allocate_cuda(std::size_t bytes);
[[nodiscard]] Result<CudaAllocation> allocate_cuda_scratch(std::size_t bytes);
[[nodiscard]] Result<void> copy_async(
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    std::string_view operation);
[[nodiscard]] Result<void> synchronize_stream(cudaStream_t stream, std::string_view operation);
[[nodiscard]] Result<void> record_cuda_event(cudaEvent_t event, cudaStream_t stream, std::string_view operation);
[[nodiscard]] Result<std::pair<cudaEvent_t, cudaEvent_t>> create_profile_events(std::string_view operation);
[[nodiscard]] Result<std::size_t> tensor_bytes(
    const nvinfer1::Dims& shape,
    session_api::TensorDataType data_type = session_api::TensorDataType::float16);
[[nodiscard]] const DeviceTensor* find_device_tensor(
    std::span<const DeviceTensor* const> tensors, std::string_view name);
[[nodiscard]] Result<DeviceTensor> allocate_device_tensor(
    std::string name, nvinfer1::Dims shape,
    session_api::TensorDataType data_type = session_api::TensorDataType::float16);
[[nodiscard]] Result<DeviceTensor> allocate_device_tensor_scratch(
    std::string name, nvinfer1::Dims shape,
    session_api::TensorDataType data_type = session_api::TensorDataType::float16);

class TensorRTStage final : public session_api::IExecutableStage {
public:
    TensorRTStage(
        session_api::ExecutableStageDescriptor descriptor,
        std::string owner_id,
        std::size_t engine_index)
        : descriptor_(std::move(descriptor)),
          owner_id_(std::move(owner_id)),
          engine_index_(engine_index) {}

    [[nodiscard]] const session_api::ExecutableStageDescriptor& descriptor()
        const noexcept override {
        return descriptor_;
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_id_;
    }

    [[nodiscard]] std::size_t engine_index() const noexcept { return engine_index_; }

private:
    session_api::ExecutableStageDescriptor descriptor_;
    std::string owner_id_;
    std::size_t engine_index_{};
};

// A normative stage: a portable program run by the exact CUDA executor on the
// session's stream instead of a TensorRT plan.
class TensorRTExactStage final : public session_api::IExecutableStage {
public:
    TensorRTExactStage(
        session_api::ExecutableStageDescriptor descriptor,
        std::string owner_id,
        deterministic::NormativeProgram program,
        std::shared_ptr<deterministic::CudaExecutor> executor)
        : descriptor_(std::move(descriptor)),
          owner_id_(std::move(owner_id)),
          program_(std::move(program)),
          executor_(std::move(executor)) {}

    [[nodiscard]] const session_api::ExecutableStageDescriptor& descriptor()
        const noexcept override {
        return descriptor_;
    }
    [[nodiscard]] std::string_view owner_id() const noexcept override { return owner_id_; }
    [[nodiscard]] const deterministic::Program& program() const noexcept { return *program_.program; }
    [[nodiscard]] deterministic::CudaExecutor& executor() const noexcept { return *executor_; }

private:
    session_api::ExecutableStageDescriptor descriptor_;
    std::string owner_id_;
    deterministic::NormativeProgram program_;  // owns the program the executor reads
    std::shared_ptr<deterministic::CudaExecutor> executor_;
};

class TensorRTCompletion final : public session_api::ICompletion {
public:
    TensorRTCompletion(
        std::string owner_id,
        cudaEvent_t event,
        std::vector<session_api::BufferHandle> retained_buffers)
        : owner_id_(std::move(owner_id)),
          retained_buffers_(std::move(retained_buffers)) {
        event_.data = event;
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_id_;
    }

    [[nodiscard]] bool ready() const noexcept override {
        return event_.data != nullptr && cudaEventQuery(event_.data) == cudaSuccess;
    }

    [[nodiscard]] Result<void> wait() override {
        if (event_.data == nullptr) return {};
        return synchronize_event(event_.data, "cudaEventSynchronize completion");
    }

    [[nodiscard]] cudaEvent_t event() const noexcept { return event_.data; }

private:
    std::string owner_id_;
    CudaEvent event_;
    std::vector<session_api::BufferHandle> retained_buffers_;
};

class SessionExecutionScope;

class TensorRTExecutionSession final : public session_api::IProviderSession {
public:
    TensorRTExecutionSession();
    ~TensorRTExecutionSession() override;

    TensorRTExecutionSession(const TensorRTExecutionSession&) = delete;
    TensorRTExecutionSession& operator=(const TensorRTExecutionSession&) = delete;

    // Prepares the runtime, stream and scratch arena and validates the bundle
    // identity, target and file digests. Codecs then load their stages with
    // load_stage() and call finalize_stages() before executing.
    [[nodiscard]] Result<void> initialize(const RuntimeConfiguration& configuration);
    [[nodiscard]] Result<session_api::ExecutableStage> load_stage(
        TensorRTStageSpec specification, std::string codec_id);
    [[nodiscard]] Result<void> finalize_stages();

    [[nodiscard]] const fs::path& bundle_root() const noexcept { return bundle_root_; }
    // File name to SHA-256 for every entry engine.sha256 lists, all verified.
    [[nodiscard]] const std::map<std::string, std::string>& bundle_files() const noexcept {
        return bundle_files_;
    }
    [[nodiscard]] std::string_view execution_mode() const noexcept {
        return context_policy_name(context_policy_);
    }
    // Routes CUDA allocations made outside a submission to this session.
    [[nodiscard]] CudaAllocationScope allocation_scope() {
        return CudaAllocationScope(
            stream_, nullptr, &scratch_arena_, false, owner_id_, stream_owner_);
    }

    [[nodiscard]] std::string_view owner_id() const noexcept override {
        return owner_id_;
    }

    [[nodiscard]] Result<session_api::BufferHandle> allocate(
        std::size_t size_bytes,
        session_api::MemoryDomain domain) override;

    [[nodiscard]] Result<session_api::ExecutableStage> load_stage(
        session_api::ExecutableStageDescriptor descriptor) override;

    // nvcr.intconv.v1 on SM 8.0 and newer.
    [[nodiscard]] bool supports_numerics_profile(std::string_view profile) const noexcept override;

    [[nodiscard]] Result<session_api::Completion> submit(
        const session_api::ExecutableStage& stage,
        std::span<const session_api::TensorView> tensors,
        std::span<const session_api::ExecutionDependency> dependencies) override;

    [[nodiscard]] Result<void> reset() override;

    [[nodiscard]] Result<std::vector<DeviceTensor>> run_stage(
        std::size_t engine_index,
        std::span<const DeviceTensor* const> inputs,
        EngineOutputStorage output_storage = EngineOutputStorage::owned);

    [[nodiscard]] Result<std::vector<DeviceTensor>> run_stage(
        const session_api::ExecutableStage& stage,
        std::span<const DeviceTensor* const> inputs,
        EngineOutputStorage output_storage = EngineOutputStorage::owned);

    [[nodiscard]] std::vector<session_api::ExecutableStage> loaded_stages() const {
        return {stages_.begin(), stages_.end()};
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    friend class SessionExecutionScope;

    [[nodiscard]] Result<session_api::Completion> submit_impl(
        const std::shared_ptr<const TensorRTStage>& stage,
        std::span<const session_api::TensorView> tensors,
        std::span<const session_api::ExecutionDependency> dependencies,
        nvinfer1::IExecutionContext* prepared_context = nullptr,
        std::chrono::steady_clock::time_point setup_start = {});
    [[nodiscard]] Result<session_api::ExecutableStage> load_validated_stage(
        session_api::ExecutableStageDescriptor descriptor, TensorRTStageSpec specification);
    [[nodiscard]] Result<session_api::ExecutableStage> load_exact_stage(
        session_api::ExecutableStageDescriptor descriptor);
    [[nodiscard]] Result<std::vector<DeviceTensor>> run_exact_stage(
        const TensorRTExactStage& stage,
        std::span<const DeviceTensor* const> inputs,
        EngineOutputStorage output_storage);

    TensorRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::vector<EngineInstance> engines_;
    std::vector<TensorRTStageSpec> stage_specs_;
    std::vector<std::shared_ptr<const TensorRTStage>> stages_;
    std::vector<std::shared_ptr<const TensorRTExactStage>> exact_stages_;
    int compute_major_{0};
    std::map<std::string, std::string> bundle_files_;
    DeviceScratchArena scratch_arena_;
    BackendProfiler profiler_{false};
    std::shared_ptr<CudaStreamOwner> stream_owner_;
    cudaStream_t stream_{};
    ContextPolicy context_policy_{ContextPolicy::per_engine};
    bool ready_{};
    bool initialized_{};
    bool profiling_enabled_{};
    bool fixed_shape_profile_{};
    fs::path bundle_root_;
    std::string owner_id_;
};

class SessionExecutionScope final {
public:
    SessionExecutionScope(
        TensorRTExecutionSession& session,
        std::string operation,
        FrameType frame_type,
        std::uint64_t frame_index,
        bool allow_cuda_graphs)
        : session_(session),
          allocation_scope_(
              session.stream_, &session.profiler_, &session.scratch_arena_,
              allow_cuda_graphs, session.owner_id_, session.stream_owner_) {
        session_.scratch_arena_.reset();
        session_.profiler_.begin(std::move(operation), frame_type, frame_index);
    }

    ~SessionExecutionScope() { finish(); }

    SessionExecutionScope(const SessionExecutionScope&) = delete;
    SessionExecutionScope& operator=(const SessionExecutionScope&) = delete;

    void finish() noexcept {
        if (finished_) return;
        session_.profiler_.finish();
        finished_ = true;
    }

private:
    TensorRTExecutionSession& session_;
    CudaAllocationScope allocation_scope_;
    bool finished_{};
};


[[nodiscard]] Result<std::vector<DeviceTensor>> run_device_stage(
    TensorRTExecutionSession& session,
    const session_api::ExecutableStage& stage,
    std::span<const DeviceTensor* const> inputs,
    EngineOutputStorage output_storage = EngineOutputStorage::owned);

[[nodiscard]] provider::ProviderDescriptor provider_descriptor();
// Creates a session with a validated bundle and no stages; the codec loads its
// stages and finalizes it.
[[nodiscard]] Result<std::shared_ptr<session_api::IProviderSession>> make_provider_session(
    const RuntimeConfiguration& configuration);

}  // namespace nvcr::tensorrt
