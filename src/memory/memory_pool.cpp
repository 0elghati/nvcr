#include "nvcr/memory/memory_pool.hpp"

#include <algorithm>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace nvcr {
namespace {

class HostMemoryResource final : public MemoryResource {
public:
    Result<void*> allocate(std::size_t bytes) override {
        if (bytes == 0) {
            return Error(ErrorCode::invalid_argument, "cannot allocate zero bytes", "memory");
        }
        try {
            return ::operator new(bytes, std::align_val_t{64});
        } catch (const std::bad_alloc&) {
            return Error(ErrorCode::resource_exhausted, "host allocation failed", "memory");
        }
    }

    void deallocate(void* pointer, std::size_t /*bytes*/) noexcept override {
        ::operator delete(pointer, std::align_val_t{64});
    }

    MemoryKind kind() const noexcept override { return MemoryKind::host; }
};

struct FreeBlock final {
    void* pointer{};
    std::size_t capacity{};
};

}  // namespace

namespace detail {

struct PoolState final {
    PoolState(std::shared_ptr<MemoryResource> memory_resource, std::size_t maximum)
        : resource(std::move(memory_resource)), capacity_bytes(maximum) {}

    ~PoolState() {
        for (const auto& block : free_blocks) {
            resource->deallocate(block.pointer, block.capacity);
        }
    }

    void return_block(void* pointer, std::size_t capacity) noexcept {
        try {
            std::scoped_lock lock(mutex);
            free_blocks.push_back({pointer, capacity});
        } catch (...) {
            resource->deallocate(pointer, capacity);
            std::scoped_lock lock(mutex);
            allocated_bytes -= capacity;
        }
    }

    std::shared_ptr<MemoryResource> resource;
    std::size_t capacity_bytes{};
    std::size_t allocated_bytes{};
    mutable std::mutex mutex;
    std::vector<FreeBlock> free_blocks;
};

}  // namespace detail

std::shared_ptr<MemoryResource> make_host_memory_resource() {
    return std::make_shared<HostMemoryResource>();
}

BufferLease::BufferLease(
    std::shared_ptr<detail::PoolState> state,
    void* pointer,
    std::size_t size,
    std::size_t capacity) noexcept
    : state_(std::move(state)), pointer_(pointer), size_(size), capacity_(capacity) {}

BufferLease::~BufferLease() { release(); }

BufferLease::BufferLease(BufferLease&& other) noexcept
    : state_(std::move(other.state_)),
      pointer_(std::exchange(other.pointer_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)) {}

BufferLease& BufferLease::operator=(BufferLease&& other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        pointer_ = std::exchange(other.pointer_, nullptr);
        size_ = std::exchange(other.size_, 0);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

std::span<std::byte> BufferLease::bytes() noexcept {
    return {static_cast<std::byte*>(pointer_), size_};
}

std::span<const std::byte> BufferLease::bytes() const noexcept {
    return {static_cast<const std::byte*>(pointer_), size_};
}

std::size_t BufferLease::capacity() const noexcept { return capacity_; }

MemoryKind BufferLease::kind() const noexcept {
    return state_ ? state_->resource->kind() : MemoryKind::host;
}

BufferLease::operator bool() const noexcept { return pointer_ != nullptr; }

void BufferLease::release() noexcept {
    if (state_ && pointer_) {
        state_->return_block(pointer_, capacity_);
    }
    pointer_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    state_.reset();
}

MemoryPool::MemoryPool(
    std::shared_ptr<MemoryResource> resource, std::size_t capacity_bytes)
    : state_(std::make_shared<detail::PoolState>(std::move(resource), capacity_bytes)) {}

Result<BufferLease> MemoryPool::acquire(std::size_t bytes) {
    if (bytes == 0) {
        return Error(ErrorCode::invalid_argument, "cannot lease zero bytes", "memory");
    }

    std::scoped_lock lock(state_->mutex);
    auto best = state_->free_blocks.end();
    for (auto it = state_->free_blocks.begin(); it != state_->free_blocks.end(); ++it) {
        if (it->capacity >= bytes &&
            (best == state_->free_blocks.end() || it->capacity < best->capacity)) {
            best = it;
        }
    }
    if (best != state_->free_blocks.end()) {
        const auto block = *best;
        state_->free_blocks.erase(best);
        return BufferLease(state_, block.pointer, bytes, block.capacity);
    }

    if (bytes > state_->capacity_bytes - std::min(state_->allocated_bytes, state_->capacity_bytes)) {
        return Error(ErrorCode::resource_exhausted, "memory pool capacity exceeded", "memory");
    }
    auto allocation = state_->resource->allocate(bytes);
    if (!allocation) {
        return allocation.error();
    }
    state_->allocated_bytes += bytes;
    return BufferLease(state_, allocation.value(), bytes, bytes);
}

std::size_t MemoryPool::allocated_bytes() const noexcept {
    std::scoped_lock lock(state_->mutex);
    return state_->allocated_bytes;
}

std::size_t MemoryPool::available_bytes() const noexcept {
    std::scoped_lock lock(state_->mutex);
    return state_->capacity_bytes - std::min(state_->allocated_bytes, state_->capacity_bytes);
}

MemoryKind MemoryPool::kind() const noexcept { return state_->resource->kind(); }

}  // namespace nvcr
