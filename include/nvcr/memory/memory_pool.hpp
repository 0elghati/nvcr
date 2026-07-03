#pragma once

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace nvcr {

enum class MemoryKind {
    host,
    pinned_host,
    device,
};

class MemoryResource {
public:
    virtual ~MemoryResource() = default;
    [[nodiscard]] virtual Result<void*> allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* pointer, std::size_t bytes) noexcept = 0;
    [[nodiscard]] virtual MemoryKind kind() const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<MemoryResource> make_host_memory_resource();

namespace detail {
struct PoolState;
}

class BufferLease final {
public:
    BufferLease() = default;
    ~BufferLease();
    BufferLease(BufferLease&& other) noexcept;
    BufferLease& operator=(BufferLease&& other) noexcept;
    BufferLease(const BufferLease&) = delete;
    BufferLease& operator=(const BufferLease&) = delete;

    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] MemoryKind kind() const noexcept;
    explicit operator bool() const noexcept;

private:
    friend class MemoryPool;
    BufferLease(
        std::shared_ptr<detail::PoolState> state,
        void* pointer,
        std::size_t size,
        std::size_t capacity) noexcept;
    void release() noexcept;

    std::shared_ptr<detail::PoolState> state_;
    void* pointer_{};
    std::size_t size_{};
    std::size_t capacity_{};
};

class MemoryPool final {
public:
    MemoryPool(std::shared_ptr<MemoryResource> resource, std::size_t capacity_bytes);

    [[nodiscard]] Result<BufferLease> acquire(std::size_t bytes);
    [[nodiscard]] std::size_t allocated_bytes() const noexcept;
    [[nodiscard]] std::size_t available_bytes() const noexcept;
    [[nodiscard]] MemoryKind kind() const noexcept;

private:
    std::shared_ptr<detail::PoolState> state_;
};

}  // namespace nvcr

