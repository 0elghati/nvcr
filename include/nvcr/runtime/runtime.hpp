#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/runtime/frame.hpp"
#include "nvcr/runtime/packet.hpp"
#include "nvcr/statistics/statistics.hpp"

#include <memory>

namespace nvcr {

enum class RuntimeState {
    initializing,
    ready,
    failed,
    stopped,
};

class Runtime final {
public:
    ~Runtime();
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] static Result<Runtime> create(
        RuntimeConfiguration configuration,
        dcvcrt::Components components);

    [[nodiscard]] Result<Packet> encode(const Frame& frame);
    [[nodiscard]] Result<Frame> decode(const Packet& packet);
    [[nodiscard]] Result<void> flush();
    [[nodiscard]] Result<void> reset();
    [[nodiscard]] RuntimeState state() const noexcept;
    [[nodiscard]] StatisticsSnapshot statistics() const noexcept;

private:
    struct Impl;
    explicit Runtime(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr

