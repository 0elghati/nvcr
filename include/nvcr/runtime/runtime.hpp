#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/codec/session.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
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

// Runtime implements IEncoderSession and IDecoderSession on the same session
// object for ergonomic symmetry.  The underlying codec::Runtime calls are
// serialized by an internal mutex so both directions are safe to drive from a
// single thread.  DCVC-RT's one-frame/one-AU behaviour is conformant with the
// send/receive contract (try_again is never returned in the current adapter).
class Runtime final : public IEncoderSession, public IDecoderSession {
public:
    ~Runtime() override;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] static Result<Runtime> create(
        RuntimeConfiguration configuration,
        codec::Components components);

    // IEncoderSession
    [[nodiscard]] Result<void> send_frame(const Frame& frame) override;
    [[nodiscard]] Result<Packet> receive_access_unit() override;
    [[nodiscard]] Result<void> flush() override;
    [[nodiscard]] Result<void> reset() override;

    // IDecoderSession
    [[nodiscard]] Result<void> send_access_unit(const Packet& packet) override;
    [[nodiscard]] Result<Frame> receive_frame() override;
    // flush() and reset() are shared with the encoder direction above.

    // Compatibility aliases — thin wrappers over send+receive; preserved so
    // existing CLI and test call-sites compile without change.
    [[nodiscard]] Result<Packet> encode(const Frame& frame);
    [[nodiscard]] Result<Frame> decode(const Packet& packet);

    [[nodiscard]] RuntimeState state() const noexcept;
    [[nodiscard]] StatisticsSnapshot statistics() const noexcept;

private:
    struct Impl;
    explicit Runtime(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr

