#pragma once

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

// Runtime is a thin facade over the encoder and decoder sessions created by the
// selected codec adapter. Calls are serialized by an internal mutex. The
// direction-specific lifecycle methods preserve independent flush/reset;
// Runtime's interface overrides remain shared compatibility operations.
class Runtime final : public IEncoderSession, public IDecoderSession {
public:
    ~Runtime() override;
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Selects the registered codec and provider named by configuration, then
    // composes their adapter and provider session into a runtime.
    [[nodiscard]] static Result<Runtime> create(RuntimeConfiguration configuration);

    // Low-level construction seam for tests and callers supplying sessions.
    [[nodiscard]] static Result<Runtime> create(
        RuntimeConfiguration configuration,
        codec::Sessions sessions);

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

    // Direction-specific lifecycle for independent drain/reset.
    [[nodiscard]] Result<void> flush_encoder();
    [[nodiscard]] Result<void> reset_encoder();
    [[nodiscard]] Result<void> flush_decoder();
    [[nodiscard]] Result<void> reset_decoder();

    [[nodiscard]] RuntimeState state() const noexcept;
    [[nodiscard]] StatisticsSnapshot statistics() const noexcept;

private:
    struct Impl;
    explicit Runtime(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr
