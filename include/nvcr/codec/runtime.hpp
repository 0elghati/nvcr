#pragma once

#include "nvcr/codec/backend.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/logging/logger.hpp"
#include "nvcr/runtime/frame.hpp"
#include "nvcr/runtime/packet.hpp"
#include "nvcr/statistics/statistics.hpp"

#include <memory>

namespace nvcr::codec {

class Runtime final {
public:
    Runtime(
        RuntimeConfiguration configuration,
        Components components,
        std::shared_ptr<LoggerFactory> logger_factory,
        std::shared_ptr<Statistics> statistics);

    [[nodiscard]] Result<void> initialize();
    [[nodiscard]] Result<Packet> encode(const Frame& frame);
    [[nodiscard]] Result<Frame> decode(const Packet& packet);
    [[nodiscard]] Result<void> flush();
    void reset() noexcept;

private:
    RuntimeConfiguration configuration_;
    Components components_;
    SequenceState encoder_state_;
    SequenceState decoder_state_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<Statistics> statistics_;
    bool initialized_{false};
};

}  // namespace nvcr::codec
