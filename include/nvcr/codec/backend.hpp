#pragma once

#include "nvcr/codec/sequence_state.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/configuration/configuration.hpp"
#include "nvcr/runtime/frame.hpp"
#include "nvcr/runtime/packet.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace nvcr::codec {

struct CodecEncodeResult final {
    std::vector<std::byte> payload;
    Frame reconstructed_frame;
    std::vector<std::byte> latent_state;
    std::uint32_t effective_qp{};
};

struct CodecDecodeResult final {
    Frame frame;
    std::vector<std::byte> latent_state;
};

// Neural video codecs interleave model execution, prior preparation, entropy
// coding, and reference-state updates. Keeping those operations behind one
// codec-level boundary prevents callers from treating them as independent
// pre/post-processing steps.
class CodecBackend {
public:
    virtual ~CodecBackend() = default;

    // Implementations must translate SDK/driver exceptions into Result errors.
    // No exception may cross this boundary.
    [[nodiscard]] virtual Result<void> initialize(
        const RuntimeConfiguration& configuration) = 0;
    [[nodiscard]] virtual Result<CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const SequenceStateView& state) = 0;
    [[nodiscard]] virtual Result<CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType frame_type,
        Timestamp timestamp,
        const SequenceStateView& state) = 0;
    [[nodiscard]] virtual Result<void> flush() = 0;
    virtual void reset() noexcept = 0;
};

struct Components final {
    std::unique_ptr<CodecBackend> codec;
};

}  // namespace nvcr::codec
