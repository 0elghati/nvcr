#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/runtime/frame.hpp"
#include "nvcr/runtime/packet.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace nvcr::codec {

struct SequenceStateView final {
    const Frame* reference_frame{};
    std::span<const std::byte> latent_state;
    std::uint64_t frame_index{};
    std::uint32_t gop_position{};
    std::uint64_t generation{};
};

class SequenceState final {
public:
    explicit SequenceState(std::uint32_t gop_size);

    [[nodiscard]] Result<FrameType> next_frame_type() const;
    [[nodiscard]] Result<void> validate_packet(FrameType frame_type) const;
    [[nodiscard]] SequenceStateView view() const noexcept;
    [[nodiscard]] Result<void> commit(
        Frame reconstructed_frame,
        std::vector<std::byte> latent_state,
        FrameType frame_type);

    void reset() noexcept;
    [[nodiscard]] std::uint64_t frame_index() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] bool has_reference() const noexcept;

private:
    std::uint32_t gop_size_;
    std::optional<Frame> reference_frame_;
    std::vector<std::byte> latent_state_;
    std::uint64_t frame_index_{};
    std::uint64_t generation_{};
    bool has_reference_{false};
};

}  // namespace nvcr::codec
