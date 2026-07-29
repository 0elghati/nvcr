#include "nvcr/codec/sequence_state.hpp"

#include <utility>

namespace nvcr::codec {

SequenceState::SequenceState(std::uint32_t gop_size) : gop_size_(gop_size) {}

Result<FrameType> SequenceState::next_frame_type() const {
    if (gop_size_ == 0) {
        return Error(ErrorCode::invalid_state, "GOP size is zero", "codec.state");
    }
    if (!has_reference_ || (frame_index_ % gop_size_) == 0) {
        return FrameType::intra;
    }
    return FrameType::predicted;
}

Result<void> SequenceState::validate_packet(FrameType frame_type) const {
    if (frame_type == FrameType::predicted && !has_reference_) {
        return Error(
            ErrorCode::invalid_state,
            "predicted frame received without a reference frame",
            "codec.state");
    }
    if (frame_type == FrameType::intra) {
        return {};
    }
    return {};
}

SequenceStateView SequenceState::view() const noexcept {
    return {
        reference_frame_ ? &*reference_frame_ : nullptr,
        latent_state_,
        frame_index_,
        gop_size_ == 0 ? 0U : static_cast<std::uint32_t>(frame_index_ % gop_size_),
        generation_,
    };
}

Result<void> SequenceState::commit(
    Frame reconstructed_frame,
    std::vector<std::byte> latent_state,
    FrameType frame_type) {
    if (reconstructed_frame.size_bytes() == 0 && frame_type == FrameType::intra) {
        return Error(
            ErrorCode::backend_error,
            "backend returned an empty reference frame",
            "codec.state");
    }
    if (frame_type == FrameType::predicted && !has_reference_) {
        return Error(
            ErrorCode::invalid_state,
            "cannot commit a predicted frame without a reference",
            "codec.state");
    }
    if (reconstructed_frame.size_bytes() != 0) {
        reference_frame_ = std::move(reconstructed_frame);
    }
    latent_state_ = std::move(latent_state);
    has_reference_ = true;
    ++frame_index_;
    return {};
}

void SequenceState::reset() noexcept {
    reference_frame_.reset();
    latent_state_.clear();
    frame_index_ = 0;
    has_reference_ = false;
    ++generation_;
}

std::uint64_t SequenceState::frame_index() const noexcept { return frame_index_; }
std::uint64_t SequenceState::generation() const noexcept { return generation_; }
bool SequenceState::has_reference() const noexcept { return has_reference_; }

}  // namespace nvcr::codec
