#include "nvcr/dcvcrt/sequence_state.hpp"

#include <utility>

namespace nvcr::dcvcrt {

SequenceState::SequenceState(std::uint32_t gop_size) : gop_size_(gop_size) {}

Result<FrameType> SequenceState::next_frame_type() const {
    if (gop_size_ == 0) {
        return Error(ErrorCode::invalid_state, "GOP size is zero", "dcvcrt.state");
    }
    if (!reference_frame_ || (frame_index_ % gop_size_) == 0) {
        return FrameType::intra;
    }
    return FrameType::predicted;
}

Result<void> SequenceState::validate_packet(FrameType frame_type) const {
    if (frame_type == FrameType::predicted && !reference_frame_) {
        return Error(
            ErrorCode::invalid_state,
            "predicted frame received without a reference frame",
            "dcvcrt.state");
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
    if (reconstructed_frame.size_bytes() == 0) {
        return Error(
            ErrorCode::backend_error,
            "backend returned an empty reference frame",
            "dcvcrt.state");
    }
    if (frame_type == FrameType::predicted && !reference_frame_) {
        return Error(
            ErrorCode::invalid_state,
            "cannot commit a predicted frame without a reference",
            "dcvcrt.state");
    }
    reference_frame_ = std::move(reconstructed_frame);
    latent_state_ = std::move(latent_state);
    ++frame_index_;
    return {};
}

void SequenceState::reset() noexcept {
    reference_frame_.reset();
    latent_state_.clear();
    frame_index_ = 0;
    ++generation_;
}

std::uint64_t SequenceState::frame_index() const noexcept { return frame_index_; }
std::uint64_t SequenceState::generation() const noexcept { return generation_; }
bool SequenceState::has_reference() const noexcept { return reference_frame_.has_value(); }

}  // namespace nvcr::dcvcrt

