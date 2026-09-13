#pragma once

#include "nvcr/common/error.hpp"
#include "nvcr/codec/sequence_state.hpp"
#include "nvcr/dcvcrt/rans_codec.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include "payload.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace nvcr::dcvcrt {

enum class IntraStage : std::uint8_t {
    analysis = 0,
    hyper_analysis,
    hyper_synthesis,
    spatial_prior_1,
    spatial_prior_2,
    spatial_prior_3,
    synthesis,
    count,
};

enum class PredictedStage : std::uint8_t {
    reference_frame = 0,
    reference_feature,
    analysis,
    hyper_analysis,
    prior,
    spatial_prior,
    synthesis,
    count,
};

class ReferenceState final {
public:
    [[nodiscard]] bool matches(const codec::SequenceStateView& state) const noexcept;
    [[nodiscard]] bool has_feature() const noexcept { return has_feature_; }

    void commit(
        const codec::SequenceStateView& state,
        bool has_feature) noexcept;
    void reset() noexcept;

private:
    bool has_frame_{};
    bool has_feature_{};
    std::uint64_t next_frame_index_{};
    std::uint64_t generation_{};
};

// Owns DCVC-RT I-frame stage identity, quantization assets, entropy state, and
// private payload syntax. Provider implementations execute the selected stage
// handles and retain ownership of tensor storage and execution mechanics.
class IntraOrchestration final {
public:
    [[nodiscard]] Result<void> initialize(const std::filesystem::path& bundle_root);

    [[nodiscard]] Result<void> bind_stages(
        std::span<const provider::experimental::ExecutableStage> loaded_stages);

    [[nodiscard]] const provider::experimental::ExecutableStage& stage(
        IntraStage stage) const;

    [[nodiscard]] std::span<const float> encoder_quantization() const noexcept {
        return q_encoder_;
    }

    [[nodiscard]] std::span<const float> decoder_quantization() const noexcept {
        return q_decoder_;
    }

    [[nodiscard]] Result<void> begin_encode(bool use_two_coders);
    [[nodiscard]] Result<void> encode_z(
        std::span<const std::int8_t> symbols,
        std::uint32_t qp,
        std::size_t per_channel_size);
    [[nodiscard]] Result<void> encode_y(std::span<const std::int16_t> indexes);
    [[nodiscard]] Result<std::vector<std::byte>> finish_encode(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t qp,
        bool use_two_coders);

    [[nodiscard]] Result<IntraPayload> parse_payload(
        std::span<const std::byte> payload) const;
    [[nodiscard]] Result<void> begin_decode(const IntraPayload& payload);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_z(
        std::size_t symbol_count,
        std::uint32_t qp,
        std::size_t per_channel_size);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_y(
        std::span<const std::uint8_t> indexes);

private:
    static constexpr std::size_t stage_count =
        static_cast<std::size_t>(IntraStage::count);

    RansCodec rans_;
    std::size_t image_z_group_{static_cast<std::size_t>(-1)};
    std::size_t gaussian_y_group_{static_cast<std::size_t>(-1)};
    std::vector<float> q_encoder_;
    std::vector<float> q_decoder_;
    std::array<provider::experimental::ExecutableStage, stage_count> stages_;
};

struct PredictedFrameDecision final {
    std::uint32_t qp{};
    bool use_frame_reference{};
};

// Owns DCVC-RT predicted-frame stage identity, temporal decisions,
// quantization assets, entropy state, and private payload syntax.
class PredictedOrchestration final {
public:
    [[nodiscard]] Result<void> initialize(const std::filesystem::path& bundle_root);

    [[nodiscard]] Result<void> bind_stages(
        std::span<const provider::experimental::ExecutableStage> loaded_stages);

    [[nodiscard]] const provider::experimental::ExecutableStage& stage(
        PredictedStage stage) const;

    [[nodiscard]] std::span<const float> encoder_quantization() const noexcept {
        return q_encoder_;
    }
    [[nodiscard]] std::span<const float> decoder_quantization() const noexcept {
        return q_decoder_;
    }
    [[nodiscard]] std::span<const float> feature_quantization() const noexcept {
        return q_feature_;
    }
    [[nodiscard]] std::span<const float> reconstruction_quantization() const noexcept {
        return q_reconstruction_;
    }

    [[nodiscard]] Result<PredictedFrameDecision> select_frame(
        std::uint32_t base_qp,
        const codec::SequenceStateView& state,
        const ReferenceState& reference) const;

    [[nodiscard]] Result<void> validate_decode_reference(
        bool use_frame_reference,
        const codec::SequenceStateView& state,
        const ReferenceState& reference) const;

    [[nodiscard]] Result<void> begin_encode(bool use_two_coders);
    [[nodiscard]] Result<void> encode_z(
        std::span<const std::int8_t> symbols,
        std::uint32_t qp,
        std::size_t per_channel_size);
    [[nodiscard]] Result<void> encode_y(std::span<const std::int16_t> indexes);
    [[nodiscard]] Result<std::vector<std::byte>> finish_encode(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t qp,
        bool use_two_coders,
        bool use_frame_reference);

    [[nodiscard]] Result<PredictedPayload> parse_payload(
        std::span<const std::byte> payload) const;
    [[nodiscard]] Result<void> begin_decode(const PredictedPayload& payload);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_z(
        std::size_t symbol_count,
        std::uint32_t qp,
        std::size_t per_channel_size);
    [[nodiscard]] Result<std::vector<std::int8_t>> decode_y(
        std::span<const std::uint8_t> indexes);

private:
    static constexpr std::size_t stage_count =
        static_cast<std::size_t>(PredictedStage::count);

    RansCodec rans_;
    std::size_t video_z_group_{static_cast<std::size_t>(-1)};
    std::size_t gaussian_y_group_{static_cast<std::size_t>(-1)};
    std::vector<float> q_encoder_;
    std::vector<float> q_decoder_;
    std::vector<float> q_feature_;
    std::vector<float> q_reconstruction_;
    std::array<provider::experimental::ExecutableStage, stage_count> stages_;
};

}  // namespace nvcr::dcvcrt
