#pragma once

#include "nvcr/common/error.hpp"
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

}  // namespace nvcr::dcvcrt
