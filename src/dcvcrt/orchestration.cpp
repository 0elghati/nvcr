#include "orchestration.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace nvcr::dcvcrt {
namespace {

constexpr std::array<std::string_view, 7> intra_stage_ids{
    "i_analysis.plan",
    "i_hyper_analysis.plan",
    "i_hyper_synthesis.plan",
    "i_spatial_prior_1.plan",
    "i_spatial_prior_2.plan",
    "i_spatial_prior_3.plan",
    "i_synthesis.plan",
};

constexpr std::array<std::string_view, 7> predicted_stage_ids{
    "p_reference_frame.plan",
    "p_reference_feature.plan",
    "p_analysis.plan",
    "p_hyper_analysis.plan",
    "p_prior.plan",
    "p_spatial_prior.plan",
    "p_synthesis.plan",
};

Error orchestration_error(std::string message) {
    return Error(ErrorCode::backend_error, std::move(message), "dcvcrt.orchestration");
}

Result<std::vector<std::byte>> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return Error(ErrorCode::io_error, "cannot open file: " + path.string(),
                     "dcvcrt.orchestration");
    }
    const auto end = input.tellg();
    if (end <= 0) {
        return Error(ErrorCode::io_error, "file is empty: " + path.string(),
                     "dcvcrt.orchestration");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        return Error(ErrorCode::io_error, "failed to read file: " + path.string(),
                     "dcvcrt.orchestration");
    }
    return bytes;
}

class AssetReader final {
public:
    AssetReader(std::filesystem::path path, std::span<const std::byte> bytes)
        : path_(std::move(path)), bytes_(bytes) {}

    [[nodiscard]] Result<void> expect_magic(std::string_view magic) {
        if (remaining() < magic.size()) return truncated();
        for (std::size_t index = 0; index < magic.size(); ++index) {
            if (std::to_integer<unsigned char>(bytes_[offset_ + index]) !=
                static_cast<unsigned char>(magic[index])) {
                return orchestration_error("invalid asset magic in " + path_.string());
            }
        }
        offset_ += magic.size();
        return {};
    }

    [[nodiscard]] Result<std::uint32_t> read_u32() {
        if (remaining() < 4U) return truncated();
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < 4U; ++index) {
            value |= static_cast<std::uint32_t>(
                std::to_integer<unsigned char>(bytes_[offset_ + index])) <<
                (8U * index);
        }
        offset_ += 4U;
        return value;
    }

    [[nodiscard]] Result<std::int32_t> read_i32() {
        auto value = read_u32();
        if (!value) return value.error();
        return std::bit_cast<std::int32_t>(value.value());
    }

    [[nodiscard]] Result<float> read_f32() {
        auto value = read_u32();
        if (!value) return value.error();
        return std::bit_cast<float>(value.value());
    }

    [[nodiscard]] Result<std::string> read_string(std::size_t size) {
        if (remaining() < size) return truncated();
        std::string output(size, char{});
        for (std::size_t index = 0; index < size; ++index) {
            output[index] = static_cast<char>(
                std::to_integer<unsigned char>(bytes_[offset_ + index]));
        }
        offset_ += size;
        return output;
    }

    [[nodiscard]] bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - offset_;
    }

    [[nodiscard]] Error truncated() const {
        return orchestration_error("truncated asset file: " + path_.string());
    }

    std::filesystem::path path_;
    std::span<const std::byte> bytes_;
    std::size_t offset_{};
};

}  // namespace

bool ReferenceState::matches(const codec::SequenceStateView& state) const noexcept {
    return has_frame_ && next_frame_index_ == state.frame_index &&
        generation_ == state.generation;
}

void ReferenceState::commit(
    const codec::SequenceStateView& state,
    bool has_feature) noexcept {
    has_frame_ = true;
    has_feature_ = has_feature;
    next_frame_index_ = state.frame_index + 1U;
    generation_ = state.generation;
}

void ReferenceState::reset() noexcept {
    has_frame_ = false;
    has_feature_ = false;
    next_frame_index_ = 0U;
    generation_ = 0U;
}

Result<void> IntraOrchestration::initialize(
    const std::filesystem::path& bundle_root) {
    const auto entropy_path = bundle_root / "i_entropy.bin";
    auto entropy_bytes = read_binary(entropy_path);
    if (!entropy_bytes) return entropy_bytes.error();
    AssetReader entropy_reader(entropy_path, entropy_bytes.value());
    auto entropy_magic = entropy_reader.expect_magic("NVCRENT1");
    if (!entropy_magic) return entropy_magic.error();
    auto group_count = entropy_reader.read_u32();
    if (!group_count) return group_count.error();
    if (group_count.value() != 2U) {
        return orchestration_error(
            "I-frame entropy asset must contain exactly two CDF groups");
    }

    rans_.clear_cdfs();
    image_z_group_ = std::numeric_limits<std::size_t>::max();
    gaussian_y_group_ = std::numeric_limits<std::size_t>::max();
    for (std::uint32_t group_index = 0; group_index < group_count.value(); ++group_index) {
        auto name_size = entropy_reader.read_u32();
        auto row_count = entropy_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!row_count) return row_count.error();
        if (name_size.value() == 0U || name_size.value() > 64U ||
            row_count.value() == 0U || row_count.value() > 16384U) {
            return orchestration_error(
                "invalid CDF group dimensions in " + entropy_path.string());
        }
        auto name = entropy_reader.read_string(name_size.value());
        if (!name) return name.error();

        RansCdfTable table;
        table.values.reserve(row_count.value());
        table.sizes.reserve(row_count.value());
        table.offsets.reserve(row_count.value());
        for (std::uint32_t row_index = 0; row_index < row_count.value(); ++row_index) {
            auto size = entropy_reader.read_i32();
            auto offset = entropy_reader.read_i32();
            if (!size) return size.error();
            if (!offset) return offset.error();
            if (size.value() < 2 || size.value() > 1024) {
                return orchestration_error(
                    "invalid CDF row size in " + entropy_path.string());
            }
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(size.value()));
            for (std::int32_t value_index = 0; value_index < size.value(); ++value_index) {
                auto value = entropy_reader.read_i32();
                if (!value) return value.error();
                values.push_back(value.value());
            }
            table.values.push_back(std::move(values));
            table.sizes.push_back(size.value());
            table.offsets.push_back(offset.value());
        }
        auto registered = rans_.add_cdf(std::move(table));
        if (!registered) return registered.error();
        if (name.value() == "image_z") {
            image_z_group_ = registered.value();
        } else if (name.value() == "gaussian_y") {
            gaussian_y_group_ = registered.value();
        } else {
            return orchestration_error("unknown CDF group " + name.value());
        }
    }
    if (!entropy_reader.done() ||
        image_z_group_ == std::numeric_limits<std::size_t>::max() ||
        gaussian_y_group_ == std::numeric_limits<std::size_t>::max()) {
        return orchestration_error("incomplete I-frame entropy asset");
    }

    const auto quant_path = bundle_root / "i_quant.bin";
    auto quant_bytes = read_binary(quant_path);
    if (!quant_bytes) return quant_bytes.error();
    AssetReader quant_reader(quant_path, quant_bytes.value());
    auto quant_magic = quant_reader.expect_magic("NVCRQNT1");
    if (!quant_magic) return quant_magic.error();
    auto qp_count = quant_reader.read_u32();
    auto channels = quant_reader.read_u32();
    if (!qp_count) return qp_count.error();
    if (!channels) return channels.error();
    if (qp_count.value() != 64U || channels.value() != 368U) {
        return orchestration_error("unexpected I-frame quantization tensor shape");
    }

    const auto value_count = static_cast<std::size_t>(qp_count.value()) * channels.value();
    q_encoder_.clear();
    q_decoder_.clear();
    q_encoder_.reserve(value_count);
    q_decoder_.reserve(value_count);
    for (auto* destination : {&q_encoder_, &q_decoder_}) {
        for (std::size_t index = 0; index < value_count; ++index) {
            auto value = quant_reader.read_f32();
            if (!value) return value.error();
            if (!std::isfinite(value.value())) {
                return orchestration_error(
                    "invalid quantization value in " + quant_path.string());
            }
            destination->push_back(value.value());
        }
    }
    if (!quant_reader.done()) {
        return orchestration_error("trailing data in " + quant_path.string());
    }
    return {};
}

Result<void> IntraOrchestration::bind_stages(
    std::span<const provider::experimental::ExecutableStage> loaded_stages) {
    std::array<provider::experimental::ExecutableStage, stage_count> selected;
    for (std::size_t expected_index = 0; expected_index < intra_stage_ids.size();
         ++expected_index) {
        for (const auto& candidate : loaded_stages) {
            if (candidate &&
                candidate->descriptor().stage_id == intra_stage_ids[expected_index]) {
                selected[expected_index] = candidate;
                break;
            }
        }
        if (!selected[expected_index]) {
            return orchestration_error(
                "provider session is missing DCVC-RT I-frame stage " +
                std::string(intra_stage_ids[expected_index]));
        }
    }
    stages_ = std::move(selected);
    return {};
}

const provider::experimental::ExecutableStage& IntraOrchestration::stage(
    IntraStage stage_id) const {
    return stages_[static_cast<std::size_t>(stage_id)];
}

Result<void> IntraOrchestration::begin_encode(bool use_two_coders) {
    rans_.reset_encoder();
    return rans_.set_use_two_coders(use_two_coders);
}

Result<void> IntraOrchestration::encode_z(
    std::span<const std::int8_t> symbols,
    std::uint32_t qp,
    std::size_t per_channel_size) {
    return rans_.encode_z(
        symbols, image_z_group_, static_cast<std::size_t>(qp) * 128U,
        per_channel_size);
}

Result<void> IntraOrchestration::encode_y(
    std::span<const std::int16_t> indexes) {
    return rans_.encode_y(indexes, gaussian_y_group_);
}

Result<std::vector<std::byte>> IntraOrchestration::finish_encode(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t qp,
    bool use_two_coders) {
    auto stream = rans_.finish_encode();
    if (!stream) return stream.error();
    return make_intra_payload(width, height, qp, use_two_coders, stream.value());
}

Result<IntraPayload> IntraOrchestration::parse_payload(
    std::span<const std::byte> payload) const {
    return parse_intra_payload(payload);
}

Result<void> IntraOrchestration::begin_decode(const IntraPayload& payload) {
    rans_.reset_encoder();
    auto mode = rans_.set_use_two_coders(payload.two_coders);
    if (!mode) return mode.error();
    return rans_.set_stream(payload.rans);
}

Result<std::vector<std::int8_t>> IntraOrchestration::decode_z(
    std::size_t symbol_count,
    std::uint32_t qp,
    std::size_t per_channel_size) {
    return rans_.decode_z(
        symbol_count, image_z_group_, static_cast<std::size_t>(qp) * 128U,
        per_channel_size);
}

Result<std::vector<std::int8_t>> IntraOrchestration::decode_y(
    std::span<const std::uint8_t> indexes) {
    return rans_.decode_y(indexes, gaussian_y_group_);
}

Result<void> PredictedOrchestration::initialize(
    const std::filesystem::path& bundle_root) {
    const auto entropy_path = bundle_root / "p_entropy.bin";
    auto entropy_bytes = read_binary(entropy_path);
    if (!entropy_bytes) return entropy_bytes.error();
    AssetReader entropy_reader(entropy_path, entropy_bytes.value());
    auto entropy_magic = entropy_reader.expect_magic("NVCRPEN1");
    if (!entropy_magic) return entropy_magic.error();
    auto group_count = entropy_reader.read_u32();
    if (!group_count) return group_count.error();
    if (group_count.value() != 2U) {
        return orchestration_error(
            "P-frame entropy asset must contain exactly two CDF groups");
    }

    rans_.clear_cdfs();
    video_z_group_ = std::numeric_limits<std::size_t>::max();
    gaussian_y_group_ = std::numeric_limits<std::size_t>::max();
    for (std::uint32_t group_index = 0; group_index < group_count.value(); ++group_index) {
        auto name_size = entropy_reader.read_u32();
        auto row_count = entropy_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!row_count) return row_count.error();
        if (name_size.value() == 0U || name_size.value() > 64U ||
            row_count.value() == 0U || row_count.value() > 16384U) {
            return orchestration_error("invalid P-frame CDF group dimensions");
        }
        auto name = entropy_reader.read_string(name_size.value());
        if (!name) return name.error();

        RansCdfTable table;
        table.values.reserve(row_count.value());
        table.sizes.reserve(row_count.value());
        table.offsets.reserve(row_count.value());
        for (std::uint32_t row_index = 0; row_index < row_count.value(); ++row_index) {
            auto size = entropy_reader.read_i32();
            auto offset = entropy_reader.read_i32();
            if (!size) return size.error();
            if (!offset) return offset.error();
            if (size.value() < 2 || size.value() > 1024) {
                return orchestration_error("invalid P-frame CDF row size");
            }
            std::vector<std::int32_t> values;
            values.reserve(static_cast<std::size_t>(size.value()));
            for (std::int32_t value_index = 0; value_index < size.value(); ++value_index) {
                auto value = entropy_reader.read_i32();
                if (!value) return value.error();
                values.push_back(value.value());
            }
            table.values.push_back(std::move(values));
            table.sizes.push_back(size.value());
            table.offsets.push_back(offset.value());
        }
        auto registered = rans_.add_cdf(std::move(table));
        if (!registered) return registered.error();
        if (name.value() == "video_z") {
            video_z_group_ = registered.value();
        } else if (name.value() == "gaussian_y") {
            gaussian_y_group_ = registered.value();
        } else {
            return orchestration_error("unknown P-frame CDF group " + name.value());
        }
    }
    if (!entropy_reader.done() ||
        video_z_group_ == std::numeric_limits<std::size_t>::max() ||
        gaussian_y_group_ == std::numeric_limits<std::size_t>::max()) {
        return orchestration_error("incomplete P-frame entropy asset");
    }

    const auto quant_path = bundle_root / "p_quant.bin";
    auto quant_bytes = read_binary(quant_path);
    if (!quant_bytes) return quant_bytes.error();
    AssetReader quant_reader(quant_path, quant_bytes.value());
    auto quant_magic = quant_reader.expect_magic("NVCRPQN1");
    if (!quant_magic) return quant_magic.error();
    auto array_count = quant_reader.read_u32();
    if (!array_count) return array_count.error();
    if (array_count.value() != 4U) {
        return orchestration_error("P-frame quant asset must contain four arrays");
    }

    q_encoder_.clear();
    q_decoder_.clear();
    q_feature_.clear();
    q_reconstruction_.clear();
    for (std::uint32_t array_index = 0; array_index < array_count.value(); ++array_index) {
        auto name_size = quant_reader.read_u32();
        auto qps = quant_reader.read_u32();
        auto channels = quant_reader.read_u32();
        if (!name_size) return name_size.error();
        if (!qps) return qps.error();
        if (!channels) return channels.error();
        auto name = quant_reader.read_string(name_size.value());
        if (!name) return name.error();
        const std::uint32_t expected_channels =
            name.value() == "q_recon" ? 320U : 256U;
        if (qps.value() != 72U || channels.value() != expected_channels) {
            return orchestration_error(
                "unexpected P-frame quantization tensor shape for " + name.value());
        }
        std::vector<float>* destination = nullptr;
        if (name.value() == "q_encoder") destination = &q_encoder_;
        else if (name.value() == "q_decoder") destination = &q_decoder_;
        else if (name.value() == "q_feature") destination = &q_feature_;
        else if (name.value() == "q_recon") destination = &q_reconstruction_;
        else return orchestration_error("unknown P-frame quant array " + name.value());
        destination->reserve(
            static_cast<std::size_t>(qps.value()) * channels.value());
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(qps.value()) * channels.value();
             ++index) {
            auto value = quant_reader.read_f32();
            if (!value) return value.error();
            if (!std::isfinite(value.value())) {
                return orchestration_error("invalid P-frame quantization value");
            }
            destination->push_back(value.value());
        }
    }
    if (!quant_reader.done() || q_encoder_.empty() || q_decoder_.empty() ||
        q_feature_.empty() || q_reconstruction_.empty()) {
        return orchestration_error("incomplete P-frame quantization asset");
    }
    return {};
}

Result<void> PredictedOrchestration::bind_stages(
    std::span<const provider::experimental::ExecutableStage> loaded_stages) {
    std::array<provider::experimental::ExecutableStage, stage_count> selected;
    for (std::size_t expected_index = 0; expected_index < predicted_stage_ids.size();
         ++expected_index) {
        for (const auto& candidate : loaded_stages) {
            if (candidate &&
                candidate->descriptor().stage_id == predicted_stage_ids[expected_index]) {
                selected[expected_index] = candidate;
                break;
            }
        }
        if (!selected[expected_index]) {
            return orchestration_error(
                "provider session is missing DCVC-RT P-frame stage " +
                std::string(predicted_stage_ids[expected_index]));
        }
    }
    stages_ = std::move(selected);
    return {};
}

const provider::experimental::ExecutableStage& PredictedOrchestration::stage(
    PredictedStage stage_id) const {
    return stages_[static_cast<std::size_t>(stage_id)];
}

Result<PredictedFrameDecision> PredictedOrchestration::select_frame(
    std::uint32_t base_qp,
    const codec::SequenceStateView& state,
    const ReferenceState& reference) const {
    if (!reference.matches(state)) {
        return Error(ErrorCode::invalid_state, "encoder device DPB is unavailable",
                     "dcvcrt.orchestration");
    }
    constexpr std::array<std::uint32_t, 8> index_map{0, 1, 0, 2, 0, 2, 0, 2};
    constexpr std::array<std::uint32_t, 3> shifts{0, 8, 4};
    if (base_qp >= 64U) {
        return orchestration_error("base P-frame QP must be in [0, 63]");
    }
    const auto qp = base_qp + shifts[index_map[state.frame_index % index_map.size()]];
    if (qp >= 72U) {
        return orchestration_error(
            "effective P-frame QP exceeds the 72-entry model table");
    }
    return PredictedFrameDecision{
        qp,
        !reference.has_feature() || (state.frame_index % 64U) == 1U,
    };
}

Result<void> PredictedOrchestration::validate_decode_reference(
    bool use_frame_reference,
    const codec::SequenceStateView& state,
    const ReferenceState& reference) const {
    if (!reference.matches(state)) {
        return Error(ErrorCode::invalid_state, "decoder device DPB is unavailable",
                     "dcvcrt.orchestration");
    }
    if (!use_frame_reference && !reference.has_feature()) {
        return Error(ErrorCode::invalid_state,
                     "P-frame feature reference is unavailable",
                     "dcvcrt.orchestration");
    }
    return {};
}

Result<void> PredictedOrchestration::begin_encode(bool use_two_coders) {
    rans_.reset_encoder();
    return rans_.set_use_two_coders(use_two_coders);
}

Result<void> PredictedOrchestration::encode_z(
    std::span<const std::int8_t> symbols,
    std::uint32_t qp,
    std::size_t per_channel_size) {
    return rans_.encode_z(
        symbols, video_z_group_, static_cast<std::size_t>(qp) * 128U,
        per_channel_size);
}

Result<void> PredictedOrchestration::encode_y(
    std::span<const std::int16_t> indexes) {
    return rans_.encode_y(indexes, gaussian_y_group_);
}

Result<std::vector<std::byte>> PredictedOrchestration::finish_encode(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t qp,
    bool use_two_coders,
    bool use_frame_reference) {
    auto stream = rans_.finish_encode();
    if (!stream) return stream.error();
    return make_predicted_payload(
        width, height, qp, use_two_coders, use_frame_reference, stream.value());
}

Result<PredictedPayload> PredictedOrchestration::parse_payload(
    std::span<const std::byte> payload) const {
    return parse_predicted_payload(payload);
}

Result<void> PredictedOrchestration::begin_decode(
    const PredictedPayload& payload) {
    rans_.reset_encoder();
    auto mode = rans_.set_use_two_coders(payload.two_coders);
    if (!mode) return mode.error();
    return rans_.set_stream(payload.rans);
}

Result<std::vector<std::int8_t>> PredictedOrchestration::decode_z(
    std::size_t symbol_count,
    std::uint32_t qp,
    std::size_t per_channel_size) {
    return rans_.decode_z(
        symbol_count, video_z_group_, static_cast<std::size_t>(qp) * 128U,
        per_channel_size);
}

Result<std::vector<std::int8_t>> PredictedOrchestration::decode_y(
    std::span<const std::uint8_t> indexes) {
    return rans_.decode_y(indexes, gaussian_y_group_);
}

}  // namespace nvcr::dcvcrt
