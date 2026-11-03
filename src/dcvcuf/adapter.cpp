#include "nvcr/dcvcuf/adapter.hpp"

#include "native_session.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/dcvcuf/backend.hpp"
#include "nvcr/runtime/registry.hpp"
#include "nvcr/statistics/statistics.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace nvcr::dcvcuf {
namespace {

constexpr std::array<std::byte, 4> intra_magic{
    std::byte{'U'}, std::byte{'F'}, std::byte{'I'}, std::byte{'1'}};
constexpr std::array<std::byte, 4> predicted_magic{
    std::byte{'U'}, std::byte{'F'}, std::byte{'P'}, std::byte{'1'}};

Error codec_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "dcvcuf");
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
}

std::uint64_t read_u64(std::span<const std::byte> input, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(
                     std::to_integer<std::uint8_t>(input[offset + shift / 8]))
                 << shift;
    return value;
}

// Upstream test_video.py schedule: frame types are decided once per step (one
// intra frame or one eight-frame HT-S group), and HT-S feature memory resets
// when the group ends at an index congruent to 1 modulo this interval.
constexpr std::uint64_t group_frames = 8;
constexpr std::uint64_t reset_interval = 32;  // upstream --reset_interval default

// Predicted payload flag byte; the upstream UF stream carries the same bit.
constexpr std::uint8_t reset_feature_memory_flag = 0x01;

struct CodecPayload final {
    std::uint8_t entropy_parallel{};
    bool reset_feature_memory{};
    std::vector<Timestamp> timestamps;
    std::span<const std::byte> bitstream;
};

std::vector<std::byte> make_payload(
    FrameType type, const NativeEncoded& encoded,
    std::span<const Timestamp> timestamps, bool reset_feature_memory = false) {
    const auto& magic = type == FrameType::intra ? intra_magic : predicted_magic;
    std::vector<std::byte> output;
    output.reserve(8 + timestamps.size() * 8 + encoded.bitstream.size());
    output.insert(output.end(), magic.begin(), magic.end());
    output.push_back(static_cast<std::byte>(timestamps.size()));
    output.push_back(static_cast<std::byte>(encoded.entropy_parallel));
    output.push_back(static_cast<std::byte>(
        reset_feature_memory ? reset_feature_memory_flag : 0U));
    output.push_back(std::byte{0});
    for (const auto timestamp : timestamps)
        append_u64(output, static_cast<std::uint64_t>(timestamp.count()));
    output.insert(output.end(), encoded.bitstream.begin(), encoded.bitstream.end());
    return output;
}

Result<CodecPayload> parse_payload(
    FrameType type, std::span<const std::byte> payload) {
    const auto& magic = type == FrameType::intra ? intra_magic : predicted_magic;
    if (payload.size() < 17 ||
        !std::equal(magic.begin(), magic.end(), payload.begin())) {
        return codec_error(
            ErrorCode::malformed_bitstream, "invalid DCVC-UF codec payload header");
    }
    const auto count = std::to_integer<std::uint8_t>(payload[4]);
    const auto entropy_parallel = std::to_integer<std::uint8_t>(payload[5]);
    const auto flags = std::to_integer<std::uint8_t>(payload[6]);
    const auto allowed_flags =
        type == FrameType::intra ? std::uint8_t{0} : reset_feature_memory_flag;
    if ((flags & ~allowed_flags) != 0 || payload[7] != std::byte{0} ||
        entropy_parallel == 0 || count == 0 || count > 8 ||
        (type == FrameType::intra && count != 1)) {
        return codec_error(
            ErrorCode::malformed_bitstream, "invalid DCVC-UF payload fields");
    }
    const auto header = 8U + static_cast<std::size_t>(count) * 8U;
    if (payload.size() <= header) {
        return codec_error(
            ErrorCode::malformed_bitstream, "DCVC-UF payload has no entropy stream");
    }
    CodecPayload parsed;
    parsed.entropy_parallel = entropy_parallel;
    parsed.reset_feature_memory = (flags & reset_feature_memory_flag) != 0;
    parsed.timestamps.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        parsed.timestamps.emplace_back(static_cast<Timestamp::rep>(
            read_u64(payload, 8U + index * 8U)));
    parsed.bitstream = payload.subspan(header);
    if (parsed.bitstream.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return codec_error(
            ErrorCode::resource_exhausted, "DCVC-UF entropy stream is too large");
    }
    return parsed;
}

Result<Packet> wrap_access_unit(
    const RuntimeConfiguration& configuration, FrameType type,
    std::uint32_t width, std::uint32_t height, Timestamp timestamp,
    std::vector<std::byte> payload) {
    AccessUnit access_unit{
        configuration.stream.bitstream_model_id,
        width,
        height,
        configuration.codec.intra_qp,
        type,
        type == FrameType::intra,
        std::move(payload)};
    auto serialized = AccessUnitIO::serialize(
        access_unit, configuration.runtime.max_packet_bytes);
    if (!serialized) return serialized.error();
    return Packet(std::move(serialized.value()), timestamp, type);
}

class Encoder final : public IEncoderSession {
public:
    Encoder(
        RuntimeConfiguration configuration,
        std::shared_ptr<NativeExecutionSession> provider,
        std::shared_ptr<Statistics> statistics)
        : configuration_(std::move(configuration)),
          provider_(std::move(provider)),
          statistics_(std::move(statistics)) {}

    Result<void> send_frame(const Frame& frame) override {
        if (flushed_)
            return codec_error(ErrorCode::invalid_state, "encoder is flushed");
        auto valid = validate_frame(frame);
        if (!valid) return valid.error();

        // Like upstream, only the first frame of a step can start an intra
        // frame; frames inside an open HT-S group always join that group.
        const auto period = static_cast<std::uint64_t>(configuration_.codec.gop_size);
        const bool intra = pending_.empty() &&
            (step_index_ == 0 || period == 1 ||
             (period > 1 && step_index_ != 1 && step_index_ % period == 1));
        if (intra) {
            const auto started = std::chrono::steady_clock::now();
            auto encoded = provider_->encode_intra(
                frame, configuration_.codec.intra_qp);
            if (!encoded) return encoded.error();
            const std::array timestamps{frame.timestamp()};
            auto packet = wrap_access_unit(
                configuration_, FrameType::intra, frame.width(), frame.height(),
                frame.timestamp(),
                make_payload(FrameType::intra, encoded.value(), timestamps));
            if (!packet) return packet.error();
            statistics_->record_encode(
                packet.value().size(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started));
            outputs_.push_back(std::move(packet.value()));
            ++step_index_;
        } else {
            pending_.push_back(frame);
            if (pending_.size() == group_frames) {
                auto emitted = emit_predicted();
                if (!emitted) return emitted.error();
            }
        }
        ++frame_index_;
        return {};
    }

    Result<Packet> receive_access_unit() override {
        if (!outputs_.empty()) {
            Packet output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_)
            return codec_error(ErrorCode::end_of_stream, "encoder is drained");
        return codec_error(ErrorCode::try_again, "encoder needs more input");
    }

    Result<void> flush() override {
        if (flushed_) return {};
        if (!pending_.empty()) {
            const auto valid_count = pending_.size();
            while (pending_.size() < group_frames) pending_.push_back(pending_.back());
            auto emitted = emit_predicted(valid_count);
            if (!emitted) return emitted.error();
        }
        flushed_ = true;
        return {};
    }

    Result<void> reset() override {
        pending_.clear();
        outputs_.clear();
        frame_index_ = 0;
        step_index_ = 0;
        flushed_ = false;
        return provider_->reset_encoder_state();
    }

private:
    Result<void> validate_frame(const Frame& frame) {
        if (frame.pixel_format() != PixelFormat::yuv420p8 ||
            frame.width() == 0 || frame.height() == 0 ||
            (frame.width() & 1U) != 0 || (frame.height() & 1U) != 0) {
            return codec_error(
                ErrorCode::invalid_argument,
                "DCVC-UF requires nonempty even-dimension YUV420P8 frames");
        }
        if (frame_index_ != 0 &&
            (frame.width() != width_ || frame.height() != height_)) {
            return codec_error(
                ErrorCode::invalid_argument,
                "DCVC-UF does not support resolution changes within a sequence");
        }
        if (frame_index_ == 0) {
            width_ = frame.width();
            height_ = frame.height();
        }
        return {};
    }

    Result<void> emit_predicted(std::size_t valid_count = group_frames) {
        const bool reset_feature_memory =
            (step_index_ + group_frames) % reset_interval == 1;
        const auto started = std::chrono::steady_clock::now();
        auto encoded = provider_->encode_predicted(
            std::span<const Frame>(pending_.data(), pending_.size()),
            configuration_.codec.intra_qp, reset_feature_memory);
        if (!encoded) return encoded.error();
        std::vector<Timestamp> timestamps;
        timestamps.reserve(valid_count);
        for (std::size_t index = 0; index < valid_count; ++index)
            timestamps.push_back(pending_[index].timestamp());
        auto packet = wrap_access_unit(
            configuration_, FrameType::predicted, width_, height_,
            timestamps.front(),
            make_payload(FrameType::predicted, encoded.value(), timestamps,
                         reset_feature_memory));
        if (!packet) return packet.error();
        statistics_->record_encode(
            packet.value().size(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started));
        outputs_.push_back(std::move(packet.value()));
        pending_.clear();
        step_index_ += valid_count;
        return {};
    }

    RuntimeConfiguration configuration_;
    std::shared_ptr<NativeExecutionSession> provider_;
    std::shared_ptr<Statistics> statistics_;
    std::vector<Frame> pending_;
    std::deque<Packet> outputs_;
    std::uint64_t frame_index_{};
    std::uint64_t step_index_{};  // upstream frame_idx at the current step start
    std::uint32_t width_{};
    std::uint32_t height_{};
    bool flushed_{};
};

class Decoder final : public IDecoderSession {
public:
    Decoder(
        RuntimeConfiguration configuration,
        std::shared_ptr<NativeExecutionSession> provider,
        std::shared_ptr<Statistics> statistics)
        : configuration_(std::move(configuration)),
          provider_(std::move(provider)),
          statistics_(std::move(statistics)) {}

    Result<void> send_access_unit(const Packet& packet) override {
        if (flushed_)
            return codec_error(ErrorCode::invalid_state, "decoder is flushed");
        if (packet.size() > configuration_.runtime.max_packet_bytes)
            return codec_error(
                ErrorCode::resource_exhausted,
                "DCVC-UF packet exceeds configured limit");
        auto access_unit = AccessUnitIO::deserialize(
            packet.data(), configuration_.runtime.max_packet_bytes);
        if (!access_unit) return access_unit.error();
        if (access_unit.value().model_id !=
                configuration_.stream.bitstream_model_id ||
            access_unit.value().frame_type != packet.frame_type()) {
            return codec_error(
                ErrorCode::malformed_bitstream,
                "DCVC-UF access-unit identity or frame type mismatch");
        }
        auto payload = parse_payload(
            access_unit.value().frame_type, access_unit.value().payload);
        if (!payload) return payload.error();

        const auto started = std::chrono::steady_clock::now();
        if (access_unit.value().frame_type == FrameType::intra) {
            auto reset = provider_->reset_decoder_state();
            if (!reset) return reset.error();
            auto decoded = provider_->decode_intra(
                payload.value().bitstream, payload.value().entropy_parallel,
                access_unit.value().qp, access_unit.value().width,
                access_unit.value().height, payload.value().timestamps.front());
            if (!decoded) return decoded.error();
            outputs_.push_back(std::move(decoded.value()));
        } else {
            auto decoded = provider_->decode_predicted(
                payload.value().bitstream, payload.value().entropy_parallel,
                access_unit.value().qp, access_unit.value().width,
                access_unit.value().height, payload.value().timestamps,
                payload.value().reset_feature_memory);
            if (!decoded) return decoded.error();
            for (auto& frame : decoded.value())
                outputs_.push_back(std::move(frame));
        }
        for (std::size_t index = 0; index < payload.value().timestamps.size(); ++index)
            statistics_->record_decode(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started) /
                payload.value().timestamps.size());
        return {};
    }

    Result<Frame> receive_frame() override {
        if (!outputs_.empty()) {
            Frame output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_)
            return codec_error(ErrorCode::end_of_stream, "decoder is drained");
        return codec_error(ErrorCode::try_again, "decoder needs more input");
    }

    Result<void> flush() override {
        flushed_ = true;
        return {};
    }

    Result<void> reset() override {
        outputs_.clear();
        flushed_ = false;
        return provider_->reset_decoder_state();
    }

private:
    RuntimeConfiguration configuration_;
    std::shared_ptr<NativeExecutionSession> provider_;
    std::shared_ptr<Statistics> statistics_;
    std::deque<Frame> outputs_;
    bool flushed_{};
};

class Adapter final : public codec::ICodecAdapter {
public:
    codec::CodecDescriptor descriptor() const override {
        return codec_descriptor();
    }
    codec::CodecCapabilities capabilities() const override {
        auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->capabilities : codec::CodecCapabilities{};
    }
    codec::OptionSchema encoder_options() const override {
        auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->encoder_options : codec::OptionSchema{};
    }
    codec::OptionSchema decoder_options() const override {
        auto entry = runtime::Registry::instance().find_codec(codec_descriptor().id);
        return entry ? entry->decoder_options : codec::OptionSchema{};
    }

    Result<codec::Sessions> create_sessions(
        const RuntimeConfiguration& configuration,
        std::shared_ptr<provider::experimental::IProviderSession> session) override {
        auto native = std::dynamic_pointer_cast<NativeExecutionSession>(session);
        if (!native)
            return codec_error(
                ErrorCode::missing_provider,
                "DCVC-UF requires the dcvcuf-upstream-cuda provider session");
        if (configuration.codec.gop_size != 1 &&
            configuration.codec.gop_size % 8 != 0) {
            return codec_error(
                ErrorCode::invalid_argument,
                "DCVC-UF HT-S requires --gop-size 1 or a multiple of 8");
        }
        auto statistics = std::make_shared<Statistics>();
        codec::Sessions output;
        output.encoder = std::make_unique<Encoder>(
            configuration, native, statistics);
        output.decoder = std::make_unique<Decoder>(
            configuration, std::move(native), statistics);
        output.statistics = std::move(statistics);
        return output;
    }
};

}  // namespace

Result<std::unique_ptr<codec::ICodecAdapter>> make_adapter() {
    return std::unique_ptr<codec::ICodecAdapter>(std::make_unique<Adapter>());
}

}  // namespace nvcr::dcvcuf
