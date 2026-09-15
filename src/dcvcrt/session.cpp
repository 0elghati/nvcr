#include "session.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/codec/sequence_state.hpp"
#include "nvcr/logging/logger.hpp"

#include <chrono>
#include <deque>
#include <exception>
#include <new>
#include <optional>
#include <utility>

namespace nvcr::dcvcrt {
namespace {

class SessionCore final {
public:
    SessionCore(
        RuntimeConfiguration configuration,
        std::unique_ptr<codec::CodecBackend> backend,
        std::shared_ptr<Logger> logger,
        std::shared_ptr<Statistics> statistics)
        : configuration_(std::move(configuration)),
          backend_(std::move(backend)),
          encoder_state_(configuration_.codec.gop_size),
          decoder_state_(configuration_.codec.gop_size),
          logger_(std::move(logger)),
          statistics_(std::move(statistics)) {}

    [[nodiscard]] Result<void> initialize() {
        auto initialized = backend_->initialize(configuration_);
        if (!initialized) return initialized.error();
        initialized_ = true;
        logger_->log(LogLevel::info, "DCVC-RT sessions initialized");
        return {};
    }

    [[nodiscard]] Result<Packet> encode(const Frame& frame) {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "session is not initialized", "dcvcrt");
        }
        if (frame.size_bytes() == 0U) {
            return Error(ErrorCode::invalid_argument, "cannot encode an empty frame", "dcvcrt");
        }
        const auto started = std::chrono::steady_clock::now();
        auto frame_type = encoder_state_.next_frame_type();
        if (!frame_type) return frame_type.error();

        auto encoded = backend_->encode(frame, frame_type.value(), encoder_state_.view());
        if (!encoded) return encoded.error();
        AccessUnit access_unit{
            configuration_.stream.bitstream_model_id,
            frame.width(),
            frame.height(),
            encoded.value().effective_qp,
            frame_type.value(),
            frame_type.value() == FrameType::intra,
            std::move(encoded.value().payload)};
        auto serialized = AccessUnitIO::serialize(
            access_unit, configuration_.runtime.max_packet_bytes);
        if (!serialized) return serialized.error();
        Packet packet(std::move(serialized.value()), frame.timestamp(), frame_type.value());
        auto committed = encoder_state_.commit(
            std::move(encoded.value().reconstructed_frame),
            std::move(encoded.value().latent_state),
            frame_type.value());
        if (!committed) return committed.error();
        statistics_->record_encode(
            packet.size(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started));
        return packet;
    }

    [[nodiscard]] Result<Frame> decode(const Packet& packet) {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "session is not initialized", "dcvcrt");
        }
        if (packet.size() > configuration_.runtime.max_packet_bytes) {
            return Error(
                ErrorCode::resource_exhausted,
                "packet exceeds configured limit",
                "dcvcrt");
        }
        std::optional<AccessUnit> access_unit;
        std::span<const std::byte> codec_payload = packet.data();
        if (AccessUnitIO::has_magic(packet.data())) {
            auto parsed = AccessUnitIO::deserialize(
                packet.data(), configuration_.runtime.max_packet_bytes);
            if (!parsed) return parsed.error();
            if (parsed.value().model_id != configuration_.stream.bitstream_model_id) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "access-unit model identity does not match the configured bundle",
                    "dcvcrt");
            }
            if (parsed.value().frame_type != packet.frame_type()) {
                return Error(
                    ErrorCode::malformed_bitstream,
                    "packet and access-unit frame types disagree",
                    "dcvcrt");
            }
            access_unit = std::move(parsed.value());
            if (access_unit->reset_state) {
                decoder_state_.reset();
            }
            codec_payload = access_unit->payload;
        } else if (!configuration_.stream.allow_legacy_access_units) {
            return Error(
                ErrorCode::malformed_bitstream,
                "legacy codec payload rejected by configuration",
                "dcvcrt");
        }
        auto valid = decoder_state_.validate_packet(packet.frame_type());
        if (!valid) return valid.error();
        const auto started = std::chrono::steady_clock::now();
        auto decoded = backend_->decode(
            codec_payload, packet.frame_type(), packet.timestamp(), decoder_state_.view());
        if (!decoded) return decoded.error();

        Frame output = std::move(decoded.value().frame);
        if (access_unit &&
            (output.width() != access_unit->width || output.height() != access_unit->height)) {
            return Error(
                ErrorCode::malformed_bitstream,
                "decoded dimensions do not match the access unit",
                "dcvcrt");
        }
        auto reference = Frame::copy_from(
            output.width(),
            output.height(),
            output.pixel_format(),
            output.data(),
            output.timestamp());
        if (!reference) return reference.error();
        auto committed = decoder_state_.commit(
            std::move(reference.value()),
            std::move(decoded.value().latent_state),
            packet.frame_type());
        if (!committed) return committed.error();
        statistics_->record_decode(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started));
        return output;
    }

    [[nodiscard]] Result<void> flush_encoder() {
        auto flushed = backend_->flush();
        if (!flushed) return flushed.error();
        reset_encoder();
        return {};
    }

    [[nodiscard]] Result<void> flush_decoder() {
        auto flushed = backend_->flush();
        if (!flushed) return flushed.error();
        reset_decoder();
        return {};
    }

    void reset_encoder() noexcept {
        encoder_state_.reset();
        backend_->reset_encoder();
        logger_->log(LogLevel::debug, "encoder sequence state reset");
    }

    void reset_decoder() noexcept {
        decoder_state_.reset();
        backend_->reset_decoder();
        logger_->log(LogLevel::debug, "decoder sequence state reset");
    }

private:
    RuntimeConfiguration configuration_;
    std::unique_ptr<codec::CodecBackend> backend_;
    codec::SequenceState encoder_state_;
    codec::SequenceState decoder_state_;
    std::shared_ptr<Logger> logger_;
    std::shared_ptr<Statistics> statistics_;
    bool initialized_{false};
};

class EncoderSession final : public IEncoderSession {
public:
    explicit EncoderSession(std::shared_ptr<SessionCore> core) : core_(std::move(core)) {}

    [[nodiscard]] Result<void> send_frame(const Frame& frame) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "encoder is flushed", "dcvcrt");
        }
        auto packet = core_->encode(frame);
        if (!packet) return packet.error();
        outputs_.push_back(std::move(packet.value()));
        return {};
    }

    [[nodiscard]] Result<Packet> receive_access_unit() override {
        if (!outputs_.empty()) {
            Packet output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) {
            return Error(ErrorCode::end_of_stream, "encoder is drained", "dcvcrt");
        }
        return Error(ErrorCode::try_again, "no encoded output available", "dcvcrt");
    }

    [[nodiscard]] Result<void> flush() override {
        if (flushed_) return {};
        auto flushed = core_->flush_encoder();
        if (!flushed) return flushed.error();
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        outputs_.clear();
        core_->reset_encoder();
        flushed_ = false;
        return {};
    }

private:
    std::shared_ptr<SessionCore> core_;
    std::deque<Packet> outputs_;
    bool flushed_{false};
};

class DecoderSession final : public IDecoderSession {
public:
    explicit DecoderSession(std::shared_ptr<SessionCore> core) : core_(std::move(core)) {}

    [[nodiscard]] Result<void> send_access_unit(const Packet& packet) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "decoder is flushed", "dcvcrt");
        }
        auto frame = core_->decode(packet);
        if (!frame) return frame.error();
        outputs_.push_back(std::move(frame.value()));
        return {};
    }

    [[nodiscard]] Result<Frame> receive_frame() override {
        if (!outputs_.empty()) {
            Frame output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) {
            return Error(ErrorCode::end_of_stream, "decoder is drained", "dcvcrt");
        }
        return Error(ErrorCode::try_again, "no decoded output available", "dcvcrt");
    }

    [[nodiscard]] Result<void> flush() override {
        if (flushed_) return {};
        auto flushed = core_->flush_decoder();
        if (!flushed) return flushed.error();
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        outputs_.clear();
        core_->reset_decoder();
        flushed_ = false;
        return {};
    }

private:
    std::shared_ptr<SessionCore> core_;
    std::deque<Frame> outputs_;
    bool flushed_{false};
};

}  // namespace

Result<codec::Sessions> make_sessions(
    RuntimeConfiguration configuration,
    std::unique_ptr<codec::CodecBackend> backend) {
    if (!backend) {
        return Error(
            ErrorCode::dependency_unavailable,
            "DCVC-RT requires a codec backend",
            "dcvcrt");
    }
    try {
        auto statistics = std::make_shared<Statistics>();
        auto logger_factory = make_default_logger_factory(configuration.runtime.log_level);
        auto core = std::make_shared<SessionCore>(
            std::move(configuration),
            std::move(backend),
            logger_factory->create("dcvcrt"),
            statistics);
        auto initialized = core->initialize();
        if (!initialized) return initialized.error();
        codec::Sessions sessions;
        sessions.encoder = std::make_unique<EncoderSession>(core);
        sessions.decoder = std::make_unique<DecoderSession>(std::move(core));
        sessions.statistics = std::move(statistics);
        return sessions;
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate DCVC-RT sessions",
            "dcvcrt");
    } catch (const std::exception& exception) {
        return Error(ErrorCode::internal_error, exception.what(), "dcvcrt");
    }
}

}  // namespace nvcr::dcvcrt
