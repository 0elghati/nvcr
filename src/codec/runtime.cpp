#include "nvcr/codec/runtime.hpp"

#include "nvcr/bitstream/access_unit.hpp"

#include <chrono>
#include <exception>
#include <new>
#include <optional>
#include <utility>

namespace nvcr::codec {
namespace {

RuntimeConfiguration normalized_configuration(RuntimeConfiguration configuration) {
    configuration.reconcile_legacy_fields();
    return configuration;
}

}  // namespace


Runtime::Runtime(
    RuntimeConfiguration configuration,
    Components components,
    std::shared_ptr<LoggerFactory> logger_factory,
    std::shared_ptr<Statistics> statistics)
    : configuration_(normalized_configuration(std::move(configuration))),
      components_(std::move(components)),
      encoder_state_(configuration_.codec.gop_size),
      decoder_state_(configuration_.codec.gop_size),
      logger_(logger_factory->create("dcvcrt")),
      statistics_(std::move(statistics)) {}

Result<void> Runtime::initialize() {
    if (initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is already initialized", "dcvcrt");
    }
    if (!components_.codec) {
        return Error(
            ErrorCode::dependency_unavailable,
            "a codec backend is required",
            "codec");
    }
    auto initialized = components_.codec->initialize(configuration_);
    if (!initialized) return initialized.error();
    initialized_ = true;
    logger_->log(LogLevel::info, "codec runtime initialized");
    return {};
}

Result<Packet> Runtime::encode(const Frame& frame) {
    if (!initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is not initialized", "dcvcrt");
    }
    if (frame.size_bytes() == 0) {
        return Error(ErrorCode::invalid_argument, "cannot encode an empty frame", "dcvcrt");
    }
    const auto started = std::chrono::steady_clock::now();
    auto frame_type = encoder_state_.next_frame_type();
    if (!frame_type) return frame_type.error();

    auto encoded = components_.codec->encode(
        frame, frame_type.value(), encoder_state_.view());
    if (!encoded) return encoded.error();
    AccessUnit access_unit{
        configuration_.stream.bitstream_model_id,
        frame.width(),
        frame.height(),
        encoded.value().effective_qp,
        frame_type.value(),
        frame_type.value() == FrameType::intra,
        std::move(encoded.value().payload)};
    auto serialized = AccessUnitIO::serialize(access_unit, configuration_.runtime.max_packet_bytes);
    if (!serialized) return serialized.error();
    Packet packet(
        std::move(serialized.value()),
        frame.timestamp(),
        frame_type.value());
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

Result<Frame> Runtime::decode(const Packet& packet) {
    if (!initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is not initialized", "dcvcrt");
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
        auto parsed = AccessUnitIO::deserialize(packet.data(), configuration_.runtime.max_packet_bytes);
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
    auto decoded = components_.codec->decode(
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

Result<void> Runtime::flush() {
    if (!initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is not initialized", "dcvcrt");
    }
    auto result = components_.codec->flush();
    if (!result) return result.error();
    reset();
    return {};
}

void Runtime::reset() noexcept {
    encoder_state_.reset();
    decoder_state_.reset();
    components_.codec->reset();
    logger_->log(LogLevel::debug, "sequence state reset");
}


namespace {

struct LegacySessionState final {
    LegacySessionState(
        RuntimeConfiguration configuration,
        Components components,
        std::shared_ptr<LoggerFactory> logger_factory,
        std::shared_ptr<Statistics> statistics)
        : runtime(
              std::move(configuration),
              std::move(components),
              std::move(logger_factory),
              std::move(statistics)) {}

    Runtime runtime;
    std::optional<Packet> pending_packet;
    std::optional<Frame> pending_frame;
    bool encoder_flushed{false};
    bool decoder_flushed{false};
};

class LegacyEncoderSession final : public IEncoderSession {
public:
    explicit LegacyEncoderSession(std::shared_ptr<LegacySessionState> state)
        : state_(std::move(state)) {}

    Result<void> send_frame(const Frame& frame) override {
        auto packet = state_->runtime.encode(frame);
        if (!packet) return packet.error();
        state_->pending_packet = std::move(packet.value());
        state_->encoder_flushed = false;
        return {};
    }

    Result<Packet> receive_access_unit() override {
        if (state_->pending_packet) {
            auto packet = std::move(*state_->pending_packet);
            state_->pending_packet.reset();
            return packet;
        }
        if (state_->encoder_flushed) {
            return Error(ErrorCode::end_of_stream, "encoder has been flushed", "codec");
        }
        return Error(ErrorCode::try_again, "no encoded output available yet", "codec");
    }

    Result<void> flush() override {
        auto flushed = state_->runtime.flush();
        if (!flushed) return flushed.error();
        state_->pending_packet.reset();
        state_->pending_frame.reset();
        state_->encoder_flushed = true;
        state_->decoder_flushed = true;
        return {};
    }

    Result<void> reset() override {
        state_->runtime.reset();
        state_->pending_packet.reset();
        state_->pending_frame.reset();
        state_->encoder_flushed = false;
        state_->decoder_flushed = false;
        return {};
    }

private:
    std::shared_ptr<LegacySessionState> state_;
};

class LegacyDecoderSession final : public IDecoderSession {
public:
    explicit LegacyDecoderSession(std::shared_ptr<LegacySessionState> state)
        : state_(std::move(state)) {}

    Result<void> send_access_unit(const Packet& packet) override {
        auto frame = state_->runtime.decode(packet);
        if (!frame) return frame.error();
        state_->pending_frame = std::move(frame.value());
        state_->decoder_flushed = false;
        return {};
    }

    Result<Frame> receive_frame() override {
        if (state_->pending_frame) {
            auto frame = std::move(*state_->pending_frame);
            state_->pending_frame.reset();
            return frame;
        }
        if (state_->decoder_flushed) {
            return Error(ErrorCode::end_of_stream, "decoder has been flushed", "codec");
        }
        return Error(ErrorCode::try_again, "no decoded output available yet", "codec");
    }

    Result<void> flush() override {
        auto flushed = state_->runtime.flush();
        if (!flushed) return flushed.error();
        state_->pending_packet.reset();
        state_->pending_frame.reset();
        state_->encoder_flushed = true;
        state_->decoder_flushed = true;
        return {};
    }

    Result<void> reset() override {
        state_->runtime.reset();
        state_->pending_packet.reset();
        state_->pending_frame.reset();
        state_->encoder_flushed = false;
        state_->decoder_flushed = false;
        return {};
    }

private:
    std::shared_ptr<LegacySessionState> state_;
};

}  // namespace

Result<Sessions> make_legacy_sessions(
    RuntimeConfiguration configuration, Components components) {
    configuration.reconcile_legacy_fields();
    try {
        auto statistics = std::make_shared<Statistics>();
        auto logger_factory = make_default_logger_factory(configuration.runtime.log_level);
        auto state = std::make_shared<LegacySessionState>(
            std::move(configuration),
            std::move(components),
            std::move(logger_factory),
            statistics);
        auto initialized = state->runtime.initialize();
        if (!initialized) return initialized.error();
        return Sessions{
            std::make_unique<LegacyEncoderSession>(state),
            std::make_unique<LegacyDecoderSession>(std::move(state)),
            std::move(statistics),
        };
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate legacy codec sessions",
            "codec");
    } catch (const std::exception& exception) {
        return Error(ErrorCode::internal_error, exception.what(), "codec");
    }
}

}  // namespace nvcr::codec
