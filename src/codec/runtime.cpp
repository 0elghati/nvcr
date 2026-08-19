#include "nvcr/codec/runtime.hpp"

#include "nvcr/bitstream/access_unit.hpp"

#include <chrono>
#include <optional>
#include <string>
#include <utility>

namespace nvcr::codec {
namespace {

std::string codec_subsystem(const RuntimeConfiguration& configuration) {
    return configuration.codec_id.empty() ? "codec" : configuration.codec_id;
}

}  // namespace


Runtime::Runtime(
    RuntimeConfiguration configuration,
    Components components,
    std::shared_ptr<LoggerFactory> logger_factory,
    std::shared_ptr<Statistics> statistics)
    : configuration_(std::move(configuration)),
      components_(std::move(components)),
      encoder_state_(configuration_.gop_size),
      decoder_state_(configuration_.gop_size),
      logger_(logger_factory->create(codec_subsystem(configuration_))),
      statistics_(std::move(statistics)) {}

Result<void> Runtime::initialize() {
    if (initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is already initialized", codec_subsystem(configuration_));
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
        return Error(ErrorCode::invalid_state, "runtime is not initialized", codec_subsystem(configuration_));
    }
    if (frame.size_bytes() == 0) {
        return Error(ErrorCode::invalid_argument, "cannot encode an empty frame", codec_subsystem(configuration_));
    }
    const auto started = std::chrono::steady_clock::now();
    auto frame_type = encoder_state_.next_frame_type();
    if (!frame_type) return frame_type.error();

    auto encoded = components_.codec->encode(
        frame, frame_type.value(), encoder_state_.view());
    if (!encoded) return encoded.error();
    AccessUnit access_unit{
        configuration_.bitstream_model_id,
        frame.width(),
        frame.height(),
        encoded.value().effective_qp,
        frame_type.value(),
        frame_type.value() == FrameType::intra,
        std::move(encoded.value().payload)};
    auto serialized = AccessUnitIO::serialize(access_unit, configuration_.max_packet_bytes);
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
        return Error(ErrorCode::invalid_state, "runtime is not initialized", codec_subsystem(configuration_));
    }
    if (packet.size() > configuration_.max_packet_bytes) {
        return Error(
            ErrorCode::resource_exhausted,
            "packet exceeds configured limit",
            codec_subsystem(configuration_));
    }
    std::optional<AccessUnit> access_unit;
    std::span<const std::byte> codec_payload = packet.data();
    if (AccessUnitIO::has_magic(packet.data())) {
        auto parsed = AccessUnitIO::deserialize(packet.data(), configuration_.max_packet_bytes);
        if (!parsed) return parsed.error();
        if (parsed.value().model_id != configuration_.bitstream_model_id) {
            return Error(
                ErrorCode::malformed_bitstream,
                "access-unit model identity does not match the configured bundle",
                codec_subsystem(configuration_));
        }
        if (parsed.value().frame_type != packet.frame_type()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "packet and access-unit frame types disagree",
                codec_subsystem(configuration_));
        }
        access_unit = std::move(parsed.value());
        if (access_unit->reset_state) {
            decoder_state_.reset();
        }
        codec_payload = access_unit->payload;
    } else if (!configuration_.allow_legacy_access_units) {
        return Error(
            ErrorCode::malformed_bitstream,
            "legacy codec payload rejected by configuration",
            codec_subsystem(configuration_));
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
            codec_subsystem(configuration_));
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
        return Error(ErrorCode::invalid_state, "runtime is not initialized", codec_subsystem(configuration_));
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

}  // namespace nvcr::codec
