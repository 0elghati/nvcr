#include "nvcr/dcvcrt/runtime.hpp"

#include <chrono>
#include <utility>

namespace nvcr::dcvcrt {

Runtime::Runtime(
    RuntimeConfiguration configuration,
    Components components,
    std::shared_ptr<LoggerFactory> logger_factory,
    std::shared_ptr<Statistics> statistics)
    : configuration_(std::move(configuration)),
      components_(std::move(components)),
      encoder_state_(configuration_.gop_size),
      decoder_state_(configuration_.gop_size),
      logger_(logger_factory->create("dcvcrt")),
      statistics_(std::move(statistics)) {}

Result<void> Runtime::initialize() {
    if (initialized_) {
        return Error(ErrorCode::invalid_state, "runtime is already initialized", "dcvcrt");
    }
    if (!components_.codec) {
        return Error(
            ErrorCode::dependency_unavailable,
            "a DCVC-RT codec backend is required",
            "dcvcrt");
    }
    auto initialized = components_.codec->initialize(configuration_);
    if (!initialized) return initialized.error();
    initialized_ = true;
    logger_->log(LogLevel::info, "DCVC-RT runtime initialized");
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
    if (encoded.value().payload.size() > configuration_.max_packet_bytes) {
        return Error(
            ErrorCode::resource_exhausted,
            "encoded packet exceeds configured limit",
            "dcvcrt");
    }

    PacketMetadata metadata{
        {"height", std::to_string(frame.height())},
        {"width", std::to_string(frame.width())},
    };
    Packet packet(
        std::move(encoded.value().payload),
        frame.timestamp(),
        frame_type.value(),
        std::move(metadata));
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
    auto valid = decoder_state_.validate_packet(packet.frame_type());
    if (!valid) return valid.error();
    const auto started = std::chrono::steady_clock::now();
    auto decoded = components_.codec->decode(
        packet.data(), packet.frame_type(), packet.timestamp(), decoder_state_.view());
    if (!decoded) return decoded.error();

    Frame output = std::move(decoded.value().frame);
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

}  // namespace nvcr::dcvcrt
