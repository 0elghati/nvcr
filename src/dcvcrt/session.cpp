#include "session.hpp"

#include "numerics.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/bitstream/numerics_identity.hpp"
#include "nvcr/codec/sequence_state.hpp"
#include "nvcr/logging/logger.hpp"

#include <chrono>
#include <deque>
#include <exception>
#include <new>
#include <optional>
#include <string>
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
        // The normative stages and their program digests are the codec's
        // declaration; the backend loads them on its provider, which refuses
        // a profile it does not implement.
        auto normative = normative_stages(configuration_);
        if (!normative) return normative.error();
        if (normative.value()) {
            identity_ = normative.value()->identity;
            auto section = NumericsIdentityIO::serialize(*identity_);
            if (!section) return section.error();
            identity_section_ = std::move(section.value());
        }
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
        // Exact streams state their numerics on random-access units; the
        // other units of the sequence inherit it and keep NVAU v1 framing.
        auto serialized = identity_ && frame_type.value() == FrameType::intra
            ? exact_access_unit(std::move(access_unit))
            : AccessUnitIO::serialize(access_unit, configuration_.runtime.max_packet_bytes);
        if (!serialized) return serialized.error();
        ++encoded_frames_;
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
        } else if (identity_) {
            return Error(ErrorCode::incompatible_target,
                         "this decoder runs exact numerics; the stream carries no numerics identity",
                         "dcvcrt");
        } else if (!configuration_.stream.allow_legacy_access_units) {
            return Error(
                ErrorCode::malformed_bitstream,
                "legacy codec payload rejected by configuration",
                "dcvcrt");
        }
        // The numerics requirement is resolved before any decode work.
        if (access_unit) {
            auto compatible = check_numerics(*access_unit);
            if (!compatible) return compatible.error();
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
        exact_sequence_ = false;
        backend_->reset_decoder();
        logger_->log(LogLevel::debug, "decoder sequence state reset");
    }

private:
    // NVAU v2 random-access unit: the codec payload and the numerics
    // identity, both required.
    [[nodiscard]] Result<std::vector<std::byte>> exact_access_unit(AccessUnit access_unit) const {
        access_unit.codec_id = configuration_.codec.id;
        access_unit.codec_profile_id = std::string(payload_profile);
        access_unit.decode_order_index = encoded_frames_;
        access_unit.presentation_order_index = encoded_frames_;
        access_unit.random_access = access_unit.frame_type == FrameType::intra;
        access_unit.sections = {
            AccessUnitSection{
                static_cast<std::uint16_t>(AccessUnitSectionType::codec_payload), 1,
                AccessUnitIO::required_section_flag, access_unit.payload},
            AccessUnitSection{
                static_cast<std::uint16_t>(AccessUnitSectionType::numerics_identity),
                NumericsIdentityIO::format_version, AccessUnitIO::required_section_flag,
                identity_section_}};
        return AccessUnitIO::serialize_sectioned(access_unit, configuration_.runtime.max_packet_bytes);
    }

    // A stream with exact numerics decodes only on a decoder that runs the
    // same profile with the same normative programs; a native stream only on
    // a native decoder. Everything else fails here, before decoding. The
    // identity is stated on random-access units; a unit without one belongs
    // to the sequence the last random-access unit opened.
    [[nodiscard]] Result<void> check_numerics(const AccessUnit& access_unit) {
        const AccessUnitSection* section = nullptr;
        for (const auto& candidate : access_unit.sections) {
            if (candidate.type != static_cast<std::uint16_t>(AccessUnitSectionType::numerics_identity)) {
                continue;
            }
            if (section != nullptr) {
                return Error(ErrorCode::malformed_bitstream,
                             "access unit has more than one numerics identity", "dcvcrt");
            }
            section = &candidate;
        }
        if (section == nullptr) {
            if (access_unit.frame_type == FrameType::intra) exact_sequence_ = false;
            if (!identity_ || (access_unit.frame_type != FrameType::intra && exact_sequence_)) return {};
            return Error(ErrorCode::incompatible_target,
                         "this decoder runs exact numerics; the stream uses native numerics", "dcvcrt");
        }
        if (access_unit.codec_profile_id != payload_profile) {
            return Error(ErrorCode::incompatible_version,
                         "unsupported DCVC-RT codec profile '" + access_unit.codec_profile_id + "'",
                         "dcvcrt");
        }
        if (access_unit.frame_type != FrameType::intra) {
            return Error(ErrorCode::malformed_bitstream,
                         "numerics identity on a unit that is not a random access point", "dcvcrt");
        }
        if (section->version != NumericsIdentityIO::format_version) {
            return Error(ErrorCode::incompatible_version,
                         "unsupported numerics identity version " + std::to_string(section->version),
                         "dcvcrt");
        }
        if ((section->flags & AccessUnitIO::required_section_flag) == 0U) {
            return Error(ErrorCode::malformed_bitstream, "numerics identity must be required", "dcvcrt");
        }
        auto identity = NumericsIdentityIO::deserialize(section->payload);
        if (!identity) return identity.error();
        if (!identity_) {
            return Error(ErrorCode::incompatible_target,
                         "stream requires " + identity.value().profile +
                             " normative stages; decode with that numerics profile and its programs",
                         "dcvcrt");
        }
        if (identity.value() != *identity_) {
            return Error(ErrorCode::incompatible_target,
                         "stream's numerics profile or normative programs differ from this decoder's",
                         "dcvcrt");
        }
        exact_sequence_ = true;
        return {};
    }

    RuntimeConfiguration configuration_;
    std::unique_ptr<codec::CodecBackend> backend_;
    std::optional<NumericsIdentity> identity_;
    std::vector<std::byte> identity_section_;
    // The last random-access unit decoded opened an exact-numerics sequence.
    bool exact_sequence_{false};
    std::uint64_t encoded_frames_{0};
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
