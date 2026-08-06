#include "test_codec.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/codec/backend.hpp"
#include "nvcr/runtime/registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::test_support {
namespace {

constexpr std::array<std::byte, 4> test_payload_magic{
    std::byte{'T'}, std::byte{'E'}, std::byte{'S'}, std::byte{'T'}};

void write_u32(std::vector<std::byte>& bytes, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        bytes.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

Result<std::uint32_t> read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4U) {
        return Error(ErrorCode::malformed_bitstream,
                     "test codec payload header is truncated", "test-codec");
    }
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(bytes[offset++])) << shift;
    }
    return value;
}

Result<std::vector<std::byte>> encode_payload(const Frame& frame) {
    std::vector<std::byte> payload;
    payload.reserve(test_payload_magic.size() + 8U + frame.size_bytes());
    payload.insert(payload.end(), test_payload_magic.begin(), test_payload_magic.end());
    write_u32(payload, frame.width());
    write_u32(payload, frame.height());
    payload.insert(payload.end(), frame.data().begin(), frame.data().end());
    return payload;
}

Result<Frame> decode_payload(std::span<const std::byte> payload, Timestamp timestamp) {
    if (payload.size() < test_payload_magic.size() + 8U ||
        !std::equal(test_payload_magic.begin(), test_payload_magic.end(), payload.begin())) {
        return Error(ErrorCode::malformed_bitstream,
                     "test codec payload magic is invalid", "test-codec");
    }
    auto width = read_u32(payload, 4U);
    auto height = read_u32(payload, 8U);
    if (!width) return width.error();
    if (!height) return height.error();
    const auto data = payload.subspan(12U);
    auto expected_size = frame_size_bytes(width.value(), height.value(), PixelFormat::yuv420p8);
    if (!expected_size) return expected_size.error();
    if (data.size() != expected_size.value()) {
        return Error(ErrorCode::malformed_bitstream,
                     "test codec payload dimensions do not match data", "test-codec");
    }
    return Frame::copy_from(width.value(), height.value(), PixelFormat::yuv420p8, data, timestamp);
}

class TestCodecBackend final : public codec::CodecBackend {
public:
    [[nodiscard]] Result<void> initialize(const RuntimeConfiguration&) override {
        initialized_ = true;
        return {};
    }

    [[nodiscard]] Result<codec::CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const codec::SequenceStateView&) override {
        static_cast<void>(frame_type);
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "test codec is not initialized", "test-codec");
        }
        auto payload = encode_payload(frame);
        if (!payload) return payload.error();
        auto reconstructed = Frame::copy_from(
            frame.width(), frame.height(), frame.pixel_format(), frame.data(), frame.timestamp());
        if (!reconstructed) return reconstructed.error();
        return codec::CodecEncodeResult{
            std::move(payload.value()), std::move(reconstructed.value()), {}, 0U};
    }

    [[nodiscard]] Result<codec::CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType,
        Timestamp timestamp,
        const codec::SequenceStateView&) override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "test codec is not initialized", "test-codec");
        }
        auto frame = decode_payload(payload, timestamp);
        if (!frame) return frame.error();
        return codec::CodecDecodeResult{std::move(frame.value()), {}};
    }

    [[nodiscard]] Result<void> flush() override { return {}; }
    void reset() noexcept override { initialized_ = false; }

private:
    bool initialized_{false};
};

class TestCodecAdapter final : public codec::ICodecAdapter {
public:
    [[nodiscard]] codec::CodecDescriptor descriptor() const override {
        return {"test-codec", "Deterministic Test Codec", 1};
    }

    [[nodiscard]] codec::CodecCapabilities capabilities() const override {
        return {
            .supports_intra = true,
            .supports_predicted = true,
            .supports_bidirectional = false,
            .supports_hierarchical = false,
            .supports_delayed_output = true,
            .min_qp = 0,
            .max_qp = 63,
        };
    }

    [[nodiscard]] codec::OptionSchema encoder_options() const override {
        return {{{"test_codec.delay", "uint", "frames delayed before output", "1", "0", "16", false}}};
    }

    [[nodiscard]] codec::OptionSchema decoder_options() const override {
        return {{{"test_codec.strict_payload", "bool", "reject malformed payloads", "true", {}, {}, false}}};
    }

    [[nodiscard]] Result<codec::Components>
    create_components(const RuntimeConfiguration&) override {
        codec::Components components;
        components.codec = std::make_unique<TestCodecBackend>();
        return components;
    }
};

Packet make_packet(const Frame& frame, std::uint64_t frame_index) {
    const auto frame_type = frame_index == 0U ? FrameType::intra : FrameType::predicted;
    AccessUnit access_unit{
        "test-codec",
        frame.width(),
        frame.height(),
        0,
        frame_type,
        frame_type == FrameType::intra,
        {}};
    auto payload = encode_payload(frame);
    if (!payload) return {};
    access_unit.payload = std::move(payload.value());
    auto wire = AccessUnitIO::serialize(access_unit);
    if (!wire) return {};
    return Packet(
        std::move(wire.value()),
        frame.timestamp(),
        frame_type,
        {{"codec_id", "test-codec"}, {"payload_syntax", "test-v1"}});
}

class TestEncoderSession final : public IEncoderSession {
public:
    explicit TestEncoderSession(std::size_t delay_frames) : delay_frames_(delay_frames) {}

    [[nodiscard]] Result<void> send_frame(const Frame& frame) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "test encoder is flushed", "test-codec");
        }
        if (frame.size_bytes() == 0U) {
            return Error(ErrorCode::invalid_argument, "test encoder received empty frame", "test-codec");
        }
        pending_.push_back(frame);
        emit_ready(false);
        return {};
    }

    [[nodiscard]] Result<Packet> receive_access_unit() override {
        if (!outputs_.empty()) {
            Packet output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) {
            return Error(ErrorCode::end_of_stream, "test encoder is drained", "test-codec");
        }
        return Error(ErrorCode::try_again, "test encoder is delaying output", "test-codec");
    }

    [[nodiscard]] Result<void> flush() override {
        if (!flushed_) {
            flushed_ = true;
            emit_ready(true);
        }
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        pending_.clear();
        outputs_.clear();
        frame_index_ = 0U;
        flushed_ = false;
        return {};
    }

private:
    void emit_ready(bool drain) {
        while (!pending_.empty() && (drain || pending_.size() > delay_frames_)) {
            outputs_.push_back(make_packet(pending_.front(), frame_index_++));
            pending_.pop_front();
        }
    }

    std::size_t delay_frames_;
    std::deque<Frame> pending_;
    std::deque<Packet> outputs_;
    std::uint64_t frame_index_{0};
    bool flushed_{false};
};

class TestDecoderSession final : public IDecoderSession {
public:
    [[nodiscard]] Result<void> send_access_unit(const Packet& packet) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "test decoder is flushed", "test-codec");
        }
        auto access_unit = AccessUnitIO::deserialize(packet.data());
        if (!access_unit) return access_unit.error();
        if (access_unit.value().model_id != "test-codec") {
            return Error(ErrorCode::malformed_bitstream, "unexpected test codec model id", "test-codec");
        }
        auto frame = decode_payload(access_unit.value().payload, packet.timestamp());
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
            return Error(ErrorCode::end_of_stream, "test decoder is drained", "test-codec");
        }
        return Error(ErrorCode::try_again, "test decoder has no output", "test-codec");
    }

    [[nodiscard]] Result<void> flush() override {
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        outputs_.clear();
        flushed_ = false;
        return {};
    }

private:
    std::deque<Frame> outputs_;
    bool flushed_{false};
};

}  // namespace

void register_test_codec() {
    auto adapter = make_test_codec_adapter();
    runtime::Registry::instance().register_codec({
        adapter->descriptor(),
        adapter->capabilities(),
        adapter->encoder_options(),
        adapter->decoder_options(),
        []() -> std::unique_ptr<codec::ICodecAdapter> {
            return make_test_codec_adapter();
        },
    });
}

std::unique_ptr<codec::ICodecAdapter> make_test_codec_adapter() {
    return std::make_unique<TestCodecAdapter>();
}

std::unique_ptr<IEncoderSession> make_test_encoder_session(std::size_t delay_frames) {
    return std::make_unique<TestEncoderSession>(delay_frames);
}

std::unique_ptr<IDecoderSession> make_test_decoder_session() {
    return std::make_unique<TestDecoderSession>();
}

}  // namespace nvcr::test_support
