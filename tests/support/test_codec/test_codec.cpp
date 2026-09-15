#include "test_codec.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/runtime/registry.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::test_support {
namespace {

constexpr std::array<std::byte, 4> test_payload_magic{
    std::byte{'T'}, std::byte{'E'}, std::byte{'S'}, std::byte{'T'}};
constexpr std::array<std::byte, 4> chunk_payload_magic{
    std::byte{'C'}, std::byte{'H'}, std::byte{'N'}, std::byte{'K'}};
constexpr std::size_t test_group_size = 8U;
constexpr std::string_view chunk_timestamps_key = "test_codec.frame_timestamps_us";

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
        return {{{"test_codec.group_size", "uint", "frames grouped per access unit", "8", "1", "8", false}}};
    }

    [[nodiscard]] codec::OptionSchema decoder_options() const override {
        return {{{"test_codec.strict_payload", "bool", "reject malformed payloads", "true", {}, {}, false}}};
    }

    [[nodiscard]] Result<codec::Sessions>
    create_sessions(
        const RuntimeConfiguration&,
        std::shared_ptr<provider::experimental::IProviderSession>
            provider_session) override {
        if (!provider_session) {
            return Error(
                ErrorCode::dependency_unavailable,
                "test codec requires a provider session",
                "test-codec");
        }
        codec::Sessions sessions;
        sessions.encoder = make_grouped_test_encoder_session();
        sessions.decoder = make_grouped_test_decoder_session();
        return sessions;
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

Result<Packet> make_chunk_packet(
    const std::deque<Frame>& frames,
    std::size_t frame_count,
    std::uint64_t chunk_index) {
    if (frame_count == 0U || frame_count > test_group_size || frames.size() < frame_count) {
        return Error(ErrorCode::invalid_argument, "invalid test chunk size", "test-codec");
    }
    const auto& first = frames.front();
    std::vector<std::byte> payload;
    payload.reserve(16U + frame_count * first.size_bytes());
    payload.insert(payload.end(), chunk_payload_magic.begin(), chunk_payload_magic.end());
    write_u32(payload, first.width());
    write_u32(payload, first.height());
    write_u32(payload, static_cast<std::uint32_t>(frame_count));
    for (std::size_t index = 0; index < frame_count; ++index) {
        const auto& frame = frames[index];
        if (frame.width() != first.width() || frame.height() != first.height() ||
            frame.pixel_format() != first.pixel_format() ||
            frame.size_bytes() != first.size_bytes()) {
            return Error(
                ErrorCode::invalid_argument,
                "test chunk frames must share one format",
                "test-codec");
        }
        payload.insert(payload.end(), frame.data().begin(), frame.data().end());
    }

    const auto frame_type = chunk_index == 0U ? FrameType::intra : FrameType::predicted;
    AccessUnit access_unit{
        "test-codec",
        first.width(),
        first.height(),
        0U,
        frame_type,
        frame_type == FrameType::intra,
        std::move(payload)};
    auto wire = AccessUnitIO::serialize(access_unit);
    if (!wire) return wire.error();

    std::string timestamps;
    for (std::size_t index = 0; index < frame_count; ++index) {
        if (!timestamps.empty()) timestamps.push_back(',');
        timestamps += std::to_string(frames[index].timestamp().count());
    }
    return Packet(
        std::move(wire.value()),
        first.timestamp(),
        frame_type,
        {{"codec_id", "test-codec"},
         {"payload_syntax", "test-chunk-v1"},
         {std::string(chunk_timestamps_key), std::move(timestamps)}});
}

Result<std::vector<Timestamp>> decode_chunk_timestamps(
    const Packet& packet, std::uint32_t frame_count) {
    const auto entry = packet.metadata().find(chunk_timestamps_key);
    if (entry == packet.metadata().end() || entry->second.empty() || entry->second.back() == ',') {
        return Error(
            ErrorCode::malformed_bitstream,
            "grouped test codec timestamps are missing or malformed",
            "test-codec");
    }

    std::vector<Timestamp> timestamps;
    timestamps.reserve(frame_count);
    std::string_view remaining = entry->second;
    while (!remaining.empty()) {
        const auto delimiter = remaining.find(',');
        const auto token = remaining.substr(0U, delimiter);
        std::int64_t value = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != token.data() + token.size()) {
            return Error(
                ErrorCode::malformed_bitstream,
                "grouped test codec timestamp is invalid",
                "test-codec");
        }
        timestamps.emplace_back(value);
        if (delimiter == std::string_view::npos) break;
        remaining.remove_prefix(delimiter + 1U);
    }
    if (timestamps.size() != frame_count || timestamps.front() != packet.timestamp()) {
        return Error(
            ErrorCode::malformed_bitstream,
            "grouped test codec timestamp count or primary timestamp is invalid",
            "test-codec");
    }
    return timestamps;
}

Result<std::vector<Frame>> decode_chunk_packet(const Packet& packet) {
    auto access_unit = AccessUnitIO::deserialize(packet.data());
    if (!access_unit) return access_unit.error();
    if (access_unit.value().model_id != "test-codec") {
        return Error(
            ErrorCode::malformed_bitstream,
            "unexpected grouped test codec model id",
            "test-codec");
    }
    const auto payload = std::span<const std::byte>(access_unit.value().payload);
    if (payload.size() < 16U ||
        !std::equal(chunk_payload_magic.begin(), chunk_payload_magic.end(), payload.begin())) {
        return Error(
            ErrorCode::malformed_bitstream,
            "grouped test codec payload header is invalid",
            "test-codec");
    }
    auto width = read_u32(payload, 4U);
    auto height = read_u32(payload, 8U);
    auto frame_count = read_u32(payload, 12U);
    if (!width) return width.error();
    if (!height) return height.error();
    if (!frame_count) return frame_count.error();
    if (frame_count.value() == 0U || frame_count.value() > test_group_size) {
        return Error(
            ErrorCode::malformed_bitstream,
            "grouped test codec frame count is invalid",
            "test-codec");
    }
    auto timestamps = decode_chunk_timestamps(packet, frame_count.value());
    if (!timestamps) return timestamps.error();
    auto frame_bytes = frame_size_bytes(
        width.value(), height.value(), PixelFormat::yuv420p8);
    if (!frame_bytes) return frame_bytes.error();
    if (payload.size() - 16U != frame_bytes.value() * frame_count.value()) {
        return Error(
            ErrorCode::malformed_bitstream,
            "grouped test codec payload size is invalid",
            "test-codec");
    }

    std::vector<Frame> frames;
    frames.reserve(frame_count.value());
    for (std::uint32_t index = 0; index < frame_count.value(); ++index) {
        auto frame = Frame::copy_from(
            width.value(),
            height.value(),
            PixelFormat::yuv420p8,
            payload.subspan(16U + index * frame_bytes.value(), frame_bytes.value()),
            timestamps.value()[index]);
        if (!frame) return frame.error();
        frames.push_back(std::move(frame.value()));
    }
    return frames;
}

class GroupedTestEncoderSession final : public IEncoderSession {
public:
    [[nodiscard]] Result<void> send_frame(const Frame& frame) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "grouped encoder is flushed", "test-codec");
        }
        if (frame.size_bytes() == 0U) {
            return Error(ErrorCode::invalid_argument, "grouped encoder received empty frame", "test-codec");
        }
        pending_.push_back(frame);
        auto emitted = emit_ready(false);
        if (!emitted) {
            pending_.pop_back();
            return emitted.error();
        }
        return {};
    }

    [[nodiscard]] Result<Packet> receive_access_unit() override {
        if (!outputs_.empty()) {
            Packet output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) {
            return Error(ErrorCode::end_of_stream, "grouped encoder is drained", "test-codec");
        }
        return Error(ErrorCode::try_again, "grouped encoder needs more input", "test-codec");
    }

    [[nodiscard]] Result<void> flush() override {
        if (flushed_) return {};
        auto emitted = emit_ready(true);
        if (!emitted) return emitted.error();
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        pending_.clear();
        outputs_.clear();
        chunk_index_ = 0U;
        flushed_ = false;
        return {};
    }

private:
    [[nodiscard]] Result<void> emit_ready(bool drain) {
        while (pending_.size() >= test_group_size || (drain && !pending_.empty())) {
            const auto count = std::min(test_group_size, pending_.size());
            auto packet = make_chunk_packet(pending_, count, chunk_index_);
            if (!packet) return packet.error();
            outputs_.push_back(std::move(packet.value()));
            for (std::size_t index = 0; index < count; ++index) pending_.pop_front();
            ++chunk_index_;
        }
        return {};
    }

    std::deque<Frame> pending_;
    std::deque<Packet> outputs_;
    std::uint64_t chunk_index_{0U};
    bool flushed_{false};
};

class GroupedTestDecoderSession final : public IDecoderSession {
public:
    [[nodiscard]] Result<void> send_access_unit(const Packet& packet) override {
        if (flushed_) {
            return Error(ErrorCode::invalid_state, "grouped decoder is flushed", "test-codec");
        }
        auto frames = decode_chunk_packet(packet);
        if (!frames) return frames.error();
        for (auto& frame : frames.value()) outputs_.push_back(std::move(frame));
        return {};
    }

    [[nodiscard]] Result<Frame> receive_frame() override {
        if (!outputs_.empty()) {
            Frame output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) {
            return Error(ErrorCode::end_of_stream, "grouped decoder is drained", "test-codec");
        }
        return Error(ErrorCode::try_again, "grouped decoder has no output", "test-codec");
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

std::unique_ptr<IEncoderSession> make_grouped_test_encoder_session() {
    return std::make_unique<GroupedTestEncoderSession>();
}

std::unique_ptr<IDecoderSession> make_grouped_test_decoder_session() {
    return std::make_unique<GroupedTestDecoderSession>();
}

}  // namespace nvcr::test_support
