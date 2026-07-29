#include <nvcr/dcvcrt/sequence_state.hpp>
#include <nvcr/experimental/fast_backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void frame_test() {
    auto frame = nvcr::Frame::create(16, 8, nvcr::PixelFormat::yuv420p8);
    expect(frame.has_value(), "valid YUV frame is created");
    expect(frame && frame.value().size_bytes() == 192, "YUV420 frame size is correct");
    expect(
        !nvcr::Frame::create(15, 8, nvcr::PixelFormat::yuv420p8),
        "odd YUV420 dimensions are rejected");
}

void packet_test() {
    nvcr::Packet original(
        {std::byte{1}, std::byte{2}, std::byte{3}},
        nvcr::Timestamp{-20},
        nvcr::FrameType::predicted,
        {{"answer", "42"}});
    auto wire = nvcr::PacketIO::serialize(original);
    expect(wire.has_value(), "packet serializes");
    if (!wire) return;
    auto decoded = nvcr::PacketIO::deserialize(wire.value());
    expect(!nvcr::PacketIO::deserialize(wire.value(), wire.value().size() - 1U),
           "packet total-size limit is enforced");
    expect(decoded.has_value(), "packet deserializes");
    if (!decoded) return;
    expect(decoded.value().timestamp() == original.timestamp(), "timestamp round-trips");
    expect(decoded.value().frame_type() == original.frame_type(), "frame type round-trips");
    expect(decoded.value().metadata() == original.metadata(), "metadata round-trips");
    expect(
        std::equal(
            decoded.value().data().begin(),
            decoded.value().data().end(),
            original.data().begin()),
        "payload round-trips");

    auto excessive_count = wire.value();
    excessive_count[16] = std::byte{0xff};
    excessive_count[17] = std::byte{0xff};
    expect(!nvcr::PacketIO::deserialize(excessive_count),
           "malicious metadata entry count is rejected before allocation");

    nvcr::PacketMetadata excessive_metadata;
    for (int index = 0; index < 17; ++index) {
        excessive_metadata.emplace(std::to_string(index), std::string(65'535, 'x'));
    }
    nvcr::Packet excessive_packet(
        {std::byte{1}}, nvcr::Timestamp{}, nvcr::FrameType::intra,
        std::move(excessive_metadata));
    expect(!nvcr::PacketIO::serialize(excessive_packet),
           "aggregate metadata limit is enforced");

    nvcr::ByteWriter malformed_metadata(1U << 21U);
    for (const auto byte : {'N', 'V', 'C', 'R'}) {
        malformed_metadata.write_u8(static_cast<std::uint8_t>(byte));
    }
    malformed_metadata.write_u16(nvcr::PacketIO::format_version);
    malformed_metadata.write_u8(static_cast<std::uint8_t>(nvcr::FrameType::intra));
    malformed_metadata.write_u8(0);
    malformed_metadata.write_u64(0);
    malformed_metadata.write_u16(17);
    const std::string oversized_key(65'535, 'k');
    for (int index = 0; index < 17; ++index) {
        malformed_metadata.write_u16(static_cast<std::uint16_t>(oversized_key.size()));
        malformed_metadata.write_u16(0);
        malformed_metadata.write_string(oversized_key);
    }
    expect(!nvcr::PacketIO::deserialize(malformed_metadata.take()),
           "parser aggregate metadata bound is enforced before payload allocation");
}

void access_unit_test() {
    nvcr::AccessUnit original{
        "dcvcrt-cvpr2025", 176, 144, 32, nvcr::FrameType::intra, true,
        {std::byte{1}, std::byte{2}, std::byte{3}}};
    auto wire = nvcr::AccessUnitIO::serialize(original);
    expect(wire.has_value(), "access unit serializes");
    if (!wire) return;
    auto decoded = nvcr::AccessUnitIO::deserialize(wire.value());
    expect(decoded.has_value(), "access unit deserializes");
    expect(decoded && decoded.value().model_id == original.model_id,
           "access-unit model identity round-trips");
    expect(decoded && decoded.value().width == 176 && decoded.value().height == 144,
           "access-unit dimensions round-trip");
    expect(decoded && decoded.value().reset_state, "access-unit reset flag round-trips");

    auto truncated = wire.value();
    truncated.pop_back();
    expect(!nvcr::AccessUnitIO::deserialize(truncated),
           "truncated access unit is rejected");
    expect(!nvcr::AccessUnitIO::deserialize(wire.value(), wire.value().size() - 1U),
           "oversized access unit is rejected before parsing");
    auto unsupported_flags = wire.value();
    unsupported_flags[7] = std::byte{0x80};
    expect(!nvcr::AccessUnitIO::deserialize(unsupported_flags),
           "unsupported access-unit flags are rejected");
    auto invalid_model = wire.value();
    invalid_model[32] = std::byte{'/'};
    expect(!nvcr::AccessUnitIO::deserialize(invalid_model),
           "non-portable model identity is rejected");

    auto invalid_dimensions = wire.value();
    invalid_dimensions[12] = std::byte{0xff};
    invalid_dimensions[13] = std::byte{0xff};
    invalid_dimensions[14] = std::byte{0xff};
    invalid_dimensions[15] = std::byte{0xff};
    expect(!nvcr::AccessUnitIO::deserialize(invalid_dimensions),
           "oversized access-unit dimensions are rejected");

    auto oversized_payload = wire.value();
    for (std::size_t index = 24; index < 32; ++index) {
        oversized_payload[index] = std::byte{0xff};
    }
    expect(!nvcr::AccessUnitIO::deserialize(oversized_payload),
           "oversized access-unit payload is rejected");

    original.reset_state = false;
    expect(!nvcr::AccessUnitIO::serialize(original),
           "intra access units must carry reset state");
    original.reset_state = true;

    original.frame_type = nvcr::FrameType::predicted;
    expect(!nvcr::AccessUnitIO::serialize(original),
           "predicted reset access unit is rejected");
    original.reset_state = false;
    original.qp = 71;
    expect(nvcr::AccessUnitIO::serialize(original).has_value(),
           "maximum effective P-frame QP is accepted");
    original.qp = 72;
    expect(!nvcr::AccessUnitIO::serialize(original),
           "out-of-range effective P-frame QP is rejected");
    original.frame_type = nvcr::FrameType::intra;
    original.reset_state = true;
    original.qp = 64;
    expect(!nvcr::AccessUnitIO::serialize(original),
           "out-of-range I-frame QP is rejected");
    original.qp = 32;
    original.width = 175;
    expect(!nvcr::AccessUnitIO::serialize(original),
           "odd YUV420 access-unit dimensions are rejected");
}

void memory_test() {
    nvcr::MemoryPool pool(nvcr::make_host_memory_resource(), 1024);
    void* first_address = nullptr;
    {
        auto lease = pool.acquire(128);
        expect(lease.has_value(), "memory lease succeeds");
        if (lease) {
            first_address = lease.value().bytes().data();
        }
    }
    auto reused = pool.acquire(64);
    expect(reused.has_value(), "released memory can be leased");
    expect(reused && reused.value().bytes().data() == first_address, "pool reuses best-fit block");
    expect(!pool.acquire(1024), "pool capacity is enforced");
}

void state_test() {
    nvcr::codec::SequenceState state(2);
    auto first_type = state.next_frame_type();
    expect(first_type && first_type.value() == nvcr::FrameType::intra, "sequence starts intra");
    expect(!state.validate_packet(nvcr::FrameType::predicted), "P frame needs a reference");

    auto reference = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
    expect(reference.has_value(), "reference frame created");
    if (!reference) return;
    expect(
        state.commit(std::move(reference.value()), {}, nvcr::FrameType::intra).has_value(),
        "intra state commits");
    auto second_type = state.next_frame_type();
    expect(
        second_type && second_type.value() == nvcr::FrameType::predicted,
        "second frame is predicted");
    state.reset();
    expect(!state.has_reference() && state.frame_index() == 0, "reset clears sequence state");

    nvcr::codec::SequenceState long_gop_state(32);
    auto long_reference = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
    expect(long_reference.has_value(), "long GOP reference frame created");
    if (!long_reference) return;
    expect(
        long_gop_state.commit(std::move(long_reference.value()), {}, nvcr::FrameType::intra)
            .has_value(),
        "long GOP intra state commits");
    for (int index = 0; index < 31; ++index) {
        auto predicted = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
        expect(predicted.has_value(), "long GOP predicted frame created");
        if (!predicted) return;
        expect(
            long_gop_state.commit(
                std::move(predicted.value()), {}, nvcr::FrameType::predicted)
                .has_value(),
            "long GOP predicted state commits");
    }
    expect(
        long_gop_state.validate_packet(nvcr::FrameType::predicted).has_value(),
        "predicted frames remain valid after 32 committed frames");
}

void experimental_fast_backend_test() {
    auto encoder = nvcr::experimental::make_fast_backend();
    auto decoder = nvcr::experimental::make_fast_backend();
    expect(encoder.has_value() && decoder.has_value(),
           "experimental fast backends are created");
    if (!encoder || !decoder) return;

    nvcr::RuntimeConfiguration configuration;
    configuration.intra_qp = 32;
    expect(encoder.value()->initialize(configuration).has_value() &&
               decoder.value()->initialize(configuration).has_value(),
           "experimental fast backends initialize");

    auto frame = nvcr::Frame::create(64, 32, nvcr::PixelFormat::yuv420p8);
    expect(frame.has_value(), "experimental fast source frame is created");
    if (!frame) return;
    for (std::size_t index = 0; index < frame.value().data().size(); ++index) {
        const auto x = index % 64U;
        const auto y = index / 64U;
        frame.value().data()[index] =
            static_cast<std::byte>((x * 3U + y * 2U) & 0xffU);
    }

    nvcr::codec::SequenceState encoder_state(32);
    nvcr::codec::SequenceState decoder_state(32);
    auto encoded = encoder.value()->encode(
        frame.value(), nvcr::FrameType::intra, encoder_state.view());
    expect(encoded.has_value(), "experimental fast intra frame encodes");
    if (!encoded) return;
    expect(encoded.value().payload.size() < frame.value().size_bytes(),
           "experimental fast intra payload is smaller than raw YUV");
    auto decoded = decoder.value()->decode(
        encoded.value().payload, nvcr::FrameType::intra,
        nvcr::Timestamp{}, decoder_state.view());
    expect(decoded.has_value(), "experimental fast intra frame decodes");
    if (!decoded) return;
    expect(decoded.value().frame.pixel_format() == nvcr::PixelFormat::yuv420p8,
           "experimental fast backend returns YUV420P8");
    expect(std::equal(
               decoded.value().frame.data().begin(),
               decoded.value().frame.data().end(),
               encoded.value().reconstructed_frame.data().begin()),
           "experimental fast encoder and decoder intra reconstruction agrees");

    std::uint8_t maximum_error = 0;
    for (std::size_t index = 0; index < frame.value().data().size(); ++index) {
        const auto source = std::to_integer<int>(frame.value().data()[index]);
        const auto reconstruction =
            std::to_integer<int>(decoded.value().frame.data()[index]);
        maximum_error = static_cast<std::uint8_t>(
            std::max<int>(maximum_error, std::abs(source - reconstruction)));
    }
    expect(maximum_error <= 4U,
           "experimental fast QP 32 reconstruction error is bounded");

    auto encoder_reference = nvcr::Frame::copy_from(
        64, 32, nvcr::PixelFormat::yuv420p8,
        encoded.value().reconstructed_frame.data());
    auto decoder_reference = nvcr::Frame::copy_from(
        64, 32, nvcr::PixelFormat::yuv420p8, decoded.value().frame.data());
    expect(encoder_reference.has_value() && decoder_reference.has_value(),
           "experimental fast references are copied");
    if (!encoder_reference || !decoder_reference) return;
    expect(encoder_state.commit(
               std::move(encoder_reference.value()), {}, nvcr::FrameType::intra)
               .has_value() &&
               decoder_state.commit(
                   std::move(decoder_reference.value()), {}, nvcr::FrameType::intra)
                   .has_value(),
           "experimental fast intra references commit");

    auto predicted = nvcr::Frame::create(64, 32, nvcr::PixelFormat::yuv420p8);
    expect(predicted.has_value(), "experimental fast predicted source is created");
    if (!predicted) return;
    for (std::size_t index = 0; index < predicted.value().data().size(); ++index) {
        const auto shifted = (index + 3U) % frame.value().data().size();
        predicted.value().data()[index] = frame.value().data()[shifted];
    }
    auto encoded_predicted = encoder.value()->encode(
        predicted.value(), nvcr::FrameType::predicted, encoder_state.view());
    expect(encoded_predicted.has_value(), "experimental fast predicted frame encodes");
    if (!encoded_predicted) return;
    expect(encoded_predicted.value().payload.size() < predicted.value().size_bytes(),
           "experimental fast predicted payload is smaller than raw YUV");
    auto decoded_predicted = decoder.value()->decode(
        encoded_predicted.value().payload, nvcr::FrameType::predicted,
        nvcr::Timestamp{}, decoder_state.view());
    expect(decoded_predicted.has_value(), "experimental fast predicted frame decodes");
    if (!decoded_predicted) return;
    expect(std::equal(
               decoded_predicted.value().frame.data().begin(),
               decoded_predicted.value().frame.data().end(),
               encoded_predicted.value().reconstructed_frame.data().begin()),
           "experimental fast encoder and decoder predicted reconstruction agrees");

    auto truncated = encoded_predicted.value().payload;
    truncated.pop_back();
    expect(!decoder.value()->decode(
               truncated, nvcr::FrameType::predicted,
               nvcr::Timestamp{}, decoder_state.view()),
           "experimental fast truncated payload is rejected");

    auto low_qp_encoder = nvcr::experimental::make_fast_backend();
    auto low_qp_decoder = nvcr::experimental::make_fast_backend();
    configuration.intra_qp = 0;
    expect(low_qp_encoder && low_qp_decoder &&
               low_qp_encoder.value()->initialize(configuration).has_value() &&
               low_qp_decoder.value()->initialize(configuration).has_value(),
           "experimental fast low-QP backends initialize");
    if (!low_qp_encoder || !low_qp_decoder) return;
    auto extreme = nvcr::Frame::create(2, 2, nvcr::PixelFormat::yuv420p8);
    expect(extreme.has_value(), "experimental fast extreme frame is created");
    if (!extreme) return;
    std::fill(extreme.value().data().begin(), extreme.value().data().end(),
              std::byte{0});
    extreme.value().data()[1] = std::byte{255};
    nvcr::codec::SequenceState low_qp_state(1);
    auto extreme_encoded = low_qp_encoder.value()->encode(
        extreme.value(), nvcr::FrameType::intra, low_qp_state.view());
    expect(extreme_encoded.has_value(),
           "experimental fast 9-bit extreme residual encodes");
    if (!extreme_encoded) return;
    auto extreme_decoded = low_qp_decoder.value()->decode(
        extreme_encoded.value().payload, nvcr::FrameType::intra,
        nvcr::Timestamp{}, low_qp_state.view());
    expect(extreme_decoded.has_value(),
           "experimental fast 9-bit extreme residual decodes");
}

}  // namespace

int main() {
    frame_test();
    packet_test();
    access_unit_test();
    memory_test();
    state_test();
    experimental_fast_backend_test();
    if (failures == 0) {
        std::cout << "All NVCR smoke tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

