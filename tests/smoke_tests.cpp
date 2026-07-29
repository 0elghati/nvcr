#include <nvcr/dcvcrt/sequence_state.hpp>
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

}  // namespace

int main() {
    frame_test();
    packet_test();
    access_unit_test();
    memory_test();
    state_test();
    if (failures == 0) {
        std::cout << "All NVCR smoke tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

