#include <nvcr/dcvcrt/sequence_state.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
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
    nvcr::dcvcrt::SequenceState state(2);
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

    nvcr::dcvcrt::SequenceState long_gop_state(32);
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
    memory_test();
    state_test();
    if (failures == 0) {
        std::cout << "All NVCR smoke tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}

