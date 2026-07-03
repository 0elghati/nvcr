#include <nvcr/dcvcrt/sequence_state.hpp>
#include <nvcr/nvcr.hpp>

#include <gtest/gtest.h>

#include <cstddef>

TEST(Frame, ComputesYuv420Storage) {
    auto frame = nvcr::Frame::create(16, 8, nvcr::PixelFormat::yuv420p8);
    ASSERT_TRUE(frame);
    EXPECT_EQ(frame.value().size_bytes(), 192U);
}

TEST(PacketIO, RoundTripsPacket) {
    nvcr::Packet packet(
        {std::byte{1}, std::byte{2}},
        nvcr::Timestamp{100},
        nvcr::FrameType::intra,
        {{"key", "value"}});
    auto encoded = nvcr::PacketIO::serialize(packet);
    ASSERT_TRUE(encoded);
    auto decoded = nvcr::PacketIO::deserialize(encoded.value());
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded.value().timestamp(), packet.timestamp());
    EXPECT_EQ(decoded.value().metadata(), packet.metadata());
    EXPECT_EQ(decoded.value().size(), packet.size());
}

TEST(MemoryPool, ReusesReleasedBlocks) {
    nvcr::MemoryPool pool(nvcr::make_host_memory_resource(), 1024);
    void* address = nullptr;
    {
        auto lease = pool.acquire(128);
        ASSERT_TRUE(lease);
        address = lease.value().bytes().data();
    }
    auto lease = pool.acquire(64);
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease.value().bytes().data(), address);
}

TEST(SequenceState, RequiresReferenceForPredictedFrames) {
    nvcr::dcvcrt::SequenceState state(32);
    EXPECT_FALSE(state.validate_packet(nvcr::FrameType::predicted));
    auto type = state.next_frame_type();
    ASSERT_TRUE(type);
    EXPECT_EQ(type.value(), nvcr::FrameType::intra);
}

TEST(SequenceState, PredictedFramesAreValidatedWithoutGopBoundaryDependency) {
    nvcr::dcvcrt::SequenceState state(32);
    auto reference = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
    ASSERT_TRUE(reference);
    ASSERT_TRUE(state.commit(std::move(reference.value()), {}, nvcr::FrameType::intra));

    auto predicted = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
    ASSERT_TRUE(predicted);
    for (int index = 0; index < 31; ++index) {
        ASSERT_TRUE(state.commit(std::move(predicted.value()), {}, nvcr::FrameType::predicted));
        predicted = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
        ASSERT_TRUE(predicted);
    }

    EXPECT_TRUE(state.validate_packet(nvcr::FrameType::predicted));
}

