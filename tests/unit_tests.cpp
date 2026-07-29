#include <nvcr/dcvcrt/sequence_state.hpp>
#include <nvcr/nvcr.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>

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
    EXPECT_FALSE(nvcr::PacketIO::deserialize(encoded.value(), encoded.value().size() - 1U));
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded.value().timestamp(), packet.timestamp());
    EXPECT_EQ(decoded.value().metadata(), packet.metadata());
    EXPECT_EQ(decoded.value().size(), packet.size());

    auto excessive_count = encoded.value();
    excessive_count[16] = std::byte{0xff};
    excessive_count[17] = std::byte{0xff};
    EXPECT_FALSE(nvcr::PacketIO::deserialize(excessive_count));

    nvcr::PacketMetadata excessive_metadata;
    for (int index = 0; index < 17; ++index) {
        excessive_metadata.emplace(std::to_string(index), std::string(65'535, 'x'));
    }
    nvcr::Packet excessive_packet(
        {std::byte{1}}, nvcr::Timestamp{}, nvcr::FrameType::intra,
        std::move(excessive_metadata));
    EXPECT_FALSE(nvcr::PacketIO::serialize(excessive_packet));

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
    EXPECT_FALSE(nvcr::PacketIO::deserialize(malformed_metadata.take()));
}

TEST(AccessUnitIO, RoundTripsAndRejectsUnsafeInputs) {
    nvcr::AccessUnit original{
        "dcvcrt-cvpr2025", 176, 144, 32, nvcr::FrameType::intra, true,
        {std::byte{1}, std::byte{2}, std::byte{3}}};
    auto wire = nvcr::AccessUnitIO::serialize(original);
    ASSERT_TRUE(wire);
    auto decoded = nvcr::AccessUnitIO::deserialize(wire.value());
    ASSERT_TRUE(decoded);
    EXPECT_EQ(decoded.value().model_id, original.model_id);
    EXPECT_EQ(decoded.value().width, 176U);
    EXPECT_TRUE(decoded.value().reset_state);

    auto truncated = wire.value();
    truncated.pop_back();
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(truncated));
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(wire.value(), wire.value().size() - 1U));
    auto unsupported_flags = wire.value();
    unsupported_flags[7] = std::byte{0x80};
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(unsupported_flags));
    auto invalid_model = wire.value();
    invalid_model[32] = std::byte{'/'};
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(invalid_model));

    auto invalid_dimensions = wire.value();
    invalid_dimensions[12] = std::byte{0xff};
    invalid_dimensions[13] = std::byte{0xff};
    invalid_dimensions[14] = std::byte{0xff};
    invalid_dimensions[15] = std::byte{0xff};
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(invalid_dimensions));

    auto oversized_payload = wire.value();
    for (std::size_t index = 24; index < 32; ++index) {
        oversized_payload[index] = std::byte{0xff};
    }
    EXPECT_FALSE(nvcr::AccessUnitIO::deserialize(oversized_payload));

    original.reset_state = false;
    EXPECT_FALSE(nvcr::AccessUnitIO::serialize(original));
    original.reset_state = true;

    original.frame_type = nvcr::FrameType::predicted;
    EXPECT_FALSE(nvcr::AccessUnitIO::serialize(original));
    original.reset_state = false;
    original.qp = 71;
    EXPECT_TRUE(nvcr::AccessUnitIO::serialize(original));
    original.qp = 72;
    EXPECT_FALSE(nvcr::AccessUnitIO::serialize(original));
    original.frame_type = nvcr::FrameType::intra;
    original.reset_state = true;
    original.qp = 64;
    EXPECT_FALSE(nvcr::AccessUnitIO::serialize(original));
    original.qp = 32;
    original.width = 175;
    EXPECT_FALSE(nvcr::AccessUnitIO::serialize(original));
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
    nvcr::codec::SequenceState state(32);
    EXPECT_FALSE(state.validate_packet(nvcr::FrameType::predicted));
    auto type = state.next_frame_type();
    ASSERT_TRUE(type);
    EXPECT_EQ(type.value(), nvcr::FrameType::intra);
}

TEST(SequenceState, PredictedFramesAreValidatedWithoutGopBoundaryDependency) {
    nvcr::codec::SequenceState state(32);
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

TEST(SequenceState, PredictedCommitCanAdvanceWithoutHostReference) {
    nvcr::codec::SequenceState state(32);
    auto reference = nvcr::Frame::create(2, 2, nvcr::PixelFormat::gray8);
    ASSERT_TRUE(reference);
    ASSERT_TRUE(state.commit(std::move(reference.value()), {}, nvcr::FrameType::intra));

    ASSERT_TRUE(state.commit(nvcr::Frame{}, {}, nvcr::FrameType::predicted));
    EXPECT_TRUE(state.has_reference());
    EXPECT_EQ(state.frame_index(), 2U);
    auto type = state.next_frame_type();
    ASSERT_TRUE(type);
    EXPECT_EQ(type.value(), nvcr::FrameType::predicted);
}
