#include <nvcr/dcvcrt/tensorrt_backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <utility>

namespace {

std::uint32_t width = 176;
std::uint32_t height = 144;

bool parse_dimension(const char* value, std::uint32_t& output) {
    const auto* end = value + std::char_traits<char>::length(value);
    const auto parsed = std::from_chars(value, end, output);
    return parsed.ec == std::errc{} && parsed.ptr == end && output >= 64U &&
        (output & 1U) == 0U;
}

void fill_test_pattern(nvcr::Frame& frame) {
    auto pixels = frame.data();
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto offset = (static_cast<std::size_t>(y) * width + x) * 3;
            pixels[offset] = static_cast<std::byte>((x * 255U) / (width - 1U));
            pixels[offset + 1] = static_cast<std::byte>((y * 255U) / (height - 1U));
            pixels[offset + 2] = static_cast<std::byte>(((x + y) * 255U) /
                                                        (width + height - 2U));
        }
    }
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    return std::to_integer<std::uint32_t>(bytes[offset]) |
        (std::to_integer<std::uint32_t>(bytes[offset + 1U]) << 8U) |
        (std::to_integer<std::uint32_t>(bytes[offset + 2U]) << 16U) |
        (std::to_integer<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 4) {
        std::cerr << "usage: nvcr_tensorrt_roundtrip_tests ENGINE_DIRECTORY [WIDTH HEIGHT]\n";
        return 2;
    }
    if (argc == 4 && (!parse_dimension(argv[2], width) || !parse_dimension(argv[3], height))) {
        std::cerr << "WIDTH and HEIGHT must be even integers of at least 64\n";
        return 2;
    }

    nvcr::RuntimeConfiguration configuration;
    configuration.intra_engine_path = std::filesystem::path(argv[1]);
    configuration.device_id = 0;
    configuration.intra_qp = 32;
    configuration.verify_encoder_reconstruction = true;
    configuration.gop_size = 2;

    constexpr nvcr::Timestamp timestamp{41'667};
    auto frame = nvcr::Frame::create(
        width, height, nvcr::PixelFormat::rgb24, timestamp);
    if (!frame) {
        std::cerr << frame.error().describe() << '\n';
        return 1;
    }
    fill_test_pattern(frame.value());

    auto verification_backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!verification_backend) {
        std::cerr << verification_backend.error().describe() << '\n';
        return 1;
    }
    auto verification_initialized =
        verification_backend.value()->initialize(configuration);
    if (!verification_initialized) {
        std::cerr << verification_initialized.error().describe() << '\n';
        return 1;
    }
    const nvcr::codec::SequenceStateView empty_state{};
    auto verification_packet = verification_backend.value()->encode(
        frame.value(), nvcr::FrameType::intra, empty_state);
    if (!verification_packet) {
        std::cerr << verification_packet.error().describe() << '\n';
        return 1;
    }
    auto verification_frame = verification_backend.value()->decode(
        verification_packet.value().payload,
        nvcr::FrameType::intra,
        timestamp,
        empty_state);
    if (!verification_frame) {
        std::cerr << verification_frame.error().describe() << '\n';
        return 1;
    }

    if (!std::ranges::equal(
            verification_packet.value().reconstructed_frame.data(),
            verification_frame.value().frame.data())) {
        std::cerr << "I-frame encoder and decoder reconstructions differ\n";
        return 1;
    }

    const nvcr::codec::SequenceStateView predicted_state{
        &verification_packet.value().reconstructed_frame,
        verification_packet.value().latent_state, 1, 1, 0};
    auto verification_predicted = verification_backend.value()->encode(
        frame.value(), nvcr::FrameType::predicted, predicted_state);
    if (!verification_predicted) {
        std::cerr << verification_predicted.error().describe() << '\n';
        return 1;
    }
    auto verification_predicted_frame = verification_backend.value()->decode(
        verification_predicted.value().payload, nvcr::FrameType::predicted,
        timestamp, predicted_state);
    if (!verification_predicted_frame) {
        std::cerr << verification_predicted_frame.error().describe() << '\n';
        return 1;
    }
    if (!std::ranges::equal(
            verification_predicted.value().reconstructed_frame.data(),
            verification_predicted_frame.value().frame.data())) {
        std::cerr << "P-frame encoder and decoder reconstructions differ\n";
        return 1;
    }
    verification_backend.value().reset();

    nvcr::RuntimeConfiguration high_qp_configuration = configuration;
    high_qp_configuration.intra_qp = 63;
    auto high_qp_backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!high_qp_backend) {
        std::cerr << high_qp_backend.error().describe() << '\n';
        return 1;
    }
    auto high_qp_initialized = high_qp_backend.value()->initialize(high_qp_configuration);
    if (!high_qp_initialized) {
        std::cerr << high_qp_initialized.error().describe() << '\n';
        return 1;
    }
    auto high_qp_intra = high_qp_backend.value()->encode(
        frame.value(), nvcr::FrameType::intra, empty_state);
    if (!high_qp_intra) {
        std::cerr << high_qp_intra.error().describe() << '\n';
        return 1;
    }
    auto high_qp_intra_decoded = high_qp_backend.value()->decode(
        high_qp_intra.value().payload, nvcr::FrameType::intra, timestamp, empty_state);
    if (!high_qp_intra_decoded) {
        std::cerr << high_qp_intra_decoded.error().describe() << '\n';
        return 1;
    }
    const nvcr::codec::SequenceStateView high_qp_state{
        &high_qp_intra.value().reconstructed_frame,
        high_qp_intra.value().latent_state, 1, 1, 0};
    auto high_qp_predicted = high_qp_backend.value()->encode(
        frame.value(), nvcr::FrameType::predicted, high_qp_state);
    if (!high_qp_predicted) {
        std::cerr << high_qp_predicted.error().describe() << '\n';
        return 1;
    }
    if (high_qp_predicted.value().effective_qp != 71U ||
        read_u32(high_qp_predicted.value().payload, 12U) != 71U) {
        std::cerr << "high base QP did not select the final predicted quant entry\n";
        return 1;
    }
    auto high_qp_predicted_decoded = high_qp_backend.value()->decode(
        high_qp_predicted.value().payload, nvcr::FrameType::predicted,
        timestamp, high_qp_state);
    if (!high_qp_predicted_decoded || !std::ranges::equal(
            high_qp_predicted.value().reconstructed_frame.data(),
            high_qp_predicted_decoded.value().frame.data())) {
        std::cerr << "high-QP P-frame round trip failed\n";
        return 1;
    }
    high_qp_backend.value().reset();

    auto backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) {
        std::cerr << backend.error().describe() << '\n';
        return 1;
    }
    nvcr::codec::Components components;
    components.codec = std::move(backend.value());
    auto runtime = nvcr::Runtime::create(configuration, std::move(components));
    if (!runtime) {
        std::cerr << runtime.error().describe() << '\n';
        return 1;
    }

    auto packet = runtime.value().encode(frame.value());
    if (!packet) {
        std::cerr << packet.error().describe() << '\n';
        return 1;
    }
    if (packet.value().size() == 0 ||
        packet.value().frame_type() != nvcr::FrameType::intra) {
        std::cerr << "native encoder returned an invalid I-frame packet\n";
        return 1;
    }

    auto decoded = runtime.value().decode(packet.value());
    if (!decoded) {
        std::cerr << decoded.error().describe() << '\n';
        return 1;
    }
    if (decoded.value().width() != width || decoded.value().height() != height ||
        decoded.value().pixel_format() != nvcr::PixelFormat::yuv420p8 ||
        decoded.value().timestamp() != timestamp || decoded.value().size_bytes() == 0) {
        std::cerr << "native decoder returned an invalid reconstructed frame\n";
        return 1;
    }

    auto predicted_frame = nvcr::Frame::create(
        width, height, nvcr::PixelFormat::rgb24, nvcr::Timestamp{83'334});
    if (!predicted_frame) {
        std::cerr << predicted_frame.error().describe() << '\n';
        return 1;
    }
    fill_test_pattern(predicted_frame.value());
    predicted_frame.value().data()[0] = std::byte{0xff};
    auto predicted_packet = runtime.value().encode(predicted_frame.value());
    if (!predicted_packet) {
        std::cerr << predicted_packet.error().describe() << '\n';
        return 1;
    }
    if (predicted_packet.value().size() == 0 ||
        predicted_packet.value().frame_type() != nvcr::FrameType::predicted) {
        std::cerr << "native encoder did not emit a P frame after the initial I frame\n";
        return 1;
    }
    auto predicted_decoded = runtime.value().decode(predicted_packet.value());
    if (!predicted_decoded) {
        std::cerr << predicted_decoded.error().describe() << '\n';
        return 1;
    }
    if (predicted_decoded.value().width() != width ||
        predicted_decoded.value().height() != height ||
        predicted_decoded.value().pixel_format() != nvcr::PixelFormat::yuv420p8 ||
        predicted_decoded.value().timestamp() != nvcr::Timestamp{83'334}) {
        std::cerr << "native decoder returned an invalid P-frame reconstruction\n";
        return 1;
    }

    for (std::uint32_t index = 2; index < 4; ++index) {
        auto next_frame = nvcr::Frame::create(
            width, height, nvcr::PixelFormat::rgb24,
            nvcr::Timestamp{static_cast<std::int64_t>((index + 1U) * 41'667U)});
        if (!next_frame) {
            std::cerr << next_frame.error().describe() << '\n';
            return 1;
        }
        fill_test_pattern(next_frame.value());
        next_frame.value().data()[index] = static_cast<std::byte>(index);
        auto next_packet = runtime.value().encode(next_frame.value());
        const auto expected_type = (index % 2U) == 0U ?
            nvcr::FrameType::intra : nvcr::FrameType::predicted;
        if (!next_packet || next_packet.value().frame_type() != expected_type) {
            std::cerr << "two-GOP encode produced the wrong frame type\n";
            return 1;
        }
        auto access_unit = nvcr::AccessUnitIO::deserialize(next_packet.value().data());
        if (!access_unit || access_unit.value().model_id != "dcvcrt") {
            std::cerr << "runtime packet did not contain the expected access unit\n";
            return 1;
        }
        auto next_decoded = runtime.value().decode(next_packet.value());
        if (!next_decoded || next_decoded.value().timestamp() != next_frame.value().timestamp()) {
            std::cerr << "two-GOP decode failed\n";
            return 1;
        }
    }

    auto reset_result = runtime.value().reset();
    if (!reset_result) {
        std::cerr << reset_result.error().describe() << '\n';
        return 1;
    }
    auto reset_packet = runtime.value().encode(frame.value());
    if (!reset_packet || reset_packet.value().frame_type() != nvcr::FrameType::intra) {
        std::cerr << "runtime reset did not restart with an intra access unit\n";
        return 1;
    }
    auto reset_decoded = runtime.value().decode(reset_packet.value());
    if (!reset_decoded) {
        std::cerr << reset_decoded.error().describe() << '\n';
        return 1;
    }

    std::cout << "Native DCVC-RT I/P round trip passed: "
              << width << 'x' << height << ", " << packet.value().size()
              << " payload bytes\n";
    return 0;
}
