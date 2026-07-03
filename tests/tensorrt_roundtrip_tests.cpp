#include <nvcr/dcvcrt/tensorrt_backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <utility>

namespace {

constexpr std::uint32_t width = 176;
constexpr std::uint32_t height = 144;

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

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: nvcr_tensorrt_roundtrip_tests ENGINE_DIRECTORY\n";
        return 2;
    }

    auto backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) {
        std::cerr << backend.error().describe() << '\n';
        return 1;
    }

    nvcr::RuntimeConfiguration configuration;
    configuration.intra_engine_path = std::filesystem::path(argv[1]);
    configuration.device_id = 0;
    configuration.intra_qp = 32;
    configuration.gop_size = 32;

    nvcr::dcvcrt::Components components;
    components.codec = std::move(backend.value());
    auto runtime = nvcr::Runtime::create(configuration, std::move(components));
    if (!runtime) {
        std::cerr << runtime.error().describe() << '\n';
        return 1;
    }

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
    const nvcr::dcvcrt::SequenceStateView empty_state{};
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

    const nvcr::dcvcrt::SequenceStateView predicted_state{
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
        decoded.value().pixel_format() != nvcr::PixelFormat::rgb24 ||
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
        predicted_decoded.value().timestamp() != nvcr::Timestamp{83'334}) {
        std::cerr << "native decoder returned an invalid P-frame reconstruction\n";
        return 1;
    }

    std::cout << "Native DCVC-RT I/P round trip passed: "
              << width << 'x' << height << ", " << packet.value().size()
              << " payload bytes\n";
    return 0;
}
