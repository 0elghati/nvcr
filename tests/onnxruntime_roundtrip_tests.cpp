#include "host_backend.hpp"

#include <nvcr/dcvcrt/backend.hpp>
#include <nvcr/nvcr.hpp>
#include <nvcr/runtime/registry.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace {

namespace session_api = nvcr::provider::experimental;

std::uint32_t width = 176;
std::uint32_t height = 144;

int fail(std::string_view message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int fail(const nvcr::Error& error) {
    std::cerr << "FAIL: " << error.describe() << '\n';
    return 1;
}

bool parse_dimension(const char* value, std::uint32_t& output) {
    const auto* end = value + std::char_traits<char>::length(value);
    const auto parsed = std::from_chars(value, end, output);
    return parsed.ec == std::errc{} && parsed.ptr == end && output >= 64U &&
        (output & 1U) == 0U;
}

void fill_test_pattern(nvcr::Frame& frame, std::uint32_t shift) {
    auto pixels = frame.data();
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto offset = (static_cast<std::size_t>(y) * width + x) * 3;
            pixels[offset] = static_cast<std::byte>(((x + shift) * 255U) / (width + shift));
            pixels[offset + 1] = static_cast<std::byte>((y * 255U) / (height - 1U));
            pixels[offset + 2] = static_cast<std::byte>(((x + y) * 255U) /
                                                        (width + height - 2U));
        }
    }
}

// Mean absolute luma error against BT.709 luma of the RGB source; catches
// garbage reconstructions without asserting codec quality.
double luma_error(const nvcr::Frame& source, const nvcr::Frame& decoded) {
    const auto rgb = source.data();
    const auto yuv = decoded.data();
    double total = 0.0;
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(width) * height; ++pixel) {
        const double luma = 0.2126 * std::to_integer<int>(rgb[pixel * 3]) +
            0.7152 * std::to_integer<int>(rgb[pixel * 3 + 1]) +
            0.0722 * std::to_integer<int>(rgb[pixel * 3 + 2]);
        total += std::abs(luma - std::to_integer<int>(yuv[pixel]));
    }
    return total / (static_cast<double>(width) * height);
}

int session_contract(const nvcr::runtime::ProviderEntry& entry,
                     const nvcr::RuntimeConfiguration& configuration) {
    auto session = entry.session_factory(configuration);
    if (!session) return fail(session.error());
    auto buffer = session.value()->allocate(64, session_api::MemoryDomain::host);
    if (!buffer || buffer.value()->host_bytes().size() != 64U ||
        buffer.value()->owner_id() != session.value()->owner_id()) {
        return fail("host allocation is not host addressable or has the wrong owner");
    }
    auto device = session.value()->allocate(64, session_api::MemoryDomain::provider_device);
    if (device) return fail("device allocation was accepted");

    session_api::ExecutableStageDescriptor escaping;
    escaping.stage_id = "escape";
    escaping.artifact.component_id = "../i_analysis";
    escaping.artifact.path = std::filesystem::temp_directory_path().string();
    auto rejected = session.value()->load_stage(escaping);
    if (rejected || rejected.error().code() != nvcr::ErrorCode::invalid_argument) {
        return fail("component id outside the bundle was not rejected");
    }

    session_api::ExecutableStageDescriptor missing;
    missing.stage_id = "missing";
    missing.artifact.component_id = "nvcr_missing_stage";
    missing.artifact.path = std::filesystem::temp_directory_path().string();
    auto absent = session.value()->load_stage(missing);
    if (absent || absent.error().code() != nvcr::ErrorCode::missing_artifact) {
        return fail("missing ONNX stage was not reported as missing_artifact");
    }

    // Unknown qnn options are rejected; other targets' options are ignored.
    for (const auto& [key, value] : {std::pair{"qnn.unknown", "1"},
                                     std::pair{"qnn.graph_optimization", "fast"}}) {
        auto options = missing;
        options.provider_options = {{key, value}};
        auto invalid = session.value()->load_stage(options);
        if (invalid || invalid.error().code() != nvcr::ErrorCode::invalid_argument) {
            return fail(std::string("provider option was not rejected: ") + key);
        }
    }
    auto foreign = missing;
    foreign.provider_options = {{"tensorrt.anything", "1"}, {"qnn.htp_performance_mode", "burst"}};
    auto ignored = session.value()->load_stage(foreign);
    if (ignored || ignored.error().code() != nvcr::ErrorCode::missing_artifact) {
        return fail("valid or foreign provider options changed stage loading");
    }
    return 0;
}

nvcr::Result<std::unique_ptr<nvcr::codec::CodecBackend>> make_backend(
    const nvcr::runtime::ProviderEntry& entry,
    const nvcr::RuntimeConfiguration& configuration) {
    auto session = entry.session_factory(configuration);
    if (!session) return session.error();
    auto backend = nvcr::dcvcrt::make_host_backend(std::move(session.value()));
    if (!backend) return backend.error();
    auto initialized = backend.value()->initialize(configuration);
    if (!initialized) return initialized.error();
    return backend;
}

int backend_round_trip(
    const nvcr::runtime::ProviderEntry& entry,
    const nvcr::RuntimeConfiguration& configuration,
    const nvcr::Frame& frame) {
    constexpr nvcr::Timestamp timestamp{41'667};
    auto backend = make_backend(entry, configuration);
    if (!backend) return fail(backend.error());
    const nvcr::codec::SequenceStateView empty_state{};
    auto intra = backend.value()->encode(frame, nvcr::FrameType::intra, empty_state);
    if (!intra) return fail(intra.error());
    auto intra_decoded = backend.value()->decode(
        intra.value().payload, nvcr::FrameType::intra, timestamp, empty_state);
    if (!intra_decoded) return fail(intra_decoded.error());
    if (!std::ranges::equal(intra.value().reconstructed_frame.data(),
                            intra_decoded.value().frame.data())) {
        return fail("I-frame encoder and decoder reconstructions differ");
    }
    const auto intra_error = luma_error(frame, intra_decoded.value().frame);
    if (intra_error > 8.0) return fail("I-frame reconstruction is far from the source");

    const nvcr::codec::SequenceStateView predicted_state{
        &intra.value().reconstructed_frame, intra.value().latent_state, 1, 1, 0};
    auto predicted = backend.value()->encode(frame, nvcr::FrameType::predicted, predicted_state);
    if (!predicted) return fail(predicted.error());
    auto predicted_decoded = backend.value()->decode(
        predicted.value().payload, nvcr::FrameType::predicted, timestamp, predicted_state);
    if (!predicted_decoded) return fail(predicted_decoded.error());
    if (!std::ranges::equal(predicted.value().reconstructed_frame.data(),
                            predicted_decoded.value().frame.data())) {
        return fail("P-frame encoder and decoder reconstructions differ");
    }
    const auto predicted_error = luma_error(frame, predicted_decoded.value().frame);
    if (predicted_error > 8.0) return fail("P-frame reconstruction is far from the source");

    // Malformed payloads may fail or decode noise, but must stay in bounds
    // (run under the sanitizer build) and must not crash.
    const auto& payload = intra.value().payload;
    for (const auto cut : {std::size_t{8}, payload.size() / 2}) {
        static_cast<void>(backend.value()->decode(
            std::span(payload).first(cut), nvcr::FrameType::intra, timestamp, empty_state));
    }
    auto corrupted = payload;
    for (std::size_t index = 20; index < corrupted.size(); index += 7) {
        corrupted[index] ^= std::byte{0x5a};
    }
    static_cast<void>(backend.value()->decode(
        corrupted, nvcr::FrameType::intra, timestamp, empty_state));
    auto oversized = payload;
    oversized[5] = std::byte{0x10};  // width field becomes at least 4096
    auto rejected = backend.value()->decode(
        oversized, nvcr::FrameType::intra, timestamp, empty_state);
    if (rejected || rejected.error().code() != nvcr::ErrorCode::malformed_bitstream) {
        return fail("an out-of-range payload width was not rejected as malformed");
    }

    // A fresh session must reproduce the same bitstream.
    auto repeat = make_backend(entry, configuration);
    if (!repeat) return fail(repeat.error());
    auto repeated = repeat.value()->encode(frame, nvcr::FrameType::intra, empty_state);
    if (!repeated || repeated.value().payload != intra.value().payload) {
        return fail("a fresh session produced a different I-frame payload");
    }

    std::cout << "host backend I/P self-conformance: I " << intra.value().payload.size()
              << " bytes (luma MAE " << intra_error << "), P "
              << predicted.value().payload.size() << " bytes (luma MAE " << predicted_error
              << ")\n";
    return 0;
}

int runtime_round_trip(const nvcr::RuntimeConfiguration& configuration) {
    auto runtime = nvcr::Runtime::create(configuration);
    if (!runtime) return fail(runtime.error());
    for (std::uint32_t index = 0; index < 4; ++index) {
        auto frame = nvcr::Frame::create(
            width, height, nvcr::PixelFormat::rgb24,
            nvcr::Timestamp{static_cast<std::int64_t>((index + 1U) * 41'667U)});
        if (!frame) return fail(frame.error());
        fill_test_pattern(frame.value(), index);
        auto packet = runtime.value().encode(frame.value());
        const auto expected_type =
            index % 2U == 0U ? nvcr::FrameType::intra : nvcr::FrameType::predicted;
        if (!packet) return fail(packet.error());
        if (packet.value().frame_type() != expected_type) {
            return fail("two-GOP encode produced the wrong frame type");
        }
        auto decoded = runtime.value().decode(packet.value());
        if (!decoded) return fail(decoded.error());
        if (decoded.value().width() != width || decoded.value().height() != height ||
            decoded.value().timestamp() != frame.value().timestamp()) {
            return fail("runtime decode returned the wrong frame geometry or timestamp");
        }
    }
    auto reset = runtime.value().reset();
    if (!reset) return fail(reset.error());
    auto frame = nvcr::Frame::create(width, height, nvcr::PixelFormat::rgb24, nvcr::Timestamp{0});
    if (!frame) return fail(frame.error());
    fill_test_pattern(frame.value(), 0);
    auto packet = runtime.value().encode(frame.value());
    if (!packet || packet.value().frame_type() != nvcr::FrameType::intra) {
        return fail("runtime reset did not restart with an intra access unit");
    }
    auto decoded = runtime.value().decode(packet.value());
    if (!decoded) return fail(decoded.error());
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 3 && argc != 5) {
        std::cerr << "usage: nvcr_onnxruntime_roundtrip_tests PROVIDER_ID "
                     "[MODEL_DIRECTORY [WIDTH HEIGHT]]\n";
        return 2;
    }
    if (argc == 5 && (!parse_dimension(argv[3], width) || !parse_dimension(argv[4], height))) {
        std::cerr << "WIDTH and HEIGHT must be even integers of at least 64\n";
        return 2;
    }

    nvcr::dcvcrt::register_codec();
    nvcr::dcvcrt::register_execution_providers();
    const auto entry = nvcr::runtime::Registry::instance().find_provider(argv[1]);
    if (!entry || !entry->session_factory) {
        return fail(std::string("provider is not registered: ") + argv[1]);
    }

    nvcr::RuntimeConfiguration configuration;
    configuration.provider.id = argv[1];
    configuration.codec.intra_qp = 32;
    configuration.codec.gop_size = 2;
    configuration.codec.verify_encoder_reconstruction = true;
    if (const int status = session_contract(*entry, configuration); status != 0) return status;
    if (argc == 2) {
        std::cout << "ONNX Runtime session contract passed for " << argv[1] << '\n';
        return 0;
    }

    configuration.artifacts.intra_engine_path = std::filesystem::path(argv[2]);
    auto frame = nvcr::Frame::create(width, height, nvcr::PixelFormat::rgb24, nvcr::Timestamp{41'667});
    if (!frame) return fail(frame.error());
    fill_test_pattern(frame.value(), 0);
    if (const int status = backend_round_trip(*entry, configuration, frame.value());
        status != 0) {
        return status;
    }
    if (const int status = runtime_round_trip(configuration); status != 0) return status;
    std::cout << "ONNX Runtime DCVC-RT I/P round trip passed: " << argv[1] << ", " << width
              << 'x' << height << '\n';
    return 0;
}
