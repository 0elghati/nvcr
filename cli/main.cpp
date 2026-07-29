#include <nvcr/dcvcrt/tensorrt_backend.hpp>
#include <nvcr/experimental/fast_backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

constexpr char sequence_magic[] = {'N', 'V', 'C', 'S'};
constexpr std::uint16_t sequence_version = 1;
constexpr std::size_t maximum_record_bytes = 65U * 1024U * 1024U;

enum class Command { encode, decode };

struct Options final {
    Command command{Command::encode};
    fs::path input;
    fs::path output;
    fs::path engine_dir;
    std::string backend{"default"};
    std::string engine_profile;
    std::uint32_t width{};
    std::uint32_t height{};
    std::size_t frames{};
    std::uint32_t qp{32};
    std::uint32_t gop_size{32};
    std::int64_t frame_period_us{16'667};
    std::int32_t device_id{};
    bool profile{};
    bool verbose{};
    bool help{};
};

void usage(std::ostream& out) {
    out << "NVCR native encoder and decoder\n\n"
        << "Usage:\n"
        << "  nvcr encode -i INPUT.yuv -o OUTPUT.nvcr -s WIDTHxHEIGHT\n"
        << "              [--backend NAME] [--engine-profile NAME] [--frames N] [--qp N]\n"
        << "              [--gop-size N] [-r FPS] [--engine-dir DIR]\n"
        << "  nvcr decode -i INPUT.nvcr -o OUTPUT.yuv\n"
        << "              [--backend NAME] [--engine-profile NAME] [--frames N]\n"
        << "              [--device-id N] [--engine-dir DIR]\n\n"
        << "Input and output video use planar 8-bit YUV 4:2:0. A frame count of zero\n"
        << "means process until end of input. Normal encoding uses the configured I/P\n"
        << "GOP; --gop-size 1 explicitly requests all-intra development mode.\n\n"
        << "Options:\n"
        << "  -i, --input FILE          Input file\n"
        << "  -o, --output FILE         Output file\n"
        << "  -s, --size WIDTHxHEIGHT   Raw input dimensions (encode only)\n"
        << "      --width N             Raw input width (encode only)\n"
        << "      --height N            Raw input height (encode only)\n"
        << "      --backend NAME        Backend selector: default/dcvcrt or experimental mlvc-fast\n"
        << "      --engine-profile NAME Installed engine profile, for example 720p-fp16\n"
        << "                            (default: NVCR_ENGINE_PROFILE or active default)\n"
        << "      --engine-dir DIR      Custom TensorRT engine and entropy asset directory\n"
        << "                            (overrides backend/profile selection)\n"
        << "      --frames N            Frames to process; 0 means all (default: 0)\n"
        << "      --qp N                I-frame QP (default: 32)\n"
        << "      --gop-size N          GOP size; 1 explicitly requests all-intra (default: 32)\n"
        << "  -r, --fps FPS             Input frame rate (default: 60)\n"
        << "      --frame-period-us N   Exact timestamp interval (alternative to --fps)\n"
        << "      --device-id N         CUDA device (default: 0)\n"
        << "      --profile             Print TensorRT/CUDA per-frame profiling counters\n"
        << "  -v, --verbose             Print per-frame CLI progress and info logs\n"
        << "  -h, --help                Show this help\n";
}

template <class T>
bool parse_number(std::string_view text, T& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

bool parse_size(std::string_view text, std::uint32_t& width, std::uint32_t& height) {
    const auto separator = text.find_first_of("xX");
    return separator != std::string_view::npos &&
        parse_number(text.substr(0, separator), width) &&
        parse_number(text.substr(separator + 1), height);
}

bool parse_fps(std::string_view text, std::int64_t& period_us) {
    std::string owned(text);
    char* end = nullptr;
    errno = 0;
    const double fps = std::strtod(owned.c_str(), &end);
    if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(fps) || fps <= 0.0) {
        return false;
    }
    const double period = 1'000'000.0 / fps;
    if (period > static_cast<double>(std::numeric_limits<std::int64_t>::max())) return false;
    period_us = static_cast<std::int64_t>(std::llround(period));
    return period_us > 0;
}

fs::path default_engine_root() {
    if (const char* xdg_data_home = std::getenv("XDG_DATA_HOME")) {
        if (*xdg_data_home != '\0') return fs::path(xdg_data_home) / "nvcr" / "engines";
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home != '\0') return fs::path(home) / ".local" / "share" / "nvcr" / "engines";
    }
    return fs::path("engines");
}

std::string normalize_backend(std::string backend) {
    if (backend == "dcvc_rt" || backend == "dcvc-rt") return "dcvcrt";
    if (backend == "mlvc_fast" || backend == "mlvc-fast" || backend == "fast") return "mlvc-fast";
    return backend;
}

fs::path resolve_engine_dir(const Options& options) {
    if (!options.engine_dir.empty()) return options.engine_dir;
    if (const char* engine_dir = std::getenv("NVCR_ENGINE_DIR")) {
        if (*engine_dir != '\0') return fs::path(engine_dir);
    }

    auto root = default_engine_root();
    if (!options.engine_profile.empty()) {
        return root / "profiles" / options.backend / options.engine_profile;
    }
    return root / options.backend;
}

bool parse_options(int argc, char* argv[], Options& options) {
    if (argc < 2) {
        usage(std::cerr);
        return false;
    }
    const std::string_view command = argv[1];
    if (command == "--help" || command == "-h") {
        options.help = true;
        return true;
    }
    if (command == "encode") {
        options.command = Command::encode;
    } else if (command == "decode") {
        options.command = Command::decode;
    } else {
        std::cerr << "nvcr: unknown command: " << command << '\n';
        return false;
    }

    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto value_after = [&](std::string_view flag) -> std::string_view {
            if (++index >= argc) {
                std::cerr << "nvcr: " << flag << " requires a value\n";
                return {};
            }
            return argv[index];
        };

        if (argument == "--help" || argument == "-h") {
            options.help = true;
            return true;
        }
        if (argument == "--input" || argument == "-i") {
            options.input = value_after(argument);
            if (options.input.empty()) return false;
        } else if (argument == "--output" || argument == "-o") {
            options.output = value_after(argument);
            if (options.output.empty()) return false;
        } else if (argument == "--backend") {
            options.backend = std::string(value_after(argument));
            if (options.backend.empty()) return false;
        } else if (argument == "--engine-profile") {
            options.engine_profile = std::string(value_after(argument));
            if (options.engine_profile.empty()) return false;
        } else if (argument == "--engine-dir") {
            options.engine_dir = value_after(argument);
            if (options.engine_dir.empty()) return false;
        } else if (argument == "--size" || argument == "-s") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_size(value, options.width, options.height)) {
                std::cerr << "nvcr: invalid frame size: " << value << '\n';
                return false;
            }
        } else if (argument == "--width") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.width)) {
                std::cerr << "nvcr: invalid width: " << value << '\n';
                return false;
            }
        } else if (argument == "--height") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.height)) {
                std::cerr << "nvcr: invalid height: " << value << '\n';
                return false;
            }
        } else if (argument == "--frames") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.frames)) {
                std::cerr << "nvcr: invalid frame count: " << value << '\n';
                return false;
            }
        } else if (argument == "--qp") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.qp)) {
                std::cerr << "nvcr: invalid QP: " << value << '\n';
                return false;
            }
        } else if (argument == "--gop-size") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.gop_size) ||
                options.gop_size == 0) {
                std::cerr << "nvcr: invalid GOP size: " << value << '\n';
                return false;
            }
        } else if (argument == "--device-id") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.device_id)) {
                std::cerr << "nvcr: invalid device id: " << value << '\n';
                return false;
            }
        } else if (argument == "--fps" || argument == "-r") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_fps(value, options.frame_period_us)) {
                std::cerr << "nvcr: invalid frame rate: " << value << '\n';
                return false;
            }
        } else if (argument == "--frame-period-us") {
            const auto value = value_after(argument);
            if (value.empty() || !parse_number(value, options.frame_period_us) ||
                options.frame_period_us <= 0) {
                std::cerr << "nvcr: invalid frame period: " << value << '\n';
                return false;
            }
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--verbose" || argument == "-v") {
            options.verbose = true;
        } else {
            std::cerr << "nvcr: unknown option: " << argument << '\n';
            return false;
        }
    }
    if (const char* backend = std::getenv("NVCR_BACKEND")) {
        if (*backend != '\0' && options.backend == "default") options.backend = backend;
    }
    if (const char* profile = std::getenv("NVCR_ENGINE_PROFILE")) {
        if (*profile != '\0' && options.engine_profile.empty()) options.engine_profile = profile;
    }
    options.backend = normalize_backend(std::move(options.backend));
    options.engine_dir = resolve_engine_dir(options);
    if (options.input.empty() || options.output.empty()) {
        std::cerr << "nvcr: --input and --output are required\n";
        return false;
    }
    if (options.input == options.output) {
        std::cerr << "nvcr: input and output must be different files\n";
        return false;
    }
    if (options.qp >= 64) {
        std::cerr << "nvcr: QP must be between 0 and 63\n";
        return false;
    }
    if (options.command == Command::encode &&
        (options.width == 0 || options.height == 0 || (options.width & 1U) != 0 ||
         (options.height & 1U) != 0)) {
        std::cerr << "nvcr: encode requires a non-zero even WIDTHxHEIGHT for YUV420p8\n";
        return false;
    }
    return true;
}

std::vector<std::byte> yuv420p8_to_rgb24(
    std::span<const std::byte> yuv, std::uint32_t width, std::uint32_t height) {
    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto uv_width = width / 2U;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (height / 2U);
    const auto* y_plane = yuv.data();
    const auto* u_plane = yuv.data() + y_size;
    const auto* v_plane = yuv.data() + y_size + uv_size;
    std::vector<std::byte> rgb(y_size * 3U);
    const auto clamp = [](int value) {
        return static_cast<std::byte>(std::clamp(value, 0, 255));
    };
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto y_index = static_cast<std::size_t>(y) * width + x;
            const auto uv_index = static_cast<std::size_t>(y / 2U) * uv_width + x / 2U;
            const int luma = std::to_integer<unsigned char>(y_plane[y_index]);
            const int u = std::to_integer<unsigned char>(u_plane[uv_index]) - 128;
            const int v = std::to_integer<unsigned char>(v_plane[uv_index]) - 128;
            const int c = std::max(luma - 16, 0);
            const auto destination = y_index * 3U;
            rgb[destination] = clamp((298 * c + 409 * v + 128) >> 8);
            rgb[destination + 1] = clamp((298 * c - 100 * u - 208 * v + 128) >> 8);
            rgb[destination + 2] = clamp((298 * c + 516 * u + 128) >> 8);
        }
    }
    return rgb;
}

std::vector<std::byte> rgb24_to_yuv420p8(
    std::span<const std::byte> rgb, std::uint32_t width, std::uint32_t height) {
    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto uv_width = width / 2U;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (height / 2U);
    std::vector<std::byte> yuv(y_size + 2U * uv_size);
    auto* y_plane = yuv.data();
    auto* u_plane = yuv.data() + y_size;
    auto* v_plane = yuv.data() + y_size + uv_size;
    const auto clamp = [](int value) {
        return static_cast<std::byte>(std::clamp(value, 0, 255));
    };
    for (std::uint32_t y = 0; y < height; y += 2U) {
        for (std::uint32_t x = 0; x < width; x += 2U) {
            int u_sum = 0;
            int v_sum = 0;
            for (std::uint32_t dy = 0; dy < 2U; ++dy) {
                for (std::uint32_t dx = 0; dx < 2U; ++dx) {
                    const auto source =
                        (static_cast<std::size_t>(y + dy) * width + x + dx) * 3U;
                    const int r = std::to_integer<unsigned char>(rgb[source]);
                    const int g = std::to_integer<unsigned char>(rgb[source + 1]);
                    const int b = std::to_integer<unsigned char>(rgb[source + 2]);
                    y_plane[static_cast<std::size_t>(y + dy) * width + x + dx] =
                        clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
                    u_sum += ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    v_sum += ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                }
            }
            const auto chroma = static_cast<std::size_t>(y / 2U) * uv_width + x / 2U;
            u_plane[chroma] = clamp(u_sum / 4);
            v_plane[chroma] = clamp(v_sum / 4);
        }
    }
    return yuv;
}

bool write_u16(std::ostream& output, std::uint16_t value) {
    const char bytes[] = {
        static_cast<char>(value & 0xffU), static_cast<char>((value >> 8U) & 0xffU)};
    output.write(bytes, sizeof(bytes));
    return static_cast<bool>(output);
}

bool write_u64(std::ostream& output, std::uint64_t value) {
    char bytes[8]{};
    for (unsigned index = 0; index < 8U; ++index) {
        bytes[index] = static_cast<char>((value >> (index * 8U)) & 0xffU);
    }
    output.write(bytes, sizeof(bytes));
    return static_cast<bool>(output);
}

bool write_sequence_header(std::ostream& output) {
    output.write(sequence_magic, sizeof(sequence_magic));
    return write_u16(output, sequence_version) && write_u16(output, 0);
}

bool read_sequence_header(std::istream& input) {
    char magic[sizeof(sequence_magic)]{};
    input.read(magic, sizeof(magic));
    char fields[4]{};
    input.read(fields, sizeof(fields));
    if (!input) return false;
    if (!std::equal(std::begin(magic), std::end(magic), std::begin(sequence_magic))) return false;
    const auto version = static_cast<std::uint16_t>(static_cast<unsigned char>(fields[0])) |
        (static_cast<std::uint16_t>(static_cast<unsigned char>(fields[1])) << 8U);
    const auto flags = static_cast<std::uint16_t>(static_cast<unsigned char>(fields[2])) |
        (static_cast<std::uint16_t>(static_cast<unsigned char>(fields[3])) << 8U);
    return version == sequence_version && flags == 0;
}

enum class RecordRead { record, end, error };

RecordRead read_record(std::istream& input, std::vector<std::byte>& bytes) {
    const int first = input.get();
    if (first == std::char_traits<char>::eof()) {
        return input.eof() ? RecordRead::end : RecordRead::error;
    }
    std::uint64_t size = static_cast<unsigned char>(first);
    char rest[7]{};
    input.read(rest, sizeof(rest));
    if (!input) return RecordRead::error;
    for (unsigned index = 1; index < 8U; ++index) {
        size |= static_cast<std::uint64_t>(static_cast<unsigned char>(rest[index - 1])) <<
            (index * 8U);
    }
    if (size == 0 || size > maximum_record_bytes ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return RecordRead::error;
    }
    bytes.resize(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return input ? RecordRead::record : RecordRead::error;
}

nvcr::Result<nvcr::Runtime> create_runtime(const Options& options) {
    auto backend_name = normalize_backend(options.backend);
    auto backend = backend_name == "mlvc-fast"
        ? nvcr::experimental::make_fast_backend()
        : nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) return backend.error();
    nvcr::RuntimeConfiguration configuration;
    if (backend_name == "mlvc-fast") configuration.model_id = "mlvc-fast-v1";
    configuration.intra_engine_path = options.engine_dir;
    configuration.device_id = options.device_id;
    configuration.intra_qp = options.qp;
    configuration.gop_size = options.gop_size;
    configuration.enable_profiling = options.profile;
    configuration.log_level = options.verbose ? nvcr::LogLevel::info : nvcr::LogLevel::warning;
    nvcr::codec::Components components;
    components.codec = std::move(backend.value());
    return nvcr::Runtime::create(configuration, std::move(components));
}

int encode(const Options& options) {
    auto frame_size = nvcr::frame_size_bytes(
        options.width, options.height, nvcr::PixelFormat::yuv420p8);
    if (!frame_size) {
        std::cerr << "nvcr: " << frame_size.error().describe() << '\n';
        return 1;
    }
    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
        std::cerr << "nvcr: cannot open input: " << options.input << '\n';
        return 1;
    }
    if (options.gop_size == 1 && options.frames > 1) {
        std::cerr << "nvcr: warning: --gop-size 1 encodes every frame as an I-frame; "
                  << "do not compare this development path with warmed I/P GOP throughput\n";
    }
    auto runtime = create_runtime(options);
    if (!runtime) {
        std::cerr << "nvcr: " << runtime.error().describe() << '\n';
        return 1;
    }
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output || !write_sequence_header(output)) {
        std::cerr << "nvcr: cannot create output: " << options.output << '\n';
        return 1;
    }

    std::vector<std::byte> yuv(frame_size.value());
    std::size_t frame_index = 0;
    std::uint64_t payload_bytes = 0;
    std::chrono::nanoseconds codec_time{};
    while (options.frames == 0 || frame_index < options.frames) {
        input.read(reinterpret_cast<char*>(yuv.data()), static_cast<std::streamsize>(yuv.size()));
        const auto read = static_cast<std::size_t>(input.gcount());
        if (read == 0 && input.eof() && options.frames == 0) break;
        if (read != yuv.size()) {
            std::cerr << "nvcr: incomplete raw frame " << frame_index;
            if (options.frames != 0) std::cerr << " (requested " << options.frames << " frames)";
            std::cerr << '\n';
            return 1;
        }
        auto frame = nvcr::Frame::copy_from(
            options.width, options.height, nvcr::PixelFormat::yuv420p8, yuv,
            nvcr::Timestamp{static_cast<nvcr::Timestamp::rep>(
                static_cast<std::int64_t>(frame_index) * options.frame_period_us)});
        if (!frame) {
            std::cerr << "nvcr: " << frame.error().describe() << '\n';
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        auto packet = runtime.value().encode(frame.value());
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (!packet) {
            std::cerr << "nvcr: " << packet.error().describe() << '\n';
            return 1;
        }
        codec_time += elapsed;
        auto wire = nvcr::PacketIO::serialize(packet.value());
        if (!wire) {
            std::cerr << "nvcr: " << wire.error().describe() << '\n';
            return 1;
        }
        if (!write_u64(output, wire.value().size())) {
            std::cerr << "nvcr: failed writing packet length\n";
            return 1;
        }
        output.write(reinterpret_cast<const char*>(wire.value().data()),
                     static_cast<std::streamsize>(wire.value().size()));
        if (!output) {
            std::cerr << "nvcr: failed writing packet " << frame_index << '\n';
            return 1;
        }
        payload_bytes += packet.value().size();
        if (options.verbose) {
            const double milliseconds =
                std::chrono::duration<double, std::milli>(elapsed).count();
            std::cout << "frame " << frame_index << ": encoded " << packet.value().size()
                      << " payload bytes in " << std::fixed << std::setprecision(2)
                      << milliseconds << " ms\n";
        }
        ++frame_index;
    }
    auto flushed = runtime.value().flush();
    if (!flushed) {
        std::cerr << "nvcr: " << flushed.error().describe() << '\n';
        return 1;
    }
    const double seconds = std::chrono::duration<double>(codec_time).count();
    std::cout << "Encoded " << frame_index << " frame(s), " << payload_bytes
              << " payload bytes, codec time " << std::fixed << std::setprecision(3)
              << seconds << " s";
    if (seconds > 0.0) {
        std::cout << " (" << static_cast<double>(frame_index) / seconds << " fps)";
    }
    std::cout << "\nWrote " << options.output << '\n';
    return 0;
}

int decode(const Options& options) {
    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
        std::cerr << "nvcr: cannot open input: " << options.input << '\n';
        return 1;
    }
    if (!read_sequence_header(input)) {
        std::cerr << "nvcr: invalid or unsupported NVCR sequence header\n";
        return 1;
    }
    auto runtime = create_runtime(options);
    if (!runtime) {
        std::cerr << "nvcr: " << runtime.error().describe() << '\n';
        return 1;
    }
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "nvcr: cannot create output: " << options.output << '\n';
        return 1;
    }

    std::vector<std::byte> wire;
    std::size_t frame_index = 0;
    std::uint32_t sequence_width = 0;
    std::uint32_t sequence_height = 0;
    std::chrono::nanoseconds codec_time{};
    while (options.frames == 0 || frame_index < options.frames) {
        const auto status = read_record(input, wire);
        if (status == RecordRead::end) {
            if (options.frames == 0) break;
            std::cerr << "nvcr: bitstream ended after " << frame_index << " frame(s); requested "
                      << options.frames << '\n';
            return 1;
        }
        if (status == RecordRead::error) {
            std::cerr << "nvcr: malformed or truncated packet record " << frame_index << '\n';
            return 1;
        }
        auto packet = nvcr::PacketIO::deserialize(wire);
        if (!packet) {
            std::cerr << "nvcr: " << packet.error().describe() << '\n';
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        auto frame = runtime.value().decode(packet.value());
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (!frame) {
            std::cerr << "nvcr: " << frame.error().describe() << '\n';
            return 1;
        }
        codec_time += elapsed;
        if (frame_index == 0) {
            sequence_width = frame.value().width();
            sequence_height = frame.value().height();
            if ((sequence_width & 1U) != 0 || (sequence_height & 1U) != 0) {
                std::cerr << "nvcr: decoded dimensions are invalid for YUV420p8\n";
                return 1;
            }
        } else if (frame.value().width() != sequence_width ||
                   frame.value().height() != sequence_height) {
            std::cerr << "nvcr: raw YUV output does not support resolution changes\n";
            return 1;
        }
        if (frame.value().pixel_format() == nvcr::PixelFormat::yuv420p8) {
            output.write(reinterpret_cast<const char*>(frame.value().data().data()),
                         static_cast<std::streamsize>(frame.value().data().size()));
        } else if (frame.value().pixel_format() == nvcr::PixelFormat::rgb24) {
            const auto yuv = rgb24_to_yuv420p8(
                frame.value().data(), frame.value().width(), frame.value().height());
            output.write(reinterpret_cast<const char*>(yuv.data()),
                         static_cast<std::streamsize>(yuv.size()));
        } else {
            std::cerr << "nvcr: decoder returned an unsupported pixel format\n";
            return 1;
        }
        if (!output) {
            std::cerr << "nvcr: failed writing decoded frame " << frame_index << '\n';
            return 1;
        }
        if (options.verbose) {
            const double milliseconds =
                std::chrono::duration<double, std::milli>(elapsed).count();
            std::cout << "frame " << frame_index << ": decoded in " << std::fixed
                      << std::setprecision(2) << milliseconds << " ms\n";
        }
        ++frame_index;
    }
    auto flushed = runtime.value().flush();
    if (!flushed) {
        std::cerr << "nvcr: " << flushed.error().describe() << '\n';
        return 1;
    }
    const double seconds = std::chrono::duration<double>(codec_time).count();
    std::cout << "Decoded " << frame_index << " frame(s), codec time " << std::fixed
              << std::setprecision(3) << seconds << " s";
    if (seconds > 0.0) {
        std::cout << " (" << static_cast<double>(frame_index) / seconds << " fps)";
    }
    std::cout << "\nWrote YUV420p8 " << sequence_width << 'x' << sequence_height
              << " to " << options.output << '\n';
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "\n";
        usage(std::cerr);
        return 2;
    }
    if (options.help) {
        usage(std::cout);
        return 0;
    }
    return options.command == Command::encode ? encode(options) : decode(options);
}
