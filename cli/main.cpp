#include <nvcr/dcvcrt/adapter.hpp>
#include <nvcr/dcvcrt/tensorrt_backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <array>
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

enum class Command {
    encode,
    decode,
    codec_list,
    codec_describe,
    provider_list,
    provider_describe,
    compatibility_check,
};

struct Options final {
    Command command{Command::encode};
    fs::path input;
    fs::path output;
    fs::path engine_dir;
    fs::path quality_reference;
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
    std::string codec_id;
    std::string provider_id;
};

void usage(std::ostream& out) {
    out << "NVCR native encoder and decoder\n\n"
        << "Usage:\n"
        << "  nvcr encode -i INPUT.yuv -o OUTPUT.nvcr -s WIDTHxHEIGHT\n"
        << "              [--backend NAME] [--engine-profile NAME] [--frames N] [--qp N]\n"
        << "              [--gop-size N] [-r FPS] [--engine-dir DIR]\n"
        << "  nvcr decode -i INPUT.nvcr -o OUTPUT.yuv [--quality-metrics REFERENCE.yuv]\n"
        << "              [--backend NAME] [--engine-profile NAME] [--frames N]\n"
        << "              [--device-id N] [--engine-dir DIR]\n"
        << "  nvcr codec list\n"
        << "  nvcr codec describe CODEC_ID\n"
        << "  nvcr provider list\n"
        << "  nvcr provider describe PROVIDER_ID\n"
        << "  nvcr compatibility check --codec CODEC_ID --provider PROVIDER_ID\n\n"
        << "Input and output video use planar 8-bit YUV 4:2:0. A frame count of zero\n"
        << "means process until end of input. Normal encoding uses the configured I/P\n"
        << "GOP; --gop-size 1 explicitly requests all-intra development mode.\n\n"
        << "Options:\n"
        << "  -i, --input FILE          Input file\n"
        << "  -o, --output FILE         Output file\n"
        << "  -s, --size WIDTHxHEIGHT   Raw input dimensions (encode only)\n"
        << "      --width N             Raw input width (encode only)\n"
        << "      --height N            Raw input height (encode only)\n"
        << "      --backend NAME        Installed backend selector (default: NVCR_BACKEND or default)\n"
        << "      --engine-profile NAME Installed engine profile, for example 720p\n"
        << "                            (default: inferred from encoded dimensions)\n"
        << "      --engine-dir DIR      Custom TensorRT engine and entropy asset directory\n"
        << "                            (overrides backend/profile selection)\n"
        << "      --frames N            Frames to process; 0 means all (default: 0)\n"
        << "      --qp N                I-frame QP (default: 32)\n"
        << "      --gop-size N          GOP size; 1 explicitly requests all-intra (default: 32)\n"
        << "  -r, --fps FPS             Input frame rate (default: 60)\n"
        << "      --frame-period-us N   Exact timestamp interval (alternative to --fps)\n"
        << "      --device-id N         CUDA device (default: 0)\n"
        << "      --profile             Print TensorRT/CUDA per-frame profiling counters\n"
        << "      --quality-metrics FILE  Report decoded Y/U/V and weighted PSNR against FILE\n"
        << "  -v, --verbose             Print per-frame CLI progress and info logs\n"
        << "  -h, --help                Show this help\n\n"
        << "Without an explicit engine override, encode selects a profile from -s and\n"
        << "decode selects it from dimensions embedded in the first access unit.\n";
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
    if (const char* engine_root = std::getenv("NVCR_ENGINE_ROOT")) {
        if (*engine_root != '\0') return fs::path(engine_root);
    }
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
    return backend;
}

std::string engine_profile_for_dimensions(std::uint32_t width, std::uint32_t height) {
    if (width == 640U && height == 360U) return "360p";
    if (width == 960U && height == 540U) return "540p";
    if (width <= 176U && height <= 144U) return "qcif";
    if (width <= 352U && height <= 288U) return "cif";
    if (width <= 1280U && height <= 720U) return "720p";
    if (width <= 1920U && height <= 1080U) return "1080p";
    return {};
}

bool engine_bundle_exists(const fs::path& path) {
    std::error_code error;
    return fs::is_regular_file(path / "engine_manifest.json", error);
}

fs::path engine_profile_dir(
    const fs::path& root, std::string_view backend, std::string_view profile) {
    constexpr std::string_view precision_suffix = "-fp16";
    std::string optimization_point(profile);
    if (optimization_point.ends_with(precision_suffix)) {
        optimization_point.resize(optimization_point.size() - precision_suffix.size());
    }
    const auto installed = root / "profiles" / backend / optimization_point;
    if (engine_bundle_exists(installed)) return installed;

    const auto legacy_installed =
        root / "profiles" / backend / (optimization_point + std::string(precision_suffix));
    if (engine_bundle_exists(legacy_installed)) return legacy_installed;

    const auto build_tree = root / (std::string(backend) + "-" + optimization_point);
    if (engine_bundle_exists(build_tree)) return build_tree;

    return installed;
}

fs::path resolve_engine_dir(
    const Options& options, std::uint32_t width, std::uint32_t height) {
    if (!options.engine_dir.empty()) return options.engine_dir;
    if (const char* engine_dir = std::getenv("NVCR_ENGINE_DIR")) {
        if (*engine_dir != '\0') return fs::path(engine_dir);
    }

    const auto root = default_engine_root();
    const std::string backend = options.backend == "default" ? "dcvcrt" : options.backend;
    const std::string profile = options.engine_profile.empty()
        ? engine_profile_for_dimensions(width, height)
        : options.engine_profile;
    if (profile.empty()) return {};
    return engine_profile_dir(root, backend, profile);
}

void bootstrap_registry() {
    nvcr::dcvcrt::register_codec();
    nvcr::dcvcrt::register_tensorrt_provider();
}

int codec_list_command() {
    bootstrap_registry();
    const auto codecs = nvcr::runtime::Registry::instance().codecs();
    if (codecs.empty()) {
        std::cout << "No codecs registered\n";
        return 0;
    }
    std::cout << "Registered codecs:\n";
    for (const auto& codec : codecs) {
        std::cout << "  " << codec.descriptor.id << " ("
                  << codec.descriptor.display_name << ")\n";
    }
    return 0;
}

int codec_describe_command(std::string_view codec_id) {
    bootstrap_registry();
    const auto entry = nvcr::runtime::Registry::instance().find_codec(codec_id);
    if (!entry) {
        std::cerr << "nvcr: unknown codec: " << codec_id << '\n';
        return 1;
    }
    std::cout << "Codec: " << entry->descriptor.id << "\n"
              << "  name: " << entry->descriptor.display_name << "\n"
              << "  api version: " << entry->descriptor.api_version << "\n"
              << "  supports intra: "
              << (entry->capabilities.supports_intra ? "yes" : "no") << "\n"
              << "  supports predicted: "
              << (entry->capabilities.supports_predicted ? "yes" : "no") << "\n"
              << "  supports delayed output: "
              << (entry->capabilities.supports_delayed_output ? "yes" : "no") << "\n"
              << "  qp range: [" << entry->capabilities.min_qp << ", "
              << entry->capabilities.max_qp << "]\n"
              << "  encoder options: " << entry->encoder_options.declarations.size() << "\n"
              << "  decoder options: " << entry->decoder_options.declarations.size() << '\n';
    return 0;
}

int provider_list_command() {
    bootstrap_registry();
    const auto providers = nvcr::runtime::Registry::instance().providers();
    if (providers.empty()) {
        std::cout << "No providers registered\n";
        return 0;
    }
    std::cout << "Registered providers:\n";
    for (const auto& provider : providers) {
        std::cout << "  " << provider.descriptor.id << " ("
                  << provider.descriptor.display_name << ", version "
                  << provider.descriptor.version << ")\n";
    }
    return 0;
}

int provider_describe_command(std::string_view provider_id) {
    bootstrap_registry();
    const auto entry = nvcr::runtime::Registry::instance().find_provider(provider_id);
    if (!entry) {
        std::cerr << "nvcr: unknown provider: " << provider_id << '\n';
        return 1;
    }
    std::cout << "Provider: " << entry->descriptor.id << "\n"
              << "  name: " << entry->descriptor.display_name << "\n"
              << "  version: " << entry->descriptor.version << "\n"
              << "  fp16: " << (entry->capabilities.supports_fp16 ? "yes" : "no") << "\n"
              << "  int8: " << (entry->capabilities.supports_int8 ? "yes" : "no") << "\n"
              << "  dynamic shapes: "
              << (entry->capabilities.supports_dynamic_shapes ? "yes" : "no") << "\n"
              << "  cuda graphs: "
              << (entry->capabilities.supports_cuda_graphs ? "yes" : "no") << "\n"
              << "  target device: " << entry->capabilities.target_device << '\n';
    return 0;
}

int compatibility_check_command(std::string_view codec_id, std::string_view provider_id) {
    bootstrap_registry();
    const bool ok =
        nvcr::runtime::Registry::instance().compatible(codec_id, provider_id);
    if (ok) {
        std::cout << "compatible: codec '" << codec_id << "' with provider '"
                  << provider_id << "'\n";
        return 0;
    }
    std::cout << "incompatible: codec '" << codec_id << "' with provider '"
              << provider_id << "'\n";
    return 1;
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
    } else if (command == "codec") {
        if (argc < 3) {
            std::cerr << "nvcr: codec requires a subcommand: list | describe\n";
            return false;
        }
        const std::string_view subcommand = argv[2];
        if (subcommand == "list") {
            options.command = Command::codec_list;
        } else if (subcommand == "describe") {
            if (argc < 4) {
                std::cerr << "nvcr: codec describe requires CODEC_ID\n";
                return false;
            }
            options.command = Command::codec_describe;
            options.codec_id = argv[3];
        } else {
            std::cerr << "nvcr: unknown codec subcommand: " << subcommand << '\n';
            return false;
        }
        return true;
    } else if (command == "provider") {
        if (argc < 3) {
            std::cerr << "nvcr: provider requires a subcommand: list | describe\n";
            return false;
        }
        const std::string_view subcommand = argv[2];
        if (subcommand == "list") {
            options.command = Command::provider_list;
        } else if (subcommand == "describe") {
            if (argc < 4) {
                std::cerr << "nvcr: provider describe requires PROVIDER_ID\n";
                return false;
            }
            options.command = Command::provider_describe;
            options.provider_id = argv[3];
        } else {
            std::cerr << "nvcr: unknown provider subcommand: " << subcommand << '\n';
            return false;
        }
        return true;
    } else if (command == "compatibility") {
        if (argc < 3 || std::string_view(argv[2]) != "check") {
            std::cerr << "nvcr: compatibility requires subcommand: check\n";
            return false;
        }
        options.command = Command::compatibility_check;
    } else {
        std::cerr << "nvcr: unknown command: " << command << '\n';
        return false;
    }

    int argument_index = 2;
    if (options.command == Command::compatibility_check) {
        argument_index = 3;  // skip the `check` subcommand token
    }

    for (int index = argument_index; index < argc; ++index) {
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
        } else if (argument == "--quality-metrics") {
            options.quality_reference = value_after(argument);
            if (options.quality_reference.empty()) return false;
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--verbose" || argument == "-v") {
            options.verbose = true;
        } else if (argument == "--codec") {
            options.codec_id = std::string(value_after(argument));
            if (options.codec_id.empty()) return false;
        } else if (argument == "--provider") {
            options.provider_id = std::string(value_after(argument));
            if (options.provider_id.empty()) return false;
        } else {
            std::cerr << "nvcr: unknown option: " << argument << '\n';
            return false;
        }
    }

    if (options.command == Command::compatibility_check) {
        if (options.codec_id.empty() || options.provider_id.empty()) {
            std::cerr << "nvcr: compatibility check requires --codec and --provider\n";
            return false;
        }
        return true;
    }

    if (const char* backend = std::getenv("NVCR_BACKEND")) {
        if (*backend != '\0' && options.backend == "default") options.backend = backend;
    }
    if (const char* profile = std::getenv("NVCR_ENGINE_PROFILE")) {
        if (*profile != '\0' && options.engine_profile.empty()) options.engine_profile = profile;
    }
    constexpr std::string_view legacy_precision_suffix = "-fp16";
    if (options.engine_profile.ends_with(legacy_precision_suffix)) {
        const auto legacy = options.engine_profile;
        options.engine_profile.resize(
            options.engine_profile.size() - legacy_precision_suffix.size());
        std::cerr << "nvcr: warning: engine profile '" << legacy
                  << "' is deprecated; use '" << options.engine_profile << "'\n";
    }
    options.backend = normalize_backend(std::move(options.backend));
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
    if (options.command == Command::encode && !options.quality_reference.empty()) {
        std::cerr << "nvcr: --quality-metrics is available only for decode\n";
        return false;
    }
    if (!options.quality_reference.empty() && options.quality_reference == options.output) {
        std::cerr << "nvcr: quality reference and output must be different files\n";
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

struct QualityAccumulator final {
    std::array<long double, 3> squared_error{};
    std::array<std::uint64_t, 3> sample_count{};
    std::size_t frame_count{};

    bool add(
        std::span<const std::byte> reference,
        std::span<const std::byte> reconstructed,
        std::uint32_t width,
        std::uint32_t height) {
        const auto y_size = static_cast<std::size_t>(width) * height;
        const std::array<std::size_t, 3> plane_sizes{y_size, y_size / 4U, y_size / 4U};
        const auto expected_size = y_size + y_size / 2U;
        if (reference.size() != expected_size || reconstructed.size() != expected_size) {
            return false;
        }
        std::size_t offset = 0;
        for (std::size_t plane = 0; plane < plane_sizes.size(); ++plane) {
            for (std::size_t index = 0; index < plane_sizes[plane]; ++index) {
                const auto expected = std::to_integer<int>(reference[offset + index]);
                const auto actual = std::to_integer<int>(reconstructed[offset + index]);
                const auto difference = expected - actual;
                squared_error[plane] += static_cast<long double>(difference * difference);
            }
            sample_count[plane] += plane_sizes[plane];
            offset += plane_sizes[plane];
        }
        ++frame_count;
        return true;
    }

    [[nodiscard]] double psnr(std::size_t plane) const {
        if (sample_count[plane] == 0U) return 0.0;
        if (squared_error[plane] == 0.0L) return std::numeric_limits<double>::infinity();
        const auto mse = squared_error[plane] / static_cast<long double>(sample_count[plane]);
        return 10.0 * std::log10(65025.0 / static_cast<double>(mse));
    }
};

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

nvcr::Result<nvcr::Runtime> create_runtime(
    const Options& options, std::uint32_t width, std::uint32_t height) {
    const auto engine_dir = resolve_engine_dir(options, width, height);
    if (engine_dir.empty()) {
        return nvcr::Error(
            nvcr::ErrorCode::invalid_argument,
            "no automatic TensorRT engine profile covers " + std::to_string(width) + "x" +
                std::to_string(height) +
                "; use --engine-profile or --engine-dir for a custom bundle",
            "cli");
    }
    if (!engine_bundle_exists(engine_dir)) {
        const std::string profile = options.engine_profile.empty()
            ? engine_profile_for_dimensions(width, height)
            : options.engine_profile;
        return nvcr::Error(
            nvcr::ErrorCode::dependency_unavailable,
            "no installed TensorRT engine bundle for profile " + profile +
                "; run: nvcr-artifacts install --profile " + profile,
            "cli");
    }
    if (options.verbose) {
        std::cout << "Using TensorRT engine bundle: " << engine_dir << '\n';
    }
    auto adapter = nvcr::dcvcrt::make_adapter();
    if (!adapter) return adapter.error();
    nvcr::RuntimeConfiguration configuration;
    configuration.intra_engine_path = engine_dir;
    configuration.device_id = options.device_id;
    configuration.intra_qp = options.qp;
    configuration.gop_size = options.gop_size;
    configuration.enable_profiling = options.profile;
    configuration.log_level = options.verbose ? nvcr::LogLevel::info : nvcr::LogLevel::warning;
    auto components = adapter.value()->create_components(configuration);
    if (!components) return components.error();
    return nvcr::Runtime::create(configuration, std::move(components.value()));
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
    auto runtime = create_runtime(options, options.width, options.height);
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

    std::vector<std::byte> wire;
    const auto first_status = read_record(input, wire);
    if (first_status != RecordRead::record) {
        std::cerr << "nvcr: NVCR sequence contains no complete packet records\n";
        return 1;
    }
    auto first_packet = nvcr::PacketIO::deserialize(wire);
    if (!first_packet) {
        std::cerr << "nvcr: " << first_packet.error().describe() << '\n';
        return 1;
    }
    std::uint32_t encoded_width = 0;
    std::uint32_t encoded_height = 0;
    if (nvcr::AccessUnitIO::has_magic(first_packet.value().data())) {
        auto access_unit = nvcr::AccessUnitIO::deserialize(first_packet.value().data());
        if (!access_unit) {
            std::cerr << "nvcr: " << access_unit.error().describe() << '\n';
            return 1;
        }
        encoded_width = access_unit.value().width;
        encoded_height = access_unit.value().height;
    }
    auto runtime = create_runtime(options, encoded_width, encoded_height);
    if (!runtime) {
        std::cerr << "nvcr: " << runtime.error().describe() << '\n';
        return 1;
    }
    std::ifstream quality_reference;
    if (!options.quality_reference.empty()) {
        quality_reference.open(options.quality_reference, std::ios::binary);
        if (!quality_reference) {
            std::cerr << "nvcr: cannot open quality reference: "
                      << options.quality_reference << "\n";
            return 1;
        }
    }
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "nvcr: cannot create output: " << options.output << '\n';
        return 1;
    }

    std::size_t frame_index = 0;
    std::uint32_t sequence_width = 0;
    std::uint32_t sequence_height = 0;
    std::chrono::nanoseconds codec_time{};
    QualityAccumulator quality;
    std::vector<std::byte> reference_yuv;
    bool have_record = true;
    while (options.frames == 0 || frame_index < options.frames) {
        if (!have_record) {
            const auto status = read_record(input, wire);
            if (status == RecordRead::end) {
                if (options.frames == 0) break;
                std::cerr << "nvcr: bitstream ended after " << frame_index
                          << " frame(s); requested " << options.frames << '\n';
                return 1;
            }
            if (status == RecordRead::error) {
                std::cerr << "nvcr: malformed or truncated packet record " << frame_index << '\n';
                return 1;
            }
        }
        have_record = false;
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
        if (frame.value().pixel_format() != nvcr::PixelFormat::yuv420p8 &&
            frame.value().pixel_format() != nvcr::PixelFormat::rgb24) {
            std::cerr << "nvcr: decoder returned an unsupported pixel format\n";
            return 1;
        }
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
        std::vector<std::byte> converted_yuv;
        std::span<const std::byte> yuv = frame.value().data();
        if (frame.value().pixel_format() == nvcr::PixelFormat::rgb24) {
            converted_yuv = rgb24_to_yuv420p8(
                frame.value().data(), frame.value().width(), frame.value().height());
            yuv = converted_yuv;
        }
        if (quality_reference.is_open()) {
            reference_yuv.resize(yuv.size());
            quality_reference.read(
                reinterpret_cast<char*>(reference_yuv.data()),
                static_cast<std::streamsize>(reference_yuv.size()));
            if (static_cast<std::size_t>(quality_reference.gcount()) != reference_yuv.size()) {
                std::cerr << "nvcr: quality reference ended before decoded frame "
                          << frame_index << "\n";
                return 1;
            }
            if (!quality.add(reference_yuv, yuv, frame.value().width(), frame.value().height())) {
                std::cerr << "nvcr: quality metric frame dimensions are invalid\n";
                return 1;
            }
        }
        output.write(reinterpret_cast<const char*>(yuv.data()),
                     static_cast<std::streamsize>(yuv.size()));
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
    if (quality.frame_count > 0U) {
        const auto y_psnr = quality.psnr(0);
        const auto u_psnr = quality.psnr(1);
        const auto v_psnr = quality.psnr(2);
        const auto weighted_psnr = (6.0 * y_psnr + u_psnr + v_psnr) / 8.0;
        std::cout << "\nQuality " << quality.frame_count
                  << " frame(s): PSNR-Y " << std::setprecision(6) << y_psnr
                  << " dB, PSNR-U " << u_psnr
                  << " dB, PSNR-V " << v_psnr
                  << " dB, PSNR-YUV " << weighted_psnr << " dB";
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
    switch (options.command) {
    case Command::encode: return encode(options);
    case Command::decode: return decode(options);
    case Command::codec_list: return codec_list_command();
    case Command::codec_describe: return codec_describe_command(options.codec_id);
    case Command::provider_list: return provider_list_command();
    case Command::provider_describe: return provider_describe_command(options.provider_id);
    case Command::compatibility_check:
        return compatibility_check_command(options.codec_id, options.provider_id);
    }
    std::cerr << "nvcr: unreachable command dispatch\n";
    return 2;
}
