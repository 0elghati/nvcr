#include "nvcr/configuration/configuration.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

namespace nvcr {
namespace {

std::string trim(std::string value) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
    value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
    return value;
}

template <typename T>
Result<T> parse_integer(std::string_view value, std::string_view key) {
    T output{};
    const auto result = std::from_chars(value.data(), value.data() + value.size(), output);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        return Error(
            ErrorCode::invalid_argument,
            "invalid integer for key '" + std::string(key) + "'",
            "configuration");
    }
    return output;
}

Result<bool> parse_bool(std::string_view value, std::string_view key) {
    if (value == "true" || value == "1") {
        return true;
    }
    if (value == "false" || value == "0") {
        return false;
    }
    return Error(
        ErrorCode::invalid_argument,
        "invalid boolean for key '" + std::string(key) + "'",
        "configuration");
}

Result<LogLevel> parse_log_level(std::string_view value) {
    if (value == "trace") return LogLevel::trace;
    if (value == "debug") return LogLevel::debug;
    if (value == "info") return LogLevel::info;
    if (value == "warning" || value == "warn") return LogLevel::warning;
    if (value == "error") return LogLevel::error;
    if (value == "off") return LogLevel::off;
    return Error(ErrorCode::invalid_argument, "invalid log_level", "configuration");
}

Result<TensorRTExecutionMode> parse_execution_mode(std::string_view value) {
    if (value == "automatic" || value == "auto") {
        return TensorRTExecutionMode::automatic;
    }
    if (value == "low-memory" || value == "low_memory") {
        return TensorRTExecutionMode::low_memory;
    }
    if (value == "performance") return TensorRTExecutionMode::performance;
    return Error(ErrorCode::invalid_argument, "invalid tensorrt_execution_mode", "configuration");
}

bool valid_identifier(std::string_view value) {
    return !value.empty() && value.size() <= 128U &&
        std::ranges::all_of(value, [](unsigned char character) {
            return std::isalnum(character) != 0 || character == '.' ||
                character == '_' || character == '-';
        });
}

Result<void> apply_setting(
    RuntimeConfiguration& configuration,
    std::string_view key,
    std::string_view value) {
    if (key == "intra_engine_path") {
        configuration.intra_engine_path = value;
    } else if (key == "predicted_engine_path") {
        configuration.predicted_engine_path = value;
    } else if (key == "entropy_model_path") {
        configuration.entropy_model_path = value;
    } else if (key == "model_id") {
        configuration.model_id = value;
    } else if (key == "bitstream_model_id") {
        configuration.bitstream_model_id = value;
    } else if (key == "provider_id") {
        configuration.provider_id = value;
    } else if (key == "device_id") {
        auto parsed = parse_integer<std::int32_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.device_id = parsed.value();
    } else if (key == "intra_qp") {
        auto parsed = parse_integer<std::uint32_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.intra_qp = parsed.value();
    } else if (key == "gop_size") {
        auto parsed = parse_integer<std::uint32_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.gop_size = parsed.value();
    } else if (key == "memory_pool_bytes") {
        auto parsed = parse_integer<std::size_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.memory_pool_bytes = parsed.value();
    } else if (key == "device_arena_bytes") {
        auto parsed = parse_integer<std::size_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.device_arena_bytes = parsed.value();
    } else if (key == "max_packet_bytes") {
        auto parsed = parse_integer<std::size_t>(value, key);
        if (!parsed) return parsed.error();
        configuration.max_packet_bytes = parsed.value();
    } else if (key == "enable_profiling") {
        auto parsed = parse_bool(value, key);
        if (!parsed) return parsed.error();
        configuration.enable_profiling = parsed.value();
    } else if (key == "allow_legacy_access_units") {
        auto parsed = parse_bool(value, key);
        if (!parsed) return parsed.error();
        configuration.allow_legacy_access_units = parsed.value();
    } else if (key == "verify_encoder_reconstruction") {
        auto parsed = parse_bool(value, key);
        if (!parsed) return parsed.error();
        configuration.verify_encoder_reconstruction = parsed.value();
    } else if (key == "tensorrt_execution_mode") {
        auto parsed = parse_execution_mode(value);
        if (!parsed) return parsed.error();
        configuration.tensorrt_execution_mode = parsed.value();
    } else if (key == "log_level") {
        auto parsed = parse_log_level(value);
        if (!parsed) return parsed.error();
        configuration.log_level = parsed.value();
    } else {
        return Error(
            ErrorCode::invalid_argument,
            "unknown setting '" + std::string(key) + "'",
            "configuration");
    }
    return {};
}

}  // namespace

Result<RuntimeConfiguration> ConfigurationLoader::from_file(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        return Error(
            ErrorCode::io_error,
            "cannot open configuration file: " + path.string(),
            "configuration");
    }

    RuntimeConfiguration configuration;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(std::move(line));
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            return Error(
                ErrorCode::invalid_argument,
                "expected key=value at line " + std::to_string(line_number),
                "configuration");
        }
        const auto key = trim(line.substr(0, separator));
        const auto value = trim(line.substr(separator + 1U));
        auto applied = apply_setting(configuration, key, value);
        if (!applied) {
            return applied.error();
        }
    }
    if (input.bad()) {
        return Error(ErrorCode::io_error, "failed while reading configuration", "configuration");
    }

    auto valid = validate(configuration);
    if (!valid) {
        return valid.error();
    }
    return configuration;
}

Result<void> ConfigurationLoader::validate(const RuntimeConfiguration& configuration) {
    if (!valid_identifier(configuration.model_id)) {
        return Error(ErrorCode::invalid_argument, "invalid model_id", "configuration");
    }
    if (!valid_identifier(configuration.bitstream_model_id)) {
        return Error(ErrorCode::invalid_argument, "invalid bitstream_model_id", "configuration");
    }
    if (!valid_identifier(configuration.provider_id)) {
        return Error(ErrorCode::invalid_argument, "invalid provider_id", "configuration");
    }
    if (configuration.device_id < 0) {
        return Error(ErrorCode::invalid_argument, "device_id cannot be negative", "configuration");
    }
    if (configuration.gop_size == 0) {
        return Error(ErrorCode::invalid_argument, "gop_size must be greater than zero", "configuration");
    }
    if (configuration.intra_qp >= 64) {
        return Error(ErrorCode::invalid_argument, "intra_qp must be in [0, 63]", "configuration");
    }
    if (configuration.memory_pool_bytes == 0 || configuration.device_arena_bytes == 0 || configuration.max_packet_bytes == 0) {
        return Error(
            ErrorCode::invalid_argument,
            "memory and packet limits must be greater than zero",
            "configuration");
    }
    return {};
}

std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
    case LogLevel::trace: return "trace";
    case LogLevel::debug: return "debug";
    case LogLevel::info: return "info";
    case LogLevel::warning: return "warning";
    case LogLevel::error: return "error";
    case LogLevel::off: return "off";
    }
    return "unknown";
}

std::string_view to_string(TensorRTExecutionMode mode) noexcept {
    switch (mode) {
    case TensorRTExecutionMode::automatic: return "automatic";
    case TensorRTExecutionMode::low_memory: return "low-memory";
    case TensorRTExecutionMode::performance: return "performance";
    }
    return "unknown";
}

}  // namespace nvcr
