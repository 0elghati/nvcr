#pragma once

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace nvcr {

enum class LogLevel : std::uint8_t {
    trace,
    debug,
    info,
    warning,
    error,
    off,
};

enum class TensorRTExecutionMode : std::uint8_t {
    automatic,
    low_memory,
    performance,
};

struct RuntimeConfiguration final {
    std::filesystem::path intra_engine_path;
    std::filesystem::path predicted_engine_path;
    std::filesystem::path entropy_model_path;
    std::string model_id{"dcvcrt-cvpr2025"};
    std::int32_t device_id{0};
    std::uint32_t intra_qp{32};
    std::uint32_t gop_size{32};
    std::size_t memory_pool_bytes{256U * 1024U * 1024U};
    std::size_t device_arena_bytes{512U * 1024U * 1024U};
    std::size_t max_packet_bytes{64U * 1024U * 1024U};
    LogLevel log_level{LogLevel::info};
    TensorRTExecutionMode tensorrt_execution_mode{TensorRTExecutionMode::automatic};
    bool allow_legacy_access_units{true};
    bool verify_encoder_reconstruction{false};
    bool enable_profiling{false};
};

class ConfigurationLoader final {
public:
    [[nodiscard]] static Result<RuntimeConfiguration> from_file(
        const std::filesystem::path& path);
    [[nodiscard]] static Result<void> validate(const RuntimeConfiguration& configuration);
};

[[nodiscard]] std::string_view to_string(LogLevel level) noexcept;
[[nodiscard]] std::string_view to_string(TensorRTExecutionMode mode) noexcept;

}  // namespace nvcr

