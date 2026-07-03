#pragma once

#include "nvcr/configuration/configuration.hpp"

#include <memory>
#include <string_view>

namespace nvcr {

class Logger {
public:
    virtual ~Logger() = default;
    virtual void log(LogLevel level, std::string_view message) noexcept = 0;
    virtual void set_level(LogLevel level) noexcept = 0;
};

class LoggerFactory {
public:
    virtual ~LoggerFactory() = default;
    [[nodiscard]] virtual std::shared_ptr<Logger> create(std::string_view subsystem) = 0;
};

[[nodiscard]] std::shared_ptr<LoggerFactory> make_default_logger_factory(LogLevel level);

}  // namespace nvcr

