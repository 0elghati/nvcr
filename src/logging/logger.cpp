#include "nvcr/logging/logger.hpp"

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#if defined(NVCR_HAS_SPDLOG)
#include <spdlog/sinks/stderr_color_sinks.h>
#include <spdlog/spdlog.h>
#endif

namespace nvcr {
namespace {

#if defined(NVCR_HAS_SPDLOG)
spdlog::level::level_enum to_spdlog(LogLevel level) {
    switch (level) {
    case LogLevel::trace: return spdlog::level::trace;
    case LogLevel::debug: return spdlog::level::debug;
    case LogLevel::info: return spdlog::level::info;
    case LogLevel::warning: return spdlog::level::warn;
    case LogLevel::error: return spdlog::level::err;
    case LogLevel::off: return spdlog::level::off;
    }
    return spdlog::level::info;
}

class SpdlogLogger final : public Logger {
public:
    SpdlogLogger(
        std::string name,
        LogLevel level,
        std::shared_ptr<spdlog::sinks::sink> sink)
        : logger_(std::make_shared<spdlog::logger>(std::move(name), std::move(sink))) {
        set_level(level);
    }

    void log(LogLevel level, std::string_view message) noexcept override {
        try {
            logger_->log(to_spdlog(level), "{}", message);
        } catch (...) {
        }
    }

    void set_level(LogLevel level) noexcept override { logger_->set_level(to_spdlog(level)); }

private:
    std::shared_ptr<spdlog::logger> logger_;
};
#else
class ConsoleLogger final : public Logger {
public:
    ConsoleLogger(std::string subsystem, LogLevel level)
        : subsystem_(std::move(subsystem)), level_(level) {}

    void log(LogLevel level, std::string_view message) noexcept override {
        if (level < level_ || level_ == LogLevel::off) {
            return;
        }
        try {
            std::scoped_lock lock(mutex_);
            std::clog << "[nvcr." << subsystem_ << "] [" << to_string(level) << "] "
                      << message << '\n';
        } catch (...) {
        }
    }

    void set_level(LogLevel level) noexcept override { level_ = level; }

private:
    std::string subsystem_;
    LogLevel level_;
    std::mutex mutex_;
};
#endif

class DefaultLoggerFactory final : public LoggerFactory {
public:
    explicit DefaultLoggerFactory(LogLevel level) : level_(level)
#if defined(NVCR_HAS_SPDLOG)
        , sink_(std::make_shared<spdlog::sinks::stderr_color_sink_mt>())
#endif
    {}

    std::shared_ptr<Logger> create(std::string_view subsystem) override {
#if defined(NVCR_HAS_SPDLOG)
        return std::make_shared<SpdlogLogger>(std::string(subsystem), level_, sink_);
#else
        return std::make_shared<ConsoleLogger>(std::string(subsystem), level_);
#endif
    }

private:
    LogLevel level_;
#if defined(NVCR_HAS_SPDLOG)
    std::shared_ptr<spdlog::sinks::sink> sink_;
#endif
};

}  // namespace

std::shared_ptr<LoggerFactory> make_default_logger_factory(LogLevel level) {
    return std::make_shared<DefaultLoggerFactory>(level);
}

}  // namespace nvcr
