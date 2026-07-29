#include "nvcr/runtime/runtime.hpp"

#include "nvcr/codec/runtime.hpp"
#include "nvcr/logging/logger.hpp"
#include "nvcr/memory/memory_pool.hpp"

#include <mutex>
#include <new>
#include <utility>

namespace nvcr {

struct Runtime::Impl final {
    Impl(
        RuntimeConfiguration runtime_configuration,
        codec::Components components,
        std::shared_ptr<LoggerFactory> logger_factory)
        : configuration(std::move(runtime_configuration)),
          logger(logger_factory->create("runtime")),
          stats(std::make_shared<Statistics>()),
          host_pool(make_host_memory_resource(), configuration.memory_pool_bytes),
          codec(configuration, std::move(components), std::move(logger_factory), stats) {}

    RuntimeConfiguration configuration;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<Statistics> stats;
    MemoryPool host_pool;
    codec::Runtime codec;
    mutable std::mutex mutex;
    RuntimeState state{RuntimeState::initializing};
};

Runtime::Runtime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result<Runtime> Runtime::create(
    RuntimeConfiguration configuration, codec::Components components) {
    auto valid = ConfigurationLoader::validate(configuration);
    if (!valid) {
        return valid.error();
    }
    if (!components.codec) {
        return Error(
            ErrorCode::dependency_unavailable,
            "a codec backend must be provided",
            "runtime");
    }

    try {
        auto logger_factory = make_default_logger_factory(configuration.log_level);
        auto impl = std::make_unique<Impl>(
            std::move(configuration), std::move(components), std::move(logger_factory));
        auto initialized = impl->codec.initialize();
        if (!initialized) {
            impl->state = RuntimeState::failed;
            return initialized.error();
        }
        impl->state = RuntimeState::ready;
        impl->logger->log(LogLevel::info, "NVCR runtime is ready");
        return Runtime(std::move(impl));
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate runtime resources",
            "runtime");
    } catch (const std::exception& exception) {
        return Error(ErrorCode::internal_error, exception.what(), "runtime");
    }
}

Result<Packet> Runtime::encode(const Frame& frame) {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->codec.encode(frame);
    } catch (const std::exception& exception) {
        return Error(ErrorCode::backend_error, exception.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<Frame> Runtime::decode(const Packet& packet) {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->codec.decode(packet);
    } catch (const std::exception& exception) {
        return Error(ErrorCode::backend_error, exception.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<void> Runtime::flush() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->codec.flush();
    } catch (const std::exception& exception) {
        return Error(ErrorCode::backend_error, exception.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<void> Runtime::reset() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    impl_->codec.reset();
    return {};
}

RuntimeState Runtime::state() const noexcept {
    if (!impl_) {
        return RuntimeState::stopped;
    }
    std::scoped_lock lock(impl_->mutex);
    return impl_->state;
}

StatisticsSnapshot Runtime::statistics() const noexcept {
    if (!impl_) {
        return {};
    }
    return impl_->stats->snapshot();
}

}  // namespace nvcr
