#include "nvcr/runtime/runtime.hpp"

#include "nvcr/logging/logger.hpp"
#include "nvcr/memory/memory_pool.hpp"
#include "nvcr/runtime/registry.hpp"

#include <mutex>
#include <new>
#include <utility>

namespace nvcr {

struct Runtime::Impl final {
    Impl(
        RuntimeConfiguration runtime_configuration,
        codec::Sessions sessions,
        std::shared_ptr<LoggerFactory> logger_factory)
        : configuration(std::move(runtime_configuration)),
          logger(logger_factory->create("runtime")),
          stats(sessions.statistics ? std::move(sessions.statistics) :
                                      std::make_shared<Statistics>()),
          host_pool(make_host_memory_resource(), configuration.runtime.memory_pool_bytes),
          encoder(std::move(sessions.encoder)),
          decoder(std::move(sessions.decoder)) {}

    RuntimeConfiguration configuration;
    std::shared_ptr<Logger> logger;
    std::shared_ptr<Statistics> stats;
    MemoryPool host_pool;
    std::unique_ptr<IEncoderSession> encoder;
    std::unique_ptr<IDecoderSession> decoder;
    mutable std::mutex mutex;
    RuntimeState state{RuntimeState::initializing};
};

Runtime::Runtime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result<Runtime> Runtime::create(RuntimeConfiguration configuration) {
    auto valid = ConfigurationLoader::validate(configuration);
    if (!valid) return valid.error();

    auto& registry = runtime::Registry::instance();
    auto adapter = registry.create_codec(configuration.codec.id);
    if (!adapter) return adapter.error();
    if (!registry.compatible(configuration.codec.id, configuration.provider.id)) {
        return Error(
            ErrorCode::missing_provider,
            "codec/provider selection is not registered: " + configuration.codec.id +
                "/" + configuration.provider.id,
            "runtime");
    }

    runtime::RuntimeServices services(registry, configuration.provider.id);
    auto provider_session = services.create_provider_session(configuration);
    if (!provider_session) return provider_session.error();
    auto sessions = adapter.value()->create_sessions(
        configuration, std::move(provider_session.value()));
    if (!sessions) return sessions.error();
    return create(std::move(configuration), std::move(sessions.value()));
}

Result<Runtime> Runtime::create(
    RuntimeConfiguration configuration, codec::Sessions sessions) {
    auto valid = ConfigurationLoader::validate(configuration);
    if (!valid) {
        return valid.error();
    }
    if (!sessions.encoder || !sessions.decoder) {
        return Error(
            ErrorCode::dependency_unavailable,
            "codec encoder and decoder sessions must be provided",
            "runtime");
    }

    try {
        auto logger_factory = make_default_logger_factory(configuration.runtime.log_level);
        auto impl = std::make_unique<Impl>(
            std::move(configuration), std::move(sessions), std::move(logger_factory));
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

// ---------------------------------------------------------------------------
// IEncoderSession
// ---------------------------------------------------------------------------

Result<void> Runtime::send_frame(const Frame& frame) {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    if (frame.size_bytes() == 0) {
        return Error(ErrorCode::invalid_argument, "cannot encode an empty frame", "runtime");
    }
    try {
        return impl_->encoder->send_frame(frame);
    } catch (const std::exception& e) {
        return Error(ErrorCode::backend_error, e.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<Packet> Runtime::receive_access_unit() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    return impl_->encoder->receive_access_unit();
}

// ---------------------------------------------------------------------------
// IDecoderSession
// ---------------------------------------------------------------------------

Result<void> Runtime::send_access_unit(const Packet& packet) {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->decoder->send_access_unit(packet);
    } catch (const std::exception& e) {
        return Error(ErrorCode::backend_error, e.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<Frame> Runtime::receive_frame() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    return impl_->decoder->receive_frame();
}

// ---------------------------------------------------------------------------
// Shared flush / reset (both directions)
// ---------------------------------------------------------------------------

Result<void> Runtime::flush() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        auto encoder = impl_->encoder->flush();
        auto decoder = impl_->decoder->flush();
        if (!encoder) return encoder.error();
        return decoder;
    } catch (const std::exception& e) {
        return Error(ErrorCode::backend_error, e.what(), "runtime");
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
    auto encoder = impl_->encoder->reset();
    auto decoder = impl_->decoder->reset();
    if (!encoder) return encoder.error();
    return decoder;
}

// ---------------------------------------------------------------------------
// Compatibility aliases (preserved for CLI / tests that call encode/decode)
// ---------------------------------------------------------------------------

Result<Packet> Runtime::encode(const Frame& frame) {
    auto sent = send_frame(frame);
    if (!sent) return sent.error();
    return receive_access_unit();
}

Result<Frame> Runtime::decode(const Packet& packet) {
    auto sent = send_access_unit(packet);
    if (!sent) return sent.error();
    return receive_frame();
}

Result<void> Runtime::flush_encoder() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->encoder->flush();
    } catch (const std::exception& e) {
        return Error(ErrorCode::backend_error, e.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<void> Runtime::reset_encoder() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    return impl_->encoder->reset();
}

Result<void> Runtime::flush_decoder() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    try {
        return impl_->decoder->flush();
    } catch (const std::exception& e) {
        return Error(ErrorCode::backend_error, e.what(), "runtime");
    } catch (...) {
        return Error(ErrorCode::backend_error, "unknown backend failure", "runtime");
    }
}

Result<void> Runtime::reset_decoder() {
    if (!impl_) {
        return Error(ErrorCode::invalid_state, "runtime was moved from", "runtime");
    }
    std::scoped_lock lock(impl_->mutex);
    if (impl_->state != RuntimeState::ready) {
        return Error(ErrorCode::invalid_state, "runtime is not ready", "runtime");
    }
    return impl_->decoder->reset();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

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
