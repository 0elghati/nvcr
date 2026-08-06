#include "nvcr/runtime/runtime.hpp"

#include "nvcr/codec/runtime.hpp"
#include "nvcr/dcvcrt/backend.hpp"
#include "nvcr/dcvcrt/tensorrt_backend.hpp"
#include "nvcr/logging/logger.hpp"
#include "nvcr/memory/memory_pool.hpp"

#include <mutex>
#include <new>
#include <optional>
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

    // Pending output queued by send_frame / send_access_unit.  DCVC-RT is
    // one-frame/one-unit so at most one item is ever pending, but the queue
    // keeps the send/receive split clean for future multi-output codecs.
    std::optional<Packet> pending_encoded;
    std::optional<Frame>  pending_decoded;
    bool encoder_flushed{false};
    bool decoder_flushed{false};
};

Runtime::Runtime(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

Result<Runtime> Runtime::create(
    RuntimeConfiguration configuration, codec::Components components) {
    // Bootstrap built-in registry entries before validation and initialization.
    dcvcrt::register_codec();
#if defined(NVCR_HAS_TENSORRT)
    dcvcrt::register_tensorrt_provider();
#endif

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
        auto result = impl_->codec.encode(frame);
        if (!result) return result.error();
        impl_->pending_encoded = std::move(result.value());
        impl_->encoder_flushed = false;
        return {};
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
    if (impl_->pending_encoded) {
        Packet p = std::move(*impl_->pending_encoded);
        impl_->pending_encoded.reset();
        return p;
    }
    if (impl_->encoder_flushed) {
        return Error(ErrorCode::end_of_stream, "encoder has been flushed", "runtime");
    }
    return Error(ErrorCode::try_again, "no encoded output available yet", "runtime");
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
        auto result = impl_->codec.decode(packet);
        if (!result) return result.error();
        impl_->pending_decoded = std::move(result.value());
        impl_->decoder_flushed = false;
        return {};
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
    if (impl_->pending_decoded) {
        Frame f = std::move(*impl_->pending_decoded);
        impl_->pending_decoded.reset();
        return f;
    }
    if (impl_->decoder_flushed) {
        return Error(ErrorCode::end_of_stream, "decoder has been flushed", "runtime");
    }
    return Error(ErrorCode::try_again, "no decoded output available yet", "runtime");
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
        auto result = impl_->codec.flush();
        if (!result) return result;
        impl_->encoder_flushed = true;
        impl_->decoder_flushed = true;
        return {};
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
    impl_->codec.reset();
    impl_->pending_encoded.reset();
    impl_->pending_decoded.reset();
    impl_->encoder_flushed = false;
    impl_->decoder_flushed = false;
    return {};
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

