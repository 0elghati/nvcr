#pragma once

#include <string>
#include <string_view>
#include <optional>
#include <utility>
#include <variant>

namespace nvcr {

enum class ErrorCode {
    invalid_argument,
    invalid_state,
    io_error,
    malformed_bitstream,
    resource_exhausted,
    dependency_unavailable,
    backend_error,
    not_implemented,
    internal_error,
    // Session-API status values (Phase 2).  Returned by send/receive calls;
    // these are not error conditions but signals about the codec pipeline state.
    try_again,            // no output available yet; send more input
    end_of_stream,        // codec has emitted all output after flush()
    missing_artifact,     // required engine/model artifact is absent
    incompatible_target,  // provider or target device is not compatible
};

class Error final {
public:
    Error(ErrorCode code, std::string message, std::string subsystem = {});

    [[nodiscard]] ErrorCode code() const noexcept;
    [[nodiscard]] std::string_view message() const noexcept;
    [[nodiscard]] std::string_view subsystem() const noexcept;
    [[nodiscard]] std::string describe() const;

private:
    ErrorCode code_;
    std::string message_;
    std::string subsystem_;
};

template <typename T>
class [[nodiscard]] Result final {
public:
    Result(T value) : storage_(std::move(value)) {}
    Result(Error error) : storage_(std::move(error)) {}

    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(storage_);
    }
    explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T& value() & { return std::get<T>(storage_); }
    [[nodiscard]] const T& value() const& { return std::get<T>(storage_); }
    [[nodiscard]] T&& value() && { return std::get<T>(std::move(storage_)); }
    [[nodiscard]] Error& error() & { return std::get<Error>(storage_); }
    [[nodiscard]] const Error& error() const& { return std::get<Error>(storage_); }

private:
    std::variant<T, Error> storage_;
};

template <>
class [[nodiscard]] Result<void> final {
public:
    Result() = default;
    Result(Error error) : error_(std::move(error)) {}

    [[nodiscard]] bool has_value() const noexcept { return !error_.has_value(); }
    explicit operator bool() const noexcept { return has_value(); }
    [[nodiscard]] Error& error() & { return *error_; }
    [[nodiscard]] const Error& error() const& { return *error_; }

private:
    std::optional<Error> error_;
};

[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

}  // namespace nvcr
