#include "nvcr/common/error.hpp"

namespace nvcr {

Error::Error(ErrorCode code, std::string message, std::string subsystem)
    : code_(code), message_(std::move(message)), subsystem_(std::move(subsystem)) {}

ErrorCode Error::code() const noexcept { return code_; }
std::string_view Error::message() const noexcept { return message_; }
std::string_view Error::subsystem() const noexcept { return subsystem_; }

std::string Error::describe() const {
    if (subsystem_.empty()) {
        return std::string(to_string(code_)) + ": " + message_;
    }
    return subsystem_ + ": " + std::string(to_string(code_)) + ": " + message_;
}

std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::invalid_argument: return "invalid argument";
    case ErrorCode::invalid_state: return "invalid state";
    case ErrorCode::io_error: return "I/O error";
    case ErrorCode::malformed_bitstream: return "malformed bitstream";
    case ErrorCode::resource_exhausted: return "resource exhausted";
    case ErrorCode::dependency_unavailable: return "dependency unavailable";
    case ErrorCode::backend_error: return "backend error";
    case ErrorCode::not_implemented: return "not implemented";
    case ErrorCode::internal_error: return "internal error";
    case ErrorCode::try_again: return "try again";
    case ErrorCode::end_of_stream: return "end of stream";
    case ErrorCode::missing_artifact: return "missing artifact";
    case ErrorCode::incompatible_target: return "incompatible target";
    }
    return "unknown error";
}

}  // namespace nvcr

