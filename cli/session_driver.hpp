#pragma once

#include <nvcr/runtime/runtime.hpp>

#include <chrono>
#include <cstddef>
#include <limits>

namespace nvcr::cli {

enum class DrainState {
    needs_input,
    end_of_stream,
    output_limit,
};

namespace detail {

template <class Operation>
auto time_codec_operation(Operation&& operation, std::chrono::nanoseconds& codec_time) {
    const auto start = std::chrono::steady_clock::now();
    auto result = operation();
    codec_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start);
    return result;
}

template <class Receive, class Emit>
Result<DrainState> drain_outputs(
    Receive&& receive,
    Emit& emit,
    bool flushed,
    std::size_t output_limit,
    std::chrono::nanoseconds& codec_time) {
    std::size_t output_count = 0;
    while (output_count < output_limit) {
        auto output = time_codec_operation(receive, codec_time);
        if (output) {
            auto emitted = emit(std::move(output).value());
            if (!emitted) return emitted.error();
            ++output_count;
            continue;
        }
        if (output.error().code() == ErrorCode::try_again) {
            if (!flushed) return DrainState::needs_input;
            return Error(
                ErrorCode::invalid_state,
                "codec returned try_again after flush",
                "cli-session-driver");
        }
        if (output.error().code() == ErrorCode::end_of_stream && flushed) {
            return DrainState::end_of_stream;
        }
        return output.error();
    }
    return DrainState::output_limit;
}

}  // namespace detail

template <class Emit>
Result<DrainState> send_frame_and_drain(
    Runtime& runtime,
    const Frame& frame,
    Emit& emit,
    std::chrono::nanoseconds& codec_time) {
    auto sent = detail::time_codec_operation(
        [&] { return runtime.send_frame(frame); }, codec_time);
    if (!sent) return sent.error();
    return detail::drain_outputs(
        [&] { return runtime.receive_access_unit(); },
        emit,
        false,
        std::numeric_limits<std::size_t>::max(),
        codec_time);
}

template <class Emit>
Result<DrainState> flush_and_drain_encoder(
    Runtime& runtime,
    Emit& emit,
    std::chrono::nanoseconds& codec_time) {
    auto flushed = detail::time_codec_operation(
        [&] { return runtime.flush_encoder(); }, codec_time);
    if (!flushed) return flushed.error();
    return detail::drain_outputs(
        [&] { return runtime.receive_access_unit(); },
        emit,
        true,
        std::numeric_limits<std::size_t>::max(),
        codec_time);
}

template <class Emit>
Result<DrainState> send_access_unit_and_drain(
    Runtime& runtime,
    const Packet& packet,
    Emit& emit,
    std::size_t output_limit,
    std::chrono::nanoseconds& codec_time) {
    auto sent = detail::time_codec_operation(
        [&] { return runtime.send_access_unit(packet); }, codec_time);
    if (!sent) return sent.error();
    return detail::drain_outputs(
        [&] { return runtime.receive_frame(); },
        emit,
        false,
        output_limit,
        codec_time);
}

template <class Emit>
Result<DrainState> flush_and_drain_decoder(
    Runtime& runtime,
    Emit& emit,
    std::size_t output_limit,
    std::chrono::nanoseconds& codec_time) {
    auto flushed = detail::time_codec_operation(
        [&] { return runtime.flush_decoder(); }, codec_time);
    if (!flushed) return flushed.error();
    return detail::drain_outputs(
        [&] { return runtime.receive_frame(); },
        emit,
        true,
        output_limit,
        codec_time);
}

}  // namespace nvcr::cli
