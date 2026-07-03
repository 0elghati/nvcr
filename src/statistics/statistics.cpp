#include "nvcr/statistics/statistics.hpp"

namespace nvcr {
namespace {

double fps(std::uint64_t frame_count, std::chrono::nanoseconds elapsed) noexcept {
    if (frame_count == 0 || elapsed.count() <= 0) {
        return 0.0;
    }
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    return static_cast<double>(frame_count) / seconds;
}

}  // namespace

double StatisticsSnapshot::encoding_fps() const noexcept {
    return fps(frames_encoded, encode_time);
}

double StatisticsSnapshot::decoding_fps() const noexcept {
    return fps(frames_decoded, decode_time);
}

double StatisticsSnapshot::average_bits_per_frame() const noexcept {
    if (frames_encoded == 0) {
        return 0.0;
    }
    return static_cast<double>(bytes_encoded) * 8.0 / static_cast<double>(frames_encoded);
}

void Statistics::record_encode(
    std::size_t packet_bytes, std::chrono::nanoseconds duration) noexcept {
    frames_encoded_.fetch_add(1, std::memory_order_relaxed);
    bytes_encoded_.fetch_add(static_cast<std::uint64_t>(packet_bytes), std::memory_order_relaxed);
    encode_nanoseconds_.fetch_add(duration.count(), std::memory_order_relaxed);
}

void Statistics::record_decode(std::chrono::nanoseconds duration) noexcept {
    frames_decoded_.fetch_add(1, std::memory_order_relaxed);
    decode_nanoseconds_.fetch_add(duration.count(), std::memory_order_relaxed);
}

StatisticsSnapshot Statistics::snapshot() const noexcept {
    return {
        frames_encoded_.load(std::memory_order_relaxed),
        frames_decoded_.load(std::memory_order_relaxed),
        bytes_encoded_.load(std::memory_order_relaxed),
        std::chrono::nanoseconds(encode_nanoseconds_.load(std::memory_order_relaxed)),
        std::chrono::nanoseconds(decode_nanoseconds_.load(std::memory_order_relaxed)),
    };
}

void Statistics::reset() noexcept {
    frames_encoded_.store(0, std::memory_order_relaxed);
    frames_decoded_.store(0, std::memory_order_relaxed);
    bytes_encoded_.store(0, std::memory_order_relaxed);
    encode_nanoseconds_.store(0, std::memory_order_relaxed);
    decode_nanoseconds_.store(0, std::memory_order_relaxed);
}

}  // namespace nvcr
