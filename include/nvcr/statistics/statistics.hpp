#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace nvcr {

struct StatisticsSnapshot final {
    std::uint64_t frames_encoded{};
    std::uint64_t frames_decoded{};
    std::uint64_t bytes_encoded{};
    std::chrono::nanoseconds encode_time{};
    std::chrono::nanoseconds decode_time{};

    [[nodiscard]] double encoding_fps() const noexcept;
    [[nodiscard]] double decoding_fps() const noexcept;
    [[nodiscard]] double average_bits_per_frame() const noexcept;
};

class Statistics final {
public:
    void record_encode(std::size_t packet_bytes, std::chrono::nanoseconds duration) noexcept;
    void record_decode(std::chrono::nanoseconds duration) noexcept;
    [[nodiscard]] StatisticsSnapshot snapshot() const noexcept;
    void reset() noexcept;

private:
    std::atomic<std::uint64_t> frames_encoded_{0};
    std::atomic<std::uint64_t> frames_decoded_{0};
    std::atomic<std::uint64_t> bytes_encoded_{0};
    std::atomic<std::int64_t> encode_nanoseconds_{0};
    std::atomic<std::int64_t> decode_nanoseconds_{0};
};

}  // namespace nvcr

