#include <nvcr/dcvcrt/rans_codec.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t cdf_rows = 128;
constexpr std::size_t y_stage_symbols = 256 * 68 * 120 / 4;
constexpr std::size_t z_channel_symbols = 17 * 30;
constexpr std::size_t z_channels = 128;

nvcr::dcvcrt::RansCdfTable make_cdf() {
    nvcr::dcvcrt::RansCdfTable table;
    table.values.resize(cdf_rows);
    table.sizes.resize(cdf_rows, 257);
    table.offsets.resize(cdf_rows, -128);
    for (auto& row : table.values) {
        row.resize(257);
        std::array<std::int32_t, 256> frequencies{};
        frequencies.fill(1);
        constexpr std::int32_t central_frequency = (65536 - 256) / 16;
        for (std::size_t index = 120; index < 136; ++index) {
            frequencies[index] += central_frequency;
        }
        frequencies[135] += 65536 - 256 - central_frequency * 16;
        row[0] = 0;
        for (std::size_t index = 0; index < frequencies.size(); ++index) {
            row[index + 1] = row[index] + frequencies[index];
        }
    }
    return table;
}

std::uint32_t next_random(std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

std::vector<std::int16_t> make_y_symbols(
    std::size_t count,
    std::uint32_t& random_state) {
    std::vector<std::int16_t> output(count);
    for (auto& combined : output) {
        const auto symbol = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(next_random(random_state) % 16) - 8);
        const auto row = static_cast<std::uint8_t>(next_random(random_state) % cdf_rows);
        combined = static_cast<std::int16_t>(
            (static_cast<std::uint16_t>(symbol) << 8U) | row);
    }
    return output;
}

std::vector<std::uint8_t> indexes_of(std::span<const std::int16_t> combined) {
    std::vector<std::uint8_t> output(combined.size());
    std::transform(combined.begin(), combined.end(), output.begin(), [](std::int16_t value) {
        return static_cast<std::uint8_t>(value);
    });
    return output;
}

template <typename T>
bool require(const nvcr::Result<T>& result, std::string_view operation) {
    if (result) return true;
    std::cerr << operation << ": " << result.error().message() << '\n';
    return false;
}

bool run_case(bool two_coders, std::size_t iterations) {
    nvcr::dcvcrt::RansCodec codec;
    auto mode = codec.set_use_two_coders(two_coders);
    auto group = codec.add_cdf(make_cdf());
    if (!require(mode, "set coder mode") || !require(group, "add CDF")) return false;

    std::uint32_t random_state = 0x4e564352U;
    std::array<std::vector<std::int16_t>, 4> y;
    std::array<std::vector<std::uint8_t>, 4> indexes;
    for (std::size_t stage = 0; stage < y.size(); ++stage) {
        y[stage] = make_y_symbols(y_stage_symbols, random_state);
        indexes[stage] = indexes_of(y[stage]);
    }
    std::vector<std::int8_t> z(z_channels * z_channel_symbols);
    for (auto& symbol : z) {
        symbol = static_cast<std::int8_t>(next_random(random_state) % 16) - 8;
    }

    std::size_t encoded_bytes = 0;
    const auto run_once = [&]() -> bool {
        codec.reset_encoder();
        auto encoded_z = codec.encode_z(z, group.value(), 0, z_channel_symbols);
        if (!require(encoded_z, "encode z")) return false;
        for (const auto& stage : y) {
            auto encoded_y = codec.encode_y(stage, group.value());
            if (!require(encoded_y, "encode y")) return false;
        }
        auto stream = codec.finish_encode();
        if (!require(stream, "finish encode")) return false;
        encoded_bytes = stream.value().size();

        auto stream_set = codec.set_stream(stream.value());
        if (!require(stream_set, "set stream")) return false;
        auto decoded_z = codec.decode_z(z.size(), group.value(), 0, z_channel_symbols);
        if (!require(decoded_z, "decode z")) return false;
        for (const auto& stage : indexes) {
            auto decoded_y = codec.decode_y(stage, group.value());
            if (!require(decoded_y, "decode y")) return false;
        }
        return true;
    };

    if (!run_once()) return false;
    const auto started = Clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        if (!run_once()) return false;
    }
    const auto elapsed = Clock::now() - started;
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    const auto symbols_per_iteration = z.size() + y.size() * y_stage_symbols;
    const auto mega_symbols_per_second =
        static_cast<double>(symbols_per_iteration * iterations) / seconds / 1'000'000.0;
    const auto milliseconds = seconds * 1000.0 / static_cast<double>(iterations);
    std::cout << (two_coders ? "two coders" : "one coder ")
              << ": " << milliseconds << " ms/round-trip, "
              << mega_symbols_per_second << " Msymbol/s, "
              << encoded_bytes << " bytes\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t iterations = 5;
    if (argc == 2) {
        const auto parsed = std::strtoul(argv[1], nullptr, 10);
        if (parsed == 0) {
            std::cerr << "iteration count must be positive\n";
            return 2;
        }
        iterations = parsed;
    }
    return run_case(false, iterations) && run_case(true, iterations) ? 0 : 1;
}
