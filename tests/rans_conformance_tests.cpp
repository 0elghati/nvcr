#include <nvcr/dcvcrt/rans_codec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

nvcr::dcvcrt::RansCdfTable make_cdf() {
    return {
        {{0, 8192, 32768, 57344, 65536}, {0, 16384, 32768, 49152, 65536}},
        {5, 5},
        {-1, -1},
    };
}

nvcr::dcvcrt::RansCdfTable make_wide_cdf() {
    std::vector<std::int32_t> row(257);
    std::array<std::int32_t, 256> frequencies{};
    frequencies.fill(1);
    constexpr std::int32_t central_frequency = (65536 - 256) / 16;
    for (std::size_t index = 120; index < 136; ++index) {
        frequencies[index] += central_frequency;
    }
    frequencies[135] += 65536 - 256 - central_frequency * 16;
    for (std::size_t index = 0; index < frequencies.size(); ++index) {
        row[index + 1] = row[index] + frequencies[index];
    }
    return {{{std::move(row)}}, {257}, {-128}};
}

std::vector<std::int16_t> combine(
    std::span<const std::int8_t> symbols,
    std::span<const std::uint8_t> indexes) {
    std::vector<std::int16_t> output;
    output.reserve(symbols.size());
    for (std::size_t index = 0; index < symbols.size(); ++index) {
        const auto bits = static_cast<std::uint16_t>(static_cast<std::uint8_t>(symbols[index]));
        output.push_back(static_cast<std::int16_t>((bits << 8U) | indexes[index]));
    }
    return output;
}

template <std::size_t Size>
std::vector<std::byte> bytes(const std::array<std::uint8_t, Size>& values) {
    std::vector<std::byte> output(values.size());
    std::transform(values.begin(), values.end(), output.begin(), [](std::uint8_t value) {
        return static_cast<std::byte>(value);
    });
    return output;
}

void golden_test(bool split) {
    constexpr std::array<std::int8_t, 12> y_values{
        -2, -1, 0, 1, 2, 5, -6, 0, 1, -1, 3, 0};
    constexpr std::array<std::uint8_t, 12> indexes{
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    constexpr std::array<std::int8_t, 12> z_values{
        -1, 0, 1, 2, 0, -1, 1, 3, -2, 0, 1, -1};
    constexpr std::array<std::uint8_t, 14> single_golden{
        0xfd, 0xe3, 0x68, 0x08, 0xdd, 0x78, 0x16,
        0x66, 0x59, 0x42, 0x38, 0x41, 0x16, 0x00};
    constexpr std::array<std::uint8_t, 17> split_golden{
        0x44, 0xf1, 0xca, 0x00, 0xb5, 0x08, 0x5a, 0x80, 0x00,
        0x16, 0x41, 0xe9, 0x49, 0x05, 0x56, 0xf7, 0xca};

    nvcr::dcvcrt::RansCodec codec;
    auto mode = codec.set_use_two_coders(split);
    expect(mode.has_value(), "rANS coder mode is selected");
    auto group = codec.add_cdf(make_cdf());
    expect(group.has_value() && group.value() == 0, "rANS CDF group is installed");
    if (!mode || !group) return;

    const auto combined = combine(y_values, indexes);
    expect(codec.encode_y(combined, group.value()).has_value(), "rANS y symbols encode");
    expect(codec.encode_z(z_values, group.value(), 0, 6).has_value(), "rANS z symbols encode");
    auto encoded = codec.finish_encode();
    expect(encoded.has_value(), "rANS stream finishes");
    if (!encoded) return;

    const auto expected = split ? bytes(split_golden) : bytes(single_golden);
    expect(encoded.value() == expected, "rANS stream is byte-identical to upstream Python");
    expect(codec.set_stream(encoded.value()).has_value(), "rANS stream is accepted for decode");
    auto decoded_y = codec.decode_y(indexes, group.value());
    auto decoded_z = codec.decode_z(z_values.size(), group.value(), 0, 6);
    expect(decoded_y.has_value(), "rANS y symbols decode");
    expect(decoded_z.has_value(), "rANS z symbols decode");
    if (decoded_y) {
        expect(std::ranges::equal(decoded_y.value(), y_values), "decoded y matches upstream");
    }
    if (decoded_z) {
        expect(std::ranges::equal(decoded_z.value(), z_values), "decoded z matches upstream");
    }
    expect(codec.finish_decode().has_value(), "upstream rANS stream ends at its initial state");
}

void wide_cdf_test() {
    std::vector<std::int8_t> symbols(4096);
    std::vector<std::uint8_t> indexes(symbols.size(), 0);
    for (std::size_t index = 0; index < symbols.size(); ++index) {
        symbols[index] = static_cast<std::int8_t>(index % 16) - 8;
    }

    nvcr::dcvcrt::RansCodec codec;
    auto group = codec.add_cdf(make_wide_cdf());
    expect(group.has_value(), "wide rANS CDF is installed");
    if (!group) return;
    const auto combined = combine(symbols, indexes);
    expect(codec.encode_y(combined, group.value()).has_value(), "wide-CDF symbols encode");
    auto encoded = codec.finish_encode();
    expect(encoded.has_value(), "wide-CDF stream finishes");
    if (!encoded) return;
    expect(codec.set_stream(encoded.value()).has_value(), "wide-CDF stream is accepted");
    auto decoded = codec.decode_y(indexes, group.value());
    expect(decoded.has_value(), "wide-CDF symbols decode");
    if (decoded) {
        expect(std::ranges::equal(decoded.value(), symbols), "wide-CDF round trip matches");
    }
}

void validation_test() {
    nvcr::dcvcrt::RansCodec codec;
    nvcr::dcvcrt::RansCdfTable invalid{{{0, 1, 65535}}, {3}, {0}};
    expect(!codec.add_cdf(std::move(invalid)), "invalid CDF is rejected");
    const std::array<std::byte, 3> short_stream{};
    expect(!codec.set_stream(short_stream), "short rANS state is rejected");
    auto finish = codec.finish_decode();
    expect(!finish && finish.error().code() == nvcr::ErrorCode::invalid_state,
           "finishing without a stream is an invalid state");
    // The encoder always flushes a state in [2^23, 2^31).
    const std::array<std::byte, 8> zero_state{};
    auto zero = codec.set_stream(zero_state);
    expect(!zero && zero.error().code() == nvcr::ErrorCode::malformed_bitstream,
           "rANS stream with an impossible initial state is rejected");
}

// Tables shaped like DCVC-RT's Gaussian rows: offset -8, 17 direct values and a
// final escape symbol whose values are coded in bypass mode.
nvcr::dcvcrt::RansCdfTable make_gaussian_table(std::size_t rows, std::int32_t escape_frequency) {
    constexpr std::int32_t direct_values = 17;
    constexpr std::int32_t cdf_size = direct_values + 2;
    nvcr::dcvcrt::RansCdfTable table;
    for (std::size_t row = 0; row < rows; ++row) {
        const double scale = 0.15 * std::pow(1.09, static_cast<double>(row));
        std::array<double, direct_values> weights{};
        double total_weight = 0.0;
        for (std::int32_t value = 0; value < direct_values; ++value) {
            weights[static_cast<std::size_t>(value)] =
                std::exp(-std::abs(value - 8) / scale) + 1e-9;
            total_weight += weights[static_cast<std::size_t>(value)];
        }
        const std::int32_t budget = 65536 - escape_frequency;
        std::array<std::int32_t, direct_values + 1> frequencies{};
        std::int32_t assigned = 0;
        for (std::int32_t value = 0; value < direct_values; ++value) {
            const auto frequency = std::max<std::int32_t>(
                1, static_cast<std::int32_t>(
                       std::floor(budget * weights[static_cast<std::size_t>(value)] / total_weight)));
            frequencies[static_cast<std::size_t>(value)] = frequency;
            assigned += frequency;
        }
        frequencies[8] += budget - assigned;
        frequencies[direct_values] = escape_frequency;
        std::vector<std::int32_t> cdf(cdf_size, 0);
        for (std::size_t index = 0; index + 1 < cdf.size(); ++index) {
            cdf[index + 1] = cdf[index] + frequencies[index];
        }
        table.values.push_back(std::move(cdf));
        table.sizes.push_back(cdf_size);
        table.offsets.push_back(-8);
    }
    return table;
}

struct EntropyCase {
    std::vector<std::int8_t> y;
    std::vector<std::uint8_t> indexes;
    std::vector<std::int8_t> z;
    std::size_t z_per_channel{0};
    std::vector<std::byte> stream;
};

std::int8_t sample_symbol(std::mt19937& random, std::size_t row) {
    // Mostly small values with occasional far tails that force bypass coding.
    const auto draw = random();
    if (draw % 97 == 0) return static_cast<std::int8_t>(static_cast<int>(random() % 256) - 128);
    const auto spread = 1 + static_cast<int>(row / 8);
    return static_cast<std::int8_t>(static_cast<int>(random() % (2 * spread + 1)) - spread);
}

constexpr std::size_t table_rows = 64;

// Encodes z followed by two y parts, matching DCVC-RT's per-frame order.
EntropyCase encode_case(nvcr::dcvcrt::RansCodec& codec, std::size_t group, std::uint32_t seed,
                        std::size_t y_count, std::size_t z_channels, std::size_t z_per_channel) {
    std::mt19937 random(seed);
    EntropyCase value;
    value.z_per_channel = z_per_channel;
    value.y.resize(y_count);
    value.indexes.resize(y_count);
    for (std::size_t index = 0; index < y_count; ++index) {
        value.indexes[index] = static_cast<std::uint8_t>(random() % table_rows);
        value.y[index] = sample_symbol(random, value.indexes[index]);
    }
    value.z.resize(z_channels * z_per_channel);
    for (std::size_t index = 0; index < value.z.size(); ++index) {
        value.z[index] = sample_symbol(random, index / z_per_channel);
    }
    expect(codec.encode_z(value.z, group, 0, z_per_channel).has_value(), "case z encodes");
    const auto half = y_count / 2;
    const auto combined = combine(value.y, value.indexes);
    expect(codec.encode_y(std::span(combined).first(half), group).has_value(), "case y0 encodes");
    expect(codec.encode_y(std::span(combined).subspan(half), group).has_value(), "case y1 encodes");
    auto encoded = codec.finish_encode();
    expect(encoded.has_value(), "case stream finishes");
    if (encoded) value.stream = std::move(encoded.value());
    codec.reset_encoder();
    return value;
}

enum class DecodeOutcome { exact, rejected, silent_corruption };

// Decodes one frame's worth of symbols and classifies the outcome. Crashing or
// reading past the stream is caught by the sanitizer build of this test.
DecodeOutcome decode_case(nvcr::dcvcrt::RansCodec& codec, std::size_t group,
                          const EntropyCase& expected, std::span<const std::byte> stream,
                          std::span<const std::uint8_t> indexes) {
    if (!codec.set_stream(stream)) return DecodeOutcome::rejected;
    auto z = codec.decode_z(expected.z.size(), group, 0, expected.z_per_channel);
    if (!z) {
        expect(z.error().code() == nvcr::ErrorCode::malformed_bitstream,
               "corrupt z decode reports malformed_bitstream");
        return DecodeOutcome::rejected;
    }
    const auto half = indexes.size() / 2;
    auto y0 = codec.decode_y(indexes.first(half), group);
    if (!y0) {
        expect(y0.error().code() == nvcr::ErrorCode::malformed_bitstream,
               "corrupt y decode reports malformed_bitstream");
        return DecodeOutcome::rejected;
    }
    auto y1 = codec.decode_y(indexes.subspan(half), group);
    if (!y1) {
        expect(y1.error().code() == nvcr::ErrorCode::malformed_bitstream,
               "corrupt y decode reports malformed_bitstream");
        return DecodeOutcome::rejected;
    }
    auto finished = codec.finish_decode();
    if (!finished) {
        expect(finished.error().code() == nvcr::ErrorCode::malformed_bitstream,
               "desynchronized stream reports malformed_bitstream");
        return DecodeOutcome::rejected;
    }
    std::vector<std::int8_t> y(y0.value());
    y.insert(y.end(), y1.value().begin(), y1.value().end());
    return y == expected.y && z.value() == expected.z ? DecodeOutcome::exact
                                                      : DecodeOutcome::silent_corruption;
}

struct CodecFixture {
    nvcr::dcvcrt::RansCodec codec;
    std::size_t group{0};
    bool ready{false};

    CodecFixture(bool split, std::int32_t escape_frequency) {
        auto mode = codec.set_use_two_coders(split);
        auto installed = codec.add_cdf(make_gaussian_table(table_rows, escape_frequency));
        ready = mode.has_value() && installed.has_value();
        if (installed) group = installed.value();
        expect(ready, "entropy fixture is configured");
    }
};

void random_round_trip_test(bool split) {
    CodecFixture fixture(split, 64);
    if (!fixture.ready) return;
    for (std::uint32_t seed = 1; seed <= 8; ++seed) {
        const auto value = encode_case(fixture.codec, fixture.group, seed, 6000 + seed, 32, 40);
        expect(decode_case(fixture.codec, fixture.group, value, value.stream, value.indexes) ==
                   DecodeOutcome::exact,
               "random rANS stream round trips and ends at its initial state");
    }
}

void worst_case_bypass_test(bool split) {
    // Every symbol takes the rarest escape and a long bypass; upstream sized the
    // encoder buffer at one byte per symbol, which this overflows.
    CodecFixture fixture(split, 1);
    if (!fixture.ready) return;
    EntropyCase value;
    value.z_per_channel = 16;
    value.z.assign(64 * 16, std::int8_t{-128});
    value.y.resize(4096);
    value.indexes.resize(value.y.size());
    for (std::size_t index = 0; index < value.y.size(); ++index) {
        value.y[index] = index % 2 == 0 ? std::int8_t{127} : std::int8_t{-128};
        value.indexes[index] = static_cast<std::uint8_t>(index % table_rows);
    }
    const auto half = value.y.size() / 2;
    const auto combined = combine(value.y, value.indexes);
    expect(fixture.codec.encode_z(value.z, fixture.group, 0, value.z_per_channel).has_value(),
           "worst-case z encodes");
    expect(fixture.codec.encode_y(std::span(combined).first(half), fixture.group).has_value(),
           "worst-case y0 encodes");
    expect(fixture.codec.encode_y(std::span(combined).subspan(half), fixture.group).has_value(),
           "worst-case y1 encodes");
    auto encoded = fixture.codec.finish_encode();
    expect(encoded.has_value(), "worst-case stream finishes");
    if (!encoded) return;
    value.stream = std::move(encoded.value());
    const auto symbols = value.y.size() + value.z.size();
    expect(value.stream.size() > symbols, "worst-case stream exceeds one byte per symbol");
    expect(decode_case(fixture.codec, fixture.group, value, value.stream, value.indexes) ==
               DecodeOutcome::exact,
           "worst-case bypass stream round trips");
    fixture.codec.reset_encoder();
}

void expect_never_silent(DecodeOutcome outcome, std::string_view message, std::size_t& rejected) {
    expect(outcome != DecodeOutcome::silent_corruption, message);
    if (outcome == DecodeOutcome::rejected) ++rejected;
}

void truncation_test(bool split) {
    CodecFixture fixture(split, 64);
    if (!fixture.ready) return;
    const auto value = encode_case(fixture.codec, fixture.group, 101, 3000, 16, 30);
    std::size_t rejected = 0;
    std::size_t trials = 0;
    for (std::size_t size = 0; size < value.stream.size(); size += 1 + size / 64) {
        ++trials;
        expect_never_silent(
            decode_case(fixture.codec, fixture.group, value,
                        std::span(value.stream).first(size), value.indexes),
            "truncated rANS stream never decodes silently", rejected);
    }
    expect(rejected == trials, "every truncated rANS stream is rejected");
    std::vector<std::byte> extended(value.stream);
    extended.push_back(std::byte{0});
    expect(decode_case(fixture.codec, fixture.group, value, extended, value.indexes) ==
               DecodeOutcome::rejected,
           "rANS stream with trailing bytes is rejected");
}

void corruption_test(bool split) {
    // Random bit flips must never crash or read past the stream. Most desync
    // the coder and are rejected. A flip that lands only in raw bypass bits
    // changes a tail value without desynchronizing; detecting that needs a
    // stream checksum, which the entropy layer does not carry.
    CodecFixture fixture(split, 64);
    if (!fixture.ready) return;
    const auto value = encode_case(fixture.codec, fixture.group, 202, 3000, 16, 30);
    std::mt19937 random(303);
    constexpr std::size_t trials = 300;
    std::size_t rejected = 0;
    for (std::size_t trial = 0; trial < trials; ++trial) {
        auto corrupted = value.stream;
        const auto position = random() % corrupted.size();
        corrupted[position] ^= static_cast<std::byte>(1U << (random() % 8));
        if (decode_case(fixture.codec, fixture.group, value, corrupted, value.indexes) ==
            DecodeOutcome::rejected) {
            ++rejected;
        }
    }
    expect(rejected * 10 >= trials * 9, "at least 90% of corrupted rANS streams are rejected");
}

void index_mismatch_test(bool split) {
    // The cross-GPU failure: the decoder computes a neighbouring entropy index
    // for one symbol. It must fail cleanly instead of reconstructing garbage.
    CodecFixture fixture(split, 64);
    if (!fixture.ready) return;
    const auto value = encode_case(fixture.codec, fixture.group, 404, 6000, 16, 30);
    std::size_t rejected = 0;
    std::size_t trials = 0;
    for (std::size_t position = 0; position < value.indexes.size(); position += 37) {
        auto indexes = value.indexes;
        indexes[position] = static_cast<std::uint8_t>(
            indexes[position] == table_rows - 1 ? indexes[position] - 1 : indexes[position] + 1);
        ++trials;
        expect_never_silent(
            decode_case(fixture.codec, fixture.group, value, value.stream, indexes),
            "neighbouring entropy index never decodes silently", rejected);
    }
    expect(rejected > trials / 2, "most neighbouring entropy indexes are rejected");
    expect(decode_case(fixture.codec, fixture.group, value, value.stream, value.indexes) ==
               DecodeOutcome::exact,
           "decoder recovers for the next correctly indexed stream");
}

}  // namespace

int main() {
    golden_test(false);
    golden_test(true);
    wide_cdf_test();
    validation_test();
    for (const bool split : {false, true}) {
        random_round_trip_test(split);
        worst_case_bypass_test(split);
        truncation_test(split);
        corruption_test(split);
        index_mismatch_test(split);
    }
    if (failures == 0) std::cout << "DCVC-RT rANS conformance passed\n";
    return failures == 0 ? 0 : 1;
}
