#include <nvcr/dcvcrt/rans_codec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
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
}

}  // namespace

int main() {
    golden_test(false);
    golden_test(true);
    wide_cdf_test();
    validation_test();
    if (failures == 0) std::cout << "DCVC-RT rANS conformance passed\n";
    return failures == 0 ? 0 : 1;
}
