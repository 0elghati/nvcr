#include "uf_rans_codec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <span>
#include <vector>

namespace {

nvcr::dcvcuf::UfRansCdfTable make_cdf(std::size_t rows) {
    nvcr::dcvcuf::UfRansCdfTable table;
    table.values.resize(rows, {0, 32768, 65536});
    table.sizes.resize(rows, 3);
    return table;
}

void check(bool value) {
    if (!value) std::abort();
}

void round_trip(std::size_t parallel) {
    nvcr::dcvcuf::UfRansCodec encoder;
    check(encoder.set_entropy_coder_parallel(parallel).has_value());
    auto z_group = encoder.add_cdf(make_cdf(64));
    auto y_group = encoder.add_cdf(make_cdf(64));
    check(z_group.has_value() && z_group.value() == 0);
    check(y_group.has_value() && y_group.value() == 1);

    const std::size_t y_count = parallel * 7;
    std::vector<std::int16_t> y(y_count);
    for (std::size_t i = 0; i < y.size(); ++i) {
        const auto value = static_cast<std::int8_t>((i & 1U) ? -1 : 1);
        y[i] = static_cast<std::int16_t>(
            static_cast<std::uint16_t>(static_cast<std::uint8_t>(value)) << 8);
    }
    check(encoder.encode_y(y, y_group.value()).has_value());

    constexpr std::size_t channels = 8;
    const std::size_t z_count = parallel * channels * 2;
    std::vector<std::int8_t> z(z_count);
    for (std::size_t i = 0; i < z.size(); ++i)
        z[i] = static_cast<std::int8_t>((static_cast<int>(i * 3) % 11) - 5);
    check(encoder.encode_z(z, z_group.value(), 0, channels).has_value());

    auto stream = encoder.finish_encode();
    check(stream.has_value() && !stream.value().empty());

    nvcr::dcvcuf::UfRansCodec decoder;
    check(decoder.set_entropy_coder_parallel(parallel).has_value());
    auto decoder_z_group = decoder.add_cdf(make_cdf(64));
    auto decoder_y_group = decoder.add_cdf(make_cdf(64));
    check(decoder_z_group.has_value() && decoder_y_group.has_value());
    check(decoder.set_stream(stream.value()).has_value());

    auto decoded_z =
        decoder.decode_z(z.size(), decoder_z_group.value(), 0, channels);
    check(decoded_z.has_value() && decoded_z.value() == z);

    const std::vector<std::uint8_t> indexes(y.size(), 0);
    auto decoded_y = decoder.decode_y(indexes, decoder_y_group.value());
    std::vector<std::int8_t> expected_y(y.size());
    for (std::size_t i = 0; i < expected_y.size(); ++i)
        expected_y[i] = static_cast<std::int8_t>((i & 1U) ? -1 : 1);
    check(decoded_y.has_value() && decoded_y.value() == expected_y);
    check(decoder.finish_decode().has_value());

    encoder.reset_encoder();
    check(encoder.encode_y(y, y_group.value()).has_value());
    check(encoder.encode_z(z, z_group.value(), 0, channels).has_value());
    auto repeated = encoder.finish_encode();
    check(repeated.has_value() && repeated.value() == stream.value());
}

void rejects_truncated_stream() {
    nvcr::dcvcuf::UfRansCodec decoder;
    check(decoder.set_entropy_coder_parallel(1).has_value());
    check(decoder.add_cdf(make_cdf(256)).has_value());
    const std::vector<std::byte> stream(4, std::byte{0});
    check(decoder.set_stream(stream).has_value());
    auto decoded = decoder.decode_z(128, 0, 0, 1);
    check(!decoded.has_value() &&
          decoded.error().code() == nvcr::ErrorCode::malformed_bitstream);
}

// Rows of different widths so CDF row choice matters, as with UF's scales.
nvcr::dcvcuf::UfRansCdfTable make_shaped_cdf(std::size_t rows) {
    nvcr::dcvcuf::UfRansCdfTable table;
    constexpr std::size_t width = 19;  // UF's CDF storage width
    for (std::size_t row = 0; row < rows; ++row) {
        const auto size = static_cast<std::int32_t>(3 + row % (width - 2));
        std::vector<std::int32_t> values(width, 0);
        // A geometric-ish shape whose spread grows with the row, like UF's
        // scale-indexed Gaussian tables.
        const double decay = 0.25 + 0.7 * static_cast<double>(row) / static_cast<double>(rows);
        std::vector<double> mass(static_cast<std::size_t>(size - 1));
        double total = 0;
        for (std::size_t i = 0; i < mass.size(); ++i) {
            mass[i] = 1.0;
            for (std::size_t k = 0; k < i; ++k) mass[i] *= decay;
            total += mass[i];
        }
        std::int32_t cumulative = 0;
        const auto symbols = static_cast<std::int32_t>(mass.size());
        for (std::int32_t i = 0; i < symbols; ++i) {
            const auto remaining = symbols - i;
            auto frequency = static_cast<std::int32_t>(
                mass[static_cast<std::size_t>(i)] / total * (65536 - symbols));
            frequency = std::max(frequency, 1);
            frequency = std::min(frequency, 65536 - cumulative - (remaining - 1));
            if (i + 1 == symbols) frequency = 65536 - cumulative;
            cumulative += frequency;
            values[static_cast<std::size_t>(i + 1)] = cumulative;
        }
        for (std::size_t i = static_cast<std::size_t>(size); i < width; ++i)
            values[i] = 65536 + static_cast<std::int32_t>(i);  // unused padding
        table.values.push_back(std::move(values));
        table.sizes.push_back(size);
    }
    return table;
}

struct Fixture {
    std::size_t parallel{};
    std::vector<std::int8_t> z;
    std::vector<std::int8_t> y;
    std::vector<std::uint8_t> rows;  // y entropy indexes
    std::vector<std::byte> stream;
};

constexpr std::size_t z_rows = 64;
constexpr std::size_t z_channels = 16;
constexpr std::size_t y_rows = 128;

Fixture encode_fixture(std::size_t parallel, std::uint32_t seed) {
    std::mt19937 random(seed);
    Fixture fixture;
    fixture.parallel = parallel;
    std::geometric_distribution<int> magnitude(0.35);
    std::uniform_int_distribution<int> sign(0, 1);
    std::uniform_int_distribution<int> rare(0, 63);
    const auto symbol = [&] {
        // Mostly small values, some needing the bypass path, a few extremes.
        int value = std::min(magnitude(random), 127);
        if (rare(random) == 0) value = 100 + rare(random);
        if (sign(random)) value = -value - (rare(random) == 1 ? 1 : 0);
        return static_cast<std::int8_t>(std::max(-128, std::min(127, value)));
    };
    fixture.z.resize(z_channels * 24);
    for (auto& value : fixture.z) value = symbol();
    fixture.y.resize(3000);
    fixture.rows.resize(fixture.y.size());
    std::uniform_int_distribution<int> row(0, y_rows - 1);
    for (std::size_t i = 0; i < fixture.y.size(); ++i) {
        fixture.y[i] = symbol();
        fixture.rows[i] = static_cast<std::uint8_t>(row(random));
    }

    nvcr::dcvcuf::UfRansCodec encoder;
    check(encoder.set_entropy_coder_parallel(parallel).has_value());
    check(encoder.add_cdf(make_shaped_cdf(z_rows)).has_value());
    check(encoder.add_cdf(make_shaped_cdf(y_rows)).has_value());
    std::vector<std::int16_t> packed(fixture.y.size());
    for (std::size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<std::int16_t>(
            (static_cast<std::uint16_t>(static_cast<std::uint8_t>(fixture.y[i])) << 8) |
            fixture.rows[i]);
    // Upstream order: y first, then z, into one stream per coder.
    check(encoder.encode_y(packed, 1).has_value());
    check(encoder.encode_z(fixture.z, 0, 0, z_channels).has_value());
    auto stream = encoder.finish_encode();
    check(stream.has_value());
    fixture.stream = std::move(stream.value());
    return fixture;
}

enum class Decoded { exact, rejected, silent_garbage };

// Decodes `stream` as a UF decoder would and classifies the result. Exact
// means every call succeeded, the end-of-stream check passed and the symbols
// equal the encoder's; rejected means some step returned malformed_bitstream.
Decoded decode_fixture(const Fixture& fixture, std::span<const std::byte> stream,
                       std::span<const std::uint8_t> rows, std::size_t y_count) {
    nvcr::dcvcuf::UfRansCodec decoder;
    check(decoder.set_entropy_coder_parallel(fixture.parallel).has_value());
    check(decoder.add_cdf(make_shaped_cdf(z_rows)).has_value());
    check(decoder.add_cdf(make_shaped_cdf(y_rows)).has_value());
    const auto rejected = [](const nvcr::Error& error) {
        check(error.code() == nvcr::ErrorCode::malformed_bitstream);
        return Decoded::rejected;
    };
    if (auto set = decoder.set_stream(stream); !set) return rejected(set.error());
    auto z = decoder.decode_z(fixture.z.size(), 0, 0, z_channels);
    if (!z) return rejected(z.error());
    auto y = decoder.decode_y(rows.first(y_count), 1);
    if (!y) return rejected(y.error());
    if (auto finished = decoder.finish_decode(); !finished) return rejected(finished.error());
    const bool same = z.value() == fixture.z && y_count == fixture.y.size() &&
                      std::equal(y.value().begin(), y.value().end(), fixture.y.begin());
    return same ? Decoded::exact : Decoded::silent_garbage;
}

void bounded_decoding(std::size_t parallel) {
    const auto fixture = encode_fixture(parallel, 0x5eed0000U + static_cast<std::uint32_t>(parallel));
    const std::span<const std::byte> stream(fixture.stream);

    // The encoder's own parameters decode exactly and pass the end check.
    check(decode_fixture(fixture, stream, fixture.rows, fixture.y.size()) == Decoded::exact);

    // Every truncation is rejected; none reads past the stream (ASan).
    for (std::size_t size = 0; size < stream.size(); ++size)
        check(decode_fixture(fixture, stream.first(size), fixture.rows, fixture.y.size()) ==
              Decoded::rejected);

    // Trailing bytes the encoder did not write are rejected by the end check.
    auto extended = fixture.stream;
    extended.push_back(std::byte{0x5a});
    check(decode_fixture(fixture, extended, fixture.rows, fixture.y.size()) ==
          Decoded::rejected);

    // Decoding fewer symbols than were coded leaves the stream unfinished.
    check(decode_fixture(fixture, stream, fixture.rows, fixture.y.size() - 1) ==
          Decoded::rejected);

    // Corrupted bytes never read out of bounds (ASan/UBSan) and are mostly
    // rejected. The UF format carries no integrity field: a flipped bit that
    // lands in raw bypass bits, for example, is a valid encoding of another
    // symbol and cannot be detected by entropy decoding. Counted, not required.
    std::mt19937 random(static_cast<std::uint32_t>(parallel));
    std::size_t flips_rejected = 0;
    std::size_t flips_undetected = 0;
    for (std::size_t byte = 0; byte < stream.size(); byte += 3) {
        auto corrupted = fixture.stream;
        corrupted[byte] ^= static_cast<std::byte>(1U << (random() % 8));
        const auto result = decode_fixture(fixture, corrupted, fixture.rows, fixture.y.size());
        flips_rejected += result == Decoded::rejected ? 1 : 0;
        flips_undetected += result == Decoded::silent_garbage ? 1 : 0;
    }
    std::size_t noise_rejected = 0;
    constexpr int noise_trials = 50;
    for (int trial = 0; trial < noise_trials; ++trial) {
        auto noise = fixture.stream;
        for (auto& value : noise) value = static_cast<std::byte>(random() & 0xffU);
        noise_rejected += decode_fixture(fixture, noise, fixture.rows, fixture.y.size()) ==
                                  Decoded::rejected
                              ? 1
                              : 0;
    }
    const auto flips = (stream.size() + 2) / 3;
    std::printf("parallel %zu: %zu bytes; %zu bit flips: rejected %zu, undetected %zu, exact %zu; "
                "random streams rejected %zu/%d\n",
                parallel, stream.size(), flips, flips_rejected, flips_undetected,
                flips - flips_rejected - flips_undetected, noise_rejected, noise_trials);
    std::fflush(stdout);
    check(noise_rejected == noise_trials);
    check(flips_rejected * 2 > flips);

    // A decoder that computes a different entropy index at one position, as a
    // provider with different numerics does, is rejected unless the other row
    // holds the coded slot with the same start and frequency: then the rANS
    // state evolves identically and only the symbol's meaning can change (see
    // scripts/backends/dcvcuf/check_uf_cdf_aliasing.py for UF's real tables).
    const auto y_table = make_shaped_cdf(y_rows);
    const auto aliased = [&](std::size_t position, std::uint8_t other) {
        const auto symbol = static_cast<int>(fixture.y[position]);
        const auto row = fixture.rows[position];
        const int value = std::abs(symbol) * 2 - (symbol > 0 ? 1 : 0);
        const auto slot = static_cast<std::size_t>(std::min(value, y_table.sizes[row] - 2));
        const auto& coded = y_table.values[row];
        const auto& alternative = y_table.values[other];
        for (std::size_t index = 0; index + 1 < static_cast<std::size_t>(y_table.sizes[other]); ++index)
            if (alternative[index] == coded[slot] && alternative[index + 1] == coded[slot + 1])
                return true;
        return false;
    };
    std::size_t divergent = 0;
    std::size_t skipped = 0;
    for (std::size_t position = 0; position < fixture.rows.size(); position += 37) {
        auto rows = fixture.rows;
        rows[position] = static_cast<std::uint8_t>((rows[position] + 1) % y_rows);
        if (aliased(position, rows[position])) {
            ++skipped;
            continue;
        }
        check(decode_fixture(fixture, stream, rows, fixture.y.size()) == Decoded::rejected);
        ++divergent;
    }
    std::printf("parallel %zu: %zu single-index divergences rejected, %zu aliased slots skipped\n",
                parallel, divergent, skipped);
    std::fflush(stdout);
    check(divergent > 50);
}

void rejects_invalid_initial_state() {
    nvcr::dcvcuf::UfRansCodec decoder;
    check(decoder.set_entropy_coder_parallel(1).has_value());
    check(decoder.add_cdf(make_cdf(256)).has_value());
    // 0xffffffff is above the largest state an encoder can flush.
    const std::vector<std::byte> stream(16, std::byte{0xff});
    check(decoder.set_stream(stream).has_value());
    auto decoded = decoder.decode_z(4, 0, 0, 1);
    check(!decoded.has_value() &&
          decoded.error().code() == nvcr::ErrorCode::malformed_bitstream);
}

void finish_requires_stream() {
    nvcr::dcvcuf::UfRansCodec decoder;
    check(!decoder.finish_decode().has_value());
}

}  // namespace

int main() {
    for (std::size_t parallel = 1; parallel <= 8; ++parallel)
        round_trip(parallel);
    rejects_truncated_stream();
    rejects_invalid_initial_state();
    finish_requires_stream();
    for (std::size_t parallel = 1; parallel <= 8; ++parallel)
        bounded_decoding(parallel);
    return 0;
}
