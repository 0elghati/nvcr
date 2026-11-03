#include "uf_rans_codec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

}  // namespace

int main() {
    for (std::size_t parallel = 1; parallel <= 8; ++parallel)
        round_trip(parallel);
    rejects_truncated_stream();
    return 0;
}
