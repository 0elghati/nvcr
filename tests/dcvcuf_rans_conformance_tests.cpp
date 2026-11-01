#include "uf_rans_codec.hpp"

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

nvcr::dcvcuf::UfRansCdfTable make_cdf(std::size_t rows) {
    nvcr::dcvcuf::UfRansCdfTable table;
    table.values.resize(rows, {0, 32768, 65536});
    table.sizes.resize(rows, 3);
    table.offsets.resize(rows, 0);
    return table;
}

void check(bool value) { if (!value) std::abort(); }

void round_trip(std::size_t parallel) {
    nvcr::dcvcuf::UfRansCodec encoder;
    check(encoder.set_entropy_coder_parallel(parallel).has_value());
    check(encoder.add_cdf(make_cdf(64)).has_value());

    const std::size_t y_count = parallel * 7;
    std::vector<std::int16_t> y(y_count);
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = static_cast<std::int16_t>(((i & 1U) ? -1 : 1) << 8);
    check(encoder.encode_y(y, 0).has_value());

    constexpr std::size_t per_channel = 4;
    const std::size_t z_count = parallel * per_channel * 2;
    std::vector<std::int8_t> z(z_count);
    for (std::size_t i = 0; i < z.size(); ++i)
        z[i] = static_cast<std::int8_t>((static_cast<int>(i * 3) % 11) - 5);
    check(encoder.encode_z(z, 0, 0, per_channel).has_value());

    auto stream = encoder.finish_encode();
    check(stream.has_value() && !stream.value().empty());

    nvcr::dcvcuf::UfRansCodec decoder;
    check(decoder.set_entropy_coder_parallel(parallel).has_value());
    check(decoder.add_cdf(make_cdf(64)).has_value());
    check(decoder.set_stream(stream.value()).has_value());
    const std::vector<std::uint8_t> indexes(y.size(), 0);
    auto decoded_y = decoder.decode_y(indexes, 0);
    std::vector<std::int8_t> expected_y(y.size());
    for (std::size_t i = 0; i < expected_y.size(); ++i)
        expected_y[i] = static_cast<std::int8_t>((i & 1U) ? -1 : 1);
    check(decoded_y.has_value() && decoded_y.value() == expected_y);
    auto decoded_z = decoder.decode_z(z.size(), 0, 0, per_channel);
    check(decoded_z.has_value() && decoded_z.value() == z);

    encoder.reset_encoder();
    check(encoder.encode_y(y, 0).has_value());
    check(encoder.encode_z(z, 0, 0, per_channel).has_value());
    auto repeated = encoder.finish_encode();
    check(repeated.has_value() && repeated.value() == stream.value());
}

}  // namespace

int main() {
    for (std::size_t parallel = 1; parallel <= 8; ++parallel) round_trip(parallel);
    return 0;
}
