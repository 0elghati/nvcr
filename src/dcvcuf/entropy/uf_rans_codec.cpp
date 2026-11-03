#include "uf_rans_codec.hpp"

#include "rans.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace nvcr::dcvcuf {
namespace {

namespace uf = upstream_rans;

constexpr std::size_t max_parallel = 8;
constexpr std::size_t z_cdf_group = 0;
constexpr std::size_t y_cdf_group = 1;
constexpr std::int32_t cdf_total = 1 << 16;

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), "dcvcuf.rans");
}

Error malformed(std::string message) {
    return Error(ErrorCode::malformed_bitstream, std::move(message), "dcvcuf.rans");
}

Result<void> validate_cdf(const UfRansCdfTable& table) {
    if (table.values.empty() || table.values.size() != table.sizes.size())
        return invalid("CDF rows and sizes must have equal nonzero length");
    const auto width = table.values.front().size();
    if (width < 2) return invalid("CDF rows must have at least two entries");
    for (std::size_t row = 0; row < table.values.size(); ++row) {
        const auto size = table.sizes[row];
        const auto& values = table.values[row];
        if (values.size() != width)
            return invalid("UF CDF storage must be rectangular");
        if (size < 2 || static_cast<std::size_t>(size) > width ||
            size - 2 > std::numeric_limits<std::int8_t>::max() ||
            values.front() != 0 ||
            values[static_cast<std::size_t>(size - 1)] != cdf_total)
            return invalid("CDF row is malformed");
        for (std::int32_t i = 1; i < size; ++i) {
            if (values[static_cast<std::size_t>(i)] <=
                values[static_cast<std::size_t>(i - 1)])
                return invalid("CDF row is not strictly increasing");
        }
    }
    return {};
}

Result<int> checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return invalid(std::string(name) + " exceeds the UF entropy ABI");
    return static_cast<int>(value);
}

std::shared_ptr<std::vector<std::vector<uf::RansSymbol>>> make_symbols(
    const UfRansCdfTable& table) {
    auto result =
        std::make_shared<std::vector<std::vector<uf::RansSymbol>>>(table.values.size());
    for (std::size_t row = 0; row < table.values.size(); ++row) {
        auto& symbols = (*result)[row];
        symbols.resize(table.values[row].size());
        for (std::size_t i = 0; i + 1 < table.values[row].size(); ++i) {
            symbols[i] = {
                static_cast<std::uint16_t>(table.values[row][i]),
                static_cast<std::uint16_t>(table.values[row][i + 1] -
                                           table.values[row][i])};
        }
    }
    return result;
}

std::shared_ptr<std::vector<std::vector<std::int32_t>>> make_cdfs(
    const UfRansCdfTable& table) {
    return std::make_shared<std::vector<std::vector<std::int32_t>>>(table.values);
}

std::shared_ptr<std::vector<std::int8_t>> make_max_values(
    const UfRansCdfTable& table) {
    auto result = std::make_shared<std::vector<std::int8_t>>();
    result->reserve(table.sizes.size());
    for (const auto size : table.sizes)
        result->push_back(static_cast<std::int8_t>(size - 2));
    return result;
}

std::size_t identical_tail(const std::vector<std::uint8_t>& first,
                           const std::vector<std::uint8_t>& second) {
    const auto check = std::min<std::size_t>({first.size(), second.size(), 8});
    std::size_t identical = 0;
    while (identical < check && first[first.size() - 1 - identical] == 0 &&
           second[second.size() - 1 - identical] == 0)
        ++identical;
    if (identical == 0 && !first.empty() && !second.empty() &&
        first.back() == second.back())
        identical = 1;
    return identical;
}

std::shared_ptr<std::vector<std::uint8_t>> copy_bytes(
    std::span<const std::uint8_t> values) {
    return std::make_shared<std::vector<std::uint8_t>>(values.begin(), values.end());
}

}  // namespace

struct UfRansCodec::Impl final {
    std::size_t parallel{1};
    std::vector<std::unique_ptr<uf::RansEncoderLib>> encoders;
    std::vector<std::unique_ptr<uf::RansDecoderLib>> decoders;
    std::vector<std::size_t> cdf_rows;
    bool encoder_has_symbols{false};
    bool encoder_finished{false};
    bool stream_set{false};

    void make_coders() {
        encoders.clear();
        decoders.clear();
        encoders.reserve(parallel);
        decoders.reserve(parallel);
        for (std::size_t i = 0; i < parallel; ++i) {
            auto encoder = std::make_unique<uf::RansEncoderLib>();
            encoder->reset();
            encoders.push_back(std::move(encoder));
            decoders.push_back(std::make_unique<uf::RansDecoderLib>());
        }
    }
};

UfRansCodec::UfRansCodec() : impl_(std::make_unique<Impl>()) {
    impl_->make_coders();
}
UfRansCodec::~UfRansCodec() = default;
UfRansCodec::UfRansCodec(UfRansCodec&&) noexcept = default;
UfRansCodec& UfRansCodec::operator=(UfRansCodec&&) noexcept = default;

Result<void> UfRansCodec::set_entropy_coder_parallel(std::size_t count) {
    if (count == 0 || count > max_parallel)
        return invalid("UF entropy parallelism must be 1..8");
    if (impl_->encoder_has_symbols || impl_->encoder_finished ||
        !impl_->cdf_rows.empty() || impl_->stream_set)
        return Error(ErrorCode::invalid_state,
                     "clear the UF entropy session before changing parallelism",
                     "dcvcuf.rans");
    impl_->parallel = count;
    impl_->make_coders();
    return {};
}

std::size_t UfRansCodec::entropy_coder_parallel() const noexcept {
    return impl_->parallel;
}

Result<std::size_t> UfRansCodec::add_cdf(UfRansCdfTable table) {
    auto valid = validate_cdf(table);
    if (!valid) return valid.error();
    const auto group = impl_->cdf_rows.size();
    if (group > y_cdf_group)
        return invalid("UF accepts exactly two CDF groups in z, y order");

    auto symbols = make_symbols(table);
    auto cdfs = make_cdfs(table);
    auto max_values = make_max_values(table);
    const auto group_int = static_cast<int>(group);
    for (std::size_t i = 0; i < impl_->parallel; ++i) {
        impl_->encoders[i]->set_cdf(symbols, max_values, group_int);
        impl_->decoders[i]->set_cdf(cdfs, max_values, group_int);
    }
    impl_->cdf_rows.push_back(table.values.size());
    return group;
}

void UfRansCodec::clear_cdfs() noexcept {
    impl_->cdf_rows.clear();
    impl_->encoder_has_symbols = false;
    impl_->encoder_finished = false;
    impl_->stream_set = false;
    impl_->make_coders();
}

Result<void> UfRansCodec::encode_y(std::span<const std::int16_t> symbols,
                                   std::size_t cdf_group) {
    if (cdf_group != y_cdf_group || impl_->cdf_rows.size() <= y_cdf_group)
        return invalid("UF y entropy requires CDF group 1");
    if (impl_->encoder_finished)
        return Error(ErrorCode::invalid_state, "reset encoder before reuse",
                     "dcvcuf.rans");
    for (const auto symbol : symbols) {
        const auto row = static_cast<std::uint16_t>(symbol) & 0xffU;
        if (row >= impl_->cdf_rows[y_cdf_group])
            return invalid("UF y symbol has invalid CDF row");
    }
    if (symbols.empty()) return {};

    auto total = checked_int(symbols.size(), "y symbol count");
    if (!total) return total.error();
    auto shared =
        std::make_shared<std::vector<std::int16_t>>(symbols.begin(), symbols.end());
    const int base = total.value() / static_cast<int>(impl_->parallel);
    for (std::size_t i = 0; i + 1 < impl_->parallel; ++i) {
        const auto offset = base * static_cast<int>(i);
        impl_->encoders[i]->encode_y(shared, base, offset);
    }
    const auto offset = base * static_cast<int>(impl_->parallel - 1);
    impl_->encoders.back()->encode_y(shared, total.value() - offset, offset);
    impl_->encoder_has_symbols = true;
    return {};
}

Result<void> UfRansCodec::encode_z(std::span<const std::int8_t> symbols,
                                   std::size_t cdf_group,
                                   std::size_t cdf_offset,
                                   std::size_t channel_count) {
    if (cdf_group != z_cdf_group || impl_->cdf_rows.empty())
        return invalid("UF z entropy requires CDF group 0");
    if (channel_count == 0 || cdf_offset > impl_->cdf_rows[z_cdf_group] ||
        channel_count > impl_->cdf_rows[z_cdf_group] - cdf_offset)
        return invalid("UF z CDF range is invalid");
    if (impl_->encoder_finished)
        return Error(ErrorCode::invalid_state, "reset encoder before reuse",
                     "dcvcuf.rans");
    if (symbols.empty()) return {};

    auto total = checked_int(symbols.size(), "z symbol count");
    auto offset_result = checked_int(cdf_offset, "z CDF offset");
    auto channels = checked_int(channel_count, "z channel count");
    if (!total) return total.error();
    if (!offset_result) return offset_result.error();
    if (!channels) return channels.error();

    auto shared =
        std::make_shared<std::vector<std::int8_t>>(symbols.begin(), symbols.end());
    const int base = total.value() / static_cast<int>(impl_->parallel);
    for (std::size_t i = 0; i + 1 < impl_->parallel; ++i) {
        const auto symbol_offset = base * static_cast<int>(i);
        impl_->encoders[i]->encode_z(shared, base, symbol_offset,
                                     offset_result.value(), channels.value());
    }
    const auto symbol_offset = base * static_cast<int>(impl_->parallel - 1);
    impl_->encoders.back()->encode_z(shared, total.value() - symbol_offset,
                                     symbol_offset, offset_result.value(),
                                     channels.value());
    impl_->encoder_has_symbols = true;
    return {};
}

Result<std::vector<std::byte>> UfRansCodec::finish_encode() {
    if (impl_->encoder_finished)
        return Error(ErrorCode::invalid_state,
                     "encode session is already finished", "dcvcuf.rans");
    if (!impl_->encoder_has_symbols) {
        impl_->encoder_finished = true;
        return std::vector<std::byte>{};
    }

    for (auto& encoder : impl_->encoders) encoder->flush();
    std::vector<std::vector<std::uint8_t>> streams;
    streams.reserve(impl_->parallel);
    for (auto& encoder : impl_->encoders)
        streams.push_back(*encoder->get_encoded_stream());

    if (impl_->parallel == 1) {
        impl_->encoder_finished = true;
        std::vector<std::byte> output(streams[0].size());
        std::memcpy(output.data(), streams[0].data(), output.size());
        return output;
    }

    const auto pairs = impl_->parallel / 2;
    const bool has_tail = impl_->parallel % 2 != 0;
    std::vector<std::size_t> groups(pairs);
    std::vector<std::size_t> overlaps(pairs);
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        overlaps[pair] =
            identical_tail(streams[2 * pair], streams[2 * pair + 1]);
        groups[pair] = streams[2 * pair].size() +
                       streams[2 * pair + 1].size() - overlaps[pair];
    }
    const auto offset_count = pairs - 1 + (has_tail ? 1 : 0);
    const auto header_size = offset_count * sizeof(std::int32_t);
    std::size_t total_size = header_size;
    for (const auto size : groups) total_size += size;
    if (has_tail) total_size += streams.back().size();
    if (total_size >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return Error(ErrorCode::resource_exhausted,
                     "UF entropy stream is too large", "dcvcuf.rans");

    std::vector<std::byte> output(total_size);
    auto* bytes = reinterpret_cast<std::uint8_t*>(output.data());
    std::size_t cumulative = groups[0];
    for (std::size_t i = 0; i < offset_count; ++i) {
        const auto value = static_cast<std::int32_t>(cumulative);
        std::memcpy(bytes + i * sizeof(value), &value, sizeof(value));
        if (i + 1 < pairs) cumulative += groups[i + 1];
    }

    std::size_t position = header_size;
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        const auto& forward = streams[2 * pair];
        const auto& reverse = streams[2 * pair + 1];
        std::memcpy(bytes + position, forward.data(), forward.size());
        std::reverse_copy(
            reverse.begin(),
            reverse.end() - static_cast<std::ptrdiff_t>(overlaps[pair]),
            bytes + position + forward.size());
        position += groups[pair];
    }
    if (has_tail)
        std::memcpy(bytes + position, streams.back().data(),
                    streams.back().size());

    impl_->encoder_finished = true;
    return output;
}

void UfRansCodec::reset_encoder() noexcept {
    for (auto& encoder : impl_->encoders) encoder->reset();
    impl_->encoder_has_symbols = false;
    impl_->encoder_finished = false;
}

Result<void> UfRansCodec::set_stream(std::span<const std::byte> stream) {
    if (stream.size() < sizeof(std::uint32_t))
        return malformed("UF entropy stream is truncated");
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(stream.data());

    if (impl_->parallel == 1 || impl_->parallel == 2) {
        auto shared =
            copy_bytes(std::span<const std::uint8_t>(bytes, stream.size()));
        impl_->decoders[0]->set_stream(shared);
        if (impl_->parallel == 2) {
            auto reverse = std::make_shared<std::vector<std::uint8_t>>(
                shared->rbegin(), shared->rend());
            impl_->decoders[1]->set_stream(reverse);
        }
        impl_->stream_set = true;
        return {};
    }

    const auto pairs = impl_->parallel / 2;
    const bool has_tail = impl_->parallel % 2 != 0;
    const auto offset_count = pairs - 1 + (has_tail ? 1 : 0);
    const auto header_size = offset_count * sizeof(std::int32_t);
    if (stream.size() < header_size)
        return malformed("UF entropy header is truncated");

    std::vector<std::size_t> offsets(offset_count);
    for (std::size_t i = 0; i < offset_count; ++i) {
        std::int32_t value = 0;
        std::memcpy(&value, bytes + i * sizeof(value), sizeof(value));
        if (value < 0) return malformed("UF entropy offset is negative");
        offsets[i] = static_cast<std::size_t>(value);
    }
    const auto payload_size = stream.size() - header_size;
    if (offsets.empty() || offsets.front() < sizeof(std::uint32_t) ||
        offsets.back() > payload_size ||
        !std::is_sorted(offsets.begin(), offsets.end()))
        return malformed("UF entropy offsets are invalid");

    const auto* payload = bytes + header_size;
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        const auto begin = pair == 0 ? 0 : offsets[pair - 1];
        const auto end =
            pair + 1 < pairs
                ? offsets[pair]
                : (has_tail ? offsets.back() : payload_size);
        if (end - begin < sizeof(std::uint32_t) || end > payload_size)
            return malformed("UF entropy group is invalid");
        auto group = copy_bytes(std::span<const std::uint8_t>(
            payload + begin, end - begin));
        impl_->decoders[2 * pair]->set_stream(group);
        auto reverse = std::make_shared<std::vector<std::uint8_t>>(
            group->rbegin(), group->rend());
        impl_->decoders[2 * pair + 1]->set_stream(reverse);
    }
    if (has_tail) {
        const auto begin = offsets.back();
        if (payload_size - begin < sizeof(std::uint32_t))
            return malformed("UF entropy tail is truncated");
        impl_->decoders.back()->set_stream(copy_bytes(
            std::span<const std::uint8_t>(payload + begin,
                                          payload_size - begin)));
    }
    impl_->stream_set = true;
    return {};
}

Result<std::vector<std::int8_t>> UfRansCodec::decode_y(
    std::span<const std::uint8_t> indexes, std::size_t cdf_group) {
    if (!impl_->stream_set)
        return Error(ErrorCode::invalid_state,
                     "set UF stream before decoding", "dcvcuf.rans");
    if (cdf_group != y_cdf_group || impl_->cdf_rows.size() <= y_cdf_group)
        return invalid("UF y entropy requires CDF group 1");
    for (const auto index : indexes) {
        if (index >= impl_->cdf_rows[y_cdf_group])
            return invalid("UF y index exceeds CDF rows");
    }
    if (indexes.empty()) return std::vector<std::int8_t>{};

    auto total = checked_int(indexes.size(), "y index count");
    if (!total) return total.error();
    auto shared = copy_bytes(indexes);
    std::vector<std::int8_t> output(indexes.size());
    const int base = total.value() / static_cast<int>(impl_->parallel);
    for (std::size_t i = 0; i + 1 < impl_->parallel; ++i) {
        const auto offset = base * static_cast<int>(i);
        impl_->decoders[i]->decode_y(output.data(), shared, base, offset);
    }
    const auto offset = base * static_cast<int>(impl_->parallel - 1);
    impl_->decoders.back()->decode_y(output.data(), shared,
                                     total.value() - offset, offset);
    bool complete = true;
    for (auto& decoder : impl_->decoders)
        complete = decoder->wait_for_decoding_finish() && complete;
    if (!complete) return malformed("UF entropy stream ended early");
    return output;
}

Result<std::vector<std::int8_t>> UfRansCodec::decode_z(
    std::size_t symbol_count, std::size_t cdf_group,
    std::size_t cdf_offset, std::size_t channel_count) {
    if (!impl_->stream_set)
        return Error(ErrorCode::invalid_state,
                     "set UF stream before decoding", "dcvcuf.rans");
    if (cdf_group != z_cdf_group || impl_->cdf_rows.empty())
        return invalid("UF z entropy requires CDF group 0");
    if (channel_count == 0 || cdf_offset > impl_->cdf_rows[z_cdf_group] ||
        channel_count > impl_->cdf_rows[z_cdf_group] - cdf_offset)
        return invalid("UF z CDF range is invalid");
    if (symbol_count == 0) return std::vector<std::int8_t>{};

    auto total = checked_int(symbol_count, "z symbol count");
    auto offset_result = checked_int(cdf_offset, "z CDF offset");
    auto channels = checked_int(channel_count, "z channel count");
    if (!total) return total.error();
    if (!offset_result) return offset_result.error();
    if (!channels) return channels.error();

    std::vector<std::int8_t> output(symbol_count);
    const int base = total.value() / static_cast<int>(impl_->parallel);
    for (std::size_t i = 0; i + 1 < impl_->parallel; ++i) {
        const auto symbol_offset = base * static_cast<int>(i);
        impl_->decoders[i]->decode_z(
            output.data(), base, symbol_offset, offset_result.value(),
            channels.value());
    }
    const auto symbol_offset = base * static_cast<int>(impl_->parallel - 1);
    impl_->decoders.back()->decode_z(
        output.data(), total.value() - symbol_offset, symbol_offset,
        offset_result.value(), channels.value());
    bool complete = true;
    for (auto& decoder : impl_->decoders)
        complete = decoder->wait_for_decoding_finish() && complete;
    if (!complete) return malformed("UF entropy stream ended early");
    return output;
}

}  // namespace nvcr::dcvcuf
