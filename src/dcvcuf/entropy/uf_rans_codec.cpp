#include "uf_rans_codec.hpp"

#include "rans.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace nvcr::dcvcuf {
namespace {

constexpr std::size_t max_parallel = 8;
constexpr std::int32_t cdf_total = 1 << 16;

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), "dcvcuf.rans");
}

Error malformed(std::string message) {
    return Error(ErrorCode::malformed_bitstream, std::move(message), "dcvcuf.rans");
}

Result<void> validate_cdf(const UfRansCdfTable& table) {
    if (table.values.empty() || table.values.size() != table.sizes.size() ||
        table.values.size() != table.offsets.size())
        return invalid("CDF groups must contain equally sized row metadata");
    for (std::size_t row = 0; row < table.values.size(); ++row) {
        const auto size = table.sizes[row];
        const auto& values = table.values[row];
        if (size < 2 || static_cast<std::size_t>(size) > values.size() || values.front() != 0 ||
            values[static_cast<std::size_t>(size - 1)] != cdf_total)
            return invalid("CDF row is malformed");
        for (std::int32_t i = 1; i < size; ++i) {
            if (values[static_cast<std::size_t>(i)] <= values[static_cast<std::size_t>(i - 1)])
                return invalid("CDF row is not strictly increasing");
        }
    }
    return {};
}

std::shared_ptr<std::vector<std::vector<std::int32_t>>> copy_values(
    const UfRansCdfTable& table) {
    return std::make_shared<std::vector<std::vector<std::int32_t>>>(table.values);
}

std::shared_ptr<std::vector<std::int32_t>> copy_ints(const std::vector<std::int32_t>& values) {
    return std::make_shared<std::vector<std::int32_t>>(values);
}

std::shared_ptr<std::vector<std::int16_t>> copy_i16(std::span<const std::int16_t> values) {
    return std::make_shared<std::vector<std::int16_t>>(values.begin(), values.end());
}

std::shared_ptr<std::vector<std::int8_t>> copy_i8(std::span<const std::int8_t> values) {
    return std::make_shared<std::vector<std::int8_t>>(values.begin(), values.end());
}

std::shared_ptr<std::vector<std::uint8_t>> copy_u8(std::span<const std::uint8_t> values) {
    return std::make_shared<std::vector<std::uint8_t>>(values.begin(), values.end());
}

std::size_t chunk_size(std::size_t total, std::size_t count) { return total / count; }

std::size_t identical_tail(const std::vector<std::uint8_t>& first,
                           const std::vector<std::uint8_t>& second) {
    const auto check = std::min<std::size_t>({first.size(), second.size(), 8});
    std::size_t identical = 0;
    while (identical < check && first[first.size() - 1 - identical] == 0 &&
           second[second.size() - 1 - identical] == 0)
        ++identical;
    if (identical == 0 && !first.empty() && first.back() == second.back()) identical = 1;
    return identical;
}

}  // namespace

struct UfRansCodec::Impl final {
    std::size_t parallel{1};
    std::vector<std::unique_ptr<RansEncoderLib>> encoders;
    std::vector<std::unique_ptr<RansDecoderLib>> decoders;
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
            encoders.push_back(std::make_unique<RansEncoderLib>());
            decoders.push_back(std::make_unique<RansDecoderLib>());
        }
    }
};

UfRansCodec::UfRansCodec() : impl_(std::make_unique<Impl>()) { impl_->make_coders(); }
UfRansCodec::~UfRansCodec() = default;
UfRansCodec::UfRansCodec(UfRansCodec&&) noexcept = default;
UfRansCodec& UfRansCodec::operator=(UfRansCodec&&) noexcept = default;

Result<void> UfRansCodec::set_entropy_coder_parallel(std::size_t count) {
    if (count == 0 || count > max_parallel) return invalid("UF entropy parallelism must be 1..8");
    if (impl_->encoder_has_symbols || impl_->encoder_finished || !impl_->cdf_rows.empty())
        return Error(ErrorCode::invalid_state,
                     "reset encoder and clear CDFs before changing parallelism", "dcvcuf.rans");
    impl_->parallel = count;
    impl_->make_coders();
    return {};
}

std::size_t UfRansCodec::entropy_coder_parallel() const noexcept { return impl_->parallel; }

Result<std::size_t> UfRansCodec::add_cdf(UfRansCdfTable table) {
    auto valid = validate_cdf(table);
    if (!valid) return valid.error();
    const auto values = copy_values(table);
    const auto sizes = copy_ints(table.sizes);
    const auto offsets = copy_ints(table.offsets);
    const auto rows = table.values.size();
    int group = -1;
    try {
        for (std::size_t i = 0; i < impl_->parallel; ++i) {
            const auto current = impl_->encoders[i]->add_cdf(values, sizes, offsets);
            impl_->decoders[i]->add_cdf(values, sizes, offsets);
            if (i == 0) group = current;
            if (current != group) return Error(ErrorCode::internal_error, "CDF group mismatch",
                                               "dcvcuf.rans");
        }
    } catch (...) {
        return Error(ErrorCode::internal_error, "adding UF CDF failed", "dcvcuf.rans");
    }
    if (group < 0 || static_cast<std::size_t>(group) != impl_->cdf_rows.size())
        return Error(ErrorCode::internal_error, "unexpected UF CDF group", "dcvcuf.rans");
    impl_->cdf_rows.push_back(rows);
    return static_cast<std::size_t>(group);
}

void UfRansCodec::clear_cdfs() noexcept {
    for (auto& encoder : impl_->encoders) {
        encoder->empty_cdf_buffer();
        encoder->reset();
    }
    for (auto& decoder : impl_->decoders) decoder->empty_cdf_buffer();
    impl_->cdf_rows.clear();
    impl_->encoder_has_symbols = false;
    impl_->encoder_finished = false;
    impl_->stream_set = false;
}

Result<void> UfRansCodec::encode_y(std::span<const std::int16_t> symbols, std::size_t cdf_group) {
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown UF y CDF group");
    if (impl_->encoder_finished) return Error(ErrorCode::invalid_state, "reset encoder before reuse",
                                              "dcvcuf.rans");
    for (const auto symbol : symbols) {
        const auto row = static_cast<std::uint16_t>(symbol) & 0xffU;
        if (row >= impl_->cdf_rows[cdf_group]) return invalid("UF y symbol has invalid CDF row");
    }
    if (symbols.empty()) return {};
    if (symbols.size() < impl_->parallel) return invalid("UF y stream is smaller than coder count");
    const auto base = chunk_size(symbols.size(), impl_->parallel);
    for (std::size_t i = 0; i < impl_->parallel; ++i) {
        const auto begin = i * base;
        const auto size = i + 1 == impl_->parallel ? symbols.size() - begin : base;
        impl_->encoders[i]->encode_y(copy_i16(symbols.subspan(begin, size)), static_cast<int>(cdf_group));
    }
    impl_->encoder_has_symbols = true;
    return {};
}

Result<void> UfRansCodec::encode_z(std::span<const std::int8_t> symbols, std::size_t cdf_group,
                                   std::size_t start_offset, std::size_t per_channel_size) {
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown UF z CDF group");
    if (per_channel_size == 0 || symbols.empty() || symbols.size() % per_channel_size != 0)
        return invalid("UF z symbols must contain complete channels");
    const auto channels = symbols.size() / per_channel_size;
    if (start_offset > impl_->cdf_rows[cdf_group] || channels > impl_->cdf_rows[cdf_group] - start_offset)
        return invalid("UF z symbols exceed CDF rows");
    if (impl_->encoder_finished) return Error(ErrorCode::invalid_state, "reset encoder before reuse",
                                              "dcvcuf.rans");
    if (symbols.size() < impl_->parallel || chunk_size(symbols.size(), impl_->parallel) % per_channel_size)
        return invalid("UF z stream cannot split on channel boundaries");
    const auto base = chunk_size(symbols.size(), impl_->parallel);
    for (std::size_t i = 0; i < impl_->parallel; ++i) {
        const auto begin = i * base;
        const auto size = i + 1 == impl_->parallel ? symbols.size() - begin : base;
        impl_->encoders[i]->encode_z(copy_i8(symbols.subspan(begin, size)), static_cast<int>(cdf_group),
                                     static_cast<int>(start_offset + begin / per_channel_size),
                                     static_cast<int>(per_channel_size));
    }
    impl_->encoder_has_symbols = true;
    return {};
}

Result<std::vector<std::byte>> UfRansCodec::finish_encode() {
    if (impl_->encoder_finished) return Error(ErrorCode::invalid_state, "encode session is already finished",
                                              "dcvcuf.rans");
    if (!impl_->encoder_has_symbols) {
        impl_->encoder_finished = true;
        return std::vector<std::byte>{};
    }
    for (auto& encoder : impl_->encoders) encoder->flush();
    std::vector<std::vector<std::uint8_t>> streams;
    streams.reserve(impl_->parallel);
    for (auto& encoder : impl_->encoders) {
        auto stream = encoder->get_encoded_stream();
        streams.push_back(*stream);
    }
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
    for (std::size_t p = 0; p < pairs; ++p) {
        overlaps[p] = identical_tail(streams[2 * p], streams[2 * p + 1]);
        groups[p] = streams[2 * p].size() + streams[2 * p + 1].size() - overlaps[p];
    }
    const auto offsets = pairs - 1 + (has_tail ? 1 : 0);
    if (groups.empty()) return Error(ErrorCode::internal_error, "UF coder partition is empty", "dcvcuf.rans");
    const auto header = offsets * sizeof(std::int32_t);
    std::size_t total = header;
    for (const auto size : groups) total += size;
    if (has_tail) total += streams.back().size();
    if (total > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return Error(ErrorCode::resource_exhausted, "UF entropy stream is too large", "dcvcuf.rans");
    std::vector<std::byte> output(total);
    auto* bytes = reinterpret_cast<std::uint8_t*>(output.data());
    std::size_t cumulative = groups[0];
    for (std::size_t i = 0; i < offsets; ++i) {
        const auto value = static_cast<std::int32_t>(cumulative);
        std::memcpy(bytes + i * sizeof(value), &value, sizeof(value));
        if (i + 1 < pairs) cumulative += groups[i + 1];
    }
    std::size_t position = header;
    for (std::size_t p = 0; p < pairs; ++p) {
        std::memcpy(bytes + position, streams[2 * p].data(), streams[2 * p].size());
        std::reverse_copy(streams[2 * p + 1].begin(),
                          streams[2 * p + 1].end() - static_cast<std::ptrdiff_t>(overlaps[p]),
                          bytes + position + streams[2 * p].size());
        position += groups[p];
    }
    if (has_tail) std::memcpy(bytes + position, streams.back().data(), streams.back().size());
    impl_->encoder_finished = true;
    return output;
}

void UfRansCodec::reset_encoder() noexcept {
    for (auto& encoder : impl_->encoders) encoder->reset();
    impl_->encoder_has_symbols = false;
    impl_->encoder_finished = false;
}

Result<void> UfRansCodec::set_stream(std::span<const std::byte> stream) {
    if (stream.empty()) return malformed("UF entropy stream is empty");
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(stream.data());
    if (impl_->parallel == 1 || impl_->parallel == 2) {
        auto shared = copy_u8(std::span<const std::uint8_t>(bytes, stream.size()));
        impl_->decoders[0]->set_stream(shared);
        if (impl_->parallel == 2) {
            auto reverse = std::make_shared<std::vector<std::uint8_t>>(shared->rbegin(), shared->rend());
            impl_->decoders[1]->set_stream(reverse);
        }
        impl_->stream_set = true;
        return {};
    }
    const auto pairs = impl_->parallel / 2;
    const bool has_tail = impl_->parallel % 2 != 0;
    const auto offset_count = pairs - 1 + (has_tail ? 1 : 0);
    const auto header = offset_count * sizeof(std::int32_t);
    if (stream.size() < header) return malformed("UF entropy header is truncated");
    std::vector<std::size_t> offsets(offset_count);
    for (std::size_t i = 0; i < offset_count; ++i) {
        std::int32_t value = 0;
        std::memcpy(&value, bytes + i * sizeof(value), sizeof(value));
        if (value < 0) return malformed("UF entropy offset is negative");
        offsets[i] = static_cast<std::size_t>(value);
    }
    const auto payload_size = stream.size() - header;
    if (offsets.front() == 0 || offsets.back() > payload_size ||
        !std::is_sorted(offsets.begin(), offsets.end()))
        return malformed("UF entropy offsets are invalid");
    const std::uint8_t* payload = bytes + header;
    for (std::size_t p = 0; p < pairs; ++p) {
        const auto begin = p == 0 ? 0 : offsets[p - 1];
        const auto end = p + 1 < pairs ? offsets[p] : (has_tail ? offsets.back() : payload_size);
        if (end <= begin || end > payload_size) return malformed("UF entropy group is invalid");
        auto group = copy_u8(std::span<const std::uint8_t>(payload + begin, end - begin));
        impl_->decoders[2 * p]->set_stream(group);
        auto reverse = std::make_shared<std::vector<std::uint8_t>>(group->rbegin(), group->rend());
        impl_->decoders[2 * p + 1]->set_stream(reverse);
    }
    if (has_tail) {
        const auto begin = offsets.back();
        if (begin >= payload_size) return malformed("UF entropy tail is empty");
        impl_->decoders.back()->set_stream(copy_u8(
            std::span<const std::uint8_t>(payload + begin, payload_size - begin)));
    }
    impl_->stream_set = true;
    return {};
}

Result<std::vector<std::int8_t>> UfRansCodec::decode_y(std::span<const std::uint8_t> indexes,
                                                      std::size_t cdf_group) {
    if (!impl_->stream_set) return Error(ErrorCode::invalid_state, "set UF stream before decoding", "dcvcuf.rans");
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown UF y CDF group");
    if (indexes.empty() || indexes.size() < impl_->parallel) return invalid("UF y indexes are too short");
    const auto base = chunk_size(indexes.size(), impl_->parallel);
    std::vector<std::int8_t> output;
    output.reserve(indexes.size());
    for (std::size_t i = 0; i < impl_->parallel; ++i) {
        const auto begin = i * base;
        const auto size = i + 1 == impl_->parallel ? indexes.size() - begin : base;
        for (const auto index : indexes.subspan(begin, size))
            if (index >= impl_->cdf_rows[cdf_group]) return invalid("UF y index exceeds CDF rows");
        impl_->decoders[i]->decode_y(copy_u8(indexes.subspan(begin, size)), static_cast<int>(cdf_group));
        auto decoded = impl_->decoders[i]->get_decoded_tensor();
        output.insert(output.end(), decoded->begin(), decoded->end());
    }
    return output;
}

Result<std::vector<std::int8_t>> UfRansCodec::decode_z(std::size_t symbol_count,
                                                      std::size_t cdf_group,
                                                      std::size_t start_offset,
                                                      std::size_t per_channel_size) {
    if (!impl_->stream_set) return Error(ErrorCode::invalid_state, "set UF stream before decoding", "dcvcuf.rans");
    if (cdf_group >= impl_->cdf_rows.size() || per_channel_size == 0 || symbol_count == 0 ||
        symbol_count % per_channel_size != 0)
        return invalid("UF z decode shape is invalid");
    const auto channels = symbol_count / per_channel_size;
    if (start_offset > impl_->cdf_rows[cdf_group] || channels > impl_->cdf_rows[cdf_group] - start_offset)
        return invalid("UF z decode exceeds CDF rows");
    if (symbol_count < impl_->parallel || chunk_size(symbol_count, impl_->parallel) % per_channel_size)
        return invalid("UF z decode cannot split on channel boundaries");
    const auto base = chunk_size(symbol_count, impl_->parallel);
    std::vector<std::int8_t> output;
    output.reserve(symbol_count);
    for (std::size_t i = 0; i < impl_->parallel; ++i) {
        const auto begin = i * base;
        const auto size = i + 1 == impl_->parallel ? symbol_count - begin : base;
        impl_->decoders[i]->decode_z(static_cast<int>(size), static_cast<int>(cdf_group),
                                     static_cast<int>(start_offset + begin / per_channel_size),
                                     static_cast<int>(per_channel_size));
        auto decoded = impl_->decoders[i]->get_decoded_tensor();
        output.insert(output.end(), decoded->begin(), decoded->end());
    }
    return output;
}

}  // namespace nvcr::dcvcuf
