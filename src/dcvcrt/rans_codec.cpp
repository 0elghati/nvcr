#include "nvcr/dcvcrt/rans_codec.hpp"

#include "rans.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace nvcr::dcvcrt {
namespace {

constexpr std::int32_t cdf_total = 1 << 16;

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), "dcvcrt.rans");
}

Error failure(std::string_view operation, const std::exception& exception) {
    return Error(
        ErrorCode::internal_error,
        std::string(operation) + " failed: " + exception.what(),
        "dcvcrt.rans");
}

Result<void> validate_cdf(const RansCdfTable& table) {
    if (table.values.empty()) return invalid("CDF table cannot be empty");
    if (table.values.size() != table.sizes.size() ||
        table.values.size() != table.offsets.size()) {
        return invalid("CDF values, sizes, and offsets must have the same row count");
    }
    for (std::size_t row_index = 0; row_index < table.values.size(); ++row_index) {
        const auto& row = table.values[row_index];
        const auto declared_size = table.sizes[row_index];
        if (declared_size < 2 || static_cast<std::size_t>(declared_size) > row.size()) {
            return invalid("CDF row " + std::to_string(row_index) + " has an invalid size");
        }
        if (row.front() != 0 || row[static_cast<std::size_t>(declared_size - 1)] != cdf_total) {
            return invalid("CDF row " + std::to_string(row_index) + " must span [0, 65536]");
        }
        for (std::int32_t index = 1; index < declared_size; ++index) {
            if (row[static_cast<std::size_t>(index)] <=
                row[static_cast<std::size_t>(index - 1)]) {
                return invalid("CDF row " + std::to_string(row_index) + " is not increasing");
            }
        }
    }
    return {};
}

template <typename T>
std::shared_ptr<std::vector<T>> copy_span(std::span<const T> values) {
    return std::make_shared<std::vector<T>>(values.begin(), values.end());
}

}  // namespace

struct RansCodec::Impl final {
    Impl()
        : encoder0(std::make_unique<RansEncoderLib>()),
          encoder1(std::make_unique<RansEncoderLibMultiThread>()),
          decoder0(std::make_unique<RansDecoderLib>()),
          decoder1(std::make_unique<RansDecoderLibMultiThread>()) {}

    std::unique_ptr<RansEncoderLib> encoder0;
    std::unique_ptr<RansEncoderLibMultiThread> encoder1;
    std::unique_ptr<RansDecoderLib> decoder0;
    std::unique_ptr<RansDecoderLibMultiThread> decoder1;
    std::vector<std::size_t> cdf_rows;
    bool use_two_coders{false};
    bool encoder_has_symbols{false};
    bool encoder_finished{false};
    bool stream_set{false};
};

RansCodec::RansCodec() : impl_(std::make_unique<Impl>()) {}
RansCodec::~RansCodec() = default;
RansCodec::RansCodec(RansCodec&&) noexcept = default;
RansCodec& RansCodec::operator=(RansCodec&&) noexcept = default;

Result<void> RansCodec::set_use_two_coders(bool enabled) {
    if (impl_->encoder_has_symbols || impl_->encoder_finished) {
        return Error(
            ErrorCode::invalid_state,
            "coder mode cannot change during an encode session",
            "dcvcrt.rans");
    }
    impl_->use_two_coders = enabled;
    return {};
}

bool RansCodec::uses_two_coders() const noexcept { return impl_->use_two_coders; }

Result<std::size_t> RansCodec::add_cdf(RansCdfTable table) {
    auto valid = validate_cdf(table);
    if (!valid) return valid.error();
    try {
        auto values = std::make_shared<std::vector<std::vector<std::int32_t>>>(
            std::move(table.values));
        auto sizes = std::make_shared<std::vector<std::int32_t>>(std::move(table.sizes));
        auto offsets = std::make_shared<std::vector<std::int32_t>>(std::move(table.offsets));
        const auto rows = values->size();
        const int group = impl_->encoder0->add_cdf(values, sizes, offsets);
        const int encoder1_group = impl_->encoder1->add_cdf(values, sizes, offsets);
        const int decoder0_group = impl_->decoder0->add_cdf(values, sizes, offsets);
        const int decoder1_group = impl_->decoder1->add_cdf(values, sizes, offsets);
        if (group < 0 || encoder1_group != group || decoder0_group != group ||
            decoder1_group != group || static_cast<std::size_t>(group) != impl_->cdf_rows.size()) {
            return Error(ErrorCode::internal_error, "upstream CDF group mismatch", "dcvcrt.rans");
        }
        impl_->cdf_rows.push_back(rows);
        return static_cast<std::size_t>(group);
    } catch (const std::exception& exception) {
        return failure("adding CDF", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "adding CDF failed", "dcvcrt.rans");
    }
}

void RansCodec::clear_cdfs() noexcept {
    impl_->encoder0->empty_cdf_buffer();
    impl_->encoder1->empty_cdf_buffer();
    impl_->decoder0->empty_cdf_buffer();
    impl_->decoder1->empty_cdf_buffer();
    impl_->cdf_rows.clear();
    reset_encoder();
    impl_->stream_set = false;
}

Result<void> RansCodec::encode_y(
    std::span<const std::int16_t> combined_symbols,
    std::size_t cdf_group) {
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown CDF group");
    if (impl_->encoder_finished) {
        return Error(ErrorCode::invalid_state, "reset encoder before reuse", "dcvcrt.rans");
    }
    for (const auto combined : combined_symbols) {
        const auto row = static_cast<std::uint16_t>(combined) & 0xffU;
        if (row >= impl_->cdf_rows[cdf_group]) return invalid("y symbol has invalid CDF index");
    }
    if (combined_symbols.empty()) return {};
    if (impl_->use_two_coders && combined_symbols.size() < 2) {
        return invalid("two-coder y input must contain at least two symbols");
    }
    try {
        const auto half = combined_symbols.size() / 2;
        impl_->encoder0->encode_y(copy_span(combined_symbols.first(
            impl_->use_two_coders ? half : combined_symbols.size())),
            static_cast<int>(cdf_group));
        if (impl_->use_two_coders) {
            impl_->encoder1->encode_y(
                copy_span(combined_symbols.subspan(half)), static_cast<int>(cdf_group));
        }
        impl_->encoder_has_symbols = true;
        return {};
    } catch (const std::exception& exception) {
        return failure("encoding y", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "encoding y failed", "dcvcrt.rans");
    }
}

Result<void> RansCodec::encode_z(
    std::span<const std::int8_t> symbols,
    std::size_t cdf_group,
    std::size_t start_offset,
    std::size_t per_channel_size) {
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown CDF group");
    if (impl_->encoder_finished) {
        return Error(ErrorCode::invalid_state, "reset encoder before reuse", "dcvcrt.rans");
    }
    if (per_channel_size == 0 || symbols.size() % per_channel_size != 0) {
        return invalid("z symbols must contain complete non-empty channels");
    }
    const auto channel_count = symbols.size() / per_channel_size;
    if (start_offset > impl_->cdf_rows[cdf_group] ||
        channel_count > impl_->cdf_rows[cdf_group] - start_offset) {
        return invalid("z symbols exceed CDF rows");
    }
    if (symbols.empty()) return {};
    const auto half = symbols.size() / 2;
    if (impl_->use_two_coders && (half == 0 || half % per_channel_size != 0)) {
        return invalid("two-coder z input must split on a channel boundary");
    }
    try {
        impl_->encoder0->encode_z(
            copy_span(symbols.first(impl_->use_two_coders ? half : symbols.size())),
            static_cast<int>(cdf_group),
            static_cast<int>(start_offset),
            static_cast<int>(per_channel_size));
        if (impl_->use_two_coders) {
            const auto channel_half = half / per_channel_size;
            impl_->encoder1->encode_z(
                copy_span(symbols.subspan(half)),
                static_cast<int>(cdf_group),
                static_cast<int>(start_offset + channel_half),
                static_cast<int>(per_channel_size));
        }
        impl_->encoder_has_symbols = true;
        return {};
    } catch (const std::exception& exception) {
        return failure("encoding z", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "encoding z failed", "dcvcrt.rans");
    }
}

Result<std::vector<std::byte>> RansCodec::finish_encode() {
    if (impl_->encoder_finished) {
        return Error(ErrorCode::invalid_state, "encode session is already finished", "dcvcrt.rans");
    }
    if (!impl_->encoder_has_symbols) {
        impl_->encoder_finished = true;
        return std::vector<std::byte>{};
    }
    try {
        if (impl_->use_two_coders) impl_->encoder1->flush();
        impl_->encoder0->flush();
        const auto stream0 = impl_->encoder0->get_encoded_stream();
        if (!impl_->use_two_coders) {
            impl_->encoder_finished = true;
            std::vector<std::byte> output(stream0->size());
            std::memcpy(output.data(), stream0->data(), stream0->size());
            return output;
        }
        std::vector<std::uint8_t> merged(stream0->begin(), stream0->end());
        {
            const auto stream1 = impl_->encoder1->get_encoded_stream();
            if (stream0->empty() || stream1->empty()) {
                return Error(ErrorCode::internal_error, "empty split rANS stream", "dcvcrt.rans");
            }
            std::size_t identical = 0;
            const auto check = std::min<std::size_t>({stream0->size(), stream1->size(), 8});
            while (identical < check && (*stream0)[stream0->size() - 1 - identical] == 0 &&
                   (*stream1)[stream1->size() - 1 - identical] == 0) {
                ++identical;
            }
            if (identical == 0 && stream0->back() == stream1->back()) identical = 1;
            merged.reserve(stream0->size() + stream1->size() - identical);
            std::reverse_copy(
                stream1->begin(), stream1->end() - static_cast<std::ptrdiff_t>(identical),
                std::back_inserter(merged));
        }
        impl_->encoder_finished = true;
        std::vector<std::byte> output(merged.size());
        std::memcpy(output.data(), merged.data(), merged.size());
        return output;
    } catch (const std::exception& exception) {
        return failure("finishing encode", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "finishing encode failed", "dcvcrt.rans");
    }
}

void RansCodec::reset_encoder() noexcept {
    impl_->encoder0->reset();
    impl_->encoder1->reset();
    impl_->encoder_has_symbols = false;
    impl_->encoder_finished = false;
}

Result<void> RansCodec::set_stream(std::span<const std::byte> bitstream) {
    if (bitstream.size() < 4) {
        return Error(ErrorCode::malformed_bitstream, "rANS stream is shorter than its state", "dcvcrt.rans");
    }
    try {
        auto stream0 = std::make_shared<std::vector<std::uint8_t>>(bitstream.size());
        std::memcpy(stream0->data(), bitstream.data(), bitstream.size());
        impl_->decoder0->set_stream(stream0);
        if (impl_->use_two_coders) {
            auto stream1 = std::make_shared<std::vector<std::uint8_t>>(
                stream0->rbegin(), stream0->rend());
            impl_->decoder1->set_stream(stream1);
        }
        impl_->stream_set = true;
        return {};
    } catch (const std::exception& exception) {
        return failure("setting stream", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "setting stream failed", "dcvcrt.rans");
    }
}

Result<std::vector<std::int8_t>> RansCodec::decode_y(
    std::span<const std::uint8_t> indexes,
    std::size_t cdf_group) {
    if (!impl_->stream_set) {
        return Error(ErrorCode::invalid_state, "set rANS stream before decoding", "dcvcrt.rans");
    }
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown CDF group");
    for (const auto row : indexes) {
        if (row >= impl_->cdf_rows[cdf_group]) return invalid("y decode index exceeds CDF rows");
    }
    if (indexes.empty()) return std::vector<std::int8_t>{};
    if (impl_->use_two_coders && indexes.size() < 2) {
        return invalid("two-coder y indexes must contain at least two entries");
    }
    try {
        const auto half = indexes.size() / 2;
        if (impl_->use_two_coders) {
            impl_->decoder1->decode_y(
                copy_span(indexes.subspan(half)), static_cast<int>(cdf_group));
        }
        impl_->decoder0->decode_y(
            copy_span(indexes.first(impl_->use_two_coders ? half : indexes.size())),
            static_cast<int>(cdf_group));
        const auto result0 = impl_->decoder0->get_decoded_tensor();
        std::vector<std::int8_t> output = std::move(*result0);
        if (impl_->use_two_coders) {
            const auto result1 = impl_->decoder1->get_decoded_tensor();
            output.insert(output.end(), result1->begin(), result1->end());
        }
        return output;
    } catch (const std::exception& exception) {
        return failure("decoding y", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "decoding y failed", "dcvcrt.rans");
    }
}

Result<std::vector<std::int8_t>> RansCodec::decode_z(
    std::size_t symbol_count,
    std::size_t cdf_group,
    std::size_t start_offset,
    std::size_t per_channel_size) {
    if (!impl_->stream_set) {
        return Error(ErrorCode::invalid_state, "set rANS stream before decoding", "dcvcrt.rans");
    }
    if (cdf_group >= impl_->cdf_rows.size()) return invalid("unknown CDF group");
    if (per_channel_size == 0 || symbol_count % per_channel_size != 0) {
        return invalid("z decode count must contain complete non-empty channels");
    }
    const auto channel_count = symbol_count / per_channel_size;
    if (start_offset > impl_->cdf_rows[cdf_group] ||
        channel_count > impl_->cdf_rows[cdf_group] - start_offset) {
        return invalid("z decode count exceeds CDF rows");
    }
    if (symbol_count == 0) return std::vector<std::int8_t>{};
    const auto half = symbol_count / 2;
    if (impl_->use_two_coders && (half == 0 || half % per_channel_size != 0)) {
        return invalid("two-coder z decode must split on a channel boundary");
    }
    try {
        if (impl_->use_two_coders) {
            const auto channel_half = half / per_channel_size;
            impl_->decoder1->decode_z(
                static_cast<int>(symbol_count - half),
                static_cast<int>(cdf_group),
                static_cast<int>(start_offset + channel_half),
                static_cast<int>(per_channel_size));
        }
        impl_->decoder0->decode_z(
            static_cast<int>(impl_->use_two_coders ? half : symbol_count),
            static_cast<int>(cdf_group),
            static_cast<int>(start_offset),
            static_cast<int>(per_channel_size));
        const auto result0 = impl_->decoder0->get_decoded_tensor();
        std::vector<std::int8_t> output = std::move(*result0);
        if (impl_->use_two_coders) {
            const auto result1 = impl_->decoder1->get_decoded_tensor();
            output.insert(output.end(), result1->begin(), result1->end());
        }
        return output;
    } catch (const std::exception& exception) {
        return failure("decoding z", exception);
    } catch (...) {
        return Error(ErrorCode::internal_error, "decoding z failed", "dcvcrt.rans");
    }
}

}  // namespace nvcr::dcvcrt
