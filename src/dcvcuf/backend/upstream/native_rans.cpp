#include "native_rans.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template<class T>
T checked(nvcr::Result<T> result) {
    if (!result) throw std::runtime_error(result.error().describe());
    return std::move(result.value());
}

void checked(nvcr::Result<void> result) {
    if (!result) throw std::runtime_error(result.error().describe());
}

nvcr::dcvcuf::UfRansCdfTable make_table(
    const std::shared_ptr<std::vector<std::int32_t>>& values,
    const std::shared_ptr<std::vector<std::int32_t>>& sizes) {
    if (!values || !sizes || sizes->empty() || values->size() % sizes->size() != 0)
        throw std::invalid_argument("invalid DCVC-UF CDF storage");
    const auto width = values->size() / sizes->size();
    nvcr::dcvcuf::UfRansCdfTable table;
    table.sizes = *sizes;
    table.values.resize(sizes->size());
    for (std::size_t row = 0; row < sizes->size(); ++row) {
        const auto begin = values->begin() + static_cast<std::ptrdiff_t>(row * width);
        table.values[row].assign(begin, begin + static_cast<std::ptrdiff_t>(width));
    }
    return table;
}

void set_table(std::vector<nvcr::dcvcuf::UfRansCdfTable>& cdfs,
               nvcr::dcvcuf::UfRansCdfTable table, int index) {
    if (index < 0 || index > 1) throw std::invalid_argument("invalid DCVC-UF CDF index");
    if (cdfs.size() <= static_cast<std::size_t>(index))
        cdfs.resize(static_cast<std::size_t>(index) + 1);
    cdfs[static_cast<std::size_t>(index)] = std::move(table);
}

template<class Codec>
void configure(Codec& codec,
               const std::vector<nvcr::dcvcuf::UfRansCdfTable>& cdfs,
               int parallel) {
    if (parallel < 1 || parallel > MAX_EC_PARALLEL)
        throw std::invalid_argument("DCVC-UF entropy parallelism must be 1..8");
    checked(codec.set_entropy_coder_parallel(static_cast<std::size_t>(parallel)));
    if (cdfs.size() != 2)
        throw std::invalid_argument("DCVC-UF requires z and y CDF tables");
    for (const auto& cdf : cdfs) checked(codec.add_cdf(cdf));
}

}  // namespace

void RansEncoder::rebuild() {
    codec_ = std::make_unique<nvcr::dcvcuf::UfRansCodec>();
    configure(*codec_, cdfs_, parallel_);
    encoded_.clear();
}

void RansEncoder::set_cdf(
    const std::shared_ptr<std::vector<std::int32_t>>& values,
    const std::shared_ptr<std::vector<std::int32_t>>& sizes, int index) {
    set_table(cdfs_, make_table(values, sizes), index);
}

void RansEncoder::set_entropy_coder_parallel(int count) {
    parallel_ = count;
    rebuild();
}

void RansEncoder::reset() {
    if (codec_) codec_->reset_encoder();
    encoded_.clear();
}

void RansEncoder::encode_y(
    const std::shared_ptr<std::vector<std::int16_t>>& symbols, int size) {
    if (!codec_ || !symbols || size < 0 ||
        static_cast<std::size_t>(size) > symbols->size())
        throw std::invalid_argument("invalid DCVC-UF y symbols");
    checked(codec_->encode_y(
        std::span<const std::int16_t>(symbols->data(), static_cast<std::size_t>(size)), 1));
}

void RansEncoder::encode_z(
    const std::shared_ptr<std::vector<std::int8_t>>& symbols,
    int cdf_offset, int channels) {
    if (!codec_ || !symbols || cdf_offset < 0 || channels <= 0)
        throw std::invalid_argument("invalid DCVC-UF z symbols");
    checked(codec_->encode_z(*symbols, 0, static_cast<std::size_t>(cdf_offset),
                             static_cast<std::size_t>(channels)));
}

void RansEncoder::flush() {
    auto bytes = checked(codec_->finish_encode());
    encoded_.resize(bytes.size());
    if (!bytes.empty()) std::memcpy(encoded_.data(), bytes.data(), bytes.size());
}

const std::vector<std::uint8_t>& RansEncoder::get_encoded_stream() const noexcept {
    return encoded_;
}

void RansDecoder::rebuild() {
    codec_ = std::make_unique<nvcr::dcvcuf::UfRansCodec>();
    configure(*codec_, cdfs_, parallel_);
    decoded_->clear();
}

void RansDecoder::set_cdf(
    const std::shared_ptr<std::vector<std::int32_t>>& values,
    const std::shared_ptr<std::vector<std::int32_t>>& sizes, int index) {
    set_table(cdfs_, make_table(values, sizes), index);
}

void RansDecoder::set_entropy_coder_parallel(int count) {
    parallel_ = count;
    rebuild();
}

void RansDecoder::fail(nvcr::Error error, std::size_t symbols) {
    if (!error_) error_ = std::move(error);
    decoded_->assign(symbols, 0);
}

void RansDecoder::set_stream(const std::uint8_t* data, int size) {
    error_.reset();
    decoded_->clear();
    if (!codec_ || !data || size <= 0) {
        fail(nvcr::Error(nvcr::ErrorCode::malformed_bitstream,
                         "DCVC-UF entropy stream is empty", "dcvcuf.rans"),
             0);
        return;
    }
    auto set = codec_->set_stream(std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data), static_cast<std::size_t>(size)));
    if (!set) fail(set.error(), 0);
}

void RansDecoder::decode_z(int total_size, int cdf_offset, int channels) {
    const auto count = static_cast<std::size_t>(std::max(total_size, 0));
    if (error_) return fail(*error_, count);
    if (!codec_ || total_size < 0 || cdf_offset < 0 || channels <= 0)
        return fail(nvcr::Error(nvcr::ErrorCode::invalid_argument,
                                "invalid DCVC-UF z decode request", "dcvcuf.rans"),
                    count);
    auto decoded = codec_->decode_z(
        count, 0, static_cast<std::size_t>(cdf_offset), static_cast<std::size_t>(channels));
    if (!decoded) return fail(decoded.error(), count);
    *decoded_ = std::move(decoded.value());
}

void RansDecoder::decode_y(
    const std::shared_ptr<std::vector<std::uint8_t>>& indexes, int size) {
    const auto count = static_cast<std::size_t>(std::max(size, 0));
    if (error_) return fail(*error_, count);
    if (!codec_ || !indexes || size < 0 || count > indexes->size())
        return fail(nvcr::Error(nvcr::ErrorCode::invalid_argument,
                                "invalid DCVC-UF y decode request", "dcvcuf.rans"),
                    count);
    auto decoded = codec_->decode_y(std::span<const std::uint8_t>(indexes->data(), count), 1);
    if (!decoded) return fail(decoded.error(), count);
    *decoded_ = std::move(decoded.value());
}

nvcr::Result<void> RansDecoder::finish() {
    if (error_) {
        auto error = std::move(*error_);
        error_.reset();
        return error;
    }
    if (!codec_)
        return nvcr::Error(nvcr::ErrorCode::invalid_state,
                           "DCVC-UF entropy decoder is not configured", "dcvcuf.rans");
    return codec_->finish_decode();
}

std::shared_ptr<std::vector<std::int8_t>>
RansDecoder::get_decoded_tensor_cpp() const noexcept {
    return decoded_;
}
