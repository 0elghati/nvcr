#pragma once

#include "uf_rans_codec.hpp"

#include <cstdint>
#include <memory>
#include <vector>

inline constexpr int MAX_EC_PARALLEL = 8;

class RansEncoder final {
public:
    void encode_y(const std::shared_ptr<std::vector<std::int16_t>>& symbols, int size);
    void encode_z(const std::shared_ptr<std::vector<std::int8_t>>& symbols,
                  int cdf_offset, int channels);
    void flush();
    const std::vector<std::uint8_t>& get_encoded_stream() const noexcept;
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<std::int32_t>>& values,
                 const std::shared_ptr<std::vector<std::int32_t>>& sizes,
                 int index);
    void set_entropy_coder_parallel(int count);

private:
    void rebuild();
    std::vector<nvcr::dcvcuf::UfRansCdfTable> cdfs_;
    std::unique_ptr<nvcr::dcvcuf::UfRansCodec> codec_;
    std::vector<std::uint8_t> encoded_;
    int parallel_{1};
};

class RansDecoder final {
public:
    void decode_y(const std::shared_ptr<std::vector<std::uint8_t>>& indexes, int size);
    void decode_z(int total_size, int cdf_offset, int channels);
    std::shared_ptr<std::vector<std::int8_t>> get_decoded_tensor_cpp() const noexcept;
    void set_cdf(const std::shared_ptr<std::vector<std::int32_t>>& values,
                 const std::shared_ptr<std::vector<std::int32_t>>& sizes,
                 int index);
    void set_entropy_coder_parallel(int count);
    void set_stream(const std::uint8_t* data, int size);

private:
    void rebuild();
    std::vector<nvcr::dcvcuf::UfRansCdfTable> cdfs_;
    std::unique_ptr<nvcr::dcvcuf::UfRansCodec> codec_;
    std::shared_ptr<std::vector<std::int8_t>> decoded_{
        std::make_shared<std::vector<std::int8_t>>()};
    int parallel_{1};
};
