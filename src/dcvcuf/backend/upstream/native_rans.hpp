#pragma once

#include "uf_rans_codec.hpp"

#include <cstdint>
#include <memory>
#include <optional>
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

// Upstream calls set_stream, decode_z and decode_y on its entropy worker
// thread, where an exception would terminate the process. Stream-driven
// failures are therefore recorded instead of thrown: the first one sticks until
// the next set_stream, later calls return zero symbols of the requested count
// (so upstream's copies stay in bounds), and the NVCR session reads the result
// through finish() once upstream's decompress returns.
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
    // The recorded failure, else the end-of-stream state check.
    [[nodiscard]] nvcr::Result<void> finish();

private:
    void rebuild();
    void fail(nvcr::Error error, std::size_t symbols);
    std::vector<nvcr::dcvcuf::UfRansCdfTable> cdfs_;
    std::unique_ptr<nvcr::dcvcuf::UfRansCodec> codec_;
    std::optional<nvcr::Error> error_;
    std::shared_ptr<std::vector<std::int8_t>> decoded_{
        std::make_shared<std::vector<std::int8_t>>()};
    int parallel_{1};
};
