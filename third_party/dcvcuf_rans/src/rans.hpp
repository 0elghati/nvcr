// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Modified for NVCR: isolated in the DCVC-UF codec namespace; bounded decoding
// with end-of-stream state queries (see UPSTREAM.md).

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nvcr::dcvcuf::upstream_rans {

struct RansSymbol {
    uint16_t start;
    uint16_t range;
};

enum class WorkType {
    EncodeDecodeY,
    EncodeDecodeZ,
    Flush,
};

struct PendingTask {
    WorkType workType;
    std::shared_ptr<std::vector<int16_t>> symbols_y;
    std::shared_ptr<std::vector<int8_t>> symbols_z;
    std::shared_ptr<std::vector<uint8_t>> indexes;
    int8_t* decoded_ptr{ nullptr };
    int total_size{ 0 };
    int cdf_offset{ 0 };
    int symbol_size{ 0 };
    int symbol_offset{ 0 };
    int ch{ 0 };

    PendingTask() = default;
    PendingTask(PendingTask&&) noexcept = default;
    PendingTask& operator=(PendingTask&&) noexcept = default;

    PendingTask(const PendingTask&) = delete;
    PendingTask& operator=(const PendingTask&) = delete;
};

using RansState = uint32_t;

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class RansEncoderLib {
public:
    RansEncoderLib();
    virtual ~RansEncoderLib();

    RansEncoderLib(const RansEncoderLib&) = delete;
    RansEncoderLib(RansEncoderLib&&) = delete;
    RansEncoderLib& operator=(const RansEncoderLib&) = delete;
    RansEncoderLib& operator=(RansEncoderLib&&) = delete;

    void check_buffer_capacity(int symbol_count);
    void encode_y(const std::shared_ptr<std::vector<int16_t>>& symbols, const int symbol_size,
                  const int symbol_offset);
    void encode_y_internal(const std::shared_ptr<std::vector<int16_t>>& symbols,
                           const int symbol_size, const int symbol_offset);
    void encode_z(const std::shared_ptr<std::vector<int8_t>>& symbols, const int symbol_size,
                  const int symbol_offset, const int cdf_offset, const int ch);
    void encode_z_internal(const std::shared_ptr<std::vector<int8_t>>& symbols, const int symbol_size,
                           const int symbol_offset, const int cdf_offset, const int ch);
    void flush();
    std::shared_ptr<std::vector<uint8_t>> get_encoded_stream();
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<std::vector<RansSymbol>>>& ransSymbols,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);
    void worker();

private:
    static const int max_stream_buffer_size = 10 * 1000 * 1000;
    uint8_t* m_stream_buffer{ nullptr };
    RansState m_rans;
    uint8_t* m_ptr{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> m_stream;

    std::vector<std::shared_ptr<std::vector<std::vector<RansSymbol>>>> m_ransSymbols;
    std::vector<std::shared_ptr<std::vector<int8_t>>> m_max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};

// Bounded stream reader: past the end it yields zeros and flags the overrun.
struct ByteReader {
    const uint8_t* ptr{ nullptr };
    const uint8_t* end{ nullptr };
    bool overrun{ false };

    uint32_t next()
    {
        if (ptr == end) {
            overrun = true;
            return 0;
        }
        return *ptr++;
    }
};

class RansDecoderLib {
public:
    RansDecoderLib();
    virtual ~RansDecoderLib();

    RansDecoderLib(const RansDecoderLib&) = delete;
    RansDecoderLib(RansDecoderLib&&) = delete;
    RansDecoderLib& operator=(const RansDecoderLib&) = delete;
    RansDecoderLib& operator=(RansDecoderLib&&) = delete;

    void decode_y(int8_t* decoded_ptr, const std::shared_ptr<std::vector<uint8_t>>& indexes,
                  const int symbol_size, const int symbol_offset);
    void decode_y_internal(int8_t* decoded_ptr, const std::shared_ptr<std::vector<uint8_t>>& indexes,
                           const int symbol_size, const int symbol_offset);
    void decode_z(int8_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                  const int cdf_offset, const int ch);
    void decode_z_internal(int8_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                           const int cdf_offset, const int ch);
    void set_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);
    void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);
    // False when the stream ended before all symbols were decoded, its
    // initial state was invalid, or a decoded symbol was impossible (sticky
    // until the next set_stream).
    bool wait_for_decoding_finish();
    // NVCR: a stream decoded with the encoder's exact parameters ends at the
    // encoder's initial state with all of its bytes consumed. Call only after
    // wait_for_decoding_finish().
    bool at_initial_state() const;
    size_t consumed_bytes() const;
    void worker();

private:
    RansState m_rans{ 0 };
    ByteReader m_in;
    bool m_corrupt{ false };
    std::shared_ptr<std::vector<uint8_t>> m_stream;

    std::vector<std::shared_ptr<std::vector<std::vector<int32_t>>>> m_cdfs;
    std::vector<std::shared_ptr<std::vector<int8_t>>> m_max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};

}  // namespace nvcr::dcvcuf::upstream_rans
