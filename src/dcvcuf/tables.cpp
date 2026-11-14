#include "tables.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace nvcr::dcvcuf {
namespace {

constexpr int qp_count = 64;

struct TableReader final {
    std::vector<char> bytes;
    std::size_t offset{};

    template <class T>
    std::vector<T> take(std::size_t count) {
        std::vector<T> values(count);
        const auto size = count * sizeof(T);
        if (offset + size > bytes.size()) throw std::runtime_error("tables.bin is truncated");
        std::memcpy(values.data(), bytes.data() + offset, size);
        offset += size;
        return values;
    }

    UfRansCdfTable cdf(std::size_t rows) {
        auto values = take<std::int32_t>(rows * 19);
        auto sizes = take<std::int32_t>(rows);
        UfRansCdfTable table;
        table.sizes = sizes;
        for (std::size_t row = 0; row < rows; ++row) {
            table.values.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(row * 19),
                                      values.begin() + static_cast<std::ptrdiff_t>(row * 19 + 19));
        }
        return table;
    }

    void q(Tables& tables, const std::string& name, int width) {
        tables.q[name] = take<std::uint16_t>(static_cast<std::size_t>(qp_count) * width);
        tables.q_width[name] = width;
    }
};

}  // namespace

std::pair<Tables, Tables> read_tables(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    TableReader reader{{std::istreambuf_iterator<char>(file), {}}};
    Tables image;
    Tables video;
    reader.q(image, "q_scale_enc", 384);
    reader.q(image, "q_scale_dec", 384);
    reader.q(image, "q_scale_y_enc", 256);
    reader.q(image, "q_scale_y_dec", 256);
    image.z = reader.cdf(8192);
    image.y = reader.cdf(128);
    reader.q(video, "q_encoder", 512);
    reader.q(video, "q_decoder", 512);
    reader.q(video, "q_feature", 512);
    video.z = reader.cdf(8192);
    video.y = reader.cdf(128);
    if (reader.offset != reader.bytes.size()) throw std::runtime_error("tables.bin has extra bytes");
    return {std::move(image), std::move(video)};
}

}  // namespace nvcr::dcvcuf
