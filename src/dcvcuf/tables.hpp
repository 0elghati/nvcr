#pragma once

// DCVC-UF's per-model tables from a bundle's tables.bin
// (scripts/backends/dcvcuf/add_tensorrt_tables.py): the per-QP quantization
// rows (FP16) and the entropy coder's z and y CDFs, for the image and the HT-S
// model.

#include "entropy/uf_rans_codec.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace nvcr::dcvcuf {

struct Tables final {
    std::map<std::string, std::vector<std::uint16_t>> q;  // name -> 64 rows of FP16
    std::map<std::string, int> q_width;
    UfRansCdfTable z;
    UfRansCdfTable y;
};

// {image, video}; throws std::runtime_error on a malformed file.
[[nodiscard]] std::pair<Tables, Tables> read_tables(const std::filesystem::path& path);

}  // namespace nvcr::dcvcuf
