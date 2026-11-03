#pragma once

#include "nvcr/common/error.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace nvcr::detail {

[[nodiscard]] Result<std::string> sha256_file(
    const std::filesystem::path& path,
    std::string_view subsystem);

// Digest of the files' bytes concatenated in order.
[[nodiscard]] Result<std::string> sha256_files(
    std::span<const std::filesystem::path> paths,
    std::string_view subsystem);

// Digest of an in-memory string, as lowercase hex.
[[nodiscard]] std::string sha256_text(std::string_view text);

}  // namespace nvcr::detail
