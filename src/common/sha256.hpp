#pragma once

#include "nvcr/common/error.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace nvcr::detail {

[[nodiscard]] Result<std::string> sha256_file(
    const std::filesystem::path& path,
    std::string_view subsystem);

}  // namespace nvcr::detail
