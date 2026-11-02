#pragma once

// Minimal JSON document reader for codec-owned artifact metadata.
//
// Numbers keep both an exact signed integer (when the token is an integer that
// fits) and a double. Nesting depth is bounded so malformed metadata cannot
// exhaust the stack.

#include "nvcr/common/error.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nvcr::detail::json {

struct Value final {
    enum class Kind : std::uint8_t {
        null_value,
        boolean,
        number,
        string,
        array,
        object,
    };

    Kind kind{Kind::null_value};
    bool boolean_value{false};
    double number_value{0.0};
    std::optional<std::int64_t> integer_value;
    std::string string_value;
    std::vector<Value> array_value;
    std::map<std::string, Value, std::less<>> object_value;

    [[nodiscard]] bool is_null() const noexcept { return kind == Kind::null_value; }
    [[nodiscard]] const Value* find(std::string_view key) const noexcept;
};

[[nodiscard]] Result<Value> parse(std::string_view document, std::string_view subsystem);
[[nodiscard]] Result<Value> parse_file(
    const std::filesystem::path& path, std::string_view subsystem);

}  // namespace nvcr::detail::json
