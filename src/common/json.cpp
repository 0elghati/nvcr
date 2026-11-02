#include "json.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <new>
#include <utility>

namespace nvcr::detail::json {
namespace {

constexpr std::size_t maximum_depth = 64;

class Parser final {
public:
    Parser(std::string_view input, std::string_view subsystem)
        : input_(input), subsystem_(subsystem) {}

    [[nodiscard]] Result<Value> parse() {
        auto value = parse_value(0);
        if (!value) return value.error();
        skip_whitespace();
        if (offset_ != input_.size()) {
            return fail("trailing JSON data at offset " + std::to_string(offset_));
        }
        return value;
    }

private:
    [[nodiscard]] Error fail(std::string message) const {
        return Error(ErrorCode::invalid_argument, std::move(message), std::string(subsystem_));
    }

    void skip_whitespace() noexcept {
        while (offset_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[offset_])) != 0) {
            ++offset_;
        }
    }

    bool consume(char expected) noexcept {
        if (offset_ < input_.size() && input_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    [[nodiscard]] Result<Value> parse_value(std::size_t depth) {
        if (depth > maximum_depth) return fail("JSON nesting is too deep");
        skip_whitespace();
        if (offset_ >= input_.size()) return fail("unexpected end of JSON");
        const char next = input_[offset_];
        if (next == '{') return parse_object(depth);
        if (next == '[') return parse_array(depth);
        if (next == '"') {
            auto string = parse_string();
            if (!string) return string.error();
            Value value;
            value.kind = Value::Kind::string;
            value.string_value = std::move(string.value());
            return value;
        }
        if (next == 't') return parse_literal("true", Value::Kind::boolean, true);
        if (next == 'f') return parse_literal("false", Value::Kind::boolean, false);
        if (next == 'n') return parse_literal("null", Value::Kind::null_value, false);
        if (next == '-' || (next >= '0' && next <= '9')) return parse_number();
        return fail("unexpected JSON token at offset " + std::to_string(offset_));
    }

    [[nodiscard]] Result<Value> parse_object(std::size_t depth) {
        ++offset_;
        Value value;
        value.kind = Value::Kind::object;
        skip_whitespace();
        if (consume('}')) return value;
        while (true) {
            skip_whitespace();
            if (offset_ >= input_.size() || input_[offset_] != '"') {
                return fail("object key must be a JSON string");
            }
            auto key = parse_string();
            if (!key) return key.error();
            skip_whitespace();
            if (!consume(':')) return fail("object key is missing ':'");
            auto member = parse_value(depth + 1);
            if (!member) return member.error();
            if (!value.object_value.emplace(std::move(key.value()), std::move(member.value()))
                     .second) {
                return fail("duplicate object key");
            }
            skip_whitespace();
            if (consume('}')) return value;
            if (!consume(',')) return fail("object member is missing ','");
        }
    }

    [[nodiscard]] Result<Value> parse_array(std::size_t depth) {
        ++offset_;
        Value value;
        value.kind = Value::Kind::array;
        skip_whitespace();
        if (consume(']')) return value;
        while (true) {
            auto element = parse_value(depth + 1);
            if (!element) return element.error();
            value.array_value.push_back(std::move(element.value()));
            skip_whitespace();
            if (consume(']')) return value;
            if (!consume(',')) return fail("array element is missing ','");
        }
    }

    [[nodiscard]] Result<Value> parse_literal(
        std::string_view literal, Value::Kind kind, bool boolean_value) {
        if (input_.substr(offset_, literal.size()) != literal) {
            return fail("invalid JSON literal at offset " + std::to_string(offset_));
        }
        offset_ += literal.size();
        Value value;
        value.kind = kind;
        value.boolean_value = boolean_value;
        return value;
    }

    [[nodiscard]] Result<Value> parse_number() {
        const std::size_t start = offset_;
        consume('-');
        const std::size_t digits = offset_;
        while (offset_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[offset_])) != 0) {
            ++offset_;
        }
        if (offset_ == digits) return fail("invalid JSON number at offset " + std::to_string(start));
        if (offset_ - digits > 1U && input_[digits] == '0') {
            return fail("JSON number has a leading zero at offset " + std::to_string(start));
        }
        bool integral = true;
        if (consume('.')) {
            integral = false;
            const std::size_t fraction = offset_;
            while (offset_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[offset_])) != 0) {
                ++offset_;
            }
            if (offset_ == fraction) return fail("JSON number has an empty fraction");
        }
        if (offset_ < input_.size() && (input_[offset_] == 'e' || input_[offset_] == 'E')) {
            integral = false;
            ++offset_;
            if (!consume('+')) consume('-');
            const std::size_t exponent = offset_;
            while (offset_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[offset_])) != 0) {
                ++offset_;
            }
            if (offset_ == exponent) return fail("JSON number has an empty exponent");
        }

        const char* first = input_.data() + start;
        const char* last = input_.data() + offset_;
        Value value;
        value.kind = Value::Kind::number;
        const auto parsed = std::from_chars(first, last, value.number_value);
        if (parsed.ec != std::errc{} || parsed.ptr != last || !std::isfinite(value.number_value)) {
            return fail("invalid JSON number at offset " + std::to_string(start));
        }
        if (integral) {
            std::int64_t integer = 0;
            const auto exact = std::from_chars(first, last, integer);
            if (exact.ec == std::errc{} && exact.ptr == last) value.integer_value = integer;
        }
        return value;
    }

    [[nodiscard]] Result<std::string> parse_string() {
        if (!consume('"')) return fail("expected JSON string");
        std::string value;
        while (offset_ < input_.size()) {
            const char character = input_[offset_++];
            if (character == '"') return value;
            if (static_cast<unsigned char>(character) < 0x20U) {
                return fail("control character in JSON string");
            }
            if (character != '\\') {
                value.push_back(character);
                continue;
            }
            if (offset_ >= input_.size()) return fail("unterminated JSON escape");
            const char escaped = input_[offset_++];
            switch (escaped) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u': {
                // Metadata strings are identifiers and paths; non-ASCII escapes
                // are rejected rather than transcoded.
                if (input_.size() - offset_ < 4U) return fail("truncated unicode escape");
                std::uint32_t code_point = 0;
                const auto parsed = std::from_chars(
                    input_.data() + offset_, input_.data() + offset_ + 4, code_point, 16);
                if (parsed.ec != std::errc{} || parsed.ptr != input_.data() + offset_ + 4 ||
                    code_point > 0x7fU) {
                    return fail("unsupported unicode escape");
                }
                offset_ += 4;
                value.push_back(static_cast<char>(code_point));
                break;
            }
            default: return fail("invalid JSON escape");
            }
        }
        return fail("unterminated JSON string");
    }

    std::string_view input_;
    std::string_view subsystem_;
    std::size_t offset_{0};
};

}  // namespace

const Value* Value::find(std::string_view key) const noexcept {
    if (kind != Kind::object) return nullptr;
    const auto found = object_value.find(key);
    return found == object_value.end() ? nullptr : &found->second;
}

Result<Value> parse(std::string_view document, std::string_view subsystem) {
    try {
        return Parser(document, subsystem).parse();
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "JSON document is too large",
                     std::string(subsystem));
    }
}

Result<Value> parse_file(const std::filesystem::path& path, std::string_view subsystem) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Error(ErrorCode::missing_artifact, "cannot open " + path.string(),
                     std::string(subsystem));
    }
    std::string document{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        return Error(ErrorCode::io_error, "cannot read " + path.string(), std::string(subsystem));
    }
    auto parsed = parse(document, subsystem);
    if (!parsed) {
        return Error(parsed.error().code(),
                     path.filename().string() + ": " + std::string(parsed.error().message()),
                     std::string(subsystem));
    }
    return parsed;
}

}  // namespace nvcr::detail::json
