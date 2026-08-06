#include "nvcr/artifacts/catalog.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace nvcr::artifacts {
namespace {

Error catalog_error(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), "artifact-catalog");
}

struct JsonValue final {
    enum class Kind : std::uint8_t {
        null_value,
        boolean,
        integer,
        string,
        array,
        object,
    };

    Kind kind{Kind::null_value};
    bool boolean_value{false};
    std::uint64_t integer_value{0};
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::map<std::string, JsonValue, std::less<>> object_value;
};

class JsonParser final {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    [[nodiscard]] Result<JsonValue> parse() {
        skip_whitespace();
        auto value = parse_value();
        if (!value) return value.error();
        skip_whitespace();
        if (offset_ != input_.size()) {
            return fail("trailing JSON data at offset " + std::to_string(offset_));
        }
        return value;
    }

private:
    [[nodiscard]] Error fail(std::string message) const {
        return catalog_error(std::move(message));
    }

    void skip_whitespace() noexcept {
        while (offset_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[offset_])) != 0) {
            ++offset_;
        }
    }

    [[nodiscard]] Result<JsonValue> parse_value() {
        skip_whitespace();
        if (offset_ >= input_.size()) return fail("unexpected end of JSON");
        switch (input_[offset_]) {
        case '{': return parse_object();
        case '[': return parse_array();
        case '"': {
            auto string = parse_string();
            if (!string) return string.error();
            JsonValue value;
            value.kind = JsonValue::Kind::string;
            value.string_value = std::move(string.value());
            return value;
        }
        case 't': return parse_literal("true", JsonValue::Kind::boolean, true);
        case 'f': return parse_literal("false", JsonValue::Kind::boolean, false);
        case 'n': return parse_null();
        default:
            if (input_[offset_] >= '0' && input_[offset_] <= '9') {
                return parse_integer();
            }
            return fail("unexpected JSON token at offset " + std::to_string(offset_));
        }
    }

    [[nodiscard]] Result<JsonValue> parse_object() {
        ++offset_;
        JsonValue value;
        value.kind = JsonValue::Kind::object;
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
            auto member = parse_value();
            if (!member) return member.error();
            if (!value.object_value.emplace(std::move(key.value()), std::move(member.value())).second) {
                return fail("duplicate object key");
            }
            skip_whitespace();
            if (consume('}')) return value;
            if (!consume(',')) return fail("object member is missing ','");
        }
    }

    [[nodiscard]] Result<JsonValue> parse_array() {
        ++offset_;
        JsonValue value;
        value.kind = JsonValue::Kind::array;
        skip_whitespace();
        if (consume(']')) return value;

        while (true) {
            auto element = parse_value();
            if (!element) return element.error();
            value.array_value.push_back(std::move(element.value()));
            skip_whitespace();
            if (consume(']')) return value;
            if (!consume(',')) return fail("array element is missing ','");
        }
    }

    [[nodiscard]] Result<JsonValue> parse_literal(
        std::string_view literal, JsonValue::Kind kind, bool boolean_value) {
        if (input_.substr(offset_, literal.size()) != literal) {
            return fail("invalid JSON literal at offset " + std::to_string(offset_));
        }
        offset_ += literal.size();
        JsonValue value;
        value.kind = kind;
        value.boolean_value = boolean_value;
        return value;
    }

    [[nodiscard]] Result<JsonValue> parse_null() {
        return parse_literal("null", JsonValue::Kind::null_value, false);
    }

    [[nodiscard]] Result<JsonValue> parse_integer() {
        const std::size_t start = offset_;
        while (offset_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[offset_])) != 0) {
            ++offset_;
        }
        if (offset_ - start > 1U && input_[start] == '0') {
            return fail("JSON integer has a leading zero at offset " + std::to_string(start));
        }
        if (offset_ < input_.size() &&
            (input_[offset_] == '.' || input_[offset_] == 'e' || input_[offset_] == 'E')) {
            return fail("catalog values must use integer JSON numbers");
        }
        std::uint64_t integer = 0;
        const auto parsed = std::from_chars(
            input_.data() + start, input_.data() + offset_, integer);
        if (parsed.ec != std::errc{} || parsed.ptr != input_.data() + offset_) {
            return fail("invalid JSON integer at offset " + std::to_string(start));
        }
        JsonValue value;
        value.kind = JsonValue::Kind::integer;
        value.integer_value = integer;
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
                auto code_point = parse_unicode_escape();
                if (!code_point) return code_point.error();
                append_utf8(value, code_point.value());
                break;
            }
            default: return fail("invalid JSON escape");
            }
        }
        return fail("unterminated JSON string");
    }

    [[nodiscard]] Result<std::uint32_t> parse_unicode_escape() {
        if (input_.size() - offset_ < 4U) return fail("truncated unicode escape");
        std::uint32_t code_point = 0;
        for (unsigned index = 0; index < 4U; ++index) {
            const char character = input_[offset_++];
            code_point <<= 4U;
            if (character >= '0' && character <= '9') {
                code_point |= static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                code_point |= static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                code_point |= static_cast<std::uint32_t>(character - 'A' + 10);
            } else {
                return fail("invalid unicode escape");
            }
        }
        if (code_point >= 0xd800U && code_point <= 0xdfffU) {
            return fail("surrogate unicode escapes are not supported");
        }
        return code_point;
    }

    static void append_utf8(std::string& output, std::uint32_t code_point) {
        if (code_point <= 0x7fU) {
            output.push_back(static_cast<char>(code_point));
        } else if (code_point <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (code_point >> 6U)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xe0U | (code_point >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
        }
    }

    bool consume(char expected) noexcept {
        if (offset_ < input_.size() && input_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    std::string_view input_;
    std::size_t offset_{0};
};

const JsonValue* member(const JsonValue& object, std::string_view key) {
    if (object.kind != JsonValue::Kind::object) return nullptr;
    const auto iterator = object.object_value.find(key);
    return iterator == object.object_value.end() ? nullptr : &iterator->second;
}

Result<std::string> required_string(
    const JsonValue& object, std::string_view key, std::string_view context) {
    const auto* value = member(object, key);
    if (value == nullptr || value->kind != JsonValue::Kind::string || value->string_value.empty()) {
        return catalog_error(std::string(context) + " requires non-empty string '" +
                             std::string(key) + "'");
    }
    return value->string_value;
}

Result<std::uint64_t> required_integer(
    const JsonValue& object, std::string_view key, std::string_view context) {
    const auto* value = member(object, key);
    if (value == nullptr || value->kind != JsonValue::Kind::integer) {
        return catalog_error(std::string(context) + " requires integer '" +
                             std::string(key) + "'");
    }
    return value->integer_value;
}

bool safe_identifier(std::string_view value) {
    if (value.empty() || value.size() > 128U) return false;
    return std::ranges::all_of(value, [](unsigned char character) {
        return std::isalnum(character) != 0 || character == '.' ||
            character == '_' || character == '-';
    });
}

bool valid_hex_digest(std::string_view value) {
    return value.size() == 64U && std::ranges::all_of(value, [](unsigned char character) {
        return std::isxdigit(character) != 0 &&
            (std::isdigit(character) != 0 || std::islower(character) != 0);
    });
}

Result<std::vector<std::byte>> decode_digest(std::string_view value) {
    if (!valid_hex_digest(value)) return catalog_error("catalog SHA-256 digest is invalid");
    std::vector<std::byte> digest;
    digest.reserve(32U);
    for (std::size_t index = 0; index < value.size(); index += 2U) {
        unsigned byte = 0;
        const auto parsed = std::from_chars(
            value.data() + index, value.data() + index + 2U, byte, 16);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + index + 2U) {
            return catalog_error("catalog SHA-256 digest is invalid");
        }
        digest.push_back(static_cast<std::byte>(byte));
    }
    return digest;
}

std::string expected_filename(const CatalogEntry& entry) {
    const std::string architecture = entry.architecture == "x86_64" ? "amd64" : "arm64";
    std::string target = entry.target_profile_id;
    if (entry.hardware_compatibility == HardwareCompatibility::same_compute_capability) {
        target = "linux-" + architecture + "-sm" +
            std::to_string(entry.compute_capability_major) +
            std::to_string(entry.compute_capability_minor);
    } else if (entry.hardware_compatibility == HardwareCompatibility::ampere_plus) {
        target = "linux-" + architecture + "-ampere-plus";
    }
    return "nvcr-engines-" + target + "-" + entry.model_profile_id + "-" +
        entry.profile + ".tar.gz";
}

Result<HardwareCompatibility> parse_compatibility(
    const JsonValue& object, std::string_view context) {
    const auto* value = member(object, "hardware_compatibility");
    if (value == nullptr) return HardwareCompatibility::exact;
    if (value->kind != JsonValue::Kind::string) {
        return catalog_error(std::string(context) + " hardware_compatibility must be a string");
    }
    if (value->string_value == "exact") return HardwareCompatibility::exact;
    if (value->string_value == "same_compute_capability") {
        return HardwareCompatibility::same_compute_capability;
    }
    if (value->string_value == "ampere_plus") return HardwareCompatibility::ampere_plus;
    return catalog_error(std::string(context) + " has invalid hardware_compatibility");
}

Result<CatalogEntry> parse_entry(const JsonValue& value, std::size_t index) {
    const std::string context = "catalog asset " + std::to_string(index);
    if (value.kind != JsonValue::Kind::object) {
        return catalog_error(context + " must be an object");
    }
    CatalogEntry entry;
    auto set_string = [&](std::string& destination, std::string_view key) -> Result<void> {
        auto result = required_string(value, key, context);
        if (!result) return result.error();
        destination = std::move(result.value());
        return {};
    };
    if (auto result = set_string(entry.backend, "backend"); !result) return result.error();
    if (auto result = set_string(entry.model_profile_id, "model_profile_id"); !result) return result.error();
    if (auto result = set_string(entry.target_profile_id, "target_profile_id"); !result) return result.error();
    if (auto result = set_string(entry.profile, "profile"); !result) return result.error();
    if (auto result = set_string(entry.precision, "precision"); !result) return result.error();
    if (auto result = set_string(entry.operating_system, "operating_system"); !result) return result.error();
    if (auto result = set_string(entry.architecture, "architecture"); !result) return result.error();
    if (auto result = set_string(entry.device_name, "device_name"); !result) return result.error();
    if (auto result = set_string(entry.filename, "filename"); !result) return result.error();
    if (auto result = set_string(entry.sha256, "sha256"); !result) return result.error();

    auto set_integer = [&](std::uint32_t& destination, std::string_view key) -> Result<void> {
        auto result = required_integer(value, key, context);
        if (!result) return result.error();
        if (result.value() > std::numeric_limits<std::uint32_t>::max()) {
            return catalog_error(context + " integer is out of range: " + std::string(key));
        }
        destination = static_cast<std::uint32_t>(result.value());
        return {};
    };
    auto size = required_integer(value, "size_bytes", context);
    if (!size) return size.error();
    entry.size_bytes = size.value();
    if (auto result = set_integer(entry.compute_capability_major, "compute_capability_major"); !result) return result.error();
    if (auto result = set_integer(entry.compute_capability_minor, "compute_capability_minor"); !result) return result.error();
    if (auto result = set_integer(entry.multiprocessor_count, "multiprocessor_count"); !result) return result.error();
    if (auto result = set_integer(entry.cuda_runtime_version, "cuda_runtime_version"); !result) return result.error();
    if (auto result = set_integer(entry.tensorrt_version_major, "tensorrt_version_major"); !result) return result.error();
    if (auto result = set_integer(entry.tensorrt_version_minor, "tensorrt_version_minor"); !result) return result.error();
    if (auto result = set_integer(entry.tensorrt_version_patch, "tensorrt_version_patch"); !result) return result.error();
    auto compatibility = parse_compatibility(value, context);
    if (!compatibility) return compatibility.error();
    entry.hardware_compatibility = compatibility.value();

    if (!safe_identifier(entry.backend) || !safe_identifier(entry.model_profile_id) ||
        !safe_identifier(entry.target_profile_id)) {
        return catalog_error(context + " has an unsafe identity field");
    }
    static constexpr std::string_view profiles[] = {
        "qcif", "cif", "360p", "540p", "720p", "1080p"};
    if (std::ranges::find(profiles, entry.profile) == std::end(profiles)) {
        return catalog_error(context + " has an unsupported profile");
    }
    if (entry.precision != "fp16") {
        return catalog_error(context + " precision must be fp16");
    }
    if (entry.operating_system != "linux" ||
        (entry.architecture != "x86_64" && entry.architecture != "aarch64")) {
        return catalog_error(context + " has an unsupported platform");
    }
    if (entry.architecture == "aarch64" &&
        entry.hardware_compatibility != HardwareCompatibility::exact) {
        return catalog_error(context + " uses compatibility mode on AArch64");
    }
    if (entry.size_bytes == 0U || entry.compute_capability_major == 0U ||
        entry.multiprocessor_count == 0U || entry.cuda_runtime_version == 0U ||
        entry.tensorrt_version_major == 0U || !valid_hex_digest(entry.sha256)) {
        return catalog_error(context + " has invalid size, runtime, capability, or digest data");
    }
    if (entry.filename != expected_filename(entry)) {
        return catalog_error(context + " filename does not match its identity");
    }
    return entry;
}

Result<std::string> read_file(const std::filesystem::path& path) {
    std::error_code status;
    const auto size = std::filesystem::file_size(path, status);
    if (status) return Error(ErrorCode::io_error, "cannot stat catalog: " + path.string(), "artifact-catalog");
    if (size > 16U * 1024U * 1024U) {
        return Error(ErrorCode::resource_exhausted, "catalog exceeds 16 MiB", "artifact-catalog");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) return Error(ErrorCode::io_error, "cannot open catalog: " + path.string(), "artifact-catalog");
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        return Error(ErrorCode::io_error, "cannot read catalog: " + path.string(), "artifact-catalog");
    }
    return contents.str();
}

}  // namespace

Result<Catalog> Catalog::from_json(std::string_view document) {
    try {
        JsonParser parser(document);
        auto root = parser.parse();
        if (!root) return root.error();
        auto schema = required_string(root.value(), "schema", "catalog");
        if (!schema) return schema.error();
        if (schema.value() != catalog_schema) {
            return catalog_error("catalog must use schema " + std::string(catalog_schema));
        }
        const auto* assets = member(root.value(), "assets");
        if (assets == nullptr || assets->kind != JsonValue::Kind::array) {
            return catalog_error("catalog requires an assets array");
        }

        Catalog output;
        std::set<std::string> identities;
        std::set<std::string> filenames;
        for (std::size_t index = 0; index < assets->array_value.size(); ++index) {
            auto entry = parse_entry(assets->array_value[index], index);
            if (!entry) return entry.error();
            const auto identity = entry.value().backend + "\x1f" +
                entry.value().model_profile_id + "\x1f" +
                entry.value().target_profile_id + "\x1f" + entry.value().profile + "\x1f" +
                std::to_string(static_cast<unsigned>(entry.value().hardware_compatibility));
            if (!identities.insert(identity).second) {
                return catalog_error("catalog contains duplicate asset identity");
            }
            if (!filenames.insert(entry.value().filename).second) {
                return catalog_error("catalog contains duplicate asset filename");
            }
            output.entries_.push_back(std::move(entry.value()));
        }
        return output;
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "catalog allocation failed", "artifact-catalog");
    }
}

Result<Catalog> Catalog::from_file(const std::filesystem::path& path) {
    auto document = read_file(path);
    if (!document) return document.error();
    return from_json(document.value());
}

Result<void> Catalog::append_candidates(
    Resolver& resolver,
    const CatalogLoadOptions& options) const {
    if (options.codec_id.empty() || options.provider_id.empty()) {
        return catalog_error("catalog candidate conversion requires codec_id and provider_id");
    }
    for (const auto& entry : entries_) {
        auto digest = decode_digest(entry.sha256);
        if (!digest) return digest.error();
        ArtifactCandidate candidate;
        candidate.artifact.codec_id = options.codec_id;
        candidate.artifact.model_set_id = entry.model_profile_id;
        candidate.artifact.component_id = entry.profile;
        candidate.artifact.provider_id = options.provider_id;
        candidate.artifact.precision = entry.precision;
        candidate.artifact.expected_digest = digest.value();
        candidate.target_profile_id = entry.target_profile_id;
        candidate.target_device = entry.device_name;
        candidate.compute_capability = std::to_string(entry.compute_capability_major) + "." +
            std::to_string(entry.compute_capability_minor);
        candidate.operating_system = entry.operating_system;
        candidate.architecture = entry.architecture;
        candidate.runtime_version = std::to_string(entry.tensorrt_version_major) + "." +
            std::to_string(entry.tensorrt_version_minor) + "." +
            std::to_string(entry.tensorrt_version_patch);
        candidate.cuda_runtime_version = entry.cuda_runtime_version;
        candidate.hardware_compatibility = entry.hardware_compatibility;
        candidate.digest = std::move(digest.value());
        candidate.license_restricted = options.mark_license_restricted;
        if (options.artifact_root.empty()) {
            candidate.artifact.path = entry.filename;
            candidate.available = false;
        } else {
            const auto path = options.artifact_root / entry.filename;
            candidate.artifact.path = path.string();
            std::error_code status;
            candidate.available = std::filesystem::is_regular_file(path, status) && !status;
        }
        resolver.add(std::move(candidate));
    }
    return {};
}

}  // namespace nvcr::artifacts
