#include "nvcr/bitstream/numerics_identity.hpp"

#include "../common/sha256.hpp"

#include <algorithm>
#include <set>
#include <string_view>

namespace nvcr {
namespace {

constexpr std::size_t maximum_id_bytes = 128;
constexpr std::size_t maximum_stages = 64;
constexpr std::size_t digest_bytes = 32;

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, std::move(message), "numerics-identity");
}

Error malformed(std::string message) {
    return Error(ErrorCode::malformed_bitstream, std::move(message), "numerics-identity");
}

bool valid_id(std::string_view id) noexcept {
    return !id.empty() && id.size() <= maximum_id_bytes &&
        std::all_of(id.begin(), id.end(), [](char character) {
            const auto value = static_cast<unsigned char>(character);
            return value > 0x20U && value < 0x7fU;
        });
}

int hex_value(char character) noexcept {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    return -1;
}

}  // namespace

Result<NumericsIdentity> NumericsIdentityIO::make(
    std::string profile, std::span<const NormativeStageIdentity> stages) {
    if (!valid_id(profile)) return invalid("numerics profile id is invalid");
    if (stages.empty() || stages.size() > maximum_stages) {
        return invalid("numerics identity must cover 1 to 64 normative stages");
    }
    std::set<std::string_view> seen;
    std::string canonical;
    for (const auto& stage : stages) {
        if (!valid_id(stage.stage_id)) return invalid("normative stage id is invalid");
        if (!seen.insert(stage.stage_id).second) return invalid("normative stage listed twice");
        canonical.push_back(static_cast<char>(stage.stage_id.size()));
        canonical += stage.stage_id;
        for (const auto byte : stage.program_digest) canonical.push_back(static_cast<char>(byte));
    }
    const auto hex = detail::sha256_text(canonical);
    NumericsIdentity identity;
    identity.profile = std::move(profile);
    for (std::size_t index = 0; index < digest_bytes; ++index) {
        const int high = hex_value(hex[2 * index]);
        const int low = hex_value(hex[2 * index + 1]);
        if (high < 0 || low < 0) return invalid("program-set digest is not SHA-256");
        identity.program_set_digest[index] = static_cast<std::byte>((high << 4) | low);
    }
    return identity;
}

Result<std::vector<std::byte>> NumericsIdentityIO::serialize(const NumericsIdentity& identity) {
    if (!valid_id(identity.profile)) return invalid("numerics profile id is invalid");
    std::vector<std::byte> bytes;
    bytes.reserve(1 + identity.profile.size() + digest_bytes);
    bytes.push_back(static_cast<std::byte>(identity.profile.size()));
    for (const char character : identity.profile) bytes.push_back(static_cast<std::byte>(character));
    bytes.insert(bytes.end(), identity.program_set_digest.begin(), identity.program_set_digest.end());
    return bytes;
}

Result<NumericsIdentity> NumericsIdentityIO::deserialize(std::span<const std::byte> bytes) {
    if (bytes.empty()) return malformed("numerics identity is truncated");
    const auto length = static_cast<std::size_t>(bytes[0]);
    if (bytes.size() < 1 + length + digest_bytes) return malformed("numerics identity is truncated");
    if (bytes.size() > 1 + length + digest_bytes) return malformed("numerics identity has trailing bytes");
    NumericsIdentity identity;
    for (const auto byte : bytes.subspan(1, length)) identity.profile.push_back(static_cast<char>(byte));
    if (!valid_id(identity.profile)) return malformed("numerics identity contains an invalid profile id");
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(1 + length), digest_bytes,
                identity.program_set_digest.begin());
    return identity;
}

}  // namespace nvcr
