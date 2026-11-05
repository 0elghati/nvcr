#pragma once

#include "nvcr/common/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nvcr {

// Payload of the NVAU v2 numerics_identity section (issue #216). A stream that
// carries it decodes only on a decoder that runs every normative stage with
// exactly the same programs under the named NVCR numerics profile. The codec
// decides which stages are normative; the runtime defines the format. It is a
// sequence property: encoders write it on random-access units, and the
// following units of the sequence inherit it.
//
// Version 1:
//   u8  profile length (1..128), profile id bytes (printable ASCII)
//   32  program-set digest
// The program-set digest is the SHA-256 of, for every normative stage in the
// codec's declared order: u8 stage id length (1..128), stage id bytes, and
// the 32-byte SHA-256 of the stage's program. Trailing bytes are malformed.
struct NormativeStageIdentity final {
    std::string stage_id;
    std::array<std::byte, 32> program_digest{};

    friend bool operator==(const NormativeStageIdentity&, const NormativeStageIdentity&) = default;
};

struct NumericsIdentity final {
    std::string profile;
    std::array<std::byte, 32> program_set_digest{};

    friend bool operator==(const NumericsIdentity&, const NumericsIdentity&) = default;
};

class NumericsIdentityIO final {
public:
    static constexpr std::uint16_t format_version = 1;

    // Identity of a profile and its normative stages (1..64, unique ids).
    [[nodiscard]] static Result<NumericsIdentity> make(
        std::string profile, std::span<const NormativeStageIdentity> stages);
    [[nodiscard]] static Result<std::vector<std::byte>> serialize(const NumericsIdentity& identity);
    [[nodiscard]] static Result<NumericsIdentity> deserialize(std::span<const std::byte> bytes);
};

}  // namespace nvcr
