#pragma once

// A codec session's numerics identity on its access units (issue #216).
//
// A stream encoded under an exact numerics profile states the profile and the
// digests of its normative programs on every random-access unit, as a required
// NVAU v2 section; the other units of the sequence inherit it. A decoder
// accepts a sequence only when the stream's identity equals its own, and a
// native decoder only native sequences, so a mismatch fails before any decode
// work. Shared by every codec that runs under an exact profile.

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/bitstream/numerics_identity.hpp"
#include "nvcr/common/error.hpp"
#include "nvcr/provider/experimental/session.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nvcr::codec {

// The stages a codec declares normative under an exact profile, and the
// identity they give its streams (the profile and their program digests).
struct NormativeStages final {
    NumericsIdentity identity;
    std::vector<provider::experimental::ExecutableStageDescriptor> descriptors;
};

class StreamNumerics final {
public:
    // codec names the codec in errors; payload_profile is the codec profile id
    // its exact random-access units carry.
    StreamNumerics(std::string codec, std::string payload_profile);

    // nullopt: native numerics.
    [[nodiscard]] Result<void> set(std::optional<NumericsIdentity> identity);
    [[nodiscard]] bool exact() const noexcept { return identity_.has_value(); }

    // A random-access unit of an exact stream: the codec payload and the
    // identity, both required sections, at the given decode index.
    [[nodiscard]] Result<std::vector<std::byte>> serialize_random_access(
        AccessUnit access_unit, std::string codec_id, std::uint64_t index, std::size_t max_bytes) const;

    // Whether this decoder may decode the unit; called before decoding it.
    [[nodiscard]] Result<void> check(const AccessUnit& access_unit);
    // The decoder was reset: the next sequence states its own numerics.
    void reset() noexcept { exact_sequence_ = false; }

private:
    std::string codec_;
    std::string payload_profile_;
    std::optional<NumericsIdentity> identity_;
    std::vector<std::byte> section_;
    bool exact_sequence_{false};  // the last random-access unit decoded opened an exact sequence
};

}  // namespace nvcr::codec
