#include "stream_numerics.hpp"

#include <utility>

namespace nvcr::codec {

StreamNumerics::StreamNumerics(std::string codec, std::string payload_profile)
    : codec_(std::move(codec)), payload_profile_(std::move(payload_profile)) {}

Result<void> StreamNumerics::set(std::optional<NumericsIdentity> identity) {
    section_.clear();
    identity_ = std::move(identity);
    if (!identity_) return {};
    auto section = NumericsIdentityIO::serialize(*identity_);
    if (!section) return section.error();
    section_ = std::move(section.value());
    return {};
}

Result<std::vector<std::byte>> StreamNumerics::serialize_random_access(
    AccessUnit access_unit, std::string codec_id, std::uint64_t index, std::size_t max_bytes) const {
    access_unit.codec_id = std::move(codec_id);
    access_unit.codec_profile_id = payload_profile_;
    access_unit.decode_order_index = index;
    access_unit.presentation_order_index = index;
    access_unit.random_access = true;
    access_unit.sections = {
        AccessUnitSection{static_cast<std::uint16_t>(AccessUnitSectionType::codec_payload), 1,
                          AccessUnitIO::required_section_flag, access_unit.payload},
        AccessUnitSection{static_cast<std::uint16_t>(AccessUnitSectionType::numerics_identity),
                          NumericsIdentityIO::format_version, AccessUnitIO::required_section_flag, section_}};
    return AccessUnitIO::serialize_sectioned(access_unit, max_bytes);
}

// A stream with exact numerics decodes only on a decoder that runs the same
// profile with the same normative programs; a native stream only on a native
// decoder. The identity is stated on random-access units; a unit without one
// belongs to the sequence the last random-access unit opened.
Result<void> StreamNumerics::check(const AccessUnit& access_unit) {
    const AccessUnitSection* section = nullptr;
    for (const auto& candidate : access_unit.sections) {
        if (candidate.type != static_cast<std::uint16_t>(AccessUnitSectionType::numerics_identity)) continue;
        if (section != nullptr) {
            return Error(ErrorCode::malformed_bitstream, "access unit has more than one numerics identity", codec_);
        }
        section = &candidate;
    }
    if (section == nullptr) {
        if (access_unit.frame_type == FrameType::intra) exact_sequence_ = false;
        if (!identity_ || (access_unit.frame_type != FrameType::intra && exact_sequence_)) return {};
        return Error(ErrorCode::incompatible_target, "this decoder runs exact numerics; the stream uses native numerics",
                     codec_);
    }
    if (access_unit.codec_profile_id != payload_profile_) {
        return Error(ErrorCode::incompatible_version,
                     "unsupported " + codec_ + " codec profile '" + access_unit.codec_profile_id + "'", codec_);
    }
    if (access_unit.frame_type != FrameType::intra) {
        return Error(ErrorCode::malformed_bitstream, "numerics identity on a unit that is not a random access point",
                     codec_);
    }
    if (section->version != NumericsIdentityIO::format_version) {
        return Error(ErrorCode::incompatible_version,
                     "unsupported numerics identity version " + std::to_string(section->version), codec_);
    }
    if ((section->flags & AccessUnitIO::required_section_flag) == 0U) {
        return Error(ErrorCode::malformed_bitstream, "numerics identity must be required", codec_);
    }
    auto identity = NumericsIdentityIO::deserialize(section->payload);
    if (!identity) return identity.error();
    if (!identity_) {
        return Error(ErrorCode::incompatible_target,
                     "stream requires " + identity.value().profile +
                         " normative stages; decode with that numerics profile and its programs",
                     codec_);
    }
    if (identity.value() != *identity_) {
        return Error(ErrorCode::incompatible_target,
                     "stream's numerics profile or normative programs differ from this decoder's", codec_);
    }
    exact_sequence_ = true;
    return {};
}

}  // namespace nvcr::codec
