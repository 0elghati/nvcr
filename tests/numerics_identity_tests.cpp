// NVAU numerics_identity section (issue #216): a decoder must reproduce the
// encoder's normative stages exactly, so the section that names them must
// round-trip and every malformed form must be rejected before decoding.

#include <nvcr/bitstream/numerics_identity.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<nvcr::NormativeStageIdentity> sample_stages() {
    std::vector<nvcr::NormativeStageIdentity> stages;
    for (const char* stage : {"i_hyper_synthesis", "i_synthesis", "p_prior", "p_synthesis"}) {
        nvcr::NormativeStageIdentity entry;
        entry.stage_id = stage;
        for (std::size_t index = 0; index < entry.program_digest.size(); ++index) {
            entry.program_digest[index] =
                static_cast<std::byte>((index * 37U + std::string_view(stage).size()) & 0xffU);
        }
        stages.push_back(entry);
    }
    return stages;
}

nvcr::NumericsIdentity sample() {
    auto identity = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", sample_stages());
    expect(identity.has_value(), "a valid identity is made");
    return identity ? identity.value() : nvcr::NumericsIdentity{};
}

bool rejected_as_malformed(std::span<const std::byte> bytes) {
    auto parsed = nvcr::NumericsIdentityIO::deserialize(bytes);
    return !parsed && parsed.error().code() == nvcr::ErrorCode::malformed_bitstream;
}

void round_trip_test() {
    const auto identity = sample();
    auto bytes = nvcr::NumericsIdentityIO::serialize(identity);
    expect(bytes.has_value(), "a valid identity serializes");
    if (!bytes) return;
    expect(bytes.value().size() == 1 + identity.profile.size() + 32, "the identity is profile plus one digest");
    auto parsed = nvcr::NumericsIdentityIO::deserialize(bytes.value());
    expect(parsed.has_value() && parsed.value() == identity, "the identity round-trips");
}

void program_set_digest_test() {
    const auto stages = sample_stages();
    auto base = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", stages);
    auto again = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", stages);
    expect(base && again && base.value() == again.value(), "the program-set digest is deterministic");
    auto flipped = stages;
    flipped[2].program_digest[31] ^= std::byte{1};
    auto changed = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", flipped);
    expect(changed && base && !(changed.value() == base.value()),
           "a one-bit program digest change changes the identity");
    auto renamed = stages;
    renamed[1].stage_id = "i_synthesis2";
    auto other = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", renamed);
    expect(other && base && !(other.value() == base.value()), "a renamed stage changes the identity");
    auto reordered = stages;
    std::swap(reordered[0], reordered[1]);
    auto swapped = nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", reordered);
    expect(swapped && base && !(swapped.value() == base.value()), "stage order is part of the identity");
    auto profile = nvcr::NumericsIdentityIO::make("nvcr.other.v1", stages);
    expect(profile && base && !(profile.value() == base.value()), "the profile is part of the identity");
}

void truncation_and_trailing_test() {
    auto bytes = nvcr::NumericsIdentityIO::serialize(sample());
    if (!bytes) return;
    const auto& full = bytes.value();
    bool every_truncation_rejected = true;
    for (std::size_t length = 0; length < full.size(); ++length) {
        every_truncation_rejected &= rejected_as_malformed(std::span(full).first(length));
    }
    expect(every_truncation_rejected, "every truncation is malformed");
    auto trailing = full;
    trailing.push_back(std::byte{0});
    expect(rejected_as_malformed(trailing), "trailing bytes are malformed");
}

void invalid_content_test() {
    auto stages = sample_stages();
    auto duplicate = stages;
    duplicate[1].stage_id = duplicate[0].stage_id;
    expect(!nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", duplicate), "duplicate stages are refused");
    expect(!nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", {}), "an identity needs normative stages");
    expect(!nvcr::NumericsIdentityIO::make("nvcr intconv.v1", stages), "ids with spaces are refused");
    std::vector<nvcr::NormativeStageIdentity> too_many(65);
    for (std::size_t index = 0; index < too_many.size(); ++index) {
        too_many[index].stage_id = "s" + std::to_string(index);
    }
    expect(!nvcr::NumericsIdentityIO::make("nvcr.intconv.v1", too_many), "more than 64 stages are refused");

    std::vector<std::byte> control{std::byte{1}, std::byte{0x07}};
    control.insert(control.end(), 32, std::byte{0});
    expect(rejected_as_malformed(control), "control characters in the profile id are malformed");
    std::vector<std::byte> empty_profile{std::byte{0}};
    empty_profile.insert(empty_profile.end(), 32, std::byte{0});
    expect(rejected_as_malformed(empty_profile), "an empty profile id is malformed");
}

}  // namespace

int main() {
    round_trip_test();
    program_set_digest_test();
    truncation_and_trailing_test();
    invalid_content_test();
    if (failures == 0) std::cout << "numerics identity conformance passed\n";
    return failures == 0 ? 0 : 1;
}
