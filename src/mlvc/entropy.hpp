#pragma once

// MLVC entropy coding through Microsoft's msrtc_rans, configured as MLVC's
// conversion coder does (video/conversion/_coder.py): RansByte, 16 symbol bits,
// 2 bypass bits. One rANS byte stream per frame; y_raw_1, then y_raw_0, then
// z_raw are pushed, so a decoder reads z, y_raw_0, y_raw_1 and then requires
// the stream to be exhausted.

#include "bundle.hpp"

#include "nvcr/common/error.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nvcr::mlvc {

struct LatentSymbols final {
    std::vector<std::int32_t> z;
    std::vector<std::int32_t> y_0;
    std::vector<std::int32_t> y_1;
};

class EntropyCoder final {
public:
    [[nodiscard]] static Result<std::unique_ptr<EntropyCoder>> create(const BundleModel& model);
    ~EntropyCoder();
    EntropyCoder(const EntropyCoder&) = delete;
    EntropyCoder& operator=(const EntropyCoder&) = delete;

    // q_index is the unshifted Q; it selects the z distributions.
    [[nodiscard]] Result<std::vector<std::byte>> encode(
        const LatentSymbols& symbols, std::uint32_t q_index) const;
    [[nodiscard]] Result<LatentSymbols> decode(
        std::span<const std::byte> payload, std::uint32_t q_index) const;

private:
    struct Impl;
    explicit EntropyCoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace nvcr::mlvc
