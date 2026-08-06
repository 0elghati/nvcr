#include <nvcr/nvcr.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const auto bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(data),
        size);
    (void)nvcr::AccessUnitIO::deserialize(bytes, 1U << 20U);
    (void)nvcr::PacketIO::deserialize(bytes, 1U << 20U);
    return 0;
}
