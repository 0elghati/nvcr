#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

int main() {
    constexpr std::size_t parser_limit = 4096U;
    std::mt19937_64 random(0x4e564352U);
    std::uniform_int_distribution<std::size_t> size_distribution(0U, parser_limit * 2U);
    std::uniform_int_distribution<unsigned> byte_distribution(0U, 255U);

    for (std::size_t iteration = 0; iteration < 10'000U; ++iteration) {
        std::vector<std::byte> input(size_distribution(random));
        for (auto& byte : input) {
            byte = static_cast<std::byte>(byte_distribution(random));
        }
        if (input.size() >= 4U && iteration % 3U == 0U) {
            constexpr std::byte packet_magic[]{
                std::byte{'N'}, std::byte{'V'}, std::byte{'C'}, std::byte{'R'}};
            std::copy(std::begin(packet_magic), std::end(packet_magic), input.begin());
        }
        if (input.size() >= 4U && iteration % 5U == 0U) {
            constexpr std::byte access_magic[]{
                std::byte{'N'}, std::byte{'V'}, std::byte{'A'}, std::byte{'U'}};
            std::copy(std::begin(access_magic), std::end(access_magic), input.begin());
        }
        static_cast<void>(nvcr::PacketIO::deserialize(input, parser_limit));
        static_cast<void>(nvcr::AccessUnitIO::deserialize(input, parser_limit));
    }

    nvcr::Packet packet(
        {std::byte{1}, std::byte{2}, std::byte{3}},
        nvcr::Timestamp{42},
        nvcr::FrameType::intra,
        {{"model", "dcvcrt-cvpr2025"}});
    auto packet_wire = nvcr::PacketIO::serialize(packet);
    if (!packet_wire) {
        std::cerr << packet_wire.error().describe() << '\n';
        return 1;
    }
    for (std::size_t size = 0; size <= packet_wire.value().size(); ++size) {
        static_cast<void>(nvcr::PacketIO::deserialize(
            std::span(packet_wire.value()).first(size), parser_limit));
    }

    nvcr::AccessUnit access_unit{
        "dcvcrt-cvpr2025", 176, 144, 71, nvcr::FrameType::predicted, false,
        {std::byte{1}, std::byte{2}, std::byte{3}}};
    auto access_wire = nvcr::AccessUnitIO::serialize(access_unit, parser_limit);
    if (!access_wire) {
        std::cerr << access_wire.error().describe() << '\n';
        return 1;
    }
    for (std::size_t size = 0; size <= access_wire.value().size(); ++size) {
        static_cast<void>(nvcr::AccessUnitIO::deserialize(
            std::span(access_wire.value()).first(size), parser_limit));
    }

    std::cout << "NVCR deterministic parser fuzz/boundary corpus passed\n";
    return 0;
}
