#include <nvcr/nvcr.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    nvcr::Packet packet(
        std::vector<std::byte>{std::byte{0x01}, std::byte{0x02}, std::byte{0x03}},
        nvcr::Timestamp{41'667},
        nvcr::FrameType::intra,
        {{"sequence", "example"}});

    auto encoded = nvcr::PacketIO::serialize(packet);
    if (!encoded) {
        std::cerr << encoded.error().describe() << '\n';
        return 1;
    }
    auto decoded = nvcr::PacketIO::deserialize(encoded.value());
    if (!decoded) {
        std::cerr << decoded.error().describe() << '\n';
        return 1;
    }
    std::cout << "Round-tripped " << decoded.value().size() << " compressed bytes\n";
    return 0;
}

