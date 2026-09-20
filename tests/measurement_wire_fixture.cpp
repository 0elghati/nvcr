#include <nvcr/nvcr.hpp>
#include "payload.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::filesystem::create_directories(argv[1]);
    for (unsigned version : {1U, 2U}) {
        for (unsigned metadata_count : {0U, 2U}) {
            const std::vector<std::byte> entropy{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
            nvcr::AccessUnit au;
            au.model_id = "dcvcrt";
            au.codec_id = "dcvcrt";
            au.codec_profile_id = "fp16";
            au.width = 64; au.height = 64; au.qp = 21;
            au.frame_type = nvcr::FrameType::intra; au.reset_state = true;
            au.payload = nvcr::dcvcrt::make_intra_payload(64, 64, 21, false, entropy);
            if (version == 2) {
                au.sections = {
                    {1, 1, nvcr::AccessUnitIO::required_section_flag, au.payload},
                    {5, 1, 0, {std::byte{42}, std::byte{43}}}};
            }
            auto access = version == 1 ? nvcr::AccessUnitIO::serialize(au) : nvcr::AccessUnitIO::serialize_sectioned(au);
            if (!access) { std::cerr << access.error().describe(); return 1; }
            nvcr::PacketMetadata metadata;
            if (metadata_count) { metadata["key"] = "value"; metadata["another-key"] = std::string(41, 'x'); }
            nvcr::Packet packet(access.value(), nvcr::Timestamp{0}, nvcr::FrameType::intra, metadata);
            auto wire = nvcr::PacketIO::serialize(packet);
            if (!wire) return 1;
            auto path = std::filesystem::path(argv[1]) / ("v" + std::to_string(version) + "-m" + std::to_string(metadata_count) + ".nvcr");
            std::ofstream out(path, std::ios::binary);
            out.write("NVCS\1\0\0\0", 8);
            const auto size = static_cast<std::uint64_t>(wire.value().size());
            for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>((size >> (8U*i)) & 255U));
            out.write(reinterpret_cast<const char*>(wire.value().data()), static_cast<std::streamsize>(size));
            if (!out) return 1;
        }
    }
    return 0;
}
