#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>
#include <iostream>
#include <span>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::byte> golden_v1_access_unit() {
    nvcr::AccessUnit unit{
        "dcvcrt",
        176,
        144,
        32,
        nvcr::FrameType::intra,
        true,
        {std::byte{0xAA}, std::byte{0xBB}},
    };
    auto wire = nvcr::AccessUnitIO::serialize(unit);
    if (!wire) return {};
    return wire.value();
}

constexpr std::array<std::byte, 40> expected_v1{
    std::byte{'N'}, std::byte{'V'}, std::byte{'A'}, std::byte{'U'},
    std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x01},
    std::byte{0x06}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0xb0}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0x90}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0x20}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{'d'}, std::byte{'c'}, std::byte{'v'}, std::byte{'c'},
    std::byte{'r'}, std::byte{'t'}, std::byte{0xaa}, std::byte{0xbb}};

void v1_roundtrip_and_prefix() {
    const auto wire = golden_v1_access_unit();
    expect(!wire.empty(), "v1 access unit serializes");
    if (wire.empty()) return;

    // Golden prefix checks preserve documented wire compatibility.
    expect(wire.size() >= 6, "v1 wire has fixed prefix");
    expect(wire[0] == std::byte{'N'} && wire[1] == std::byte{'V'} &&
           wire[2] == std::byte{'A'} && wire[3] == std::byte{'U'},
           "v1 magic bytes are NVAU");
    expect(wire[4] == std::byte{0x01} && wire[5] == std::byte{0x00},
           "v1 format version remains 1");
        expect(std::equal(wire.begin(), wire.end(), expected_v1.begin(), expected_v1.end()),
            "v1 wire bytes match the fixed golden vector");

    auto decoded = nvcr::AccessUnitIO::deserialize(wire);
    expect(decoded.has_value(), "v1 access unit deserializes");
    if (decoded) {
        expect(decoded.value().model_id == "dcvcrt", "model id round-trips");
        expect(decoded.value().width == 176 && decoded.value().height == 144,
               "dimensions round-trip");
    }
}

void v2_roundtrip_and_prefix() {
    nvcr::AccessUnit unit{
        "dcvcrt",
        176,
        144,
        63,
        nvcr::FrameType::predicted,
        false,
        {std::byte{0x01}},
    };
    unit.codec_id = "dcvcrt";
    unit.codec_profile_id = "dcvcrt";
    unit.decode_order_index = 3;
    unit.presentation_order_index = 3;
    unit.sections = {
        {static_cast<std::uint16_t>(nvcr::AccessUnitSectionType::codec_payload),
         1,
         nvcr::AccessUnitIO::required_section_flag,
         unit.payload},
    };

    auto wire = nvcr::AccessUnitIO::serialize_sectioned(unit);
    expect(wire.has_value(), "v2 access unit serializes");
    if (!wire) return;

    expect(wire.value().size() >= 6, "v2 wire has fixed prefix");
    expect(wire.value()[0] == std::byte{'N'} && wire.value()[1] == std::byte{'V'} &&
           wire.value()[2] == std::byte{'A'} && wire.value()[3] == std::byte{'U'},
           "v2 magic bytes are NVAU");
    expect(wire.value()[4] == std::byte{0x02} && wire.value()[5] == std::byte{0x00},
           "v2 format version remains 2");

    constexpr std::array<std::byte, 99> expected_v2{
        std::byte{'N'}, std::byte{'V'}, std::byte{'A'}, std::byte{'U'},
        std::byte{0x02}, std::byte{0x00}, std::byte{0x40}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x01}, std::byte{0x01}, std::byte{0x08}, std::byte{0x00},
        std::byte{0xb0}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x90}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x3f}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x03}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x03}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x01}, std::byte{0x00},
        std::byte{0x06}, std::byte{0x00}, std::byte{0x06}, std::byte{0x00},
        std::byte{0x06}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x63}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{'d'}, std::byte{'c'}, std::byte{'v'}, std::byte{'c'},
        std::byte{'r'}, std::byte{'t'}, std::byte{'d'}, std::byte{'c'},
        std::byte{'v'}, std::byte{'c'}, std::byte{'r'}, std::byte{'t'},
        std::byte{'d'}, std::byte{'c'}, std::byte{'v'}, std::byte{'c'},
        std::byte{'r'}, std::byte{'t'},
        std::byte{0x01}, std::byte{0x00},
        std::byte{0x01}, std::byte{0x00},
        std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x01}};
    expect(std::equal(
               wire.value().begin(), wire.value().end(), expected_v2.begin(), expected_v2.end()),
           "v2 wire bytes match the fixed golden vector");

    auto decoded = nvcr::AccessUnitIO::deserialize(wire.value());
    expect(decoded.has_value(), "v2 access unit deserializes");
}

void malformed_and_bounds() {
    const auto wire = golden_v1_access_unit();
    if (wire.empty()) return;

    for (std::size_t size = 0; size < wire.size(); ++size) {
        auto sliced = std::span<const std::byte>(wire.data(), size);
        auto parsed = nvcr::AccessUnitIO::deserialize(sliced);
        if (size < 32) {
            expect(!parsed.has_value(), "truncated header is rejected");
        }
    }

    auto oversized = wire;
    if (oversized.size() > 24) {
        for (std::size_t i = 24; i < 32 && i < oversized.size(); ++i) {
            oversized[i] = std::byte{0xff};
        }
        expect(!nvcr::AccessUnitIO::deserialize(oversized).has_value(),
               "oversized payload length is rejected");
    }
}

}  // namespace

int main() {
    v1_roundtrip_and_prefix();
    v2_roundtrip_and_prefix();
    malformed_and_bounds();
    if (failures == 0) {
        std::cout << "NVCR format contract tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
