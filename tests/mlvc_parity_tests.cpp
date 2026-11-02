// MLVC upstream-runtime parity gate.
//
// Compares NVCR's MLVC codec with a reference vector recorded by MLVC's own
// conversion runtime (scripts/backends/mlvc/reference_vectors.py) on the same
// bundle and execution provider:
//   1. every NVCR access-unit payload equals the upstream `.mlvc` record;
//   2. decoding NVCR's stream reproduces the upstream 8-bit YUV byte for byte;
//   3. decoding the upstream records reproduces the same YUV; and
//   4. encoder and decoder reset restart the sequence identically.
// With a trailing "decode-only" argument only (3) runs. That mode checks
// upstream streams NVCR cannot encode, such as rate-controlled streams with
// dropped frames.

#include <nvcr/bitstream/access_unit.hpp>
#include <nvcr/codec/adapter.hpp>
#include <nvcr/dcvcrt/backend.hpp>
#include <nvcr/mlvc/codec.hpp>
#include <nvcr/nvcr.hpp>
#include <nvcr/runtime/registry.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

int fail(std::string_view message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int fail(const nvcr::Error& error) {
    std::cerr << "FAIL: " << error.describe() << '\n';
    return 1;
}

bool parse_uint(const char* text, std::uint32_t& output) {
    const auto* end = text + std::char_traits<char>::length(text);
    const auto parsed = std::from_chars(text, end, output);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

std::vector<std::byte> read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    std::vector<std::byte> output(bytes.size());
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        output[index] = static_cast<std::byte>(bytes[index]);
    }
    return output;
}

// Splits an upstream `.mlvc` file into whole records (header included).
bool split_records(std::span<const std::byte> file, std::vector<std::vector<std::byte>>& records) {
    std::size_t offset = 0;
    while (offset < file.size()) {
        if (file.size() - offset < 8U) return false;
        std::uint32_t length = 0;
        for (unsigned index = 0; index < 4U; ++index) {
            length |= std::to_integer<std::uint32_t>(file[offset + 4U + index]) << (8U * index);
        }
        if (file.size() - offset - 8U < length) return false;
        const auto record = file.subspan(offset, 8U + length);
        records.emplace_back(record.begin(), record.end());
        offset += 8U + length;
    }
    return true;
}

struct Codec final {
    nvcr::codec::Sessions sessions;
};

nvcr::Result<Codec> create_codec(const nvcr::RuntimeConfiguration& configuration) {
    const auto provider = nvcr::runtime::Registry::instance().find_provider(configuration.provider.id);
    if (!provider || !provider->session_factory) {
        return nvcr::Error(nvcr::ErrorCode::missing_provider,
                           "provider is not registered: " + configuration.provider.id);
    }
    auto session = provider->session_factory(configuration);
    if (!session) return session.error();
    auto adapter = nvcr::runtime::Registry::instance().create_codec("mlvc");
    if (!adapter) return adapter.error();
    auto sessions = adapter.value()->create_sessions(configuration, std::move(session.value()));
    if (!sessions) return sessions.error();
    return Codec{std::move(sessions.value())};
}

nvcr::Result<std::vector<std::byte>> decode_all(
    nvcr::IDecoderSession& decoder, const std::vector<nvcr::Packet>& packets) {
    std::vector<std::byte> output;
    for (const auto& packet : packets) {
        auto sent = decoder.send_access_unit(packet);
        if (!sent) return sent.error();
        while (true) {
            auto frame = decoder.receive_frame();
            if (!frame) {
                if (frame.error().code() == nvcr::ErrorCode::try_again) break;
                return frame.error();
            }
            const auto bytes = frame.value().data();
            output.insert(output.end(), bytes.begin(), bytes.end());
        }
    }
    return output;
}

std::size_t first_difference(std::span<const std::byte> left, std::span<const std::byte> right) {
    const auto count = std::min(left.size(), right.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (left[index] != right[index]) return index;
    }
    return count;
}

}  // namespace

int main(int argc, char* argv[]) {
    const bool decode_only = argc == 10 && std::string_view(argv[9]) == "decode-only";
    if (argc != 9 && !decode_only) {
        std::cerr << "usage: nvcr_mlvc_parity_tests PROVIDER_ID BUNDLE_DIR REFERENCE_DIR "
                     "INPUT_YUV WIDTH HEIGHT FRAMES IFRAME_PERIOD|model [decode-only]\n"
                     "REFERENCE_DIR holds output.mlvc and output.yuv from "
                     "scripts/backends/mlvc/reference_vectors.py\n";
        return 2;
    }
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t frames = 0;
    std::uint32_t iframe_period = nvcr::mlvc::use_model_iframe_period;
    if (!parse_uint(argv[5], width) || !parse_uint(argv[6], height) ||
        !parse_uint(argv[7], frames) ||
        (std::string_view(argv[8]) != "model" && !parse_uint(argv[8], iframe_period))) {
        return fail("invalid numeric argument");
    }
    const fs::path reference_dir(argv[3]);
    const auto upstream_file = read_file(reference_dir / "output.mlvc");
    const auto upstream_yuv = read_file(reference_dir / "output.yuv");
    std::vector<std::vector<std::byte>> upstream_records;
    if (!split_records(upstream_file, upstream_records) || upstream_records.size() != frames) {
        return fail("reference output.mlvc does not hold one record per frame");
    }
    std::int32_t q_index = 0;
    std::memcpy(&q_index, upstream_records.front().data(), sizeof(q_index));
    const auto input = read_file(argv[4]);
    const std::size_t frame_bytes = static_cast<std::size_t>(width) * height * 3U / 2U;
    if (input.size() < frame_bytes * frames || upstream_yuv.size() != frame_bytes * frames) {
        return fail("input or reference YUV is shorter than the frame count");
    }

    nvcr::mlvc::register_codec();
    nvcr::dcvcrt::register_execution_providers();
    nvcr::RuntimeConfiguration configuration;
    configuration.codec.id = "mlvc";
    configuration.provider.id = argv[1];
    configuration.artifacts.intra_engine_path = fs::path(argv[2]);
    configuration.codec.intra_qp = static_cast<std::uint32_t>(q_index);
    configuration.codec.gop_size = iframe_period;
    auto codec = create_codec(configuration);
    if (!codec) return fail(codec.error());
    auto& encoder = *codec.value().sessions.encoder;
    auto& decoder = *codec.value().sessions.decoder;

    if (decode_only) {
        // The encoder supplies only the access-unit identity of frame 0.
        auto first = nvcr::Frame::copy_from(
            width, height, nvcr::PixelFormat::yuv420p8, std::span(input).first(frame_bytes));
        if (!first) return fail(first.error());
        auto sent = encoder.send_frame(first.value());
        if (!sent) return fail(sent.error());
        auto packet = encoder.receive_access_unit();
        if (!packet) return fail(packet.error());
        auto header = nvcr::AccessUnitIO::deserialize(packet.value().data(), 1U << 26U);
        if (!header) return fail(header.error());
        // Intra positions follow MLVC's I period over coded (non-dropped)
        // frames; "model" assumes the bundle's period exceeds the clip.
        std::vector<nvcr::Packet> upstream_packets;
        std::uint64_t coded = 0;
        for (std::uint32_t index = 0; index < frames; ++index) {
            std::int32_t record_q = 0;
            std::memcpy(&record_q, upstream_records[index].data(), sizeof(record_q));
            const bool dropped = record_q < 0;
            const bool intra = !dropped &&
                (index == 0U || (iframe_period != nvcr::mlvc::use_model_iframe_period &&
                                 coded % iframe_period == 0U));
            if (intra) coded = 0;
            if (!dropped) ++coded;
            auto unit = header.value();
            unit.qp = dropped ? 0U : static_cast<std::uint32_t>(record_q);
            unit.frame_type = intra ? nvcr::FrameType::intra : nvcr::FrameType::predicted;
            unit.reset_state = intra;
            unit.payload = upstream_records[index];
            auto bytes = nvcr::AccessUnitIO::serialize(unit, 1U << 26U);
            if (!bytes) return fail(bytes.error());
            upstream_packets.emplace_back(
                std::move(bytes.value()),
                nvcr::Timestamp{static_cast<std::int64_t>(index) * 33'333}, unit.frame_type);
        }
        auto decoded = decode_all(decoder, upstream_packets);
        if (!decoded) return fail(decoded.error());
        if (decoded.value() != upstream_yuv) {
            return fail("decoding upstream records diverges in frame " +
                        std::to_string(first_difference(decoded.value(), upstream_yuv) /
                                       frame_bytes));
        }
        std::cout << "MLVC upstream decode parity passed: " << argv[1] << ", " << frames
                  << " frames\n";
        return 0;
    }

    // 1. Encoder parity.
    std::vector<nvcr::Packet> packets;
    for (std::uint32_t index = 0; index < frames; ++index) {
        auto frame = nvcr::Frame::copy_from(
            width, height, nvcr::PixelFormat::yuv420p8,
            std::span(input).subspan(index * frame_bytes, frame_bytes),
            nvcr::Timestamp{static_cast<std::int64_t>(index) * 33'333});
        if (!frame) return fail(frame.error());
        auto sent = encoder.send_frame(frame.value());
        if (!sent) return fail(sent.error());
        auto packet = encoder.receive_access_unit();
        if (!packet) return fail(packet.error());
        packets.push_back(std::move(packet.value()));
    }
    std::size_t encoder_matches = 0;
    std::size_t first_mismatch = frames;
    std::vector<nvcr::Packet> upstream_packets;
    for (std::uint32_t index = 0; index < frames; ++index) {
        auto access_unit = nvcr::AccessUnitIO::deserialize(packets[index].data(), 1U << 26U);
        if (!access_unit) return fail(access_unit.error());
        if (access_unit.value().payload == upstream_records[index]) {
            ++encoder_matches;
        } else if (first_mismatch == frames) {
            first_mismatch = index;
        }
        auto wrapped = access_unit.value();
        wrapped.payload = upstream_records[index];
        auto bytes = nvcr::AccessUnitIO::serialize(wrapped, 1U << 26U);
        if (!bytes) return fail(bytes.error());
        upstream_packets.emplace_back(std::move(bytes.value()), packets[index].timestamp(),
                                      packets[index].frame_type());
    }
    std::cout << "encoder records identical to upstream: " << encoder_matches << '/' << frames
              << '\n';

    // 2. Decoder parity on NVCR's stream; 3. on the upstream records.
    auto decoded = decode_all(decoder, packets);
    if (!decoded) return fail(decoded.error());
    const bool own_matches = decoded.value() == upstream_yuv;
    std::cout << "decoding NVCR stream reproduces upstream YUV: " << (own_matches ? "yes" : "no");
    if (!own_matches) {
        std::cout << " (first difference in frame "
                  << first_difference(decoded.value(), upstream_yuv) / frame_bytes << ')';
    }
    std::cout << '\n';
    auto reset = decoder.reset();
    if (!reset) return fail(reset.error());
    auto from_upstream = decode_all(decoder, upstream_packets);
    if (!from_upstream) return fail(from_upstream.error());
    const bool upstream_matches = from_upstream.value() == upstream_yuv;
    std::cout << "decoding upstream records reproduces upstream YUV: "
              << (upstream_matches ? "yes" : "no");
    if (!upstream_matches) {
        std::cout << " (first difference in frame "
                  << first_difference(from_upstream.value(), upstream_yuv) / frame_bytes << ')';
    }
    std::cout << '\n';

    // 4. Reset restarts both directions identically.
    reset = encoder.reset();
    if (!reset) return fail(reset.error());
    reset = decoder.reset();
    if (!reset) return fail(reset.error());
    const std::uint32_t restart = std::min<std::uint32_t>(frames, 3U);
    std::vector<nvcr::Packet> restarted;
    for (std::uint32_t index = 0; index < restart; ++index) {
        auto frame = nvcr::Frame::copy_from(
            width, height, nvcr::PixelFormat::yuv420p8,
            std::span(input).subspan(index * frame_bytes, frame_bytes));
        if (!frame) return fail(frame.error());
        auto sent = encoder.send_frame(frame.value());
        if (!sent) return fail(sent.error());
        auto packet = encoder.receive_access_unit();
        if (!packet) return fail(packet.error());
        if (!std::equal(packet.value().data().begin(), packet.value().data().end(),
                        packets[index].data().begin(), packets[index].data().end())) {
            return fail("encoder reset does not restart the sequence identically");
        }
        restarted.push_back(std::move(packet.value()));
    }
    auto redecoded = decode_all(decoder, restarted);
    if (!redecoded) return fail(redecoded.error());
    if (!std::equal(redecoded.value().begin(), redecoded.value().end(), upstream_yuv.begin())) {
        return fail("decoder reset does not restart the sequence identically");
    }

    // 5. Malformed access units fail without crashing; a flipped payload byte
    // may still decode, but it must not take the decoder down.
    const auto mutate = [&](std::size_t index, auto&& change) {
        auto unit = nvcr::AccessUnitIO::deserialize(packets[index].data(), 1U << 26U).value();
        change(unit);
        auto bytes = nvcr::AccessUnitIO::serialize(unit, 1U << 26U).value();
        return nvcr::Packet(std::move(bytes), packets[index].timestamp(), unit.frame_type);
    };
    const auto rejects = [&](const std::vector<nvcr::Packet>& stream) {
        if (!decoder.reset()) return false;
        for (const auto& packet : stream) {
            if (!decoder.send_access_unit(packet)) return true;
            while (decoder.receive_frame()) {
            }
        }
        return false;
    };
    if (!rejects({mutate(0, [](nvcr::AccessUnit& unit) { unit.payload.resize(6); })})) {
        return fail("truncated MLVC record was accepted");
    }
    if (!rejects({mutate(0, [](nvcr::AccessUnit& unit) { unit.payload.pop_back(); })})) {
        return fail("MLVC record with a wrong length was accepted");
    }
    if (!rejects({mutate(0, [](nvcr::AccessUnit& unit) { unit.model_id = "unknown-1x1"; })})) {
        return fail("unknown MLVC model identity was accepted");
    }
    if (frames > 1U && !rejects({packets[1]})) {
        return fail("predicted access unit without an intra frame was accepted");
    }
    if (!rejects({mutate(0, [](nvcr::AccessUnit& unit) { unit.qp ^= 1U; })})) {
        return fail("access-unit Q index that disagrees with the record was accepted");
    }
    for (std::size_t offset = 8; offset < 48U; offset += 5U) {
        std::vector<nvcr::Packet> corrupted{mutate(0, [&](nvcr::AccessUnit& unit) {
            if (offset < unit.payload.size()) unit.payload[offset] ^= std::byte{0x5a};
        })};
        if (frames > 1U) corrupted.push_back(packets[1]);
        (void)rejects(corrupted);
    }

    if (encoder_matches != frames) {
        return fail("encoder diverges from upstream at frame " + std::to_string(first_mismatch));
    }
    if (!own_matches || !upstream_matches) return fail("decoder diverges from upstream");
    std::cout << "MLVC upstream-runtime parity passed: " << argv[1] << ", " << frames
              << " frames, q " << q_index << '\n';
    return 0;
}
