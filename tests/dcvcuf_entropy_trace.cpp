// C13 cross-decode: decodes one .nvcr file on either DCVC-UF execution path
// and records every UF entropy decode call, so two decoders of the same bytes
// can be compared (scripts/backends/dcvcuf/compare_entropy_traces.py).
//
// Decoding follows the production CLI: access units in file order through the
// public Runtime API, every emitted frame appended to the output YUV as soon as
// it is emitted, and a stop at the first access unit that fails. The observer
// from src/dcvcuf/diagnostics.hpp receives the entropy calls from the codec's
// UF rANS coder, which both paths share.
//
// Usage: BUNDLE INPUT.nvcr DECODED.yuv TRACE.bin
// Exit status: 0 every access unit decoded, 3 an access unit failed with an
// NVCR error (the trace records which one and why), 2 usage or harness error.
//
// Trace (little-endian): "UFTR", u32 version 1, then records, each a u8 tag:
//   1 unit begin   u32 unit, u8 frame type (0 intra, 1 predicted), u32 bytes
//   2 entropy call u8 call (0 z, 1 y), u8 complete, u64 count,
//                  count u8 indexes (y only), count i8 symbols
//   3 stream end   u8 valid (the end-of-stream state check)
//   4 unit end     u8 ok, u32 frames emitted, u32 error code, u32 length, message
//   5 trace end    u32 units attempted, u64 frames written

#include "diagnostics.hpp"

#include <nvcr/dcvcrt/backend.hpp>
#include <nvcr/dcvcuf/backend.hpp>
#include <nvcr/nvcr.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <span>
#include <string_view>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace diag = nvcr::dcvcuf::diagnostics;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

class TraceWriter final : public diag::Observer {
public:
    explicit TraceWriter(const std::filesystem::path& path)
        : output_(path, std::ios::binary | std::ios::trunc) {
        if (!output_) fail("cannot create " + path.string());
        output_.write("UFTR", 4);
        u32(1);
    }

    void unit_begin(std::uint32_t unit, nvcr::FrameType type, std::size_t bytes) {
        const std::lock_guard lock(mutex_);
        u8(1);
        u32(unit);
        u8(type == nvcr::FrameType::intra ? 0 : 1);
        u32(static_cast<std::uint32_t>(bytes));
    }
    void unit_end(bool ok, std::uint32_t frames, std::uint32_t code, const std::string& message) {
        const std::lock_guard lock(mutex_);
        u8(4);
        u8(ok ? 1 : 0);
        u32(frames);
        u32(code);
        u32(static_cast<std::uint32_t>(message.size()));
        output_.write(message.data(), static_cast<std::streamsize>(message.size()));
    }
    void trace_end(std::uint32_t units, std::uint64_t frames) {
        const std::lock_guard lock(mutex_);
        u8(5);
        u32(units);
        u64(frames);
        output_.flush();
        if (!output_) fail("trace write failed");
    }

    void step(diag::Direction, diag::StepKind, bool) override {}
    void tensor(diag::Direction, std::string_view, std::span<const std::int64_t>,
                std::span<const std::byte>) override {}
    void frames(diag::Direction, std::span<const nvcr::Frame>) override {}

    void entropy(diag::EntropyCall call, std::span<const std::uint8_t> indexes,
                 std::span<const std::int8_t> symbols, bool complete) override {
        const std::lock_guard lock(mutex_);
        u8(2);
        u8(call == diag::EntropyCall::z ? 0 : 1);
        u8(complete ? 1 : 0);
        u64(symbols.size());
        if (call == diag::EntropyCall::y) {
            if (indexes.size() != symbols.size()) fail("y indexes and symbols differ in count");
            output_.write(reinterpret_cast<const char*>(indexes.data()),
                          static_cast<std::streamsize>(indexes.size()));
        }
        output_.write(reinterpret_cast<const char*>(symbols.data()),
                      static_cast<std::streamsize>(symbols.size()));
    }
    void entropy_end(bool valid) override {
        const std::lock_guard lock(mutex_);
        u8(3);
        u8(valid ? 1 : 0);
    }

private:
    void u8(std::uint8_t value) { output_.put(static_cast<char>(value)); }
    void u32(std::uint32_t value) { little(value, 4); }
    void u64(std::uint64_t value) { little(value, 8); }
    void little(std::uint64_t value, std::size_t bytes) {
        std::array<char, 8> buffer{};
        for (std::size_t index = 0; index < bytes; ++index)
            buffer[index] = static_cast<char>((value >> (index * 8)) & 0xffU);
        output_.write(buffer.data(), static_cast<std::streamsize>(bytes));
    }

    std::mutex mutex_;  // fused-path calls arrive on upstream's worker thread
    std::ofstream output_;
};

class InstalledObserver final {
public:
    explicit InstalledObserver(diag::Observer* observer) { diag::set_observer(observer); }
    ~InstalledObserver() { diag::set_observer(nullptr); }
    InstalledObserver(const InstalledObserver&) = delete;
    InstalledObserver& operator=(const InstalledObserver&) = delete;
};

// The CLI's .nvcr layout: NVCS v1 header, then u64 length + PacketIO record.
class StreamReader final {
public:
    explicit StreamReader(const std::filesystem::path& path) : input_(path, std::ios::binary) {
        if (!input_) fail("cannot open " + path.string());
        std::array<char, 8> header{};
        input_.read(header.data(), header.size());
        static constexpr std::array<char, 8> expected{'N', 'V', 'C', 'S', 1, 0, 0, 0};
        if (!input_ || header != expected) fail("unsupported NVCR sequence header");
    }
    // False at a clean end of file; throws on a truncated record.
    bool next(std::vector<std::byte>& record) {
        std::array<unsigned char, 8> length{};
        input_.read(reinterpret_cast<char*>(length.data()), length.size());
        if (input_.gcount() == 0 && input_.eof()) return false;
        if (input_.gcount() != static_cast<std::streamsize>(length.size()))
            fail("truncated record length");
        std::uint64_t size = 0;
        for (std::size_t index = 0; index < length.size(); ++index)
            size |= static_cast<std::uint64_t>(length[index]) << (index * 8);
        if (size > (std::uint64_t{1} << 31)) fail("record length exceeds 2 GiB");
        record.resize(static_cast<std::size_t>(size));
        input_.read(reinterpret_cast<char*>(record.data()), static_cast<std::streamsize>(size));
        if (input_.gcount() != static_cast<std::streamsize>(size)) fail("truncated record");
        return true;
    }

private:
    std::ifstream input_;
};

struct Outcome {
    bool ok{true};
    std::uint32_t frames{};
    nvcr::ErrorCode code{};
    std::string message;
};

void record_error(Outcome& outcome, const nvcr::Error& error) {
    outcome.ok = false;
    outcome.code = error.code();
    outcome.message = error.describe();
}

// Appends every frame the decoder has ready; a non-retry error ends the unit.
void drain(nvcr::Runtime& runtime, std::ofstream& yuv, Outcome& outcome,
           std::uint64_t& written) {
    for (;;) {
        auto frame = runtime.receive_frame();
        if (!frame) {
            const auto code = frame.error().code();
            if (code != nvcr::ErrorCode::try_again && code != nvcr::ErrorCode::end_of_stream)
                record_error(outcome, frame.error());
            return;
        }
        const auto bytes = frame.value().data();
        yuv.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!yuv) fail("decoded YUV write failed");
        ++outcome.frames;
        ++written;
    }
}

int run(char** argv) {
    const std::filesystem::path bundle = argv[1];
    const std::filesystem::path input = argv[2];
    const std::filesystem::path decoded = argv[3];
    const std::filesystem::path trace_path = argv[4];

    nvcr::dcvcrt::register_execution_providers();  // registers the generic tensorrt provider
    nvcr::dcvcuf::register_codec();
    nvcr::dcvcuf::register_execution_provider();

    TraceWriter trace(trace_path);
    const InstalledObserver installed(&trace);
    auto configuration = nvcr::dcvcuf::runtime_configuration(bundle);
    if (!configuration) fail("configuration: " + configuration.error().describe());
    configuration.value().runtime.log_level = nvcr::LogLevel::warning;
    auto runtime = nvcr::Runtime::create(configuration.value());
    if (!runtime) fail("runtime: " + runtime.error().describe());

    StreamReader reader(input);
    std::ofstream yuv(decoded, std::ios::binary | std::ios::trunc);
    if (!yuv) fail("cannot create " + decoded.string());

    std::vector<std::byte> record;
    std::uint32_t unit = 0;
    std::uint64_t written = 0;
    bool failed = false;
    while (!failed && reader.next(record)) {
        auto packet = nvcr::PacketIO::deserialize(record);
        Outcome outcome;
        if (!packet) {
            trace.unit_begin(unit, nvcr::FrameType::intra, record.size());
            record_error(outcome, packet.error());
        } else {
            trace.unit_begin(unit, packet.value().frame_type(), packet.value().size());
            auto sent = runtime.value().send_access_unit(packet.value());
            if (!sent) record_error(outcome, sent.error());
            else drain(runtime.value(), yuv, outcome, written);
        }
        trace.unit_end(outcome.ok, outcome.frames, static_cast<std::uint32_t>(outcome.code),
                       outcome.message);
        if (!outcome.ok) {
            std::cerr << "unit " << unit << ": " << outcome.message << '\n';
            failed = true;
        }
        ++unit;
    }
    if (!failed) {
        Outcome outcome;
        trace.unit_begin(unit, nvcr::FrameType::predicted, 0);  // decoder flush
        auto flushed = runtime.value().flush_decoder();
        if (!flushed) record_error(outcome, flushed.error());
        else drain(runtime.value(), yuv, outcome, written);
        trace.unit_end(outcome.ok, outcome.frames, static_cast<std::uint32_t>(outcome.code),
                       outcome.message);
        failed = !outcome.ok;
        if (failed) std::cerr << "flush: " << outcome.message << '\n';
    }
    yuv.flush();
    if (!yuv) fail("decoded YUV write failed");
    trace.trace_end(unit, written);
    std::cout << (failed ? "FAILED" : "COMPLETE") << " units=" << unit << " frames=" << written
              << '\n';
    return failed ? 3 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " BUNDLE INPUT.nvcr DECODED.yuv TRACE.bin\n";
        return 2;
    }
    try {
        return run(argv);
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 2;
    }
}
