// C13 G1: the DCVC-UF encoder's internal reconstruction equals the decoder's
// output byte for byte, and (#194) reset and drain with a partly filled
// eight-frame group behave as the session contract requires.
//
// Drives the registered dcvc-uf codec through the public Runtime API on either
// execution path (a native .pt bundle or a TensorRT stage bundle), with the
// test-only observer from src/dcvcuf/diagnostics.hpp installed. Each access
// unit is decoded as soon as it is emitted, and the decoder's step is compared
// with the encoder's step that produced it:
//
//   intra:     encoder reconstructed frame == decoded frame
//   HT-S group memory, context, y_hat and feature tensors identical, and the
//              frames from the reconstruction head on the encoder's feature
//              == the decoded frames (the real frames of a padded tail)
//
// Usage:
//   g1:            BUNDLE INPUT.yuv WIDTH HEIGHT FRAMES QP GOP STREAM.nvcr DECODED.yuv REPORT.json
//   partial-group: BUNDLE INPUT.yuv WIDTH HEIGHT QP REPORT.json
// Exit status: 0 pass, 1 identity or contract failure, 2 usage or runtime error.

#include "diagnostics.hpp"

#include <nvcr/dcvcrt/backend.hpp>
#include <nvcr/dcvcuf/backend.hpp>
#include <nvcr/nvcr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace diag = nvcr::dcvcuf::diagnostics;
constexpr std::int64_t frame_period_us = 16'667;  // the CLI default, so streams match its output

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

template <class T>
T value(nvcr::Result<T> result, const char* what) {
    if (!result) fail(std::string(what) + ": " + result.error().describe());
    return std::move(result.value());
}
void value(nvcr::Result<void> result, const char* what) {
    if (!result) fail(std::string(what) + ": " + result.error().describe());
}

std::string json_escape(const std::string& text) {
    std::string output;
    for (const char character : text) {
        if (character == '"' || character == '\\') output.push_back('\\');
        output.push_back(character);
    }
    return output;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16U;
    std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    std::uint32_t mantissa = bits & 0x3ffU;
    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            out = sign | (exponent << 23U) | ((mantissa & 0x3ffU) << 13U);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        out = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
    }
    float result = 0;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

struct Tensor final {
    std::vector<std::int64_t> shape;
    std::vector<std::byte> bytes;
};

struct Step final {
    diag::StepKind kind{};
    bool reset{};
    std::map<std::string, Tensor> tensors;
    std::vector<std::string> order;  // tensor names in recording order
    std::vector<nvcr::Frame> frames;
};

class Recorder final : public diag::Observer {
public:
    // Negative control (NVCR_DCVCUF_G1_CONTROL=feature|frame): flips one byte
    // of the first encoder HT-S feature or group frame, so the check is shown
    // to fail for the reason it tests.
    std::string control;

    void step(diag::Direction direction, diag::StepKind kind, bool reset) override {
        queue(direction).push_back(Step{kind, reset, {}, {}, {}});
    }
    void tensor(diag::Direction direction, std::string_view name,
                std::span<const std::int64_t> shape, std::span<const std::byte> bytes) override {
        auto& step = current(direction);
        if (!step.tensors.contains(std::string(name))) step.order.emplace_back(name);
        Tensor value{{shape.begin(), shape.end()}, {bytes.begin(), bytes.end()}};
        if (control == "feature" && direction == diag::Direction::encoder && name == "feature" &&
            !value.bytes.empty()) {
            value.bytes[value.bytes.size() / 2] ^= std::byte{0x01};
            control.clear();
        }
        step.tensors[std::string(name)] = std::move(value);
    }
    void frames(diag::Direction direction, std::span<const nvcr::Frame> frames) override {
        auto& step = current(direction);
        const auto first = step.frames.size();
        step.frames.insert(step.frames.end(), frames.begin(), frames.end());
        if (control == "frame" && direction == diag::Direction::encoder &&
            step.kind == diag::StepKind::predicted && step.frames.size() > first + 1) {
            auto data = step.frames[first + 1].data();
            data[data.size() / 3] ^= std::byte{0x01};
            control.clear();
        }
    }

    std::deque<Step>& queue(diag::Direction direction) {
        return direction == diag::Direction::encoder ? encoder_ : decoder_;
    }

private:
    Step& current(diag::Direction direction) {
        auto& steps = queue(direction);
        if (steps.empty()) fail("diagnostic record before its step");
        return steps.back();
    }
    std::deque<Step> encoder_;
    std::deque<Step> decoder_;
};

struct Mismatch final {
    std::uint64_t step{};
    std::uint64_t first_frame{};
    std::string kind;
    std::string stage;
    std::string tensor;
    std::uint64_t frame{};  // global frame number for frame mismatches
    std::uint64_t differing{};        // bytes (frames) or FP16 elements (tensors)
    std::uint64_t value_differing{};  // FP16 elements whose values differ (not only the sign of zero)
    std::uint64_t total{};
    std::uint64_t first_index{};
    double max_abs{};

    std::string json() const {
        std::ostringstream out;
        out << "{\"step\":" << step << ",\"step_first_frame\":" << first_frame << ",\"kind\":\""
            << kind << "\",\"stage\":\"" << json_escape(stage) << "\",\"tensor\":\""
            << json_escape(tensor) << "\",\"frame\":" << frame << ",\"differing_elements\":"
            << differing << ",\"value_differing_elements\":" << value_differing
            << ",\"elements\":" << total << ",\"first_differing_index\":"
            << first_index << ",\"max_abs_difference\":" << max_abs << "}";
        return out.str();
    }
};

// Upstream stage that produces each compared tensor (dmc_hts_proxy.cpp).
std::string stage_of(const std::string& name);
const std::map<std::string, std::string> producing_stage{
    {"memory", "feature_adaptor_i/feature_adaptor_m (reference memory for this group)"},
    {"context", "feature_extractor (context for this group)"},
    {"y_hat", "spatial-prior y reconstruction (process_with_mask / restore_y)"},
    {"feature", "decoder (HT-S synthesis feature)"},
    {"frames", "recon_head + pixel_shuffle_8 + YUV420 conversion"},
    {"z_hat", "hyper encoder + round_z (encoder) / z entropy decode (decoder)"},
    {"hyper_params", "image hyper decoder (scales and means for spatial step 0)"},
    {"common", "image hyper decoder (common prior features)"},
};
std::string stage_of(const std::string& name) {
    if (const auto found = producing_stage.find(name); found != producing_stage.end())
        return found->second;
    if (name.starts_with("spatial_prior_"))
        return name.ends_with("_input") ? "y_hat accumulated before image spatial prior step " +
                                              name.substr(14, 1)
                                        : "image spatial prior step " + name.substr(14, 1);
    return name;
}

// Byte comparison of one tensor; FP16 elements for the difference statistics.
std::optional<Mismatch> compare_tensor(const std::string& name, const Tensor& encoder,
                                       const Tensor& decoder) {
    if (encoder.shape != decoder.shape || encoder.bytes.size() != decoder.bytes.size())
        return Mismatch{.kind = "tensor-shape", .stage = stage_of(name), .tensor = name};
    if (encoder.bytes == decoder.bytes) return std::nullopt;
    Mismatch mismatch{.kind = "tensor", .stage = stage_of(name), .tensor = name};
    const auto count = encoder.bytes.size() / 2;
    mismatch.total = count;
    bool seen = false;
    for (std::size_t index = 0; index < count; ++index) {
        std::uint16_t a = 0;
        std::uint16_t b = 0;
        std::memcpy(&a, encoder.bytes.data() + index * 2, 2);
        std::memcpy(&b, decoder.bytes.data() + index * 2, 2);
        if (a == b) continue;
        ++mismatch.differing;
        const float x = half_to_float(a);
        const float y = half_to_float(b);
        if (x != y && !(std::isnan(x) && std::isnan(y))) {
            if (mismatch.value_differing++ == 0) mismatch.first_index = index;
            mismatch.max_abs = std::max(mismatch.max_abs, static_cast<double>(std::fabs(x - y)));
        } else if (!seen && mismatch.value_differing == 0) {
            mismatch.first_index = index;
        }
        seen = true;
    }
    // Equal values that differ only in the sign of zero (an encoder rounding a
    // small negative value versus a decoder rebuilding it from an integer) are
    // reported separately; they are not value differences.
    if (mismatch.value_differing == 0) mismatch.kind = "tensor-signed-zero";
    return mismatch;
}

class Comparator final {
public:
    std::uint64_t steps{};
    std::uint64_t intra_steps{};
    std::uint64_t predicted_steps{};
    std::uint64_t reset_steps{};
    std::uint64_t frames_compared{};
    std::uint64_t tensors_compared{};
    std::uint64_t tensor_bytes_compared{};
    std::uint64_t mismatched_steps{};
    std::uint64_t last_group_frames{};
    std::uint64_t signed_zero_only_tensors{};
    std::map<std::string, std::uint64_t> signed_zero_only_by_name;
    std::optional<Mismatch> first;
    std::optional<Mismatch> first_signed_zero_only;

    // Compares the oldest encoder step with the decoder step just completed.
    void compare(Recorder& recorder, std::span<const nvcr::Frame> received) {
        auto& encoder_steps = recorder.queue(diag::Direction::encoder);
        auto& decoder_steps = recorder.queue(diag::Direction::decoder);
        if (encoder_steps.empty() || decoder_steps.size() != 1)
            fail("decoder step without a matching encoder step");
        Step encoder = std::move(encoder_steps.front());
        Step decoder = std::move(decoder_steps.front());
        encoder_steps.pop_front();
        decoder_steps.pop_front();
        const auto step_index = steps++;
        const bool intra = decoder.kind == diag::StepKind::intra;
        (intra ? intra_steps : predicted_steps)++;
        if (decoder.reset) ++reset_steps;
        std::optional<Mismatch> found;
        auto note = [&](Mismatch mismatch) {
            mismatch.step = step_index;
            mismatch.first_frame = frame_cursor_;
            if (!found) found = std::move(mismatch);
        };
        if (encoder.kind != decoder.kind || encoder.reset != decoder.reset)
            note(Mismatch{.kind = "schedule", .stage = "adapter schedule", .tensor = ""});
        if (received.size() != decoder.frames.size())
            fail("decoder returned a different frame count than it reported");
        for (std::size_t index = 0; index < received.size(); ++index) {
            if (!std::ranges::equal(received[index].data(), decoder.frames[index].data()))
                fail("session output differs from the observed decoder frames");
        }
        if (!intra) {
            for (const auto* name : {"memory", "context", "y_hat", "feature"}) {
                if (!encoder.tensors.contains(name) || !decoder.tensors.contains(name))
                    fail(std::string("missing diagnostic tensor ") + name);
            }
        }
        // Every tensor both sides recorded, in the decoder's recording order.
        for (const auto& name : decoder.order) {
            const auto e = encoder.tensors.find(name);
            if (e == encoder.tensors.end()) continue;
            const auto& d = decoder.tensors.at(name);
            ++tensors_compared;
            tensor_bytes_compared += d.bytes.size();
            auto mismatch = compare_tensor(name, e->second, d);
            if (!mismatch) continue;
            if (mismatch->kind == "tensor-signed-zero") {
                ++signed_zero_only_tensors;
                ++signed_zero_only_by_name[name];
                if (!first_signed_zero_only) {
                    first_signed_zero_only = *mismatch;
                    first_signed_zero_only->step = step_index;
                    first_signed_zero_only->first_frame = frame_cursor_;
                }
                continue;
            }
            note(*mismatch);
        }
        const auto expected = intra ? std::size_t{1} : std::size_t{8};
        if (encoder.frames.size() != expected || decoder.frames.empty() ||
            decoder.frames.size() > expected)
            fail("unexpected diagnostic frame count");
        for (std::size_t index = 0; index < decoder.frames.size(); ++index) {
            const auto a = encoder.frames[index].data();
            const auto b = decoder.frames[index].data();
            ++frames_compared;
            if (std::ranges::equal(a, b)) continue;
            Mismatch mismatch{.kind = "frame", .stage = intra ? "image decoder" : "recon_head",
                              .tensor = "frames", .frame = frame_cursor_ + index};
            mismatch.stage = intra ? "image decoder + pixel_shuffle_8 + YUV420 conversion"
                                   : producing_stage.at("frames");
            mismatch.total = b.size();
            bool seen = false;
            for (std::size_t byte = 0; byte < std::min(a.size(), b.size()); ++byte) {
                if (a[byte] == b[byte]) continue;
                ++mismatch.differing;
                if (!seen) mismatch.first_index = byte;
                seen = true;
                mismatch.max_abs = std::max(
                    mismatch.max_abs,
                    std::fabs(static_cast<double>(std::to_integer<int>(a[byte])) -
                              std::to_integer<int>(b[byte])));
            }
            note(mismatch);
        }
        if (!intra) last_group_frames = decoder.frames.size();
        frame_cursor_ += decoder.frames.size();
        if (found) {
            ++mismatched_steps;
            if (!first) first = found;
        }
    }

    [[nodiscard]] std::uint64_t frames_seen() const { return frame_cursor_; }

private:
    std::uint64_t frame_cursor_{};
};

nvcr::Runtime make_runtime(const std::filesystem::path& bundle, std::uint32_t qp,
                           std::uint32_t gop) {
    auto configuration = value(nvcr::dcvcuf::runtime_configuration(bundle), "configuration");
    configuration.codec.intra_qp = qp;
    configuration.codec.gop_size = gop;
    configuration.runtime.log_level = nvcr::LogLevel::warning;
    return value(nvcr::Runtime::create(configuration), "runtime");
}

class YuvReader final {
public:
    YuvReader(const std::filesystem::path& path, std::uint32_t width, std::uint32_t height)
        : input_(path, std::ios::binary), width_(width), height_(height),
          bytes_(static_cast<std::size_t>(width) * height * 3 / 2) {
        if (!input_) fail("cannot open " + path.string());
    }
    nvcr::Frame frame(std::uint64_t index) {
        std::vector<std::byte> raw(bytes_);
        input_.seekg(static_cast<std::streamoff>(index * bytes_));
        input_.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
        if (!input_) fail("input ends before frame " + std::to_string(index));
        return value(nvcr::Frame::copy_from(
                         width_, height_, nvcr::PixelFormat::yuv420p8, raw,
                         nvcr::Timestamp{static_cast<nvcr::Timestamp::rep>(
                             static_cast<std::int64_t>(index) * frame_period_us)}),
                     "frame");
    }

private:
    std::ifstream input_;
    std::uint32_t width_;
    std::uint32_t height_;
    std::size_t bytes_;
};

// The CLI's .nvcr file layout (NVCS v1 header, then u64 length + PacketIO record).
class StreamWriter final {
public:
    explicit StreamWriter(const std::filesystem::path& path)
        : output_(path, std::ios::binary | std::ios::trunc) {
        if (!output_) fail("cannot create " + path.string());
        output_.write("NVCS", 4);
        const std::array<char, 4> fields{1, 0, 0, 0};
        output_.write(fields.data(), fields.size());
    }
    void write(const nvcr::Packet& packet) {
        auto wire = value(nvcr::PacketIO::serialize(packet), "packet serialization");
        std::array<char, 8> length{};
        for (std::size_t index = 0; index < 8; ++index)
            length[index] = static_cast<char>((wire.size() >> (index * 8)) & 0xffU);
        output_.write(length.data(), length.size());
        output_.write(reinterpret_cast<const char*>(wire.data()),
                      static_cast<std::streamsize>(wire.size()));
        bytes_ += packet.size();
        if (!output_) fail("stream write failed");
    }
    [[nodiscard]] std::uint64_t payload_bytes() const { return bytes_; }

private:
    std::ofstream output_;
    std::uint64_t bytes_{};
};

std::vector<nvcr::Packet> drain_packets(nvcr::Runtime& runtime) {
    std::vector<nvcr::Packet> output;
    for (;;) {
        auto packet = runtime.receive_access_unit();
        if (packet) {
            output.push_back(std::move(packet.value()));
            continue;
        }
        if (packet.error().code() == nvcr::ErrorCode::try_again ||
            packet.error().code() == nvcr::ErrorCode::end_of_stream)
            return output;
        fail("receive_access_unit: " + packet.error().describe());
    }
}

std::vector<nvcr::Frame> drain_frames(nvcr::Runtime& runtime) {
    std::vector<nvcr::Frame> output;
    for (;;) {
        auto frame = runtime.receive_frame();
        if (frame) {
            output.push_back(std::move(frame.value()));
            continue;
        }
        if (frame.error().code() == nvcr::ErrorCode::try_again ||
            frame.error().code() == nvcr::ErrorCode::end_of_stream)
            return output;
        fail("receive_frame: " + frame.error().describe());
    }
}

void bootstrap() {
    nvcr::dcvcrt::register_execution_providers();  // registers the generic tensorrt provider
    nvcr::dcvcuf::register_codec();
    nvcr::dcvcuf::register_execution_provider();
}

int run_g1(char** argv) {
    const std::filesystem::path bundle = argv[2];
    const std::filesystem::path input = argv[3];
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto height = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const auto frames = static_cast<std::uint64_t>(std::stoull(argv[6]));
    const auto qp = static_cast<std::uint32_t>(std::stoul(argv[7]));
    const auto gop = static_cast<std::uint32_t>(std::stoul(argv[8]));
    const std::filesystem::path stream_path = argv[9];
    const std::filesystem::path decoded_path = argv[10];
    const std::filesystem::path report_path = argv[11];

    Recorder recorder;
    if (const char* control = std::getenv("NVCR_DCVCUF_G1_CONTROL")) recorder.control = control;
    Comparator comparator;
    diag::set_observer(&recorder);
    auto runtime = make_runtime(bundle, qp, gop);
    YuvReader reader(input, width, height);
    StreamWriter stream(stream_path);
    std::ofstream decoded(decoded_path, std::ios::binary | std::ios::trunc);
    if (!decoded) fail("cannot create " + decoded_path.string());
    std::uint64_t packets = 0;
    // A decode error is a G1 failure; it is recorded with its step, and the
    // run stops because later steps depend on the failed one.
    std::optional<std::string> decode_failure;
    auto decode = [&](const nvcr::Packet& packet) {
        stream.write(packet);
        ++packets;
        if (decode_failure) return;
        auto sent = runtime.send_access_unit(packet);
        if (!sent) {
            const auto& pending = recorder.queue(diag::Direction::encoder);
            const auto& partial = recorder.queue(diag::Direction::decoder);
            const bool intra = !pending.empty() && pending.front().kind == diag::StepKind::intra;
            // Localize: the first tensor the failed decoder step recorded that
            // differs from the encoder's, in recording order.
            std::string localized = "null";
            std::size_t observed = 0;
            if (!pending.empty() && !partial.empty()) {
                for (const auto& name : partial.back().order) {
                    const auto e = pending.front().tensors.find(name);
                    if (e == pending.front().tensors.end()) continue;
                    ++observed;
                    auto mismatch =
                        compare_tensor(name, e->second, partial.back().tensors.at(name));
                    if (mismatch && mismatch->kind != "tensor-signed-zero") {
                        mismatch->step = comparator.steps;
                        mismatch->first_frame = comparator.frames_seen();
                        localized = mismatch->json();
                        break;
                    }
                }
            }
            decode_failure = "{\"step\":" + std::to_string(comparator.steps) +
                             ",\"step_first_frame\":" + std::to_string(comparator.frames_seen()) +
                             ",\"kind\":\"" + (intra ? "intra" : "predicted") +
                             "\",\"error\":\"" + json_escape(sent.error().describe()) +
                             "\",\"decoder_tensors_before_failure\":" + std::to_string(observed) +
                             ",\"first_differing_tensor\":" + localized + "}";
            return;
        }
        auto output = drain_frames(runtime);
        comparator.compare(recorder, output);
        for (const auto& frame : output)
            decoded.write(reinterpret_cast<const char*>(frame.data().data()),
                          static_cast<std::streamsize>(frame.data().size()));
    };
    for (std::uint64_t index = 0; index < frames && !decode_failure; ++index) {
        value(runtime.send_frame(reader.frame(index)), "send_frame");
        for (const auto& packet : drain_packets(runtime)) decode(packet);
    }
    if (!decode_failure) {
        value(runtime.flush_encoder(), "flush_encoder");
        for (const auto& packet : drain_packets(runtime)) decode(packet);
    }
    if (!decode_failure) {
        value(runtime.flush_decoder(), "flush_decoder");
        if (!drain_frames(runtime).empty()) fail("frames after the final drain");
    }
    diag::set_observer(nullptr);
    if (!decode_failure && (!recorder.queue(diag::Direction::encoder).empty() ||
                            !recorder.queue(diag::Direction::decoder).empty()))
        fail("unpaired diagnostic steps remain");
    decoded.close();

    const bool frames_complete = comparator.frames_seen() == frames;
    const bool pass = !decode_failure && frames_complete && comparator.mismatched_steps == 0;
    std::ofstream report(report_path);
    report << "{\"schema\":\"nvcr.dcvcuf.g1-reconstruction-identity.v1\",\"bundle\":\""
           << json_escape(bundle.string()) << "\",\"input\":\"" << json_escape(input.string())
           << "\",\"width\":" << width << ",\"height\":" << height << ",\"frames\":" << frames
           << ",\"qp\":" << qp << ",\"gop\":" << gop << ",\"access_units\":" << packets
           << ",\"payload_bytes\":" << stream.payload_bytes() << ",\"steps\":" << comparator.steps
           << ",\"intra_steps\":" << comparator.intra_steps
           << ",\"predicted_steps\":" << comparator.predicted_steps
           << ",\"feature_memory_resets\":" << comparator.reset_steps
           << ",\"last_group_frames\":" << comparator.last_group_frames
           << ",\"frames_decoded\":" << comparator.frames_seen()
           << ",\"frames_compared\":" << comparator.frames_compared
           << ",\"tensors_compared\":" << comparator.tensors_compared
           << ",\"tensor_bytes_compared\":" << comparator.tensor_bytes_compared
           << ",\"mismatched_steps\":" << comparator.mismatched_steps << ",\"first_mismatch\":"
           << (comparator.first ? comparator.first->json() : "null")
           << ",\"decode_failure\":" << decode_failure.value_or("null")
           << ",\"signed_zero_only_tensors\":" << comparator.signed_zero_only_tensors
           << ",\"signed_zero_only_by_name\":{" << [&] {
                  std::string text;
                  for (const auto& [name, count] : comparator.signed_zero_only_by_name)
                      text += (text.empty() ? "\"" : ",\"") + name + "\":" + std::to_string(count);
                  return text;
              }() << "},\"first_signed_zero_only\":"
           << (comparator.first_signed_zero_only ? comparator.first_signed_zero_only->json() : "null")
           << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
    std::cout << (pass ? "PASS" : "FAIL") << " steps=" << comparator.steps
              << " frames=" << comparator.frames_compared
              << " tensors=" << comparator.tensors_compared
              << " resets=" << comparator.reset_steps
              << " tail=" << comparator.last_group_frames
              << " mismatched_steps=" << comparator.mismatched_steps << '\n';
    if (comparator.first) std::cout << "first mismatch: " << comparator.first->json() << '\n';
    if (decode_failure) std::cout << "decode failure: " << *decode_failure << '\n';
    return pass ? 0 : 1;
}

struct Check final {
    std::string name;
    bool pass{};
    std::string detail;
};

std::vector<std::byte> concatenated(const std::vector<nvcr::Packet>& packets) {
    std::vector<std::byte> output;
    for (const auto& packet : packets) {
        auto wire = value(nvcr::PacketIO::serialize(packet), "packet serialization");
        output.insert(output.end(), wire.begin(), wire.end());
    }
    return output;
}

std::vector<nvcr::Packet> encode(nvcr::Runtime& runtime, YuvReader& reader, std::uint64_t first,
                                 std::uint64_t count) {
    std::vector<nvcr::Packet> output;
    for (std::uint64_t index = first; index < first + count; ++index) {
        value(runtime.send_frame(reader.frame(index)), "send_frame");
        auto packets = drain_packets(runtime);
        output.insert(output.end(), packets.begin(), packets.end());
    }
    value(runtime.flush_encoder(), "flush_encoder");
    auto packets = drain_packets(runtime);
    output.insert(output.end(), packets.begin(), packets.end());
    return output;
}

std::vector<std::byte> decode_all(nvcr::Runtime& runtime, const std::vector<nvcr::Packet>& packets,
                                  std::uint64_t& frames, std::vector<nvcr::Timestamp>& timestamps) {
    std::vector<std::byte> output;
    for (const auto& packet : packets) {
        value(runtime.send_access_unit(packet), "send_access_unit");
        for (const auto& frame : drain_frames(runtime)) {
            output.insert(output.end(), frame.data().begin(), frame.data().end());
            timestamps.push_back(frame.timestamp());
            ++frames;
        }
    }
    value(runtime.flush_decoder(), "flush_decoder");
    for (const auto& frame : drain_frames(runtime)) {
        output.insert(output.end(), frame.data().begin(), frame.data().end());
        timestamps.push_back(frame.timestamp());
        ++frames;
    }
    return output;
}

bool idle_encoder(nvcr::Runtime& runtime) {
    auto packet = runtime.receive_access_unit();
    return !packet && packet.error().code() == nvcr::ErrorCode::try_again;
}
bool idle_decoder(nvcr::Runtime& runtime) {
    auto frame = runtime.receive_frame();
    return !frame && frame.error().code() == nvcr::ErrorCode::try_again;
}

int run_partial_group(char** argv) {
    const std::filesystem::path bundle = argv[2];
    const std::filesystem::path input = argv[3];
    const auto width = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto height = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const auto qp = static_cast<std::uint32_t>(std::stoul(argv[6]));
    const std::filesystem::path report_path = argv[7];
    constexpr std::uint32_t gop = 65528;  // intra only at frame 0, as in the G1 runs
    YuvReader reader(input, width, height);
    std::vector<Check> checks;

    // 1. Drain with a partly filled group: I + 5 frames, then flush. The tail
    //    is padded to eight frames inside the encoder; the decoder must return
    //    exactly the five real frames, in order, and G1 identity must hold.
    {
        Recorder recorder;
        Comparator comparator;
        diag::set_observer(&recorder);
        auto runtime = make_runtime(bundle, qp, gop);
        auto packets = encode(runtime, reader, 0, 6);
        std::vector<nvcr::Timestamp> timestamps;
        for (const auto& packet : packets) {
            value(runtime.send_access_unit(packet), "send_access_unit");
            auto output = drain_frames(runtime);
            comparator.compare(recorder, output);
            for (const auto& frame : output) timestamps.push_back(frame.timestamp());
        }
        value(runtime.flush_decoder(), "flush_decoder");
        const bool drained = drain_frames(runtime).empty();
        diag::set_observer(nullptr);
        bool ordered = timestamps.size() == 6;
        for (std::size_t index = 0; ordered && index < timestamps.size(); ++index)
            ordered = timestamps[index].count() == static_cast<std::int64_t>(index) * frame_period_us;
        checks.push_back({"drain-partial-group",
                          packets.size() == 2 && drained && ordered &&
                              comparator.mismatched_steps == 0 && comparator.last_group_frames == 5,
                          "access_units=" + std::to_string(packets.size()) +
                              " frames=" + std::to_string(timestamps.size()) +
                              " tail_frames=" + std::to_string(comparator.last_group_frames) +
                              " identity_mismatched_steps=" +
                              std::to_string(comparator.mismatched_steps)});
    }

    // Reference: a fresh session encoding and decoding frames 0..16 (I, two groups).
    std::vector<nvcr::Packet> reference_packets;
    std::vector<std::byte> reference_frames;
    std::uint64_t reference_count = 0;
    {
        auto runtime = make_runtime(bundle, qp, gop);
        reference_packets = encode(runtime, reader, 0, 17);
        std::vector<nvcr::Timestamp> timestamps;
        reference_frames = decode_all(runtime, reference_packets, reference_count, timestamps);
    }

    // 2. Encoder reset with a partly filled group: different content (frames
    //    40..45: I + 5 pending), reset, then frames 0..16. The pending frames
    //    are discarded and the stream equals the fresh session's.
    {
        auto runtime = make_runtime(bundle, qp, gop);
        for (std::uint64_t index = 40; index < 46; ++index)
            value(runtime.send_frame(reader.frame(index)), "send_frame");
        value(runtime.reset_encoder(), "reset_encoder");
        const bool idle = idle_encoder(runtime);
        auto packets = encode(runtime, reader, 0, 17);
        const bool same = concatenated(packets) == concatenated(reference_packets);
        checks.push_back({"encoder-reset-partial-group", idle && same,
                          std::string("no_output_after_reset=") + (idle ? "true" : "false") +
                              " stream_equals_fresh_session=" + (same ? "true" : "false")});
    }

    // 3. Decoder reset with queued output: decode the I unit and first group
    //    of a different stream without receiving, reset, then decode the
    //    reference stream. No stale frames; output equals a fresh decoder's.
    {
        std::vector<nvcr::Packet> other;
        {
            auto encoder = make_runtime(bundle, qp, gop);
            other = encode(encoder, reader, 40, 17);
        }
        auto runtime = make_runtime(bundle, qp, gop);
        value(runtime.send_access_unit(other.at(0)), "send_access_unit");
        value(runtime.send_access_unit(other.at(1)), "send_access_unit");
        value(runtime.reset_decoder(), "reset_decoder");
        const bool idle = idle_decoder(runtime);
        std::uint64_t count = 0;
        std::vector<nvcr::Timestamp> timestamps;
        const auto frames = decode_all(runtime, reference_packets, count, timestamps);
        const bool same = frames == reference_frames && count == reference_count;
        checks.push_back({"decoder-reset-queued-output", idle && same,
                          std::string("no_output_after_reset=") + (idle ? "true" : "false") +
                              " frames_equal_fresh_decoder=" + (same ? "true" : "false") +
                              " frames=" + std::to_string(count)});
    }

    // 4. Encoder reset after a flushed partial group, then reuse.
    {
        auto runtime = make_runtime(bundle, qp, gop);
        (void)encode(runtime, reader, 40, 6);
        value(runtime.reset_encoder(), "reset_encoder");
        auto packets = encode(runtime, reader, 0, 17);
        const bool same = concatenated(packets) == concatenated(reference_packets);
        checks.push_back({"encoder-reuse-after-flushed-partial-group", same,
                          std::string("stream_equals_fresh_session=") + (same ? "true" : "false")});
    }

    bool pass = true;
    std::ofstream report(report_path);
    report << "{\"schema\":\"nvcr.dcvcuf.partial-group-lifecycle.v1\",\"bundle\":\""
           << json_escape(bundle.string()) << "\",\"input\":\"" << json_escape(input.string())
           << "\",\"width\":" << width << ",\"height\":" << height << ",\"qp\":" << qp
           << ",\"checks\":[";
    for (std::size_t index = 0; index < checks.size(); ++index) {
        const auto& check = checks[index];
        pass = pass && check.pass;
        report << (index ? "," : "") << "{\"name\":\"" << check.name << "\",\"pass\":"
               << (check.pass ? "true" : "false") << ",\"detail\":\"" << json_escape(check.detail)
               << "\"}";
        std::cout << (check.pass ? "PASS " : "FAIL ") << check.name << ": " << check.detail << '\n';
    }
    report << "],\"pass\":" << (pass ? "true" : "false") << "}\n";
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        bootstrap();
        if (mode == "g1" && argc == 12) return run_g1(argv);
        if (mode == "partial-group" && argc == 8) return run_partial_group(argv);
        std::cerr << "usage: g1 BUNDLE INPUT WIDTH HEIGHT FRAMES QP GOP STREAM DECODED REPORT\n"
                     "       partial-group BUNDLE INPUT WIDTH HEIGHT QP REPORT\n";
        return 2;
    } catch (const std::exception& exception) {
        diag::set_observer(nullptr);
        std::cerr << "error: " << exception.what() << '\n';
        return 2;
    }
}
