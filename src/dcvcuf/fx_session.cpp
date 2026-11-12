#include "fx_session.hpp"

#include "../codec/fx_stages.hpp"
#include "entropy/uf_rans_codec.hpp"
#include "numerics.hpp"
#include "tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// DCVC-UF under nvcr.fx.v1. Every stage that decides an entropy table or the
// state carried to the next unit runs as an fx program on the provider (any
// provider that claims the profile). The I-frame codes its latent in four
// spatial steps; an eight-frame HT-S chunk takes all its entropy indexes from
// one prior and codes them in one pass, its spatial steps refining only the
// means. The analysis transforms and the pictures shown for a chunk stay FP16:
// nothing reads them back. Entropy coding (UF rANS), payloads and the chunk
// schedule are DCVC-UF's usual ones.

namespace nvcr::dcvcuf {
namespace {

namespace api = provider::experimental;
using codec::fx::Bind;
using codec::fx::Buffer;
using codec::fx::run;
using Shape = std::array<std::int64_t, 4>;

constexpr std::string_view subsystem = "dcvcuf.fx";
constexpr int fa = 14;
constexpr std::int32_t one = 1 << fa;
constexpr int ch_y = 256;
constexpr int ch_z = 128;
constexpr int ch_feature = 512;
constexpr int min_symbols_per_stream = 32768;  // upstream def_const.h
constexpr int max_entropy_parallel = 8;
constexpr std::size_t group = 8;  // frames per HT-S chunk

Error fail(std::string message) { return Error(ErrorCode::backend_error, std::move(message), std::string(subsystem)); }

std::int64_t round_up(std::int64_t value, std::int64_t multiple) { return (value + multiple - 1) / multiple * multiple; }

// Upstream get_mask_4x: in channel group g (64 channels each), step k codes the
// positions whose 2 (row % 2) + column % 2 equals four_way[k][g].
constexpr std::array<std::array<int, 4>, 4> four_way{{{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2}}};

bool active(int k, int channel, std::int64_t row, std::int64_t col) {
    return four_way[static_cast<std::size_t>(k)][static_cast<std::size_t>(channel / (ch_y / 4))] ==
           2 * (row % 2) + col % 2;
}

struct Geometry final {
    std::int64_t width, height, ph, pw, yh, yw, yh4, yw4;

    static Geometry of(std::uint32_t width, std::uint32_t height) {
        Geometry g{};
        g.width = width;
        g.height = height;
        g.ph = round_up(height, 16);
        g.pw = round_up(width, 16);
        g.yh = g.ph / 16;
        g.yw = g.pw / 16;
        g.yh4 = round_up(g.yh, 4);
        g.yw4 = round_up(g.yw, 4);
        return g;
    }
    [[nodiscard]] Shape latent(std::int64_t channels) const { return {1, channels, yh, yw}; }
    [[nodiscard]] Shape feature(std::int64_t channels) const { return {1, channels, ph / 8, pw / 8}; }
    [[nodiscard]] Shape z() const { return {1, ch_z, yh4 / 4, yw4 / 4}; }
};

// One direction's carried state: the I-frame picture or the last chunk's
// feature, and the HT-S feature memory (none after an I-frame or a reset).
struct Reference final {
    std::optional<Buffer> frame;
    std::optional<Buffer> feature;
    std::optional<Buffer> memory;
    bool reset{false};  // the last chunk reset the feature memory

    void clear() {
        frame.reset();
        feature.reset();
        memory.reset();
        reset = false;
    }
};

class FxSession final : public ExecutionSession {
public:
    explicit FxSession(std::shared_ptr<api::IProviderSession> session)
        : session_(std::move(session)), owner_("dcvcuf-fx-" + std::string(session_->owner_id())) {}

    Result<void> initialize(const RuntimeConfiguration& configuration) {
        try {
            auto normative = normative_stages(configuration);
            if (!normative) return normative.error();
            if (!normative.value()) return fail("the fx session needs the nvcr.fx.v1 profile");
            for (auto descriptor : normative.value()->descriptors) {
                const auto id = descriptor.stage_id;
                auto loaded = session_->load_stage(std::move(descriptor));
                if (!loaded) return loaded.error();
                stages_[id] = std::move(loaded.value());
            }
            const auto& root = configuration.artifacts.intra_engine_path;
            using Dims = std::vector<api::DimensionBounds>;
            const api::DimensionBounds any{1, 65535};
            const auto u8 = api::TensorDataType::uint8;
            const auto f16 = api::TensorDataType::float16;
            const auto i32 = api::TensorDataType::int32;
            const auto read = api::TensorAccess::read;
            const auto write = api::TensorAccess::write;
            const auto channels = [](std::int64_t c) { return api::DimensionBounds{c, c}; };
            const std::vector<std::pair<std::string, std::vector<api::TensorContract>>> fp16{
                {"i_analysis", {{"luma", u8, read, Dims{{1, 1}, channels(1), any, any}},
                                {"chroma", u8, read, Dims{{1, 1}, channels(2), any, any}},
                                {"q_enc", f16, read, Dims{{1, 1}, channels(384), {1, 1}, {1, 1}}},
                                {"y", f16, write, Dims{{1, 1}, channels(ch_y), any, any}}}},
                {"i_hyper_analysis", {{"y_padded", f16, read, Dims{{1, 1}, channels(ch_y), any, any}},
                                      {"z", f16, write, Dims{{1, 1}, channels(ch_z), any, any}}}},
                {"p_encode", {{"luma", u8, read, Dims{{1, 1}, channels(group), any, any}},
                              {"chroma", u8, read, Dims{{1, 1}, channels(2 * group), any, any}},
                              {"context", i32, read, Dims{{1, 1}, channels(ch_feature), any, any}},
                              {"q_encoder", f16, read, Dims{{1, 1}, channels(ch_feature), {1, 1}, {1, 1}}},
                              {"y", f16, write, Dims{{1, 1}, channels(ch_y), any, any}}}},
                {"p_hyper_analysis", {{"y_padded", f16, read, Dims{{1, 1}, channels(ch_y), any, any}},
                                      {"z", f16, write, Dims{{1, 1}, channels(ch_z), any, any}}}},
                {"p_display", {{"feature", i32, read, Dims{{1, 1}, channels(ch_feature), any, any}},
                               {"luma", u8, write, Dims{{1, 1}, channels(group), any, any}},
                               {"chroma", u8, write, Dims{{1, 1}, channels(2 * group), any, any}}}}};
            for (const auto& [component, tensors] : fp16) {
                api::ExecutableStageDescriptor descriptor;
                descriptor.stage_id = component + ".plan";
                descriptor.artifact.codec_id = "dcvc-uf";
                descriptor.artifact.model_set_id = configuration.artifacts.model_id;
                descriptor.artifact.component_id = component;
                descriptor.artifact.provider_id = configuration.provider.id;
                descriptor.artifact.precision = "fp16";
                descriptor.artifact.path = root.string();
                descriptor.tensors = tensors;
                auto loaded = session_->load_stage(std::move(descriptor));
                if (!loaded) return loaded.error();
                stages_[component] = std::move(loaded.value());
            }
            std::tie(image_, video_) = read_tables(root / "tables.bin");
            return {};
        } catch (const std::exception& exception) {
            return fail(std::string("fx session initialization failed: ") + exception.what());
        }
    }

    std::string_view owner_id() const noexcept override { return owner_; }
    Result<api::BufferHandle> allocate(std::size_t, api::MemoryDomain) override {
        return fail("DCVC-UF fx buffers are codec-owned");
    }
    Result<api::ExecutableStage> load_stage(api::ExecutableStageDescriptor) override {
        return fail("DCVC-UF loads its fx stages itself");
    }
    Result<api::Completion> submit(const api::ExecutableStage&, std::span<const api::TensorView>,
                                   std::span<const api::ExecutionDependency>) override {
        return fail("DCVC-UF runs whole units, not single stages");
    }
    Result<void> reset() override { return session_->reset(); }
    Result<void> reset_encoder_state() override {
        encoder_.clear();
        return {};
    }
    Result<void> reset_decoder_state() override {
        decoder_.clear();
        return {};
    }

    Result<Encoded> encode_intra(const Frame& frame, std::uint32_t qp) override {
        return guarded<Encoded>([&] { return encode_i(frame, qp); });
    }
    Result<Frame> decode_intra(std::span<const std::byte> bitstream, std::uint8_t parallel, std::uint32_t qp,
                               std::uint32_t width, std::uint32_t height, Timestamp timestamp) override {
        return guarded<Frame>([&] { return decode_i(bitstream, parallel, qp, width, height, timestamp); });
    }
    Result<Encoded> encode_predicted(std::span<const Frame> frames, std::uint32_t qp, bool reset) override {
        if (frames.size() != group) return Error(ErrorCode::invalid_argument, "HT-S encoding requires eight frames", std::string(subsystem));
        return guarded<Encoded>([&] { return encode_p(frames, qp, reset); });
    }
    Result<std::vector<Frame>> decode_predicted(std::span<const std::byte> bitstream, std::uint8_t parallel,
                                                std::uint32_t qp, std::uint32_t width, std::uint32_t height,
                                                std::span<const Timestamp> timestamps, bool reset) override {
        if (timestamps.empty() || timestamps.size() > group) {
            return Error(ErrorCode::invalid_argument, "HT-S timestamp count must be in [1, 8]", std::string(subsystem));
        }
        return guarded<std::vector<Frame>>(
            [&] { return decode_p(bitstream, parallel, qp, width, height, timestamps, reset); });
    }

private:
    template <class T, class F>
    Result<T> guarded(F&& f) {
        try {
            return f();
        } catch (const std::exception& exception) {
            return fail(exception.what());
        }
    }

    const api::ExecutableStage& stage(const std::string& id) const { return stages_.at(id); }

    Buffer take(Shape shape, bool host, api::TensorDataType type = api::TensorDataType::int32) {
        auto made = pool_.take(*session_, shape, host, type);
        if (!made) throw std::runtime_error(std::string(made.error().message()));
        return std::move(made.value());
    }

    static void check(const Result<void>& done) {
        if (!done) throw std::runtime_error(std::string(done.error().message()));
    }

    Buffer constant(Shape shape, std::int32_t value) {
        auto t = take(shape, true);
        std::fill(t.ints().begin(), t.ints().end(), value);
        return t;
    }

    // The FP16 row of a quantization table for qp, as the stages read it.
    Buffer q_row(const Tables& tables, const std::string& name, std::uint32_t qp) {
        const auto width = tables.q_width.at(name);
        auto t = take({1, width, 1, 1}, true, api::TensorDataType::float16);
        const auto& rows = tables.q.at(name);
        std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(qp) * width, width, t.halves().begin());
        return t;
    }

    // Channel 2 (row % 2) + column % 2 of the parity masks is 1 there, 0 elsewhere.
    const Buffer& parity(const Geometry& g) {
        if (!parity_ || parity_->shape[2] != g.yh || parity_->shape[3] != g.yw) {
            auto made = codec::fx::allocate(*session_, {1, 4, g.yh, g.yw}, true);
            if (!made) throw std::runtime_error(std::string(made.error().message()));
            const auto plane = g.yh * g.yw;
            auto v = made.value().ints();
            for (std::size_t i = 0; i < v.size(); ++i) {
                const auto at = static_cast<std::int64_t>(i) % plane;
                v[i] = 2 * (at / g.yw % 2) + at % g.yw % 2 == static_cast<std::int64_t>(i) / plane ? 1 : 0;
            }
            parity_ = std::move(made.value());
        }
        return *parity_;
    }

    // The frames' 8-bit planes at the padded size, edges replicated: one luma and two chroma channels per frame.
    std::pair<Buffer, Buffer> planes(std::span<const Frame> frames, const Geometry& g) {
        const auto n = static_cast<std::int64_t>(frames.size());
        auto luma = take({1, n, g.ph, g.pw}, true, api::TensorDataType::uint8);
        auto chroma = take({1, 2 * n, g.ph / 2, g.pw / 2}, true, api::TensorDataType::uint8);
        const auto pad = [](const std::byte* in, std::int64_t width, std::int64_t height, std::byte* out,
                            std::int64_t padded_width, std::int64_t padded_height) {
            for (std::int64_t row = 0; row < padded_height; ++row, out += padded_width) {
                const auto* line = in + std::min(row, height - 1) * width;
                std::copy_n(line, width, out);
                std::fill(out + width, out + padded_width, line[width - 1]);
            }
        };
        const auto w = g.width, h = g.height;
        auto* y = luma.handle->host_bytes().data();
        auto* c = chroma.handle->host_bytes().data();
        for (std::int64_t f = 0; f < n; ++f) {
            const auto* in = frames[static_cast<std::size_t>(f)].data().data();
            pad(in, w, h, y + f * g.ph * g.pw, g.pw, g.ph);
            for (std::int64_t k = 0; k < 2; ++k) {
                pad(in + w * h + k * (w * h / 4), w / 2, h / 2, c + (2 * f + k) * (g.ph / 2) * (g.pw / 2), g.pw / 2,
                    g.ph / 2);
            }
        }
        return {std::move(luma), std::move(chroma)};
    }

    // y (FP16 from an analysis stage) -> z symbols (NCHW) through the hyper analysis on y padded to a multiple of 4.
    std::vector<std::int8_t> hyper(const Buffer& y, const Geometry& g, const std::string& component) {
        auto padded = take({1, ch_y, g.yh4, g.yw4}, true, api::TensorDataType::float16);
        auto z = take(g.z(), true, api::TensorDataType::float16);
        const auto in = y.halves();
        auto out = padded.halves();
        for (std::int64_t c = 0; c < ch_y; ++c) {
            for (std::int64_t row = 0; row < g.yh4; ++row) {
                for (std::int64_t col = 0; col < g.yw4; ++col) {
                    out[static_cast<std::size_t>((c * g.yh4 + row) * g.yw4 + col)] = in[static_cast<std::size_t>(
                        (c * g.yh + std::min(row, g.yh - 1)) * g.yw + std::min(col, g.yw - 1))];
                }
            }
        }
        check(run(*session_, stage(component), {{"y_padded", &padded}}, {{"z", &z}}));
        std::vector<std::int8_t> symbols(z.count());
        std::transform(z.halves().begin(), z.halves().end(), symbols.begin(), [](std::uint16_t v) {
            return static_cast<std::int8_t>(std::clamp(std::nearbyint(half_to_double(v)), -128.0, 127.0));
        });
        return symbols;
    }

    // z symbols NCHW <-> the coder's NHWC order (the z CDF follows the channel).
    static std::vector<std::int8_t> nhwc(std::span<const std::int8_t> z, const Shape& s, bool to_nhwc) {
        std::vector<std::int8_t> out(z.size());
        const auto plane = s[2] * s[3];
        for (std::int64_t c = 0; c < s[1]; ++c) {
            for (std::int64_t p = 0; p < plane; ++p) {
                const auto chw = static_cast<std::size_t>(c * plane + p);
                const auto hwc = static_cast<std::size_t>(p * s[1] + c);
                out[to_nhwc ? hwc : chw] = z[to_nhwc ? chw : hwc];
            }
        }
        return out;
    }

    Buffer z_fixed(std::span<const std::int8_t> symbols, const Geometry& g) {
        auto z = take(g.z(), true);
        std::transform(symbols.begin(), symbols.end(), z.ints().begin(), [](std::int8_t v) { return v * one; });
        return z;
    }

    UfRansCodec& coder(bool image, bool encoder, int parallel) {
        auto& slot = coders_[image ? 0 : 1][encoder ? 0 : 1][static_cast<std::size_t>(parallel - 1)];
        if (!slot) {
            slot.emplace();
            const auto& tables = image ? image_ : video_;
            if (!slot->set_entropy_coder_parallel(static_cast<std::size_t>(parallel)) || !slot->add_cdf(tables.z) ||
                !slot->add_cdf(tables.y)) {
                throw std::runtime_error("cannot configure the UF entropy coder");
            }
        }
        return *slot;
    }

    // Packed coder rows: signed symbol in the high byte, CDF row in the low byte; z coded last (decoded first).
    Encoded entropy_encode(bool image, const std::vector<std::vector<std::int16_t>>& steps,
                           std::span<const std::int8_t> z, const Geometry& g, std::uint32_t qp) {
        std::size_t total = 0;
        for (const auto& step : steps) total += step.size();
        const int parallel = std::clamp(static_cast<int>(total) / min_symbols_per_stream, 1, max_entropy_parallel);
        auto& codec = coder(image, true, parallel);
        codec.reset_encoder();
        for (auto step = steps.rbegin(); step != steps.rend(); ++step) check(codec.encode_y(*step, 1));
        check(codec.encode_z(nhwc(z, g.z(), true), 0, static_cast<std::size_t>(qp) * ch_z, ch_z));
        auto stream = codec.finish_encode();
        if (!stream) throw std::runtime_error(std::string(stream.error().message()));
        return Encoded{std::move(stream.value()), static_cast<std::uint8_t>(parallel)};
    }

    std::vector<std::int8_t> decode_z(UfRansCodec& codec, std::span<const std::byte> bitstream, const Geometry& g,
                                      std::uint32_t qp) {
        check(codec.set_stream(bitstream));
        const auto s = g.z();
        auto z = codec.decode_z(static_cast<std::size_t>(s[1] * s[2] * s[3]), 0, static_cast<std::size_t>(qp) * ch_z,
                                ch_z);
        if (!z) throw std::runtime_error(std::string(z.error().message()));
        return nhwc(z.value(), s, false);
    }

    static std::vector<std::uint8_t> indexes_of(const Buffer& indexes) {
        std::vector<std::uint8_t> out(indexes.count());
        std::transform(indexes.ints().begin(), indexes.ints().end(), out.begin(),
                       [](std::int32_t v) { return static_cast<std::uint8_t>(v); });
        return out;
    }

    // The coder's part of a decoded step: symbols for the indexes that are not skipped, 0 for the others.
    static void place(std::span<const std::uint8_t> index, std::span<const std::int8_t> decoded, const Buffer& symbols) {
        auto out = symbols.ints();
        std::size_t next = 0;
        for (std::size_t i = 0; i < index.size(); ++i) {
            out[i] = index[i] == skipped_index ? 0 : decoded[next++];
        }
        if (next != decoded.size()) throw std::runtime_error("decoded symbol count");
    }

    static std::vector<std::uint8_t> coded(std::span<const std::uint8_t> index) {
        std::vector<std::uint8_t> out;
        for (const auto i : index) {
            if (i != skipped_index) out.push_back(i);
        }
        return out;
    }

    // The encoder's symbol round(y_scaled - mean) of one element, clamped like upstream; skipped elements are 0.
    static std::int32_t quantize(double y_scaled, std::int32_t mean, std::uint8_t index) {
        if (index == skipped_index) return 0;
        return static_cast<std::int32_t>(std::clamp(std::nearbyint(y_scaled - static_cast<double>(mean) / one), -128.0, 127.0));
    }

    static void row(std::vector<std::int16_t>& rows, std::int32_t symbol, std::uint8_t index) {
        if (index == skipped_index) return;
        rows.push_back(static_cast<std::int16_t>(static_cast<std::uint16_t>(static_cast<std::uint8_t>(symbol)) << 8U | index));
    }

    // 8-bit 4:2:0 from an exact picture in [-1/2, 1/2] at 2^-fa: luma round(255 v), chroma round(255 mean of 2x2).
    static Frame show(const Buffer& picture, const Geometry& g, Timestamp timestamp) {
        const auto v = picture.ints();
        const auto plane = static_cast<std::size_t>(g.ph * g.pw);
        const auto w = static_cast<std::size_t>(g.width), h = static_cast<std::size_t>(g.height);
        const auto stride = static_cast<std::size_t>(g.pw);
        std::vector<std::byte> bytes(w * h * 3 / 2);
        auto* out = reinterpret_cast<std::uint8_t*>(bytes.data());
        const auto at = [&](std::size_t c, std::size_t y, std::size_t x) {
            return std::clamp<std::int64_t>(v[c * plane + y * stride + x] + one / 2, 0, one);
        };
        for (std::size_t y = 0; y < h; ++y) {
            for (std::size_t x = 0; x < w; ++x) *out++ = static_cast<std::uint8_t>((at(0, y, x) * 255 + one / 2) >> fa);
        }
        for (std::size_t c = 1; c < 3; ++c) {
            for (std::size_t y = 0; y < h; y += 2) {
                for (std::size_t x = 0; x < w; x += 2) {
                    const auto sum = at(c, y, x) + at(c, y, x + 1) + at(c, y + 1, x) + at(c, y + 1, x + 1);
                    *out++ = static_cast<std::uint8_t>((sum * 255 + 2 * one) >> (fa + 2));
                }
            }
        }
        auto frame = Frame::copy_from(static_cast<std::uint32_t>(g.width), static_cast<std::uint32_t>(g.height),
                                      PixelFormat::yuv420p8, bytes, timestamp);
        if (!frame) throw std::runtime_error(std::string(frame.error().message()));
        return std::move(frame.value());
    }

    // Frame f of p_display's 8-bit planes, cropped to the visible size.
    static Frame shown(const Buffer& luma, const Buffer& chroma, std::size_t f, const Geometry& g, Timestamp timestamp) {
        const auto w = static_cast<std::size_t>(g.width), h = static_cast<std::size_t>(g.height);
        const auto pw = static_cast<std::size_t>(g.pw), ph = static_cast<std::size_t>(g.ph);
        std::vector<std::byte> bytes(w * h * 3 / 2);
        auto* out = bytes.data();
        const auto* y = luma.handle->host_bytes().data() + f * ph * pw;
        for (std::size_t row = 0; row < h; ++row, out += w) std::copy_n(y + row * pw, w, out);
        for (std::size_t k = 0; k < 2; ++k) {
            const auto* c = chroma.handle->host_bytes().data() + (2 * f + k) * (ph / 2) * (pw / 2);
            for (std::size_t row = 0; row < h / 2; ++row, out += w / 2) std::copy_n(c + row * (pw / 2), w / 2, out);
        }
        auto frame = Frame::copy_from(static_cast<std::uint32_t>(g.width), static_cast<std::uint32_t>(g.height),
                                      PixelFormat::yuv420p8, bytes, timestamp);
        if (!frame) throw std::runtime_error(std::string(frame.error().message()));
        return std::move(frame.value());
    }

    // ---- I-frame ------------------------------------------------------------------------------------------------
    // y_scaled (encoder): y q_scale_y_enc, NCHW; decode: the coder's step decoder. Returns the exact picture and, for
    // the encoder, the coder rows of each step.
    template <class Decode>
    std::pair<Buffer, std::vector<std::vector<std::int16_t>>> intra(const Geometry& g, std::uint32_t qp,
                                                                     std::span<const std::int8_t> z_symbols,
                                                                     const std::vector<double>* y_scaled,
                                                                     Decode&& decode) {
        auto& s = *session_;
        const bool host = y_scaled != nullptr;  // the encoder reads the means
        const Shape reduced = g.latent(64);
        auto z = z_fixed(z_symbols, g);
        auto ref = constant(g.latent(1), 0);
        const auto& par = parity(g);
        auto common = take(g.latent(ch_y), false);
        auto indexes = take(reduced, true);
        auto means = take(reduced, host);
        check(run(s, stage("i_prior"), {{"z_fixed", &z}, {"latent_ref", &ref}, {"parity", &par}},
                  {{"common", &common}, {"indexes", &indexes}, {"means", &means}}));
        std::vector<std::vector<std::int16_t>> rows;
        std::optional<Buffer> y_hat;
        for (int k = 0;; ++k) {
            auto symbols = take(reduced, true);
            const auto index = indexes_of(indexes);
            if (y_scaled == nullptr) {
                place(index, decode(coded(index)), symbols);
            } else {
                // Reduced element r is, for its channel c of 64 and position, the active element of the step.
                std::vector<std::int16_t> step;
                auto out = symbols.ints();
                std::size_t r = 0;
                for (std::int64_t c = 0; c < 64; ++c) {
                    for (std::int64_t row = 0; row < g.yh; ++row) {
                        for (std::int64_t col = 0; col < g.yw; ++col, ++r) {
                            int full = 0;
                            while (!active(k, full * 64, row, col)) ++full;
                            const auto at = static_cast<std::size_t>(((full * 64 + c) * g.yh + row) * g.yw + col);
                            out[r] = quantize((*y_scaled)[at], means.ints()[r], index[r]);
                            this->row(step, out[r], index[r]);
                        }
                    }
                }
                rows.push_back(std::move(step));
            }
            if (k == 3) {
                auto picture = take({1, 3, g.ph, g.pw}, true);
                auto qp_t = constant({1, 1, 1, 1}, static_cast<std::int32_t>(qp));
                check(run(s, stage("i_synthesis"),
                          {{"y_hat", &*y_hat}, {"symbols", &symbols}, {"symbol_means", &means}, {"qp", &qp_t},
                           {"parity", &par}},
                          {{"frame", &picture}}));
                return {std::move(picture), std::move(rows)};
            }
            auto next = take(g.latent(ch_y), false);
            auto next_indexes = take(reduced, true);
            auto next_means = take(reduced, host);
            std::vector<Bind> in{{"symbols", &symbols}, {"symbol_means", &means}, {"common", &common}, {"parity", &par}};
            if (y_hat) in.push_back({"y_hat", &*y_hat});
            check(run(s, stage("i_spatial_prior_" + std::to_string(k + 1)), in,
                      {{"y_hat_next", &next}, {"indexes", &next_indexes}, {"means", &next_means}}));
            y_hat = std::move(next);
            indexes = std::move(next_indexes);
            means = std::move(next_means);
        }
    }

    Result<Encoded> encode_i(const Frame& frame, std::uint32_t qp) {
        const auto g = Geometry::of(frame.width(), frame.height());
        auto [luma, chroma] = planes(std::span(&frame, 1), g);
        auto q = q_row(image_, "q_scale_enc", qp);
        auto y = take(g.latent(ch_y), true, api::TensorDataType::float16);
        check(run(*session_, stage("i_analysis"), {{"luma", &luma}, {"chroma", &chroma}, {"q_enc", &q}}, {{"y", &y}}));
        const auto z = hyper(y, g, "i_hyper_analysis");
        auto q_y = q_row(image_, "q_scale_y_enc", qp);
        std::vector<double> scaled(y.count());
        const auto plane = static_cast<std::size_t>(g.yh * g.yw);
        for (std::size_t i = 0; i < scaled.size(); ++i) {
            scaled[i] = half_to_double(y.halves()[i]) * half_to_double(q_y.halves()[i / plane]);
        }
        auto [picture, rows] = intra(g, qp, z, &scaled, [](auto) -> std::vector<std::int8_t> { return {}; });
        auto encoded = entropy_encode(true, rows, z, g, qp);
        encoder_.clear();
        encoder_.frame = std::move(picture);
        return encoded;
    }

    Result<Frame> decode_i(std::span<const std::byte> bitstream, std::uint8_t parallel, std::uint32_t qp,
                           std::uint32_t width, std::uint32_t height, Timestamp timestamp) {
        const auto g = Geometry::of(width, height);
        auto& codec = coder(true, false, parallel);
        const auto z = decode_z(codec, bitstream, g, qp);
        auto [picture, rows] = intra(g, qp, z, nullptr, [&](const std::vector<std::uint8_t>& index) {
            auto decoded = codec.decode_y(index, 1);
            if (!decoded) throw std::runtime_error(std::string(decoded.error().message()));
            return std::move(decoded.value());
        });
        check(codec.finish_decode());
        auto shown_frame = show(picture, g, timestamp);
        decoder_.clear();
        decoder_.frame = std::move(picture);
        return shown_frame;
    }

    // ---- HT-S chunk ---------------------------------------------------------------------------------------------
    // The chunk's memory and context from the reference: the I-frame picture, the exact picture of the last chunk's
    // last frame after a reset, or the memory and the last chunk's feature.
    Buffer adapt(Reference& reference, const Geometry& g) {
        auto& s = *session_;
        auto memory = take(g.feature(ch_feature), false);
        auto context = take(g.feature(ch_feature), false);
        if (reference.memory) {
            check(run(s, stage("p_reference_memory"), {{"memory", &*reference.memory}, {"feature", &*reference.feature}},
                      {{"memory_next", &memory}, {"context", &context}}));
        } else if (reference.reset) {
            check(run(s, stage("p_reset_reference"), {{"feature", &*reference.feature}},
                      {{"memory", &memory}, {"context", &context}}));
        } else {
            if (!reference.frame) throw std::runtime_error("HT-S chunk without a reference");
            check(run(s, stage("p_reference_frame"), {{"frame", &*reference.frame}},
                      {{"memory", &memory}, {"context", &context}}));
        }
        reference.memory = std::move(memory);
        return context;
    }

    // Runs the chunk's normative stages; y_scaled (encoder): y / q_dec. Returns the decoded feature and, for the
    // encoder, the chunk's coder rows.
    template <class Decode>
    std::pair<Buffer, std::vector<std::int16_t>> chunk(Reference& reference, const Buffer& context, const Geometry& g,
                                                       std::uint32_t qp, std::span<const std::int8_t> z_symbols,
                                                       const Buffer* y, Decode&& decode) {
        auto& s = *session_;
        const bool host = y != nullptr;
        auto z = z_fixed(z_symbols, g);
        auto ref = constant(g.latent(1), 0);
        auto qp_t = constant({1, 1, 1, 1}, static_cast<std::int32_t>(qp));
        const auto& par = parity(g);
        auto q_dec = take(g.latent(ch_y), host);
        auto indexes = take(g.latent(ch_y), true);
        auto means = take(g.latent(ch_y), host);
        auto reduced = take(g.latent(ch_y), false);
        check(run(s, stage("p_prior"),
                  {{"z_fixed", &z}, {"memory", &*reference.memory}, {"qp", &qp_t}, {"latent_ref", &ref}},
                  {{"q_dec", &q_dec}, {"indexes", &indexes}, {"means", &means}, {"reduced", &reduced}}));
        const auto index = indexes_of(indexes);
        auto symbols = take(g.latent(ch_y), true);
        std::vector<std::int16_t> rows;
        if (!host) place(index, decode(coded(index)), symbols);
        std::optional<Buffer> y_hat;
        for (int k = 0;; ++k) {
            if (host) {  // step k's symbols, from y / q_dec and the step's means
                auto out = symbols.ints();
                for (std::int64_t c = 0; c < ch_y; ++c) {
                    for (std::int64_t row = 0; row < g.yh; ++row) {
                        for (std::int64_t col = 0; col < g.yw; ++col) {
                            if (!active(k, static_cast<int>(c), row, col)) continue;
                            const auto at = static_cast<std::size_t>((c * g.yh + row) * g.yw + col);
                            const double q = static_cast<double>(q_dec.ints()[at]) / one;
                            out[at] = quantize(half_to_double(y->halves()[at]) / q, means.ints()[at], index[at]);
                        }
                    }
                }
            }
            if (k == 3) break;
            auto next = take(g.latent(ch_y), false);
            auto next_means = take(g.latent(ch_y), host);
            std::vector<Bind> in{{"symbols", &symbols}, {"symbol_means", &means}, {"reduced", &reduced}, {"parity", &par}};
            if (y_hat) in.push_back({"y_hat", &*y_hat});
            check(run(s, stage("p_spatial_prior_" + std::to_string(k + 1)), in,
                      {{"y_hat_next", &next}, {"means", &next_means}}));
            y_hat = std::move(next);
            means = std::move(next_means);
        }
        auto feature = take(g.feature(ch_feature), false);
        check(run(s, stage("p_synthesis"),
                  {{"y_hat", &*y_hat}, {"symbols", &symbols}, {"symbol_means", &means}, {"q_dec", &q_dec},
                   {"context", &context}, {"qp", &qp_t}, {"parity", &par}},
                  {{"feature", &feature}}));
        if (host) {  // every coded element of the chunk, in tensor order
            for (std::size_t i = 0; i < index.size(); ++i) row(rows, symbols.ints()[i], index[i]);
        }
        return {std::move(feature), std::move(rows)};
    }

    static void advance(Reference& reference, Buffer feature, bool reset) {
        reference.frame.reset();
        reference.feature = std::move(feature);
        reference.reset = reset;
        if (reset) reference.memory.reset();
    }

    Result<Encoded> encode_p(std::span<const Frame> frames, std::uint32_t qp, bool reset) {
        const auto g = Geometry::of(frames.front().width(), frames.front().height());
        auto context = adapt(encoder_, g);
        auto [luma, chroma] = planes(frames, g);
        auto q = q_row(video_, "q_encoder", qp);
        auto y = take(g.latent(ch_y), true, api::TensorDataType::float16);
        check(run(*session_, stage("p_encode"),
                  {{"luma", &luma}, {"chroma", &chroma}, {"context", &context}, {"q_encoder", &q}}, {{"y", &y}}));
        const auto z = hyper(y, g, "p_hyper_analysis");
        auto [feature, rows] = chunk(encoder_, context, g, qp, z, &y,
                                     [](auto) -> std::vector<std::int8_t> { return {}; });
        auto encoded = entropy_encode(false, {rows}, z, g, qp);
        advance(encoder_, std::move(feature), reset);
        return encoded;
    }

    Result<std::vector<Frame>> decode_p(std::span<const std::byte> bitstream, std::uint8_t parallel, std::uint32_t qp,
                                        std::uint32_t width, std::uint32_t height, std::span<const Timestamp> timestamps,
                                        bool reset) {
        const auto g = Geometry::of(width, height);
        auto& codec = coder(false, false, parallel);
        const auto z = decode_z(codec, bitstream, g, qp);
        auto context = adapt(decoder_, g);
        auto [feature, rows] = chunk(decoder_, context, g, qp, z, nullptr, [&](const std::vector<std::uint8_t>& index) {
            auto decoded = codec.decode_y(index, 1);
            if (!decoded) throw std::runtime_error(std::string(decoded.error().message()));
            return std::move(decoded.value());
        });
        check(codec.finish_decode());
        auto luma = take({1, static_cast<std::int64_t>(group), g.ph, g.pw}, true, api::TensorDataType::uint8);
        auto chroma = take({1, 2 * static_cast<std::int64_t>(group), g.ph / 2, g.pw / 2}, true, api::TensorDataType::uint8);
        check(run(*session_, stage("p_display"), {{"feature", &feature}}, {{"luma", &luma}, {"chroma", &chroma}}));
        std::vector<Frame> frames;
        for (std::size_t f = 0; f < timestamps.size(); ++f) frames.push_back(shown(luma, chroma, f, g, timestamps[f]));
        advance(decoder_, std::move(feature), reset);
        return frames;
    }

    // IEEE binary16 bit pattern -> double (the quantization tables).
    static double half_to_double(std::uint16_t h) {
        const int exponent = (h >> 10) & 0x1f;
        const int mantissa = h & 0x3ff;
        const double sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
        if (exponent == 0) return sign * std::ldexp(mantissa, -24);
        if (exponent == 31) return mantissa != 0 ? std::nan("") : sign * HUGE_VAL;
        return sign * std::ldexp(mantissa + 1024, exponent - 25);
    }

    std::shared_ptr<api::IProviderSession> session_;
    std::string owner_;
    std::map<std::string, api::ExecutableStage, std::less<>> stages_;
    codec::fx::BufferPool pool_;
    std::optional<Buffer> parity_;
    Tables image_;
    Tables video_;
    std::array<std::array<std::array<std::optional<UfRansCodec>, static_cast<std::size_t>(max_entropy_parallel)>, 2>, 2> coders_;
    Reference encoder_;
    Reference decoder_;
};

}  // namespace

Result<std::shared_ptr<ExecutionSession>> make_fx_session(std::shared_ptr<api::IProviderSession> provider,
                                                          const RuntimeConfiguration& configuration) {
    if (!provider) return fail("the fx session needs a provider session");
    auto session = std::make_shared<FxSession>(std::move(provider));
    auto initialized = session->initialize(configuration);
    if (!initialized) return initialized.error();
    return std::shared_ptr<ExecutionSession>(std::move(session));
}

}  // namespace nvcr::dcvcuf
