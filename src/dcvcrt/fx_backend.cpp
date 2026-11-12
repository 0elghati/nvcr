#include "fx_backend.hpp"

#include "../common/json.hpp"
#include "host_ops.hpp"
#include "numerics.hpp"
#include "orchestration.hpp"

#include "nvcr/numerics/profiles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// DCVC-RT under nvcr.fx.v1. Every stage that decides an entropy table or the
// reference runs as an fx program on the provider (any provider that claims the
// profile), including the spatial steps: a stage returns a step's entropy
// indexes and means and takes its symbols back. The encoder's analysis
// transforms and the P-frame display picture, which nothing reads back, stay
// FP16. Entropy coding, payloads, QP schedule and reference decisions are
// DCVC-RT's usual orchestration, so streams keep the NVI1/NVP1 syntax.

namespace nvcr::dcvcrt {
namespace {

namespace api = provider::experimental;
using host_ops::Half;
using codec::CodecDecodeResult;
using codec::CodecEncodeResult;
using codec::SequenceStateView;

constexpr std::string_view subsystem = "dcvcrt.fx";
constexpr int fa = 14;
constexpr std::int64_t one = std::int64_t{1} << fa;

Error fail(std::string message) { return Error(ErrorCode::backend_error, std::move(message), std::string(subsystem)); }

std::int32_t round_up(std::int32_t value, std::int32_t multiple) { return (value + multiple - 1) / multiple * multiple; }

// ---- provider buffers ----------------------------------------------------------------------------------------------
struct Buffer final {
    api::BufferHandle handle;
    std::array<std::int64_t, 4> shape{};
    api::TensorDataType type{api::TensorDataType::int32};
    bool host{false};

    [[nodiscard]] std::size_t count() const {
        return static_cast<std::size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
    }
    [[nodiscard]] std::size_t bytes() const {
        return count() * (type == api::TensorDataType::int32 ? 4U : type == api::TensorDataType::float16 ? 2U : 1U);
    }
    [[nodiscard]] std::span<std::int32_t> ints() const {
        return {reinterpret_cast<std::int32_t*>(handle->host_bytes().data()), count()};
    }
    [[nodiscard]] std::span<Half> halves() const {
        return {reinterpret_cast<Half*>(handle->host_bytes().data()), count()};
    }
};

// host: the codec reads or writes the values; otherwise device memory is preferred.
Result<Buffer> allocate(api::IProviderSession& session, std::array<std::int64_t, 4> shape, bool host,
                        api::TensorDataType type = api::TensorDataType::int32) {
    Buffer buffer{nullptr, shape, type, host};
    const std::array<api::MemoryDomain, 3> domains{api::MemoryDomain::provider_device, api::MemoryDomain::pinned_host,
                                                   api::MemoryDomain::host};
    for (std::size_t i = host ? 1 : 0; i < domains.size(); ++i) {
        auto allocated = session.allocate(buffer.bytes(), domains[i]);
        if (allocated && (!host || !allocated.value()->host_bytes().empty())) {
            buffer.handle = std::move(allocated.value());
            return buffer;
        }
    }
    return fail("cannot allocate a provider buffer");
}

struct Bind final {
    std::string_view name;
    const Buffer* buffer;
};

Result<void> run(api::IProviderSession& session, const api::ExecutableStage& stage, const std::vector<Bind>& in,
                 const std::vector<Bind>& out) {
    std::vector<api::TensorView> views;
    for (const auto* list : {&in, &out}) {
        for (const auto& b : *list) {
            views.push_back({b.name, b.buffer->type, {b.buffer->shape.begin(), b.buffer->shape.end()},
                             list == &in ? api::TensorAccess::read : api::TensorAccess::write,
                             {b.buffer->handle, 0, b.buffer->bytes()}});
        }
    }
    auto completion = session.submit(stage, views, {});
    if (!completion) return completion.error();
    return completion.value()->wait();
}

// ---- codec rules ---------------------------------------------------------------------------------------------------
// Upstream DCVC-RT's spatial masks: in group g of the channels, step k codes the
// positions whose (row % 2, column % 2) equals pattern[k][g] (index 2 row + column).
constexpr std::array<std::array<int, 4>, 4> four_way{{{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2}}};
constexpr std::array<std::array<int, 2>, 2> two_way{{{0, 1}, {1, 0}}};  // 0: row == column parity

// One spatial step: for each position the single active channel of each group.
struct Step final {
    int groups;  // 4 (I-frame) or 2 (P-frame)
    int k;

    [[nodiscard]] bool active(int g, int row, int col) const {
        if (groups == 4) return four_way[static_cast<std::size_t>(k)][static_cast<std::size_t>(g)] == 2 * (row % 2) + col % 2;
        return two_way[static_cast<std::size_t>(k)][static_cast<std::size_t>(g)] == ((row % 2) != (col % 2) ? 1 : 0);
    }
};

// Visits the active element of every reduced position: f(reduced_index, full_index).
template <typename F>
void each_active(const Step& step, int channels, int height, int width, F&& f) {
    const int per_group = channels / step.groups;
    std::size_t reduced = 0;
    for (int c = 0; c < per_group; ++c) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col, ++reduced) {
                for (int g = 0; g < step.groups; ++g) {
                    if (!step.active(g, row, col)) continue;
                    f(reduced, (static_cast<std::size_t>(g * per_group + c) * static_cast<std::size_t>(height) +
                                static_cast<std::size_t>(row)) * static_cast<std::size_t>(width) +
                                   static_cast<std::size_t>(col));
                }
            }
        }
    }
}

struct Geometry final {
    std::int32_t width, height, padded_height, padded_width, yh, yw, yh4, yw4;

    static Geometry of(std::uint32_t width, std::uint32_t height) {
        Geometry g{};
        g.width = static_cast<std::int32_t>(width);
        g.height = static_cast<std::int32_t>(height);
        g.padded_height = round_up(g.height, 16);
        g.padded_width = round_up(g.width, 16);
        g.yh = g.padded_height / 16;
        g.yw = g.padded_width / 16;
        g.yh4 = round_up(g.yh, 4);
        g.yw4 = round_up(g.yw, 4);
        return g;
    }
    [[nodiscard]] std::size_t z_per_channel() const {
        return static_cast<std::size_t>(yh4 / 4) * static_cast<std::size_t>(yw4 / 4);
    }
};

struct Reference final {
    std::optional<Buffer> frame;    // exact picture (int32, 2^-fa)
    std::optional<Buffer> feature;  // exact reference feature
    std::uint32_t qp{};             // the feature's QP
    ReferenceState semantics;

    void clear() {
        frame.reset();
        feature.reset();
        semantics.reset();
    }
};

class FxBackend final : public codec::CodecBackend {
public:
    explicit FxBackend(std::shared_ptr<api::IProviderSession> session) : session_(std::move(session)) {}

    Result<void> initialize(const RuntimeConfiguration& configuration) override {
        try {
            auto normative = normative_stages(configuration);
            if (!normative) return normative.error();
            if (!normative.value()) return fail("the fx backend needs the nvcr.fx.v1 profile");
            for (auto descriptor : normative.value()->descriptors) {
                const auto id = descriptor.stage_id;
                auto loaded = session_->load_stage(std::move(descriptor));
                if (!loaded) return loaded.error();
                stages_[id] = std::move(loaded.value());
            }
            const auto& root = configuration.artifacts.intra_engine_path;
            using Dims = std::vector<api::DimensionBounds>;
            const auto any = api::DimensionBounds{1, 65535};
            const auto half = api::TensorDataType::float16;
            const auto read = api::TensorAccess::read;
            const auto write = api::TensorAccess::write;
            const std::vector<std::pair<std::string, std::vector<api::TensorContract>>> analysis{
                {"i_analysis", {{"frame", half, read, Dims{{1, 1}, {3, 3}, any, any}},
                                {"q_enc", half, read, Dims{{1, 1}, {368, 368}, {1, 1}, {1, 1}}},
                                {"y", half, write, Dims{{1, 1}, {256, 256}, any, any}}}},
                {"i_hyper_analysis", {{"y_padded", half, read, Dims{{1, 1}, {256, 256}, any, any}},
                                      {"z", half, write, Dims{{1, 1}, {128, 128}, any, any}}}},
                {"p_encode", {{"luma", api::TensorDataType::uint8, read, Dims{{1, 1}, {1, 1}, any, any}},
                              {"chroma", api::TensorDataType::uint8, read, Dims{{1, 1}, {2, 2}, any, any}},
                              {"context", api::TensorDataType::int32, read, Dims{{1, 1}, {256, 256}, any, any}},
                              {"q_encoder", half, read, Dims{{1, 1}, {256, 256}, {1, 1}, {1, 1}}},
                              {"y", half, write, Dims{{1, 1}, {128, 128}, any, any}}}},
                {"p_hyper_analysis", {{"y_padded", half, read, Dims{{1, 1}, {128, 128}, any, any}},
                                      {"z", half, write, Dims{{1, 1}, {128, 128}, any, any}}}},
                {"p_display", {{"feature", api::TensorDataType::int32, read, Dims{{1, 1}, {256, 256}, any, any}},
                               {"q_recon", half, read, Dims{{1, 1}, {320, 320}, {1, 1}, {1, 1}}},
                               {"luma", api::TensorDataType::uint8, write, Dims{{1, 1}, {1, 1}, any, any}},
                               {"chroma", api::TensorDataType::uint8, write, Dims{{1, 1}, {2, 2}, any, any}}}}};
            for (const auto& [component, tensors] : analysis) {
                api::ExecutableStageDescriptor descriptor;
                descriptor.stage_id = component + ".plan";
                descriptor.artifact.codec_id = "dcvc-rt";
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
            auto directory = configuration.artifacts.normative_program_path;
            if (directory.empty()) directory = root / normative_program_subdirectory;
            auto rules = detail::json::parse_file(directory / fx_entropy_file, subsystem);
            if (!rules) return rules.error();
            const auto* threshold = rules.value().find("skip_threshold");
            if (threshold == nullptr || threshold->integer_value.value_or(0) >= 0) {
                return fail("fx_entropy.json must have no skip threshold");
            }
            auto intra = intra_.initialize(root);
            if (!intra) return intra.error();
            auto predicted = predicted_.initialize(root);
            if (!predicted) return predicted.error();
            qp_ = configuration.codec.intra_qp;
            verify_ = configuration.codec.verify_encoder_reconstruction;
            return {};
        } catch (const std::exception& exception) {
            return fail(std::string("fx backend initialization failed: ") + exception.what());
        }
    }

    Result<CodecEncodeResult> encode(const Frame& frame, FrameType type, const SequenceStateView& state) override {
        try {
            return type == FrameType::intra ? encode_intra(frame, state) : encode_predicted(frame, state);
        } catch (const std::exception& exception) {
            return fail(std::string("fx encoding failed: ") + exception.what());
        }
    }

    Result<CodecDecodeResult> decode(std::span<const std::byte> payload, FrameType type, Timestamp timestamp,
                                     const SequenceStateView& state) override {
        try {
            return type == FrameType::intra ? decode_intra(payload, timestamp, state)
                                            : decode_predicted(payload, timestamp, state);
        } catch (const std::exception& exception) {
            return fail(std::string("fx decoding failed: ") + exception.what());
        }
    }

    Result<void> flush() override { return session_->reset(); }
    void reset() noexcept override {
        encoder_.clear();
        decoder_.clear();
    }
    void reset_encoder() noexcept override { encoder_.clear(); }
    void reset_decoder() noexcept override { decoder_.clear(); }

private:
    const api::ExecutableStage& stage(const std::string& id) const { return stages_.at(id); }

    // Storage is reused across frames: a pooled buffer is free when only the pool holds it.
    Result<Buffer> take(std::array<std::int64_t, 4> shape, bool host,
                        api::TensorDataType type = api::TensorDataType::int32) {
        Buffer wanted{nullptr, shape, type, host};
        for (const auto& pooled : pool_) {
            if (pooled.handle.use_count() == 1 && pooled.host == host && pooled.bytes() == wanted.bytes()) {
                wanted.handle = pooled.handle;
                return wanted;
            }
        }
        auto made = allocate(*session_, shape, host, type);
        if (made) pool_.push_back(made.value());
        return made;
    }

    // Full-tensor offsets of a spatial step's active elements, in the entropy coder's order.
    const std::vector<std::uint32_t>& active(const Step& step, int channels, const Geometry& g) {
        auto& offsets = active_[{step.groups, step.k, g.yh, g.yw}];
        if (offsets.empty()) {
            offsets.resize(static_cast<std::size_t>(channels / step.groups) * static_cast<std::size_t>(g.yh) *
                           static_cast<std::size_t>(g.yw));
            each_active(step, channels, g.yh, g.yw,
                        [&](std::size_t r, std::size_t i) { offsets[r] = static_cast<std::uint32_t>(i); });
        }
        return offsets;
    }

    // Encoder: y (scaled by the caller) and z symbols from the analysis stages; P frames read the exact context.
    Result<std::pair<Buffer, std::vector<std::int8_t>>> analyse(const Frame& frame, const Geometry& g, bool intra,
                                                                 std::uint32_t qp, const Buffer* context) {
        auto& s = *session_;
        const std::int64_t channels = intra ? 256 : 128;
        const std::int64_t quant = intra ? 368 : 256;
        auto q = take({1, quant, 1, 1}, true, api::TensorDataType::float16);
        if (!q) return q.error();
        const auto table = intra ? intra_.encoder_quantization() : predicted_.encoder_quantization();
        for (std::int64_t c = 0; c < quant; ++c) {
            q.value().halves()[static_cast<std::size_t>(c)] =
                host_ops::to_half(table[static_cast<std::size_t>(static_cast<std::int64_t>(qp) * quant + c)]);
        }
        auto y = take({1, channels, g.yh, g.yw}, true, api::TensorDataType::float16);
        if (!y) return y.error();
        const auto bytes = frame.data();
        const std::span<const std::uint8_t> yuv{reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
        Result<void> ran{};
        if (intra) {
            auto picture = take({1, 3, g.padded_height, g.padded_width}, true, api::TensorDataType::float16);
            if (!picture) return picture.error();
            if (!host_ops::yuv420p8_to_ycbcr_padded(yuv, g.width, g.height, picture.value().halves(), g.padded_height,
                                                    g.padded_width)) {
                return fail("input color conversion");
            }
            ran = run(s, stage("i_analysis"), {{"frame", &picture.value()}, {"q_enc", &q.value()}}, {{"y", &y.value()}});
        } else {
            const auto u8 = api::TensorDataType::uint8;
            auto luma = take({1, 1, g.padded_height, g.padded_width}, true, u8);
            auto chroma = take({1, 2, g.padded_height / 2, g.padded_width / 2}, true, u8);
            if (!luma || !chroma) return fail("cannot allocate the encoder picture");
            pad_planes(yuv, g, luma.value(), chroma.value());
            ran = run(s, stage("p_encode"),
                      {{"luma", &luma.value()}, {"chroma", &chroma.value()}, {"context", context},
                       {"q_encoder", &q.value()}},
                      {{"y", &y.value()}});
        }
        if (!ran) return ran.error();
        auto padded = take({1, channels, g.yh4, g.yw4}, true, api::TensorDataType::float16);
        auto z = take({1, 128, g.yh4 / 4, g.yw4 / 4}, true, api::TensorDataType::float16);
        if (!padded) return padded.error();
        if (!z) return z.error();
        if (!host_ops::replicate_pad(y.value().halves(), padded.value().halves(),
                                     {1, static_cast<std::int32_t>(channels), g.yh, g.yw}, g.yh4, g.yw4)) {
            return fail("replicate_pad");
        }
        ran = run(s, stage(intra ? "i_hyper_analysis" : "p_hyper_analysis"), {{"y_padded", &padded.value()}},
                  {{"z", &z.value()}});
        if (!ran) return ran.error();
        std::vector<std::int8_t> symbols(z.value().count());
        if (!host_ops::round_to_int8(z.value().halves(), symbols)) return fail("round_to_int8");
        return std::pair{std::move(y.value()), std::move(symbols)};
    }

    // The 8-bit picture at the padded size with its edges replicated: luma, and both chroma planes at half size.
    static void pad_planes(std::span<const std::uint8_t> yuv, const Geometry& g, const Buffer& luma, const Buffer& chroma) {
        const auto pad = [](const std::uint8_t* in, std::size_t width, std::size_t height, std::byte* out,
                            std::size_t padded_width, std::size_t padded_height) {
            for (std::size_t row = 0; row < padded_height; ++row, out += padded_width) {
                const auto* line = reinterpret_cast<const std::byte*>(in + std::min(row, height - 1) * width);
                std::copy_n(line, width, out);
                std::fill(out + width, out + padded_width, line[width - 1]);
            }
        };
        const auto w = static_cast<std::size_t>(g.width), h = static_cast<std::size_t>(g.height);
        const auto pw = static_cast<std::size_t>(g.padded_width), ph = static_cast<std::size_t>(g.padded_height);
        pad(yuv.data(), w, h, luma.handle->host_bytes().data(), pw, ph);
        auto* c = chroma.handle->host_bytes().data();
        pad(yuv.data() + w * h, w / 2, h / 2, c, pw / 2, ph / 2);
        pad(yuv.data() + w * h + w * h / 4, w / 2, h / 2, c + pw / 2 * (ph / 2), pw / 2, ph / 2);
    }

    Result<Buffer> z_fixed(std::span<const std::int8_t> symbols, const Geometry& g) {
        auto z = take({1, 128, g.yh4 / 4, g.yw4 / 4}, true);
        if (!z) return z.error();
        for (std::size_t i = 0; i < symbols.size(); ++i) z.value().ints()[i] = symbols[i] * static_cast<std::int32_t>(one);
        return z;
    }

    // The spatial steps' parity masks: channel 2 (row % 2) + column % 2 is 1 there, 0 elsewhere.
    Result<Buffer> parity(const Geometry& g) {
        if (!parity_ || parity_->shape[2] != g.yh || parity_->shape[3] != g.yw) {
            auto made = allocate(*session_, {1, 4, g.yh, g.yw}, true);
            if (!made) return made.error();
            auto v = made.value().ints();
            const auto plane = static_cast<std::size_t>(g.yh) * static_cast<std::size_t>(g.yw);
            for (std::size_t i = 0; i < v.size(); ++i) {
                const auto at = i % plane;
                const auto row = at / static_cast<std::size_t>(g.yw);
                const auto col = at % static_cast<std::size_t>(g.yw);
                v[i] = 2 * (row % 2) + col % 2 == i / plane ? 1 : 0;
            }
            parity_ = std::move(made.value());
        }
        return *parity_;
    }

    // The coder's part of spatial step k: symbols for the stage's indexes, decoded, or chosen by the encoder as
    // round(y_scaled - mean) at the step's active elements (and recorded as packed symbol/index rows).
    template <typename Decode>
    Result<void> code(const Step& step, const Geometry& g, int channels, const Buffer& indexes, const Buffer& means,
                      const Buffer& symbols, const std::vector<double>* y_scaled,
                      std::vector<std::vector<std::int16_t>>* rows, Decode&& decode) {
        const auto count = indexes.count();
        const auto values = indexes.ints();
        std::vector<std::uint8_t> index(count);
        std::transform(values.begin(), values.end(), index.begin(), [](std::int32_t i) { return static_cast<std::uint8_t>(i); });
        auto out = symbols.ints();
        if (y_scaled == nullptr) {
            auto decoded = decode(index);
            if (!decoded) return decoded.error();
            if (decoded.value().size() != count) return fail("decoded symbol count");
            std::copy(decoded.value().begin(), decoded.value().end(), out.begin());
            return {};
        }
        const auto& offsets = active(step, channels, g);
        std::vector<std::int16_t> row(count);
        for (std::size_t r = 0; r < count; ++r) {
            const double v = std::nearbyint((*y_scaled)[offsets[r]] - static_cast<double>(means.ints()[r]) / static_cast<double>(one));
            out[r] = static_cast<std::int32_t>(std::clamp(v, -128.0, 127.0));
            row[r] = static_cast<std::int16_t>(static_cast<std::uint16_t>(static_cast<std::uint8_t>(out[r])) << 8U | index[r]);
        }
        rows->push_back(std::move(row));
        return {};
    }

    // YUV 4:2:0 8-bit from the exact picture: luma round(255 v), chroma floor(255 mean of 2x2), clamped to [0, 255].
    Result<Frame> display(const Buffer& picture, const Geometry& g, Timestamp timestamp) {
        const auto v = picture.ints();
        const auto stride = static_cast<std::size_t>(g.padded_width);
        const auto plane = static_cast<std::size_t>(g.padded_height) * stride;
        const auto width = static_cast<std::size_t>(g.width);
        const auto height = static_cast<std::size_t>(g.height);
        std::vector<std::byte> bytes(width * height * 3 / 2);
        auto* out = reinterpret_cast<std::uint8_t*>(bytes.data());
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::int64_t p = std::clamp<std::int64_t>(v[y * stride + x], 0, one);
                *out++ = static_cast<std::uint8_t>((p * 255 + one / 2) >> fa);
            }
        }
        for (std::size_t c = 1; c < 3; ++c) {
            for (std::size_t y = 0; y < height; y += 2) {
                for (std::size_t x = 0; x < width; x += 2) {
                    const auto* q = &v[c * plane + y * stride + x];
                    const std::int64_t sum = std::int64_t{q[0]} + q[1] + q[stride] + q[stride + 1];
                    *out++ = static_cast<std::uint8_t>(std::clamp<std::int64_t>((sum * 255) >> (fa + 2), 0, 255));
                }
            }
        }
        return Frame::copy_from(static_cast<std::uint32_t>(g.width), static_cast<std::uint32_t>(g.height),
                                PixelFormat::yuv420p8, bytes, timestamp);
    }

    // A P picture p_display already converted: the visible part of its padded 8-bit planes.
    Result<Frame> display(const Buffer& luma, const Buffer& chroma, const Geometry& g, Timestamp timestamp) {
        const auto width = static_cast<std::size_t>(g.width);
        const auto height = static_cast<std::size_t>(g.height);
        const auto stride = static_cast<std::size_t>(g.padded_width);
        const auto plane = static_cast<std::size_t>(g.padded_height) / 2 * (stride / 2);
        std::vector<std::byte> bytes(width * height * 3 / 2);
        auto* out = bytes.data();
        const auto* y = luma.handle->host_bytes().data();
        for (std::size_t row = 0; row < height; ++row, out += width) std::copy_n(y + row * stride, width, out);
        const auto* c = chroma.handle->host_bytes().data();
        for (std::size_t part = 0; part < 2; ++part) {
            for (std::size_t row = 0; row < height / 2; ++row, out += width / 2) {
                std::copy_n(c + part * plane + row * (stride / 2), width / 2, out);
            }
        }
        return Frame::copy_from(static_cast<std::uint32_t>(g.width), static_cast<std::uint32_t>(g.height),
                                PixelFormat::yuv420p8, bytes, timestamp);
    }

    // Only its size matters (stages crop to it); zeros keep it inside the profile's range.
    Result<Buffer> latent_ref(const Geometry& g) {
        auto t = take({1, 1, g.yh, g.yw}, true);
        if (t) std::fill(t.value().ints().begin(), t.value().ints().end(), 0);
        return t;
    }

    Result<Buffer> qp_tensor(std::uint32_t qp) {
        auto t = take({1, 1, 1, 1}, true);
        if (t) t.value().ints()[0] = static_cast<std::int32_t>(qp);
        return t;
    }

    // ---- I-frames ----
    struct IntraResult final {
        Buffer picture;
        std::vector<std::vector<std::int16_t>> rows;
    };

    template <typename Decode>
    Result<IntraResult> intra(const Geometry& g, std::uint32_t qp, std::span<const std::int8_t> z_symbols,
                              const Buffer* y, Decode&& decode) {
        auto& s = *session_;
        const bool host = y != nullptr;  // the encoder reads q and the means
        const std::array<std::int64_t, 4> reduced{1, 64, g.yh, g.yw};
        auto z = z_fixed(z_symbols, g);
        auto ref = latent_ref(g);
        auto par = parity(g);
        auto params = take({1, 514, g.yh, g.yw}, false);
        auto common = take({1, 256, g.yh, g.yw}, false);
        auto indexes = take(reduced, true);
        auto means = take(reduced, host);
        auto q = take({1, 1, g.yh, g.yw}, host);
        if (!z || !ref || !par || !params || !common || !indexes || !means || !q) {
            return fail("cannot allocate I-frame tensors");
        }
        auto ran = run(s, stage("i_prior"), {{"z_fixed", &z.value()}, {"latent_ref", &ref.value()}, {"parity", &par.value()}},
                       {{"params", &params.value()}, {"common", &common.value()}, {"indexes", &indexes.value()},
                        {"means", &means.value()}, {"q", &q.value()}});
        if (!ran) return ran.error();
        IntraResult result;
        const std::vector<double>* y_scaled = nullptr;
        if (y != nullptr) {  // y q_enc, q_enc = 1.5 sigmoid(q) + 0.5 (encoder only)
            const auto plane = static_cast<std::size_t>(g.yh) * static_cast<std::size_t>(g.yw);
            std::vector<double> factor(plane);
            for (std::size_t i = 0; i < plane; ++i) {
                factor[i] = 1.5 / (1.0 + std::exp(-static_cast<double>(q.value().ints()[i]) / static_cast<double>(one))) + 0.5;
            }
            scaled_.resize(y->count());
            const auto values = y->halves();
            for (std::size_t i = 0; i < values.size(); ++i) {
                scaled_[i] = static_cast<double>(host_ops::to_float(values[i])) * factor[i % plane];
            }
            y_scaled = &scaled_;
        }
        std::optional<Buffer> y_hat;
        auto symbols = take(reduced, true);
        if (!symbols) return symbols.error();
        for (int k = 0;; ++k) {
            auto coded = code(Step{4, k}, g, 256, indexes.value(), means.value(), symbols.value(), y_scaled,
                              &result.rows, decode);
            if (!coded) return coded.error();
            if (k == 3) break;
            auto next = take({1, 256, g.yh, g.yw}, false);
            auto next_indexes = take(reduced, true);
            auto next_means = take(reduced, host);
            if (!next || !next_indexes || !next_means) return fail("cannot allocate I-frame tensors");
            std::vector<Bind> in{{"symbols", &symbols.value()}, {"symbol_means", &means.value()},
                                 {"common", &common.value()}, {"parity", &par.value()}};
            if (y_hat) in.push_back({"y_hat", &*y_hat});
            ran = run(s, stage("i_spatial_prior_" + std::to_string(k + 1)), in,
                      {{"y_hat_next", &next.value()}, {"indexes", &next_indexes.value()}, {"means", &next_means.value()}});
            if (!ran) return ran.error();
            y_hat = std::move(next.value());
            indexes = std::move(next_indexes);
            means = std::move(next_means);
            symbols = take(reduced, true);
            if (!symbols) return symbols.error();
        }
        auto picture = take({1, 3, g.padded_height, g.padded_width}, true);
        auto qp_t = qp_tensor(qp);
        if (!picture || !qp_t) return fail("cannot allocate the I-frame picture");
        ran = run(s, stage("i_synthesis"),
                  {{"y_hat", &*y_hat}, {"symbols", &symbols.value()}, {"symbol_means", &means.value()},
                   {"params", &params.value()}, {"qp", &qp_t.value()}, {"parity", &par.value()}},
                  {{"frame", &picture.value()}});
        if (!ran) return ran.error();
        result.picture = std::move(picture.value());
        return result;
    }

    Result<CodecEncodeResult> encode_intra(const Frame& frame, const SequenceStateView& state) {
        const auto g = Geometry::of(frame.width(), frame.height());
        auto analysed = analyse(frame, g, true, qp_, nullptr);
        if (!analysed) return analysed.error();
        auto& [y, z_symbols] = analysed.value();
        auto coded = intra(g, qp_, z_symbols, &y, [](auto&) -> Result<std::vector<std::int8_t>> { return fail("decode"); });
        if (!coded) return coded.error();
        const bool two_coders = static_cast<std::uint64_t>(g.width) * static_cast<std::uint64_t>(g.height) >= 1280ULL * 720ULL;
        auto begun = intra_.begin_encode(two_coders);
        if (!begun) return begun.error();
        auto coded_z = intra_.encode_z(z_symbols, qp_, g.z_per_channel());
        if (!coded_z) return coded_z.error();
        for (const auto& row : coded.value().rows) {
            auto coded_y = intra_.encode_y(row);
            if (!coded_y) return coded_y.error();
        }
        auto payload = intra_.finish_encode(frame.width(), frame.height(), qp_, two_coders);
        if (!payload) return payload.error();
        auto shown = display(coded.value().picture, g, frame.timestamp());
        if (!shown) return shown.error();
        encoder_.clear();
        encoder_.frame = std::move(coded.value().picture);
        encoder_.semantics.commit(state, false);
        return CodecEncodeResult{std::move(payload.value()), std::move(shown.value()), {}, qp_};
    }

    Result<CodecDecodeResult> decode_intra(std::span<const std::byte> payload, Timestamp timestamp,
                                           const SequenceStateView& state) {
        auto parsed = intra_.parse_payload(payload);
        if (!parsed) return parsed.error();
        const auto info = parsed.value();
        const auto g = Geometry::of(info.width, info.height);
        auto begun = intra_.begin_decode(info);
        if (!begun) return begun.error();
        auto z = intra_.decode_z(128 * g.z_per_channel(), info.qp, g.z_per_channel());
        if (!z) return z.error();
        auto decoded = intra(g, info.qp, z.value(), nullptr,
                             [&](std::span<const std::uint8_t> indexes) { return intra_.decode_y(indexes); });
        if (!decoded) return decoded.error();
        auto ended = intra_.end_decode();
        if (!ended) return ended.error();
        auto shown = display(decoded.value().picture, g, timestamp);
        if (!shown) return shown.error();
        decoder_.clear();
        decoder_.frame = std::move(decoded.value().picture);
        decoder_.semantics.commit(state, false);
        return CodecDecodeResult{std::move(shown.value()), {}};
    }

    // ---- P-frames ----
    struct Context final {
        Buffer temporal;
        Buffer context;
    };

    Result<Context> context(Reference& reference, bool use_frame, std::uint32_t qp, const Geometry& g) {
        auto& s = *session_;
        auto temporal = take({1, 256, g.padded_height / 8, g.padded_width / 8}, false);
        auto context = take({1, 256, g.padded_height / 8, g.padded_width / 8}, false);
        auto qp_t = qp_tensor(qp);
        if (!temporal || !context || !qp_t) return fail("cannot allocate the P-frame context");
        if (use_frame && !reference.frame && reference.feature) {  // a reset: the exact picture of the feature
            auto frame = take({1, 3, g.padded_height, g.padded_width}, false);
            auto last_qp = qp_tensor(reference.qp);
            if (!frame || !last_qp) return fail("cannot allocate the reset picture");
            auto reset = run(s, stage("p_reset_frame"), {{"feature", &*reference.feature}, {"qp", &last_qp.value()}},
                             {{"frame", &frame.value()}});
            if (!reset) return reset.error();
            reference.frame = std::move(frame.value());
        }
        if (use_frame && !reference.frame) return fail("no reference picture");
        if (!use_frame && !reference.feature) return fail("no reference feature");
        auto ran = use_frame ? run(s, stage("p_reference_frame"), {{"frame", &*reference.frame}, {"qp", &qp_t.value()}},
                                   {{"temporal_context", &temporal.value()}, {"context", &context.value()}})
                             : run(s, stage("p_reference_feature"), {{"feature", &*reference.feature}, {"qp", &qp_t.value()}},
                                   {{"temporal_context", &temporal.value()}, {"context", &context.value()}});
        if (!ran) return ran.error();
        return Context{std::move(temporal.value()), std::move(context.value())};
    }

    struct PredictedResult final {
        Buffer luma;  // the shown picture, 8-bit 4:2:0 at the padded size
        Buffer chroma;
        Buffer feature;
        std::vector<std::vector<std::int16_t>> rows;
    };

    template <typename Decode>
    // show: also the P picture (the decoder; the encoder when it verifies its reconstruction).
    Result<PredictedResult> predicted(const Geometry& g, std::uint32_t qp, const Context& ctx,
                                      std::span<const std::int8_t> z_symbols, const Buffer* y, bool show,
                                      Decode&& decode) {
        auto& s = *session_;
        const bool host = y != nullptr;  // the encoder reads q and the means
        const std::array<std::int64_t, 4> reduced{1, 64, g.yh, g.yw};
        auto z = z_fixed(z_symbols, g);
        auto ref = latent_ref(g);
        auto par = parity(g);
        auto params = take({1, 384, g.yh, g.yw}, false);
        auto indexes = take(reduced, true);
        auto means = take(reduced, host);
        auto q_dec = take({1, 128, g.yh, g.yw}, host);
        auto symbols = take(reduced, true);
        if (!z || !ref || !par || !params || !indexes || !means || !q_dec || !symbols) {
            return fail("cannot allocate P-frame tensors");
        }
        auto ran = run(s, stage("p_prior"),
                       {{"z_fixed", &z.value()}, {"temporal_context", &ctx.temporal}, {"latent_ref", &ref.value()},
                        {"parity", &par.value()}},
                       {{"params", &params.value()}, {"indexes", &indexes.value()}, {"means", &means.value()},
                        {"q", &q_dec.value()}});
        if (!ran) return ran.error();
        PredictedResult result;
        const std::vector<double>* scaled = nullptr;
        if (y != nullptr) {  // y / max(q_dec, 0.5) (encoder only)
            scaled_.resize(y->count());
            const auto values = y->halves();
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double q = std::max(static_cast<double>(q_dec.value().ints()[i]) / static_cast<double>(one), 0.5);
                scaled_[i] = static_cast<double>(host_ops::to_float(values[i])) / q;
            }
            scaled = &scaled_;
        }
        auto coded = code(Step{2, 0}, g, 128, indexes.value(), means.value(), symbols.value(), scaled, &result.rows, decode);
        if (!coded) return coded.error();
        auto y_hat = take({1, 128, g.yh, g.yw}, false);
        auto indexes_1 = take(reduced, true);
        auto means_1 = take(reduced, host);
        auto symbols_1 = take(reduced, true);
        if (!y_hat || !indexes_1 || !means_1 || !symbols_1) return fail("cannot allocate P-frame tensors");
        ran = run(s, stage("p_spatial_prior"),
                  {{"symbols", &symbols.value()}, {"symbol_means", &means.value()}, {"params", &params.value()},
                   {"parity", &par.value()}},
                  {{"y_hat_next", &y_hat.value()}, {"indexes", &indexes_1.value()}, {"means", &means_1.value()}});
        if (!ran) return ran.error();
        coded = code(Step{2, 1}, g, 128, indexes_1.value(), means_1.value(), symbols_1.value(), scaled, &result.rows,
                     decode);
        if (!coded) return coded.error();
        auto feature = take({1, 256, g.padded_height / 8, g.padded_width / 8}, false);
        const auto u8 = api::TensorDataType::uint8;
        auto luma = take({1, 1, g.padded_height, g.padded_width}, true, u8);
        auto chroma = take({1, 2, g.padded_height / 2, g.padded_width / 2}, true, u8);
        auto qp_t = qp_tensor(qp);
        if (!feature || !luma || !chroma || !qp_t) return fail("cannot allocate the P-frame outputs");
        ran = run(s, stage("p_synthesis"),
                  {{"y_hat", &y_hat.value()}, {"symbols", &symbols_1.value()}, {"symbol_means", &means_1.value()},
                   {"params", &params.value()}, {"context", &ctx.context}, {"qp", &qp_t.value()}, {"parity", &par.value()}},
                  {{"feature", &feature.value()}});
        if (!ran) return ran.error();
        result.feature = std::move(feature.value());
        if (!show) return result;
        auto q = take({1, 320, 1, 1}, true, api::TensorDataType::float16);
        if (!q) return q.error();
        const auto table = predicted_.reconstruction_quantization();
        for (std::size_t c = 0; c < 320; ++c) q.value().halves()[c] = host_ops::to_half(table[qp * 320 + c]);
        ran = run(s, stage("p_display"), {{"feature", &result.feature}, {"q_recon", &q.value()}},
                  {{"luma", &luma.value()}, {"chroma", &chroma.value()}});
        if (!ran) return ran.error();
        result.luma = std::move(luma.value());
        result.chroma = std::move(chroma.value());
        return result;
    }

    Result<CodecEncodeResult> encode_predicted(const Frame& frame, const SequenceStateView& state) {
        const auto g = Geometry::of(frame.width(), frame.height());
        auto decision = predicted_.select_frame(qp_, state, encoder_.semantics);
        if (!decision) return decision.error();
        const auto qp = decision.value().qp;
        auto ctx = context(encoder_, decision.value().use_frame_reference, qp, g);
        if (!ctx) return ctx.error();
        auto analysed = analyse(frame, g, false, qp, &ctx.value().context);
        if (!analysed) return analysed.error();
        auto& [y, z_symbols] = analysed.value();
        auto coded = predicted(g, qp, ctx.value(), z_symbols, &y, verify_,
                               [](auto&) -> Result<std::vector<std::int8_t>> { return fail("decode"); });
        if (!coded) return coded.error();
        const bool two_coders = static_cast<std::uint64_t>(g.width) * static_cast<std::uint64_t>(g.height) >= 1280ULL * 720ULL;
        auto begun = predicted_.begin_encode(two_coders);
        if (!begun) return begun.error();
        auto coded_z = predicted_.encode_z(z_symbols, qp, g.z_per_channel());
        if (!coded_z) return coded_z.error();
        for (const auto& row : coded.value().rows) {
            auto coded_y = predicted_.encode_y(row);
            if (!coded_y) return coded_y.error();
        }
        auto payload = predicted_.finish_encode(frame.width(), frame.height(), qp, two_coders,
                                                decision.value().use_frame_reference);
        if (!payload) return payload.error();
        Frame shown;  // like native: a P reconstruction only when the encoder verifies it
        if (verify_) {
            auto picture = display(coded.value().luma, coded.value().chroma, g, frame.timestamp());
            if (!picture) return picture.error();
            shown = std::move(picture.value());
        }
        encoder_.frame.reset();
        encoder_.feature = std::move(coded.value().feature);
        encoder_.qp = qp;
        encoder_.semantics.commit(state, true);
        return CodecEncodeResult{std::move(payload.value()), std::move(shown), {}, qp};
    }

    Result<CodecDecodeResult> decode_predicted(std::span<const std::byte> payload, Timestamp timestamp,
                                               const SequenceStateView& state) {
        auto parsed = predicted_.parse_payload(payload);
        if (!parsed) return parsed.error();
        const auto info = parsed.value();
        auto valid = predicted_.validate_decode_reference(info.use_frame_reference, state, decoder_.semantics);
        if (!valid) return valid.error();
        const auto g = Geometry::of(info.width, info.height);
        auto ctx = context(decoder_, info.use_frame_reference, info.qp, g);
        if (!ctx) return ctx.error();
        auto begun = predicted_.begin_decode(info);
        if (!begun) return begun.error();
        auto z = predicted_.decode_z(128 * g.z_per_channel(), info.qp, g.z_per_channel());
        if (!z) return z.error();
        auto decoded = predicted(g, info.qp, ctx.value(), z.value(), nullptr, true,
                                 [&](std::span<const std::uint8_t> indexes) { return predicted_.decode_y(indexes); });
        if (!decoded) return decoded.error();
        auto ended = predicted_.end_decode();
        if (!ended) return ended.error();
        auto shown = display(decoded.value().luma, decoded.value().chroma, g, timestamp);
        if (!shown) return shown.error();
        decoder_.frame.reset();
        decoder_.feature = std::move(decoded.value().feature);
        decoder_.qp = info.qp;
        decoder_.semantics.commit(state, true);
        return CodecDecodeResult{std::move(shown.value()), {}};
    }

    std::shared_ptr<api::IProviderSession> session_;
    std::map<std::string, api::ExecutableStage, std::less<>> stages_;
    std::vector<Buffer> pool_;
    std::map<std::array<int, 4>, std::vector<std::uint32_t>> active_;
    std::optional<Buffer> parity_;
    std::vector<double> scaled_;  // the encoder's y / step size, reused across frames
    IntraOrchestration intra_;
    PredictedOrchestration predicted_;
    Reference encoder_;
    Reference decoder_;
    std::uint32_t qp_{};
    bool verify_{};
};

}  // namespace

Result<std::unique_ptr<codec::CodecBackend>> make_fx_backend(std::shared_ptr<api::IProviderSession> session) {
    if (!session) return fail("the fx backend needs a provider session");
    return std::unique_ptr<codec::CodecBackend>(std::make_unique<FxBackend>(std::move(session)));
}

}  // namespace nvcr::dcvcrt
