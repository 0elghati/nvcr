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
// profile); the encoder's analysis transforms and the P-frame display picture,
// which nothing reads back, stay FP16. Between stages the
// codec applies its own rules on the host: the spatial masks, integer scale
// indexes from fx_entropy.json, and y_hat = sum of masked (symbol 2^fa + mean).
// Entropy coding, payloads, QP schedule and reference decisions are DCVC-RT's
// usual orchestration, so streams keep the NVI1/NVP1 syntax.

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
    [[nodiscard]] std::size_t bytes() const { return count() * (type == api::TensorDataType::int32 ? 4U : 2U); }
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

Result<void> run(api::IProviderSession& session, const api::ExecutableStage& stage, std::initializer_list<Bind> in,
                 std::initializer_list<Bind> out) {
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
struct EntropyRules final {
    std::vector<std::int64_t> edges;  // index k <=> scale >= edges[k - 1]
    std::vector<std::uint8_t> table;  // the index of every scale in [0, edges.back()]

    void build() {
        table.resize(edges.empty() || edges.back() < 0 ? 0 : static_cast<std::size_t>(edges.back()) + 1);
        for (std::size_t s = 0; s < table.size(); ++s) table[s] = search(static_cast<std::int64_t>(s));
    }
    [[nodiscard]] std::uint8_t search(std::int64_t scale) const {
        return static_cast<std::uint8_t>(std::upper_bound(edges.begin(), edges.end(), scale) - edges.begin());
    }
    [[nodiscard]] std::uint8_t index(std::int64_t scale) const {
        return scale >= 0 && static_cast<std::size_t>(scale) < table.size() ? table[static_cast<std::size_t>(scale)]
                                                                              : search(scale);
    }
};

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
                {"p_analysis", {{"frame", half, read, Dims{{1, 1}, {3, 3}, any, any}},
                                {"context", half, read, Dims{{1, 1}, {256, 256}, any, any}},
                                {"q_encoder", half, read, Dims{{1, 1}, {256, 256}, {1, 1}, {1, 1}}},
                                {"y", half, write, Dims{{1, 1}, {128, 128}, any, any}}}},
                {"p_hyper_analysis", {{"y_padded", half, read, Dims{{1, 1}, {128, 128}, any, any}},
                                      {"z", half, write, Dims{{1, 1}, {128, 128}, any, any}}}},
                {"p_display", {{"feature", api::TensorDataType::int32, read, Dims{{1, 1}, {256, 256}, any, any}},
                               {"q_recon", half, read, Dims{{1, 1}, {320, 320}, {1, 1}, {1, 1}}},
                               {"frame", api::TensorDataType::int32, write, Dims{{1, 1}, {3, 3}, any, any}}}}};
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
            const auto* edges = rules.value().find("index_edges");
            const auto* threshold = rules.value().find("skip_threshold");
            if (edges == nullptr || threshold == nullptr || threshold->integer_value.value_or(0) >= 0) {
                return fail("fx_entropy.json needs index_edges and no skip threshold");
            }
            for (const auto& edge : edges->array_value) rules_.edges.push_back(edge.integer_value.value_or(0));
            rules_.build();
            auto intra = intra_.initialize(root);
            if (!intra) return intra.error();
            auto predicted = predicted_.initialize(root);
            if (!predicted) return predicted.error();
            qp_ = configuration.codec.intra_qp;
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

    // Encoder: FP16 picture, then y (scaled by the caller) and z symbols from the analysis stages.
    Result<std::pair<Buffer, std::vector<std::int8_t>>> analyse(const Frame& frame, const Geometry& g, bool intra,
                                                                 std::uint32_t qp, const Buffer* context) {
        auto& s = *session_;
        auto picture = take({1, 3, g.padded_height, g.padded_width}, true, api::TensorDataType::float16);
        if (!picture) return picture.error();
        const auto bytes = frame.data();
        if (!host_ops::yuv420p8_to_ycbcr_padded(
                {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()}, g.width, g.height,
                picture.value().halves(), g.padded_height, g.padded_width)) {
            return fail("input color conversion");
        }
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
        Result<void> ran = intra ? run(s, stage("i_analysis"), {{"frame", &picture.value()}, {"q_enc", &q.value()}},
                                       {{"y", &y.value()}})
                                 : Result<void>{};
        if (!intra) {
            auto ctx = take({1, 256, g.padded_height / 8, g.padded_width / 8}, true, api::TensorDataType::float16);
            if (!ctx) return ctx.error();
            const auto ints = context->ints();
            auto halves = ctx.value().halves();
            for (std::size_t i = 0; i < ints.size(); ++i) {
                halves[i] = host_ops::to_half(static_cast<float>(static_cast<double>(ints[i]) / static_cast<double>(one)));
            }
            ran = run(s, stage("p_analysis"),
                      {{"frame", &picture.value()}, {"context", &ctx.value()}, {"q_encoder", &q.value()}},
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

    Result<Buffer> z_fixed(std::span<const std::int8_t> symbols, const Geometry& g) {
        auto z = take({1, 128, g.yh4 / 4, g.yw4 / 4}, true);
        if (!z) return z.error();
        for (std::size_t i = 0; i < symbols.size(); ++i) z.value().ints()[i] = symbols[i] * static_cast<std::int32_t>(one);
        return z;
    }

    // Runs the spatial steps. Encoder: y_scaled holds y / step size (float) and
    // the symbols are chosen here; decoder: decode(indexes) returns them.
    // Returns y_hat; indexes_out collects the encoder's packed symbol/index rows.
    template <typename Decode>
    Result<Buffer> spatial_steps(bool intra, const Geometry& g, const Buffer& params, const Buffer* common,
                                 const std::vector<double>* y_scaled, std::vector<std::vector<std::int16_t>>* rows,
                                 Decode&& decode) {
        auto& s = *session_;
        const int channels = intra ? 256 : 128;
        const int groups = intra ? 4 : 2;
        const auto plane = static_cast<std::size_t>(g.yh) * static_cast<std::size_t>(g.yw);
        auto y_hat = take({1, channels, g.yh, g.yw}, true);
        if (!y_hat) return y_hat.error();
        std::fill(y_hat.value().ints().begin(), y_hat.value().ints().end(), 0);
        auto scales_means = take({1, 2 * channels, g.yh, g.yw}, true);
        if (!scales_means) return scales_means.error();
        // Step 0 reads the prior's scales and means; params = [q.. | scales | means].
        const std::size_t first = (intra ? 2 : 128) * plane;
        std::span<const std::int32_t> scales = params.ints().subspan(first, static_cast<std::size_t>(channels) * plane);
        std::span<const std::int32_t> means = params.ints().subspan(first + static_cast<std::size_t>(channels) * plane,
                                                                    static_cast<std::size_t>(channels) * plane);
        const std::size_t reduced = static_cast<std::size_t>(channels / groups) * plane;
        for (int k = 0; k < groups; ++k) {
            if (k > 0) {
                const auto spatial = intra ? "i_spatial_prior_" + std::to_string(k) : std::string("p_spatial_prior");
                auto ran = intra ? run(s, stage(spatial), {{"y_hat", &y_hat.value()}, {"common", common}},
                                       {{"scales_means", &scales_means.value()}})
                                 : run(s, stage(spatial), {{"y_hat", &y_hat.value()}, {"params", &params}},
                                       {{"scales_means", &scales_means.value()}});
                if (!ran) return ran.error();
                scales = scales_means.value().ints().first(static_cast<std::size_t>(channels) * plane);
                means = scales_means.value().ints().subspan(static_cast<std::size_t>(channels) * plane);
            }
            const auto& offsets = active(Step{groups, k}, channels, g);
            std::vector<std::uint8_t> indexes(reduced);
            std::vector<std::int8_t> symbols(reduced);
            for (std::size_t r = 0; r < reduced; ++r) {
                const auto i = offsets[r];
                indexes[r] = rules_.index(scales[i]);
                if (y_scaled != nullptr) {
                    const double v = std::nearbyint((*y_scaled)[i] - static_cast<double>(means[i]) / static_cast<double>(one));
                    symbols[r] = static_cast<std::int8_t>(std::clamp(v, -128.0, 127.0));
                }
            }
            if (y_scaled == nullptr) {
                auto decoded = decode(indexes);
                if (!decoded) return decoded.error();
                symbols = std::move(decoded.value());
            } else {
                std::vector<std::int16_t> row(reduced);
                for (std::size_t r = 0; r < reduced; ++r) {
                    row[r] = static_cast<std::int16_t>(static_cast<std::uint16_t>(static_cast<std::uint8_t>(symbols[r])) << 8U |
                                                       indexes[r]);
                }
                rows->push_back(std::move(row));
            }
            auto hat = y_hat.value().ints();
            for (std::size_t r = 0; r < reduced; ++r) {
                hat[offsets[r]] += static_cast<std::int32_t>(symbols[r] * one + means[offsets[r]]);
            }
        }
        return y_hat;
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
        auto z = z_fixed(z_symbols, g);
        auto ref = latent_ref(g);
        auto params = take({1, 514, g.yh, g.yw}, true);
        auto common = take({1, 256, g.yh, g.yw}, false);
        if (!z || !ref || !params || !common) return fail("cannot allocate I-frame tensors");
        auto ran = run(s, stage("i_prior"), {{"z_fixed", &z.value()}, {"latent_ref", &ref.value()}},
                       {{"params", &params.value()}, {"common", &common.value()}});
        if (!ran) return ran.error();
        IntraResult result;
        std::optional<std::vector<double>> y_scaled;
        if (y != nullptr) {  // y q_enc, q_enc = 1.5 sigmoid(params[0]) + 0.5 (encoder only)
            const auto plane = static_cast<std::size_t>(g.yh) * static_cast<std::size_t>(g.yw);
            y_scaled.emplace(y->count());
            const auto values = y->halves();
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double p = static_cast<double>(params.value().ints()[i % plane]) / static_cast<double>(one);
                (*y_scaled)[i] = static_cast<double>(host_ops::to_float(values[i])) * (1.5 / (1.0 + std::exp(-p)) + 0.5);
            }
        }
        auto y_hat = spatial_steps(true, g, params.value(), &common.value(), y_scaled ? &*y_scaled : nullptr,
                                   &result.rows, decode);
        if (!y_hat) return y_hat.error();
        auto picture = take({1, 3, g.padded_height, g.padded_width}, true);
        auto qp_t = qp_tensor(qp);
        if (!picture || !qp_t) return fail("cannot allocate the I-frame picture");
        ran = run(s, stage("i_synthesis"),
                  {{"y_hat", &y_hat.value()}, {"params", &params.value()}, {"qp", &qp_t.value()}},
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

    Result<Context> context(Reference& reference, bool use_frame, std::uint32_t qp, const Geometry& g, bool host) {
        auto& s = *session_;
        auto temporal = take({1, 256, g.padded_height / 8, g.padded_width / 8}, false);
        auto context = take({1, 256, g.padded_height / 8, g.padded_width / 8}, host);
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
        Buffer picture;
        Buffer feature;
        std::vector<std::vector<std::int16_t>> rows;
    };

    template <typename Decode>
    Result<PredictedResult> predicted(const Geometry& g, std::uint32_t qp, const Context& ctx,
                                      std::span<const std::int8_t> z_symbols, const Buffer* y, Decode&& decode) {
        auto& s = *session_;
        auto z = z_fixed(z_symbols, g);
        auto ref = latent_ref(g);
        auto params = take({1, 384, g.yh, g.yw}, true);
        if (!z || !ref || !params) return fail("cannot allocate P-frame tensors");
        auto ran = run(s, stage("p_prior"),
                       {{"z_fixed", &z.value()}, {"temporal_context", &ctx.temporal}, {"latent_ref", &ref.value()}},
                       {{"params", &params.value()}});
        if (!ran) return ran.error();
        PredictedResult result;
        std::optional<std::vector<double>> y_scaled;
        if (y != nullptr) {  // y / max(q_dec, 0.5) (encoder only)
            y_scaled.emplace(y->count());
            const auto values = y->halves();
            for (std::size_t i = 0; i < values.size(); ++i) {
                const double q = std::max(static_cast<double>(params.value().ints()[i]) / static_cast<double>(one), 0.5);
                (*y_scaled)[i] = static_cast<double>(host_ops::to_float(values[i])) / q;
            }
        }
        auto y_hat = spatial_steps(false, g, params.value(), nullptr, y_scaled ? &*y_scaled : nullptr, &result.rows,
                                   decode);
        if (!y_hat) return y_hat.error();
        auto feature = take({1, 256, g.padded_height / 8, g.padded_width / 8}, false);
        auto picture = take({1, 3, g.padded_height, g.padded_width}, true);
        auto qp_t = qp_tensor(qp);
        if (!feature || !picture || !qp_t) return fail("cannot allocate the P-frame outputs");
        ran = run(s, stage("p_synthesis"),
                  {{"y_hat", &y_hat.value()}, {"params", &params.value()}, {"context", &ctx.context}, {"qp", &qp_t.value()}},
                  {{"feature", &feature.value()}});
        if (!ran) return ran.error();
        auto q = take({1, 320, 1, 1}, true, api::TensorDataType::float16);
        if (!q) return q.error();
        const auto table = predicted_.reconstruction_quantization();
        for (std::size_t c = 0; c < 320; ++c) q.value().halves()[c] = host_ops::to_half(table[qp * 320 + c]);
        ran = run(s, stage("p_display"), {{"feature", &feature.value()}, {"q_recon", &q.value()}},
                  {{"frame", &picture.value()}});
        if (!ran) return ran.error();
        result.picture = std::move(picture.value());
        result.feature = std::move(feature.value());
        return result;
    }

    Result<CodecEncodeResult> encode_predicted(const Frame& frame, const SequenceStateView& state) {
        const auto g = Geometry::of(frame.width(), frame.height());
        auto decision = predicted_.select_frame(qp_, state, encoder_.semantics);
        if (!decision) return decision.error();
        const auto qp = decision.value().qp;
        auto ctx = context(encoder_, decision.value().use_frame_reference, qp, g, true);
        if (!ctx) return ctx.error();
        auto analysed = analyse(frame, g, false, qp, &ctx.value().context);
        if (!analysed) return analysed.error();
        auto& [y, z_symbols] = analysed.value();
        auto coded = predicted(g, qp, ctx.value(), z_symbols, &y,
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
        auto shown = display(coded.value().picture, g, frame.timestamp());
        if (!shown) return shown.error();
        encoder_.frame.reset();
        encoder_.feature = std::move(coded.value().feature);
        encoder_.qp = qp;
        encoder_.semantics.commit(state, true);
        return CodecEncodeResult{std::move(payload.value()), std::move(shown.value()), {}, qp};
    }

    Result<CodecDecodeResult> decode_predicted(std::span<const std::byte> payload, Timestamp timestamp,
                                               const SequenceStateView& state) {
        auto parsed = predicted_.parse_payload(payload);
        if (!parsed) return parsed.error();
        const auto info = parsed.value();
        auto valid = predicted_.validate_decode_reference(info.use_frame_reference, state, decoder_.semantics);
        if (!valid) return valid.error();
        const auto g = Geometry::of(info.width, info.height);
        auto ctx = context(decoder_, info.use_frame_reference, info.qp, g, false);
        if (!ctx) return ctx.error();
        auto begun = predicted_.begin_decode(info);
        if (!begun) return begun.error();
        auto z = predicted_.decode_z(128 * g.z_per_channel(), info.qp, g.z_per_channel());
        if (!z) return z.error();
        auto decoded = predicted(g, info.qp, ctx.value(), z.value(), nullptr,
                                 [&](std::span<const std::uint8_t> indexes) { return predicted_.decode_y(indexes); });
        if (!decoded) return decoded.error();
        auto ended = predicted_.end_decode();
        if (!ended) return ended.error();
        auto shown = display(decoded.value().picture, g, timestamp);
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
    EntropyRules rules_;
    IntraOrchestration intra_;
    PredictedOrchestration predicted_;
    Reference encoder_;
    Reference decoder_;
    std::uint32_t qp_{};
};

}  // namespace

Result<std::unique_ptr<codec::CodecBackend>> make_fx_backend(std::shared_ptr<api::IProviderSession> session) {
    if (!session) return fail("the fx backend needs a provider session");
    return std::unique_ptr<codec::CodecBackend>(std::make_unique<FxBackend>(std::move(session)));
}

}  // namespace nvcr::dcvcrt
