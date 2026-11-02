#include "host_backend.hpp"

#include "nvcr/dcvcrt/backend.hpp"

#include "host_ops.hpp"
#include "orchestration.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::dcvcrt {
namespace {

namespace session_api = provider::experimental;
using host_ops::Half;
using SequenceStateView = codec::SequenceStateView;

constexpr std::string_view subsystem = "dcvcrt.host";
constexpr std::int64_t image_quant_channels = 368;
constexpr std::uint32_t minimum_frame_dimension = 64U;
constexpr std::uint32_t maximum_frame_width = 1920U;
constexpr std::uint32_t maximum_frame_height = 1080U;

Error backend_error(std::string message) {
    return Error(ErrorCode::backend_error, std::move(message), std::string(subsystem));
}

Error op_error(std::string_view operation) {
    return backend_error(std::string(operation) + " rejected its tensor shapes");
}

bool supported_frame_dimensions(std::uint32_t width, std::uint32_t height) {
    return width >= minimum_frame_dimension && height >= minimum_frame_dimension &&
        width <= maximum_frame_width && height <= maximum_frame_height &&
        (width & 1U) == 0U && (height & 1U) == 0U;
}

std::int32_t round_up(std::int32_t value, std::int32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

using Dims = std::array<std::int64_t, 4>;

Dims dims(std::int64_t channels, std::int64_t height, std::int64_t width) {
    return {1, channels, height, width};
}

host_ops::Shape4D shape4d(std::int32_t channels, std::int32_t height, std::int32_t width) {
    return {1, channels, height, width};
}

// Stage inputs and outputs live in provider buffers; intermediates that never
// reach a stage stay in plain vectors.
struct Tensor final {
    Dims shape{};
    session_api::BufferHandle buffer;
    std::span<Half> values;
};

Result<Tensor> allocate_tensor(session_api::IProviderSession& session, Dims shape) {
    std::size_t count = 1;
    for (const auto dimension : shape) count *= static_cast<std::size_t>(dimension);
    auto buffer = session.allocate(count * sizeof(Half), session_api::MemoryDomain::host);
    if (!buffer) return buffer.error();
    const auto bytes = buffer.value()->host_bytes();
    if (bytes.size() < count * sizeof(Half) ||
        reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(Half) != 0U) {
        return Error(
            ErrorCode::incompatible_target,
            "DCVC-RT host backend requires host-addressable provider buffers",
            std::string(subsystem));
    }
    return Tensor{shape, std::move(buffer.value()),
                  std::span<Half>(reinterpret_cast<Half*>(bytes.data()), count)};
}

struct Binding final {
    std::string_view name;
    const Tensor* tensor;
};

Result<void> run_stage(
    session_api::IProviderSession& session,
    const session_api::ExecutableStage& stage,
    std::initializer_list<Binding> inputs,
    std::initializer_list<Binding> outputs) {
    std::vector<session_api::TensorView> views;
    views.reserve(inputs.size() + outputs.size());
    const auto bind = [&views](const Binding& binding, session_api::TensorAccess access) {
        views.push_back(session_api::TensorView{
            binding.name,
            session_api::TensorDataType::float16,
            {binding.tensor->shape.begin(), binding.tensor->shape.end()},
            access,
            session_api::BufferSlice{
                binding.tensor->buffer, 0, binding.tensor->values.size_bytes()}});
    };
    for (const auto& binding : inputs) bind(binding, session_api::TensorAccess::read);
    for (const auto& binding : outputs) bind(binding, session_api::TensorAccess::write);
    auto completion = session.submit(stage, views, {});
    if (!completion) return completion.error();
    return completion.value()->wait();
}

Result<Tensor> quant_tensor(
    session_api::IProviderSession& session,
    std::span<const float> table,
    std::uint32_t qp,
    std::int64_t channels) {
    const auto count = static_cast<std::size_t>(channels);
    const auto first = static_cast<std::size_t>(qp) * count;
    if (first + count > table.size()) {
        return backend_error("quantization table has no entry for QP " + std::to_string(qp));
    }
    auto tensor = allocate_tensor(session, dims(channels, 1, 1));
    if (!tensor) return tensor.error();
    for (std::size_t index = 0; index < count; ++index) {
        tensor.value().values[index] = host_ops::to_half(table[first + index]);
    }
    return tensor;
}

Result<Tensor> input_frame_tensor(session_api::IProviderSession& session, const Frame& frame) {
    const auto width = static_cast<std::int32_t>(frame.width());
    const auto height = static_cast<std::int32_t>(frame.height());
    const auto padded_height = round_up(height, 16);
    const auto padded_width = round_up(width, 16);
    auto tensor = allocate_tensor(session, dims(3, padded_height, padded_width));
    if (!tensor) return tensor.error();
    const auto bytes = frame.data();
    const std::span<const std::uint8_t> input(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    const bool converted = frame.pixel_format() == PixelFormat::yuv420p8 ?
        host_ops::yuv420p8_to_ycbcr_padded(
            input, width, height, tensor.value().values, padded_height, padded_width) :
        host_ops::rgb24_to_ycbcr_padded(
            input, width, height, tensor.value().values, padded_height, padded_width);
    if (!converted) return op_error("input color conversion");
    return tensor;
}

Result<Frame> to_yuv420p8(
    const Tensor& frame_hat, std::uint32_t width, std::uint32_t height, Timestamp timestamp) {
    const auto y_size = static_cast<std::size_t>(width) * height;
    std::vector<std::byte> bytes(y_size + y_size / 2U);
    const bool converted = host_ops::ycbcr_padded_to_yuv420p8(
        frame_hat.values,
        shape4d(3, static_cast<std::int32_t>(frame_hat.shape[2]),
                static_cast<std::int32_t>(frame_hat.shape[3])),
        static_cast<std::int32_t>(width), static_cast<std::int32_t>(height),
        std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(bytes.data()), bytes.size()));
    if (!converted) return op_error("YUV420P8 reconstruction");
    return Frame::copy_from(width, height, PixelFormat::yuv420p8, bytes, timestamp);
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

// Same NVD1 layout as the TensorRT backend's decoded-intra latent state.
std::vector<std::byte> serialize_dpb(const Tensor& frame) {
    std::vector<std::byte> output;
    output.reserve(24 + frame.values.size_bytes());
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x44}, std::byte{0x31}});
    append_u32(output, static_cast<std::uint32_t>(frame.shape[2]));
    append_u32(output, static_cast<std::uint32_t>(frame.shape[3]));
    append_u32(output, 0U);
    append_u32(output, 0U);
    const auto bytes = std::as_bytes(frame.values);
    output.insert(output.end(), bytes.begin(), bytes.end());
    return output;
}

Result<Tensor> spatial_context(
    session_api::IProviderSession& session,
    std::span<const Half> y_hat,
    std::span<const Half> conditioning,
    std::int32_t y_height,
    std::int32_t y_width) {
    auto context = allocate_tensor(session, dims(512, y_height, y_width));
    if (!context) return context.error();
    auto destination = context.value().values;
    if (y_hat.size() + conditioning.size() != destination.size()) {
        return op_error("spatial context assembly");
    }
    std::copy(y_hat.begin(), y_hat.end(), destination.begin());
    std::copy(conditioning.begin(), conditioning.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(y_hat.size()));
    return context;
}

struct HostDpb final {
    std::optional<Tensor> frame;
    std::optional<Tensor> feature;
    ReferenceState semantics;

    void clear() noexcept {
        frame.reset();
        feature.reset();
        semantics.reset();
    }
};

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    HostDpb& dpb,
    session_api::IProviderSession& session,
    IntraOrchestration& orchestration);

Result<CodecEncodeResult> encode_intra(
    const Frame& frame,
    std::uint32_t qp,
    const SequenceStateView& state,
    HostDpb& dpb,
    session_api::IProviderSession& session,
    IntraOrchestration& orchestration,
    bool verify_reconstruction) {
    if (frame.pixel_format() != PixelFormat::rgb24 &&
        frame.pixel_format() != PixelFormat::yuv420p8) {
        return Error(ErrorCode::invalid_argument,
                     "native I-frame encoding requires RGB24 or YUV420P8",
                     std::string(subsystem));
    }
    if (!supported_frame_dimensions(frame.width(), frame.height())) {
        return Error(ErrorCode::invalid_argument,
                     "native I-frame dimensions must be within 64x64 and 1920x1080",
                     std::string(subsystem));
    }

    auto input = input_frame_tensor(session, frame);
    if (!input) return input.error();
    const auto y_height = static_cast<std::int32_t>(input.value().shape[2] / 16);
    const auto y_width = static_cast<std::int32_t>(input.value().shape[3] / 16);
    const auto y_height4 = round_up(y_height, 4);
    const auto y_width4 = round_up(y_width, 4);
    const auto y_shape = shape4d(256, y_height, y_width);

    auto q_enc = quant_tensor(
        session, orchestration.encoder_quantization(), qp, image_quant_channels);
    if (!q_enc) return q_enc.error();
    auto y = allocate_tensor(session, dims(256, y_height, y_width));
    if (!y) return y.error();
    auto ran = run_stage(
        session, orchestration.stage(IntraStage::analysis),
        {{"frame", &input.value()}, {"q_enc", &q_enc.value()}}, {{"y", &y.value()}});
    if (!ran) return ran.error();

    auto y_padded = allocate_tensor(session, dims(256, y_height4, y_width4));
    if (!y_padded) return y_padded.error();
    if (!host_ops::replicate_pad(
            y.value().values, y_padded.value().values, y_shape, y_height4, y_width4)) {
        return op_error("replicate_pad");
    }
    auto z = allocate_tensor(session, dims(128, y_height4 / 4, y_width4 / 4));
    if (!z) return z.error();
    ran = run_stage(
        session, orchestration.stage(IntraStage::hyper_analysis),
        {{"y_padded", &y_padded.value()}}, {{"z", &z.value()}});
    if (!ran) return ran.error();
    std::vector<std::int8_t> z_symbols(z.value().values.size());
    if (!host_ops::round_to_int8(z.value().values, z_symbols)) {
        return op_error("round_to_int8");
    }

    auto params_padded = allocate_tensor(session, dims(514, y_height4, y_width4));
    auto common_padded = allocate_tensor(session, dims(256, y_height4, y_width4));
    if (!params_padded) return params_padded.error();
    if (!common_padded) return common_padded.error();
    ran = run_stage(
        session, orchestration.stage(IntraStage::hyper_synthesis),
        {{"z_hat", &z.value()}},
        {{"params_padded", &params_padded.value()},
         {"common_padded", &common_padded.value()}});
    if (!ran) return ran.error();

    const auto spatial_count =
        static_cast<std::size_t>(y_height) * static_cast<std::size_t>(y_width);
    const auto prior_count = 256 * spatial_count;
    std::vector<Half> params(514 * spatial_count);
    std::vector<Half> common(prior_count);
    if (!host_ops::crop_spatial(
            params_padded.value().values, params, shape4d(514, y_height4, y_width4),
            y_height, y_width) ||
        !host_ops::crop_spatial(
            common_padded.value().values, common, shape4d(256, y_height4, y_width4),
            y_height, y_width)) {
        return op_error("crop_spatial");
    }
    if (!host_ops::apply_image_prior_and_quantize(params, y.value().values, spatial_count)) {
        return op_error("apply_image_prior_and_quantize");
    }
    const std::span<const Half> parameter_view(params);
    const auto q_dec_values = parameter_view.subspan(spatial_count, spatial_count);
    auto current_scales = parameter_view.subspan(2 * spatial_count, prior_count);
    auto current_means = parameter_view.subspan((2 + 256) * spatial_count, prior_count);

    const auto reduced_count = prior_count / 4;
    std::vector<Half> mask(prior_count);
    std::vector<Half> symbols_full(prior_count);
    std::vector<Half> reconstructed(prior_count);
    std::vector<Half> scales_full(prior_count);
    std::vector<Half> symbols(reduced_count);
    std::vector<Half> scales(reduced_count);
    std::array<std::vector<std::int16_t>, 4> indexes;
    std::optional<Tensor> spatial_params;
    auto y_hat = allocate_tensor(session, dims(256, y_height, y_width));
    if (!y_hat) return y_hat.error();
    for (std::int32_t stage_index = 0; stage_index < 4; ++stage_index) {
        const auto slot = static_cast<std::size_t>(stage_index);
        // The first stage's reconstruction becomes y_hat; later stages add to it.
        const std::span<Half> stage_output =
            stage_index == 0 ? y_hat.value().values : std::span<Half>(reconstructed);
        if (!host_ops::make_four_way_mask(mask, y_shape, stage_index) ||
            !host_ops::process_with_mask(
                y.value().values, current_scales, current_means, mask, {},
                symbols_full, stage_output, scales_full, 0.0F) ||
            !host_ops::reduce_channel_quarters(symbols_full, symbols, y_shape) ||
            !host_ops::reduce_channel_quarters(scales_full, scales, y_shape)) {
            return op_error("I-frame mask processing");
        }
        indexes[slot].resize(reduced_count);
        if (!host_ops::build_encode_indexes(symbols, scales, indexes[slot], 0.0F)) {
            return op_error("build_encode_indexes");
        }
        if (stage_index > 0 && !host_ops::add_in_place(y_hat.value().values, reconstructed)) {
            return op_error("add_in_place");
        }

        if (stage_index < 3) {
            auto context = spatial_context(
                session, y_hat.value().values, common, y_height, y_width);
            if (!context) return context.error();
            auto output = allocate_tensor(session, dims(512, y_height, y_width));
            if (!output) return output.error();
            const auto spatial_stage = static_cast<IntraStage>(
                static_cast<std::size_t>(IntraStage::spatial_prior_1) + slot);
            ran = run_stage(
                session, orchestration.stage(spatial_stage),
                {{"context", &context.value()}}, {{"scales_means", &output.value()}});
            if (!ran) return ran.error();
            spatial_params = std::move(output.value());
            current_scales = spatial_params->values.first(prior_count);
            current_means = spatial_params->values.subspan(prior_count, prior_count);
        }
    }

    if (!host_ops::multiply_broadcast_in_place(
            y_hat.value().values, q_dec_values, spatial_count, 256)) {
        return op_error("multiply_broadcast_in_place");
    }
    auto q_dec = quant_tensor(
        session, orchestration.decoder_quantization(), qp, image_quant_channels);
    if (!q_dec) return q_dec.error();
    auto frame_hat = allocate_tensor(
        session, dims(3, input.value().shape[2], input.value().shape[3]));
    if (!frame_hat) return frame_hat.error();
    ran = run_stage(
        session, orchestration.stage(IntraStage::synthesis),
        {{"y_hat", &y_hat.value()}, {"q_dec", &q_dec.value()}},
        {{"frame_hat", &frame_hat.value()}});
    if (!ran) return ran.error();

    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    auto mode = orchestration.begin_encode(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z.value().shape[2] * z.value().shape[3]);
    auto encoded_z = orchestration.encode_z(z_symbols, qp, per_channel);
    if (!encoded_z) return encoded_z.error();
    for (const auto& stage_indexes : indexes) {
        auto encoded_y = orchestration.encode_y(stage_indexes);
        if (!encoded_y) return encoded_y.error();
    }
    auto payload = orchestration.finish_encode(frame.width(), frame.height(), qp, two_coders);
    if (!payload) return payload.error();

    auto reconstructed_frame = to_yuv420p8(
        frame_hat.value(), frame.width(), frame.height(), frame.timestamp());
    if (!reconstructed_frame) return reconstructed_frame.error();
    dpb.clear();
    dpb.frame = std::move(frame_hat.value());
    dpb.semantics.commit(state, false);

    if (verify_reconstruction) {
        auto conformant = decode_intra(
            payload.value(), frame.timestamp(), state, dpb, session, orchestration);
        if (!conformant) return conformant.error();
        return CodecEncodeResult{
            std::move(payload.value()), std::move(conformant.value().frame),
            std::move(conformant.value().latent_state), qp};
    }
    return CodecEncodeResult{
        std::move(payload.value()), std::move(reconstructed_frame.value()), {}, qp};
}

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    HostDpb& dpb,
    session_api::IProviderSession& session,
    IntraOrchestration& orchestration) {
    auto parsed = orchestration.parse_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    const auto padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = padded_height / 16;
    const auto y_width = padded_width / 16;
    const auto y_height4 = round_up(y_height, 4);
    const auto y_width4 = round_up(y_width, 4);
    const auto y_shape = shape4d(256, y_height, y_width);

    auto stream_set = orchestration.begin_decode(info);
    if (!stream_set) return stream_set.error();
    const auto per_channel = static_cast<std::size_t>(y_height4 / 4) *
        static_cast<std::size_t>(y_width4 / 4);
    auto z_values = orchestration.decode_z(128 * per_channel, info.qp, per_channel);
    if (!z_values) return z_values.error();
    auto z = allocate_tensor(session, dims(128, y_height4 / 4, y_width4 / 4));
    if (!z) return z.error();
    if (z_values.value().size() != z.value().values.size() ||
        !host_ops::int8_to_half(z_values.value(), z.value().values)) {
        return op_error("I-frame z symbol upload");
    }

    auto params_padded = allocate_tensor(session, dims(514, y_height4, y_width4));
    auto common_padded = allocate_tensor(session, dims(256, y_height4, y_width4));
    if (!params_padded) return params_padded.error();
    if (!common_padded) return common_padded.error();
    auto ran = run_stage(
        session, orchestration.stage(IntraStage::hyper_synthesis),
        {{"z_hat", &z.value()}},
        {{"params_padded", &params_padded.value()},
         {"common_padded", &common_padded.value()}});
    if (!ran) return ran.error();

    const auto spatial_count =
        static_cast<std::size_t>(y_height) * static_cast<std::size_t>(y_width);
    const auto prior_count = 256 * spatial_count;
    std::vector<Half> params(514 * spatial_count);
    std::vector<Half> common(prior_count);
    if (!host_ops::crop_spatial(
            params_padded.value().values, params, shape4d(514, y_height4, y_width4),
            y_height, y_width) ||
        !host_ops::crop_spatial(
            common_padded.value().values, common, shape4d(256, y_height4, y_width4),
            y_height, y_width)) {
        return op_error("crop_spatial");
    }
    if (!host_ops::apply_image_prior(params, spatial_count)) {
        return op_error("apply_image_prior");
    }
    const std::span<const Half> parameter_view(params);
    const auto q_dec_values = parameter_view.subspan(spatial_count, spatial_count);
    auto current_scales = parameter_view.subspan(2 * spatial_count, prior_count);
    auto current_means = parameter_view.subspan((2 + 256) * spatial_count, prior_count);

    const auto reduced_count = prior_count / 4;
    std::vector<Half> mask(prior_count);
    std::vector<Half> scales(reduced_count);
    std::vector<std::uint8_t> indexes(reduced_count);
    std::vector<Half> symbols(reduced_count);
    std::vector<Half> reconstructed(prior_count);
    std::optional<Tensor> spatial_params;
    auto y_hat = allocate_tensor(session, dims(256, y_height, y_width));
    if (!y_hat) return y_hat.error();
    for (std::int32_t stage_index = 0; stage_index < 4; ++stage_index) {
        if (!host_ops::make_four_way_mask(mask, y_shape, stage_index) ||
            !host_ops::reduce_masked_quarters(current_scales, mask, scales, y_shape) ||
            !host_ops::build_decode_indexes(scales, indexes, 0.0F)) {
            return op_error("I-frame decode index preparation");
        }
        auto decoded = orchestration.decode_y(indexes);
        if (!decoded) return decoded.error();
        const std::span<Half> stage_output =
            stage_index == 0 ? y_hat.value().values : std::span<Half>(reconstructed);
        if (decoded.value().size() != reduced_count ||
            !host_ops::int8_to_half(decoded.value(), symbols) ||
            !host_ops::restore_four_way(symbols, current_means, mask, stage_output, y_shape)) {
            return op_error("I-frame symbol restoration");
        }
        if (stage_index > 0 && !host_ops::add_in_place(y_hat.value().values, reconstructed)) {
            return op_error("add_in_place");
        }

        if (stage_index < 3) {
            auto context = spatial_context(
                session, y_hat.value().values, common, y_height, y_width);
            if (!context) return context.error();
            auto output = allocate_tensor(session, dims(512, y_height, y_width));
            if (!output) return output.error();
            const auto spatial_stage = static_cast<IntraStage>(
                static_cast<std::size_t>(IntraStage::spatial_prior_1) +
                static_cast<std::size_t>(stage_index));
            ran = run_stage(
                session, orchestration.stage(spatial_stage),
                {{"context", &context.value()}}, {{"scales_means", &output.value()}});
            if (!ran) return ran.error();
            spatial_params = std::move(output.value());
            current_scales = spatial_params->values.first(prior_count);
            current_means = spatial_params->values.subspan(prior_count, prior_count);
        }
    }

    if (!host_ops::multiply_broadcast_in_place(
            y_hat.value().values, q_dec_values, spatial_count, 256)) {
        return op_error("multiply_broadcast_in_place");
    }
    auto q_dec = quant_tensor(
        session, orchestration.decoder_quantization(), info.qp, image_quant_channels);
    if (!q_dec) return q_dec.error();
    auto frame_hat = allocate_tensor(session, dims(3, padded_height, padded_width));
    if (!frame_hat) return frame_hat.error();
    ran = run_stage(
        session, orchestration.stage(IntraStage::synthesis),
        {{"y_hat", &y_hat.value()}, {"q_dec", &q_dec.value()}},
        {{"frame_hat", &frame_hat.value()}});
    if (!ran) return ran.error();

    auto frame = to_yuv420p8(frame_hat.value(), info.width, info.height, timestamp);
    if (!frame) return frame.error();
    auto latent_state = serialize_dpb(frame_hat.value());
    dpb.clear();
    dpb.frame = std::move(frame_hat.value());
    dpb.semantics.commit(state, false);
    return CodecDecodeResult{std::move(frame.value()), std::move(latent_state)};
}

Result<CodecEncodeResult> encode_predicted(
    const Frame& frame,
    std::uint32_t base_qp,
    const SequenceStateView& state,
    HostDpb& dpb,
    session_api::IProviderSession& session,
    PredictedOrchestration& orchestration,
    bool verify_reconstruction) {
    if (frame.pixel_format() != PixelFormat::rgb24 &&
        frame.pixel_format() != PixelFormat::yuv420p8) {
        return Error(ErrorCode::invalid_argument,
                     "native P-frame encoding requires RGB24 or YUV420P8",
                     std::string(subsystem));
    }
    if (!supported_frame_dimensions(frame.width(), frame.height())) {
        return Error(ErrorCode::invalid_argument,
                     "native P-frame dimensions must be even and within 64x64 and 1920x1080",
                     std::string(subsystem));
    }
    if (!dpb.frame) {
        return Error(ErrorCode::invalid_state, "encoder DPB is unavailable",
                     std::string(subsystem));
    }
    auto decision = orchestration.select_frame(base_qp, state, dpb.semantics);
    if (!decision) return decision.error();
    const auto qp = decision.value().qp;
    const auto use_frame_reference = decision.value().use_frame_reference;
    if (!use_frame_reference && !dpb.feature) {
        return Error(ErrorCode::invalid_state, "P-frame feature storage is unavailable",
                     std::string(subsystem));
    }

    auto input = input_frame_tensor(session, frame);
    if (!input) return input.error();
    const auto padded_height = input.value().shape[2];
    const auto padded_width = input.value().shape[3];
    const auto y_height = static_cast<std::int32_t>(padded_height / 16);
    const auto y_width = static_cast<std::int32_t>(padded_width / 16);
    const auto y_height4 = round_up(y_height, 4);
    const auto y_width4 = round_up(y_width, 4);
    const auto prior_shape = shape4d(128, y_height, y_width);
    const auto prior_count =
        128U * static_cast<std::size_t>(y_height) * static_cast<std::size_t>(y_width);

    auto q_feature = quant_tensor(session, orchestration.feature_quantization(), qp, 256);
    if (!q_feature) return q_feature.error();
    auto context = allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    auto temporal_context =
        allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    if (!context) return context.error();
    if (!temporal_context) return temporal_context.error();
    auto ran = use_frame_reference ?
        run_stage(
            session, orchestration.stage(PredictedStage::reference_frame),
            {{"reference_frame", &*dpb.frame}, {"q_feature", &q_feature.value()}},
            {{"context", &context.value()}, {"temporal_context", &temporal_context.value()}}) :
        run_stage(
            session, orchestration.stage(PredictedStage::reference_feature),
            {{"reference_feature", &*dpb.feature}, {"q_feature", &q_feature.value()}},
            {{"context", &context.value()}, {"temporal_context", &temporal_context.value()}});
    if (!ran) return ran.error();

    auto q_encoder = quant_tensor(session, orchestration.encoder_quantization(), qp, 256);
    if (!q_encoder) return q_encoder.error();
    auto y = allocate_tensor(session, dims(128, y_height, y_width));
    if (!y) return y.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::analysis),
        {{"frame", &input.value()}, {"context", &context.value()},
         {"q_encoder", &q_encoder.value()}},
        {{"y", &y.value()}});
    if (!ran) return ran.error();

    auto y_padded = allocate_tensor(session, dims(128, y_height4, y_width4));
    if (!y_padded) return y_padded.error();
    if (!host_ops::replicate_pad(
            y.value().values, y_padded.value().values, prior_shape, y_height4, y_width4)) {
        return op_error("replicate_pad");
    }
    auto z = allocate_tensor(session, dims(128, y_height4 / 4, y_width4 / 4));
    if (!z) return z.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::hyper_analysis),
        {{"y_padded", &y_padded.value()}}, {{"z", &z.value()}});
    if (!ran) return ran.error();
    std::vector<std::int8_t> z_symbols(z.value().values.size());
    if (!host_ops::round_to_int8(z.value().values, z_symbols)) {
        return op_error("round_to_int8");
    }

    auto params = allocate_tensor(session, dims(384, y_height, y_width));
    if (!params) return params.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::prior),
        {{"z_hat", &z.value()}, {"temporal_context", &temporal_context.value()}},
        {{"params", &params.value()}});
    if (!ran) return ran.error();
    const auto q_dec_values = params.value().values.first(prior_count);
    const auto scales_values = params.value().values.subspan(prior_count, prior_count);
    const auto means_values = params.value().values.subspan(2 * prior_count, prior_count);
    if (!host_ops::clamp_reciprocal_with_quant(q_dec_values, y.value().values)) {
        return op_error("clamp_reciprocal_with_quant");
    }

    const auto reduced_count = prior_count / 2;
    std::vector<Half> mask(prior_count);
    std::vector<Half> symbols_full(prior_count);
    std::vector<Half> scales_full(prior_count);
    std::vector<Half> reconstructed(prior_count);
    std::vector<Half> symbols(reduced_count);
    std::vector<Half> scales(reduced_count);
    std::vector<std::int16_t> indexes0(reduced_count);
    std::vector<std::int16_t> indexes1(reduced_count);
    auto y_hat = allocate_tensor(session, dims(128, y_height, y_width));
    if (!y_hat) return y_hat.error();
    if (!host_ops::make_two_way_mask(mask, prior_shape, 0) ||
        !host_ops::process_with_mask(
            y.value().values, scales_values, means_values, mask, {}, symbols_full,
            y_hat.value().values, scales_full, 0.0F) ||
        !host_ops::reduce_channel_halves(symbols_full, symbols, prior_shape) ||
        !host_ops::reduce_channel_halves(scales_full, scales, prior_shape) ||
        !host_ops::build_encode_indexes(symbols, scales, indexes0, 0.0F)) {
        return op_error("P-frame first mask processing");
    }

    auto context_input = spatial_context(
        session, y_hat.value().values, params.value().values, y_height, y_width);
    if (!context_input) return context_input.error();
    auto spatial_params = allocate_tensor(session, dims(256, y_height, y_width));
    if (!spatial_params) return spatial_params.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::spatial_prior),
        {{"context", &context_input.value()}},
        {{"scales_means", &spatial_params.value()}});
    if (!ran) return ran.error();
    const auto spatial_values = spatial_params.value().values;
    if (!host_ops::make_two_way_mask(mask, prior_shape, 1) ||
        !host_ops::process_with_mask(
            y.value().values, spatial_values.first(prior_count),
            spatial_values.subspan(prior_count, prior_count), mask, {}, symbols_full,
            reconstructed, scales_full, 0.0F) ||
        !host_ops::reduce_channel_halves(symbols_full, symbols, prior_shape) ||
        !host_ops::reduce_channel_halves(scales_full, scales, prior_shape) ||
        !host_ops::build_encode_indexes(symbols, scales, indexes1, 0.0F) ||
        !host_ops::add_and_multiply(y_hat.value().values, reconstructed, q_dec_values)) {
        return op_error("P-frame second mask processing");
    }

    auto q_decoder = quant_tensor(session, orchestration.decoder_quantization(), qp, 256);
    auto q_recon = quant_tensor(session, orchestration.reconstruction_quantization(), qp, 320);
    if (!q_decoder) return q_decoder.error();
    if (!q_recon) return q_recon.error();
    auto frame_hat = allocate_tensor(session, dims(3, padded_height, padded_width));
    auto feature = allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    if (!frame_hat) return frame_hat.error();
    if (!feature) return feature.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::synthesis),
        {{"y_hat", &y_hat.value()}, {"context", &context.value()},
         {"q_decoder", &q_decoder.value()}, {"q_recon", &q_recon.value()}},
        {{"frame_hat", &frame_hat.value()}, {"feature", &feature.value()}});
    if (!ran) return ran.error();

    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    auto mode = orchestration.begin_encode(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z.value().shape[2] * z.value().shape[3]);
    auto encoded_z = orchestration.encode_z(z_symbols, qp, per_channel);
    if (!encoded_z) return encoded_z.error();
    auto encoded_y0 = orchestration.encode_y(indexes0);
    if (!encoded_y0) return encoded_y0.error();
    auto encoded_y1 = orchestration.encode_y(indexes1);
    if (!encoded_y1) return encoded_y1.error();
    auto payload = orchestration.finish_encode(
        frame.width(), frame.height(), qp, two_coders, use_frame_reference);
    if (!payload) return payload.error();

    Frame reconstructed_frame;
    if (verify_reconstruction) {
        auto converted = to_yuv420p8(
            frame_hat.value(), frame.width(), frame.height(), frame.timestamp());
        if (!converted) return converted.error();
        reconstructed_frame = std::move(converted.value());
    }
    dpb.frame = std::move(frame_hat.value());
    dpb.feature = std::move(feature.value());
    dpb.semantics.commit(state, true);
    return CodecEncodeResult{
        std::move(payload.value()), std::move(reconstructed_frame), {}, qp};
}

Result<CodecDecodeResult> decode_predicted(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    HostDpb& dpb,
    session_api::IProviderSession& session,
    PredictedOrchestration& orchestration) {
    auto parsed = orchestration.parse_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    if (!dpb.frame) {
        return Error(ErrorCode::invalid_state, "decoder DPB is unavailable",
                     std::string(subsystem));
    }
    auto reference_valid = orchestration.validate_decode_reference(
        info.use_frame_reference, state, dpb.semantics);
    if (!reference_valid) return reference_valid.error();
    if (!info.use_frame_reference && !dpb.feature) {
        return Error(ErrorCode::invalid_state, "P-frame feature storage is unavailable",
                     std::string(subsystem));
    }

    const std::int64_t padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const std::int64_t padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = static_cast<std::int32_t>(padded_height / 16);
    const auto y_width = static_cast<std::int32_t>(padded_width / 16);
    const auto z_height = round_up(y_height, 4) / 4;
    const auto z_width = round_up(y_width, 4) / 4;
    const auto prior_shape = shape4d(128, y_height, y_width);
    const auto prior_count =
        128U * static_cast<std::size_t>(y_height) * static_cast<std::size_t>(y_width);

    auto q_feature = quant_tensor(
        session, orchestration.feature_quantization(), info.qp, 256);
    if (!q_feature) return q_feature.error();
    auto context = allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    auto temporal_context =
        allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    if (!context) return context.error();
    if (!temporal_context) return temporal_context.error();
    auto ran = info.use_frame_reference ?
        run_stage(
            session, orchestration.stage(PredictedStage::reference_frame),
            {{"reference_frame", &*dpb.frame}, {"q_feature", &q_feature.value()}},
            {{"context", &context.value()}, {"temporal_context", &temporal_context.value()}}) :
        run_stage(
            session, orchestration.stage(PredictedStage::reference_feature),
            {{"reference_feature", &*dpb.feature}, {"q_feature", &q_feature.value()}},
            {{"context", &context.value()}, {"temporal_context", &temporal_context.value()}});
    if (!ran) return ran.error();

    auto stream_set = orchestration.begin_decode(info);
    if (!stream_set) return stream_set.error();
    const auto per_channel =
        static_cast<std::size_t>(z_height) * static_cast<std::size_t>(z_width);
    auto z_values = orchestration.decode_z(128 * per_channel, info.qp, per_channel);
    if (!z_values) return z_values.error();
    auto z = allocate_tensor(session, dims(128, z_height, z_width));
    if (!z) return z.error();
    if (z_values.value().size() != z.value().values.size() ||
        !host_ops::int8_to_half(z_values.value(), z.value().values)) {
        return op_error("P-frame z symbol upload");
    }
    auto params = allocate_tensor(session, dims(384, y_height, y_width));
    if (!params) return params.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::prior),
        {{"z_hat", &z.value()}, {"temporal_context", &temporal_context.value()}},
        {{"params", &params.value()}});
    if (!ran) return ran.error();
    const auto q_dec_values = params.value().values.first(prior_count);
    const auto scales_values = params.value().values.subspan(prior_count, prior_count);
    const auto means_values = params.value().values.subspan(2 * prior_count, prior_count);
    if (!host_ops::clamp_min(q_dec_values, 0.5F)) return op_error("clamp_min");

    const auto reduced_count = prior_count / 2;
    std::vector<Half> mask(prior_count);
    std::vector<Half> scales(reduced_count);
    std::vector<std::uint8_t> indexes(reduced_count);
    std::vector<Half> symbols(reduced_count);
    std::vector<Half> reconstructed(prior_count);
    auto y_hat = allocate_tensor(session, dims(128, y_height, y_width));
    if (!y_hat) return y_hat.error();
    if (!host_ops::make_two_way_mask(mask, prior_shape, 0) ||
        !host_ops::reduce_masked_halves(scales_values, mask, scales, prior_shape) ||
        !host_ops::build_decode_indexes(scales, indexes, 0.0F)) {
        return op_error("P-frame first decode index preparation");
    }
    auto decoded0 = orchestration.decode_y(indexes);
    if (!decoded0) return decoded0.error();
    if (decoded0.value().size() != reduced_count ||
        !host_ops::int8_to_half(decoded0.value(), symbols) ||
        !host_ops::restore_two_way(
            symbols, means_values, mask, y_hat.value().values, prior_shape)) {
        return op_error("P-frame first symbol restoration");
    }

    auto context_input = spatial_context(
        session, y_hat.value().values, params.value().values, y_height, y_width);
    if (!context_input) return context_input.error();
    auto spatial_params = allocate_tensor(session, dims(256, y_height, y_width));
    if (!spatial_params) return spatial_params.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::spatial_prior),
        {{"context", &context_input.value()}},
        {{"scales_means", &spatial_params.value()}});
    if (!ran) return ran.error();
    const auto spatial_values = spatial_params.value().values;
    if (!host_ops::make_two_way_mask(mask, prior_shape, 1) ||
        !host_ops::reduce_masked_halves(
            spatial_values.first(prior_count), mask, scales, prior_shape) ||
        !host_ops::build_decode_indexes(scales, indexes, 0.0F)) {
        return op_error("P-frame second decode index preparation");
    }
    auto decoded1 = orchestration.decode_y(indexes);
    if (!decoded1) return decoded1.error();
    if (decoded1.value().size() != reduced_count ||
        !host_ops::int8_to_half(decoded1.value(), symbols) ||
        !host_ops::restore_two_way(
            symbols, spatial_values.subspan(prior_count, prior_count), mask,
            reconstructed, prior_shape) ||
        !host_ops::add_and_multiply(y_hat.value().values, reconstructed, q_dec_values)) {
        return op_error("P-frame second symbol restoration");
    }

    auto q_decoder = quant_tensor(
        session, orchestration.decoder_quantization(), info.qp, 256);
    auto q_recon = quant_tensor(
        session, orchestration.reconstruction_quantization(), info.qp, 320);
    if (!q_decoder) return q_decoder.error();
    if (!q_recon) return q_recon.error();
    auto frame_hat = allocate_tensor(session, dims(3, padded_height, padded_width));
    auto feature = allocate_tensor(session, dims(256, padded_height / 8, padded_width / 8));
    if (!frame_hat) return frame_hat.error();
    if (!feature) return feature.error();
    ran = run_stage(
        session, orchestration.stage(PredictedStage::synthesis),
        {{"y_hat", &y_hat.value()}, {"context", &context.value()},
         {"q_decoder", &q_decoder.value()}, {"q_recon", &q_recon.value()}},
        {{"frame_hat", &frame_hat.value()}, {"feature", &feature.value()}});
    if (!ran) return ran.error();

    auto frame = to_yuv420p8(frame_hat.value(), info.width, info.height, timestamp);
    if (!frame) return frame.error();
    dpb.frame = std::move(frame_hat.value());
    dpb.feature = std::move(feature.value());
    dpb.semantics.commit(state, true);
    return CodecDecodeResult{std::move(frame.value()), {}};
}

class HostBackend final : public CodecBackend {
public:
    explicit HostBackend(std::shared_ptr<session_api::IProviderSession> session)
        : session_(std::move(session)) {}

    ~HostBackend() override {
        if (session_) static_cast<void>(session_->reset());
    }

    Result<void> initialize(const RuntimeConfiguration& configuration) override {
        if (initialized_) {
            return Error(ErrorCode::invalid_state, "DCVC-RT host backend is already initialized",
                         std::string(subsystem));
        }
        try {
            const auto& bundle_root = configuration.artifacts.intra_engine_path;
            std::vector<session_api::ExecutableStage> stages;
            stages.reserve(intra_stage_ids.size() + predicted_stage_ids.size());
            for (const auto ids : {std::span(intra_stage_ids), std::span(predicted_stage_ids)}) {
                for (const auto stage_id : ids) {
                    session_api::ExecutableStageDescriptor descriptor;
                    descriptor.stage_id = std::string(stage_id);
                    descriptor.artifact.codec_id = "dcvc-rt";
                    descriptor.artifact.model_set_id = configuration.artifacts.model_id;
                    descriptor.artifact.component_id =
                        std::string(stage_id.substr(0, stage_id.find('.')));
                    descriptor.artifact.provider_id = configuration.provider.id;
                    descriptor.artifact.precision = "fp16";
                    descriptor.artifact.path = bundle_root.string();
                    auto loaded = session_->load_stage(std::move(descriptor));
                    if (!loaded) return loaded.error();
                    stages.push_back(std::move(loaded.value()));
                }
            }

            IntraOrchestration intra_orchestration;
            auto intra_ready = intra_orchestration.initialize(bundle_root);
            if (!intra_ready) return intra_ready.error();
            auto bound = intra_orchestration.bind_stages(stages);
            if (!bound) return bound.error();
            PredictedOrchestration predicted_orchestration;
            auto predicted_ready = predicted_orchestration.initialize(bundle_root);
            if (!predicted_ready) return predicted_ready.error();
            bound = predicted_orchestration.bind_stages(stages);
            if (!bound) return bound.error();

            intra_orchestration_ = std::move(intra_orchestration);
            predicted_orchestration_ = std::move(predicted_orchestration);
            intra_qp_ = configuration.codec.intra_qp;
            verify_encoder_reconstruction_ = configuration.codec.verify_encoder_reconstruction;
            initialized_ = true;
            return {};
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("DCVC-RT host backend initialization failed: ") + exception.what());
        } catch (...) {
            return backend_error("DCVC-RT host backend initialization failed");
        }
    }

    Result<CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const SequenceStateView& state) override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "DCVC-RT host backend is not initialized",
                         std::string(subsystem));
        }
        try {
            return frame_type == FrameType::intra ?
                encode_intra(frame, intra_qp_, state, encoder_dpb_, *session_,
                             intra_orchestration_, verify_encoder_reconstruction_) :
                encode_predicted(frame, intra_qp_, state, encoder_dpb_, *session_,
                                 predicted_orchestration_, verify_encoder_reconstruction_);
        } catch (const std::exception& exception) {
            return backend_error(std::string("native frame encoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native frame encoding failed");
        }
    }

    Result<CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType frame_type,
        Timestamp timestamp,
        const SequenceStateView& state) override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "DCVC-RT host backend is not initialized",
                         std::string(subsystem));
        }
        try {
            return frame_type == FrameType::intra ?
                decode_intra(payload, timestamp, state, decoder_dpb_, *session_,
                             intra_orchestration_) :
                decode_predicted(payload, timestamp, state, decoder_dpb_, *session_,
                                 predicted_orchestration_);
        } catch (const std::exception& exception) {
            return backend_error(std::string("native frame decoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native frame decoding failed");
        }
    }

    Result<void> flush() override {
        if (!initialized_) {
            return Error(ErrorCode::invalid_state, "DCVC-RT host backend is not initialized",
                         std::string(subsystem));
        }
        return session_->reset();
    }

    void reset() noexcept override {
        static_cast<void>(session_->reset());
        encoder_dpb_.clear();
        decoder_dpb_.clear();
    }

    void reset_encoder() noexcept override {
        static_cast<void>(session_->reset());
        encoder_dpb_.clear();
    }

    void reset_decoder() noexcept override {
        static_cast<void>(session_->reset());
        decoder_dpb_.clear();
    }

private:
    std::shared_ptr<session_api::IProviderSession> session_;
    IntraOrchestration intra_orchestration_;
    PredictedOrchestration predicted_orchestration_;
    HostDpb encoder_dpb_;
    HostDpb decoder_dpb_;
    std::uint32_t intra_qp_{};
    bool verify_encoder_reconstruction_{};
    bool initialized_{};
};

}  // namespace

Result<std::unique_ptr<codec::CodecBackend>> make_host_backend(
    std::shared_ptr<provider::experimental::IProviderSession> provider_session) {
    if (!provider_session) {
        return Error(ErrorCode::invalid_argument,
                     "DCVC-RT host backend requires a provider session",
                     std::string(subsystem));
    }
    try {
        return std::unique_ptr<codec::CodecBackend>(
            std::make_unique<HostBackend>(std::move(provider_session)));
    } catch (const std::bad_alloc&) {
        return Error(ErrorCode::resource_exhausted, "unable to allocate DCVC-RT host backend",
                     std::string(subsystem));
    }
}

}  // namespace nvcr::dcvcrt
