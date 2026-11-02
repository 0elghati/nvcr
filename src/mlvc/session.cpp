#include "session.hpp"

#include "bundle.hpp"
#include "entropy.hpp"
#include "host_ops.hpp"
#include "record.hpp"
#include "sequence.hpp"

#include "nvcr/bitstream/access_unit.hpp"
#include "nvcr/logging/logger.hpp"
#include "nvcr/mlvc/codec.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <utility>

namespace nvcr::mlvc {
namespace {

namespace session_api = provider::experimental;

constexpr std::string_view subsystem = "mlvc";

Error mlvc_error(ErrorCode code, std::string message) {
    return Error(code, std::move(message), std::string(subsystem));
}

struct Tensor final {
    std::vector<std::int64_t> shape;
    session_api::TensorDataType type{};
    session_api::BufferHandle buffer;

    [[nodiscard]] std::span<std::byte> bytes() const { return buffer->host_bytes(); }
};

struct Binding final {
    std::string_view name;
    const Tensor* tensor;
};

std::size_t element_bytes(session_api::TensorDataType type) {
    switch (type) {
    case session_api::TensorDataType::uint8:
    case session_api::TensorDataType::int8: return 1;
    case session_api::TensorDataType::float16: return 2;
    case session_api::TensorDataType::int32:
    case session_api::TensorDataType::float32: return 4;
    }
    return 0;
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dimension : shape) count *= static_cast<std::size_t>(dimension);
    return count;
}

// One loaded canvas: both ONNX parts, the entropy coder, and the zero feature
// MLVC feeds an I frame or a feature reset.
struct Canvas final {
    const BundleModel* model{};
    host_ops::LatentShape latent;
    session_api::TensorDataType float_type{};
    session_api::ExecutableStage encoder;
    session_api::ExecutableStage decoder;
    std::unique_ptr<EntropyCoder> entropy;
    std::vector<std::int64_t> image_shape;
    std::vector<std::int64_t> feature_shape;
    std::vector<std::int64_t> z_shape;
    std::vector<std::int64_t> y_shape;
    Tensor zero_feature;
};

class SharedState final {
public:
    SharedState(
        Bundle bundle,
        std::shared_ptr<session_api::IProviderSession> provider,
        std::string provider_id)
        : bundle_(std::move(bundle)),
          provider_(std::move(provider)),
          provider_id_(std::move(provider_id)) {}

    [[nodiscard]] const Bundle& bundle() const noexcept { return bundle_; }

    [[nodiscard]] Result<Tensor> allocate(
        std::vector<std::int64_t> shape, session_api::TensorDataType type) {
        const auto size = element_count(shape) * element_bytes(type);
        auto buffer = provider_->allocate(size, session_api::MemoryDomain::host);
        if (!buffer) return buffer.error();
        if (buffer.value()->host_bytes().size() < size) {
            return mlvc_error(ErrorCode::incompatible_target,
                              "MLVC requires host-addressable provider buffers");
        }
        return Tensor{std::move(shape), type, std::move(buffer.value())};
    }

    [[nodiscard]] Result<void> run(
        const session_api::ExecutableStage& stage,
        std::initializer_list<Binding> inputs,
        std::initializer_list<Binding> outputs) {
        std::vector<session_api::TensorView> views;
        views.reserve(inputs.size() + outputs.size());
        const auto bind = [&views](const Binding& binding, session_api::TensorAccess access) {
            const auto& tensor = *binding.tensor;
            views.push_back(session_api::TensorView{
                binding.name, tensor.type, tensor.shape, access,
                session_api::BufferSlice{
                    tensor.buffer, 0, element_count(tensor.shape) * element_bytes(tensor.type)}});
        };
        for (const auto& binding : inputs) bind(binding, session_api::TensorAccess::read);
        for (const auto& binding : outputs) bind(binding, session_api::TensorAccess::write);
        auto completion = provider_->submit(stage, views, {});
        if (!completion) return completion.error();
        return completion.value()->wait();
    }

    // Loads each canvas once; encoder and decoder sessions share it.
    [[nodiscard]] Result<std::shared_ptr<const Canvas>> canvas(const BundleModel& model) {
        std::scoped_lock lock(mutex_);
        if (const auto found = canvases_.find(model.canvas_id); found != canvases_.end()) {
            return found->second;
        }
        auto verified = verify_digests(model);
        if (!verified) return verified.error();

        auto canvas = std::make_shared<Canvas>();
        canvas->model = &model;
        canvas->latent = host_ops::latent_shape(model.parameters);
        canvas->float_type = model.precision == Precision::fp16 ?
            session_api::TensorDataType::float16 : session_api::TensorDataType::float32;
        const auto& p = model.parameters;
        const auto& latent = canvas->latent;
        canvas->image_shape = {1, 3, p.height, p.width};
        canvas->feature_shape = {1, p.feature_channels, p.height / p.downsample_feature,
                                 p.width / p.downsample_feature};
        canvas->z_shape = {1, latent.z_channels, latent.z_height, latent.z_width};
        canvas->y_shape = {1, latent.y_channels, latent.y_height, latent.y_width};

        const auto load = [&](const ModelPart& part, std::string_view stage,
                              std::initializer_list<std::pair<std::string_view,
                                                              const std::vector<std::int64_t>*>>
                                  inputs,
                              std::initializer_list<std::pair<std::string_view,
                                                              const std::vector<std::int64_t>*>>
                                  outputs) -> Result<session_api::ExecutableStage> {
            session_api::ExecutableStageDescriptor descriptor;
            descriptor.stage_id = std::string(stage) + "_" + model.canvas_id;
            descriptor.artifact.codec_id = std::string(codec_descriptor().id);
            descriptor.artifact.model_set_id = bundle_.name;
            descriptor.artifact.component_id = part.component_id;
            descriptor.artifact.engine_profile_id = model.canvas_id;
            descriptor.artifact.provider_id = provider_id_;
            descriptor.artifact.precision = model.precision == Precision::fp16 ? "fp16" : "fp32";
            descriptor.artifact.path = bundle_.root.string();
            const auto contract = [&](std::string_view name,
                                      const std::vector<std::int64_t>& shape,
                                      session_api::TensorAccess access) {
                session_api::TensorContract tensor;
                tensor.name = std::string(name);
                tensor.data_type = name == "q_index_shifted" ?
                    session_api::TensorDataType::int32 : canvas->float_type;
                tensor.access = access;
                for (const auto dimension : shape) tensor.dimensions.push_back({dimension, dimension});
                descriptor.tensors.push_back(std::move(tensor));
            };
            for (const auto& [name, shape] : inputs) {
                contract(name, *shape, session_api::TensorAccess::read);
            }
            for (const auto& [name, shape] : outputs) {
                contract(name, *shape, session_api::TensorAccess::write);
            }
            return provider_->load_stage(std::move(descriptor));
        };
        const std::vector<std::int64_t> q_shape{1};
        auto encoder = load(model.encoder, "mlvc_encoder",
                            {{"x", &canvas->image_shape},
                             {"ref_feature", &canvas->feature_shape},
                             {"q_index_shifted", &q_shape}},
                            {{"feature", &canvas->feature_shape},
                             {"z_raw", &canvas->z_shape},
                             {"y_raw_0", &canvas->y_shape},
                             {"y_raw_1", &canvas->y_shape}});
        if (!encoder) return encoder.error();
        auto decoder = load(model.decoder, "mlvc_decoder",
                            {{"z_raw", &canvas->z_shape},
                             {"y_raw_0", &canvas->y_shape},
                             {"y_raw_1", &canvas->y_shape},
                             {"ref_feature", &canvas->feature_shape},
                             {"q_index_shifted", &q_shape}},
                            {{"x_hat", &canvas->image_shape},
                             {"feature", &canvas->feature_shape}});
        if (!decoder) return decoder.error();
        canvas->encoder = std::move(encoder.value());
        canvas->decoder = std::move(decoder.value());

        auto entropy = EntropyCoder::create(model);
        if (!entropy) return entropy.error();
        canvas->entropy = std::move(entropy.value());

        auto zero = allocate(canvas->feature_shape, canvas->float_type);
        if (!zero) return zero.error();
        std::memset(zero.value().bytes().data(), 0, zero.value().bytes().size());
        canvas->zero_feature = std::move(zero.value());

        std::shared_ptr<const Canvas> loaded = std::move(canvas);
        canvases_.emplace(model.canvas_id, loaded);
        return loaded;
    }

private:
    Bundle bundle_;
    std::shared_ptr<session_api::IProviderSession> provider_;
    std::string provider_id_;
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<const Canvas>, std::less<>> canvases_;
};

Result<Tensor> q_tensor(SharedState& state, std::uint32_t q_index_shifted) {
    auto tensor = state.allocate({1}, session_api::TensorDataType::int32);
    if (!tensor) return tensor.error();
    const auto value = static_cast<std::int32_t>(q_index_shifted);
    std::memcpy(tensor.value().bytes().data(), &value, sizeof(value));
    return tensor;
}

Precision precision_of(const Canvas& canvas) {
    return canvas.model->precision;
}

// ModelWrapper output -> _coder.py astype(np.int32): truncation toward zero.
// A non-finite or out-of-range model output is a provider failure, not data.
Result<std::vector<std::int32_t>> to_symbols(const Tensor& tensor, Precision precision) {
    const auto count = element_count(tensor.shape);
    const auto bytes = tensor.bytes();
    std::vector<std::int32_t> symbols(count);
    for (std::size_t index = 0; index < count; ++index) {
        float value{};
        if (precision == Precision::fp16) {
            host_ops::Half half{};
            std::memcpy(&half, bytes.data() + index * sizeof(half), sizeof(half));
            value = host_ops::to_float(half);
        } else {
            std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
        }
        if (!std::isfinite(value) || value <= -2147483649.0F || value >= 2147483648.0F) {
            return mlvc_error(ErrorCode::backend_error,
                              "MLVC encoder produced a non-integer-representable latent");
        }
        symbols[index] = static_cast<std::int32_t>(value);
    }
    return symbols;
}

// Decoded int32 -> float32 -> model precision, as _base_split_model.py and
// ModelWrapper._transform_input convert them.
void from_symbols(const std::vector<std::int32_t>& symbols, Precision precision, const Tensor& tensor) {
    const auto bytes = tensor.bytes();
    for (std::size_t index = 0; index < symbols.size(); ++index) {
        const auto value = static_cast<float>(symbols[index]);
        if (precision == Precision::fp16) {
            const auto half = host_ops::to_half(value);
            std::memcpy(bytes.data() + index * sizeof(half), &half, sizeof(half));
        } else {
            std::memcpy(bytes.data() + index * sizeof(value), &value, sizeof(value));
        }
    }
}

struct Shared final {
    std::shared_ptr<SharedState> state;
    std::shared_ptr<Statistics> statistics;
    std::shared_ptr<Logger> logger;
    RuntimeConfiguration configuration;
};

class EncoderSession final : public IEncoderSession {
public:
    explicit EncoderSession(Shared shared) : shared_(std::move(shared)) {}

    [[nodiscard]] Result<void> send_frame(const Frame& frame) override {
        if (flushed_) return mlvc_error(ErrorCode::invalid_state, "encoder is flushed");
        if (frame.pixel_format() != PixelFormat::yuv420p8) {
            return mlvc_error(ErrorCode::invalid_argument, "MLVC encodes 8-bit I420 frames");
        }
        try {
            auto packet = encode(frame);
            if (!packet) return packet.error();
            outputs_.push_back(std::move(packet.value()));
            return {};
        } catch (const std::bad_alloc&) {
            return mlvc_error(ErrorCode::resource_exhausted, "MLVC encoder ran out of memory");
        } catch (const std::exception& exception) {
            return mlvc_error(ErrorCode::internal_error, exception.what());
        }
    }

    [[nodiscard]] Result<Packet> receive_access_unit() override {
        if (!outputs_.empty()) {
            Packet output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) return mlvc_error(ErrorCode::end_of_stream, "encoder is drained");
        return mlvc_error(ErrorCode::try_again, "no encoded output available");
    }

    // MLVC has no lookahead, so flushing emits nothing new.
    [[nodiscard]] Result<void> flush() override {
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        outputs_.clear();
        references_.clear();
        if (scheduler_) scheduler_->reset();
        flushed_ = false;
        shared_.logger->log(LogLevel::debug, "MLVC encoder state reset");
        return {};
    }

private:
    [[nodiscard]] Result<void> select_canvas(const Frame& frame) {
        if (canvas_ && frame.width() == placement_.frame_width &&
            frame.height() == placement_.frame_height) {
            return {};
        }
        if (scheduler_ && scheduler_->started()) {
            return mlvc_error(ErrorCode::invalid_argument,
                              "MLVC frame dimensions changed; reset the encoder first");
        }
        auto model = select_model(shared_.state->bundle(), frame.width(), frame.height());
        if (!model) return model.error();
        auto canvas = shared_.state->canvas(*model.value());
        if (!canvas) return canvas.error();
        auto placement = host_ops::place(frame.width(), frame.height(), model.value()->parameters);
        if (!placement) return placement.error();

        const auto& parameters = model.value()->parameters;
        const auto q_index = shared_.configuration.codec.intra_qp;
        if (q_index >= parameters.qp_num) {
            return mlvc_error(ErrorCode::invalid_argument,
                              "MLVC Q index must be below " + std::to_string(parameters.qp_num));
        }
        auto loop = LoopParameters::from_model(parameters);
        if (shared_.configuration.codec.gop_size != use_model_iframe_period) {
            loop.iframe_period = shared_.configuration.codec.gop_size;
        }
        canvas_ = std::move(canvas.value());
        placement_ = placement.value();
        scheduler_.emplace(loop);
        references_.clear();
        shared_.logger->log(LogLevel::info,
                            "MLVC encoder uses canvas " + model.value()->canvas_id);
        return {};
    }

    [[nodiscard]] Result<Packet> encode(const Frame& frame) {
        const auto started = std::chrono::steady_clock::now();
        auto selected = select_canvas(frame);
        if (!selected) return selected.error();
        const auto& canvas = *canvas_;
        auto& state = *shared_.state;
        const auto precision = precision_of(canvas);
        const auto q_index = shared_.configuration.codec.intra_qp;

        const auto plan = scheduler_->plan();
        if (plan.frame_class == FrameClass::intra) references_.clear();
        const Tensor* reference = &canvas.zero_feature;
        if (plan.reference && !plan.feature_reset) {
            reference = references_.find(*plan.reference);
            if (reference == nullptr) {
                return mlvc_error(ErrorCode::internal_error, "MLVC encoder reference is missing");
            }
        }

        auto x = state.allocate(canvas.image_shape, canvas.float_type);
        if (!x) return x.error();
        host_ops::prepare_input(frame.data(), placement_, precision, x.value().bytes());
        auto q = q_tensor(state, shifted_q_index(canvas.model->parameters, q_index, plan.coded_index));
        if (!q) return q.error();
        auto feature = state.allocate(canvas.feature_shape, canvas.float_type);
        if (!feature) return feature.error();
        auto z = state.allocate(canvas.z_shape, canvas.float_type);
        if (!z) return z.error();
        auto y_0 = state.allocate(canvas.y_shape, canvas.float_type);
        if (!y_0) return y_0.error();
        auto y_1 = state.allocate(canvas.y_shape, canvas.float_type);
        if (!y_1) return y_1.error();
        auto ran = state.run(
            canvas.encoder,
            {{"x", &x.value()}, {"ref_feature", reference}, {"q_index_shifted", &q.value()}},
            {{"feature", &feature.value()}, {"z_raw", &z.value()},
             {"y_raw_0", &y_0.value()}, {"y_raw_1", &y_1.value()}});
        if (!ran) return ran.error();

        LatentSymbols symbols;
        for (const auto& [tensor, target] : {
                 std::pair{&z.value(), &symbols.z}, std::pair{&y_0.value(), &symbols.y_0},
                 std::pair{&y_1.value(), &symbols.y_1}}) {
            auto converted = to_symbols(*tensor, precision);
            if (!converted) return converted.error();
            *target = std::move(converted.value());
        }
        auto payload = canvas.entropy->encode(symbols, q_index);
        if (!payload) return payload.error();
        auto record = serialize_record(static_cast<std::int32_t>(q_index), payload.value());
        if (!record) return record.error();

        // The encoder references its own feature output, not a decoded one.
        references_.save(plan.coded_index, std::move(feature.value()), plan.mark_as_ltr);
        scheduler_->commit(plan);

        const auto type = plan.frame_class == FrameClass::intra ?
            FrameType::intra : FrameType::predicted;
        AccessUnit access_unit{
            canvas.model->stream_identity, frame.width(), frame.height(), q_index, type,
            type == FrameType::intra, std::move(record.value())};
        auto serialized = AccessUnitIO::serialize(
            access_unit, shared_.configuration.runtime.max_packet_bytes);
        if (!serialized) return serialized.error();
        Packet packet(std::move(serialized.value()), frame.timestamp(), type);
        shared_.statistics->record_encode(
            packet.size(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started));
        return packet;
    }

    Shared shared_;
    std::shared_ptr<const Canvas> canvas_;
    host_ops::Placement placement_{};
    std::optional<FrameScheduler> scheduler_;
    ReferenceStore<Tensor> references_;
    std::deque<Packet> outputs_;
    bool flushed_{false};
};

class DecoderSession final : public IDecoderSession {
public:
    explicit DecoderSession(Shared shared) : shared_(std::move(shared)) {}

    [[nodiscard]] Result<void> send_access_unit(const Packet& packet) override {
        if (flushed_) return mlvc_error(ErrorCode::invalid_state, "decoder is flushed");
        try {
            auto frame = decode(packet);
            if (!frame) return frame.error();
            outputs_.push_back(std::move(frame.value()));
            return {};
        } catch (const std::bad_alloc&) {
            return mlvc_error(ErrorCode::resource_exhausted, "MLVC decoder ran out of memory");
        } catch (const std::exception& exception) {
            return mlvc_error(ErrorCode::internal_error, exception.what());
        }
    }

    [[nodiscard]] Result<Frame> receive_frame() override {
        if (!outputs_.empty()) {
            Frame output = std::move(outputs_.front());
            outputs_.pop_front();
            return output;
        }
        if (flushed_) return mlvc_error(ErrorCode::end_of_stream, "decoder is drained");
        return mlvc_error(ErrorCode::try_again, "no decoded output available");
    }

    [[nodiscard]] Result<void> flush() override {
        flushed_ = true;
        return {};
    }

    [[nodiscard]] Result<void> reset() override {
        outputs_.clear();
        references_.clear();
        if (scheduler_) scheduler_->reset();
        last_output_.reset();
        flushed_ = false;
        shared_.logger->log(LogLevel::debug, "MLVC decoder state reset");
        return {};
    }

private:
    // An intra access unit names its canvas through the stream identity; a
    // predicted one must continue the current sequence unchanged.
    [[nodiscard]] Result<void> select_canvas(const AccessUnit& access_unit) {
        if (access_unit.frame_type != FrameType::intra) {
            if (!scheduler_ || !scheduler_->started()) {
                return mlvc_error(ErrorCode::malformed_bitstream,
                                  "MLVC stream must start with an intra access unit");
            }
            if (access_unit.model_id != canvas_->model->stream_identity ||
                access_unit.width != placement_.frame_width ||
                access_unit.height != placement_.frame_height) {
                return mlvc_error(ErrorCode::malformed_bitstream,
                                  "MLVC predicted access unit changes model or dimensions");
            }
            return {};
        }
        const BundleModel* model = nullptr;
        for (const auto& candidate : shared_.state->bundle().models) {
            if (candidate.stream_identity == access_unit.model_id) model = &candidate;
        }
        if (model == nullptr) {
            return mlvc_error(ErrorCode::malformed_bitstream,
                              "access-unit model " + access_unit.model_id +
                                  " is not in the configured MLVC bundle");
        }
        auto placement = host_ops::place(access_unit.width, access_unit.height, model->parameters);
        if (!placement) {
            return mlvc_error(ErrorCode::malformed_bitstream,
                              std::string(placement.error().message()));
        }
        if (!canvas_ || canvas_->model != model) {
            auto canvas = shared_.state->canvas(*model);
            if (!canvas) return canvas.error();
            canvas_ = std::move(canvas.value());
            scheduler_.emplace(LoopParameters::from_model(model->parameters));
        }
        placement_ = placement.value();
        return {};
    }

    [[nodiscard]] Result<Frame> decode(const Packet& packet) {
        if (packet.size() > shared_.configuration.runtime.max_packet_bytes) {
            return mlvc_error(ErrorCode::resource_exhausted, "packet exceeds configured limit");
        }
        if (!AccessUnitIO::has_magic(packet.data())) {
            return mlvc_error(ErrorCode::malformed_bitstream, "MLVC packets must be NVAU access units");
        }
        auto parsed = AccessUnitIO::deserialize(
            packet.data(), shared_.configuration.runtime.max_packet_bytes);
        if (!parsed) return parsed.error();
        const auto& access_unit = parsed.value();
        const bool intra = access_unit.frame_type == FrameType::intra;
        if (access_unit.frame_type != packet.frame_type() || access_unit.reset_state != intra) {
            return mlvc_error(ErrorCode::malformed_bitstream,
                              "packet frame type and access-unit flags disagree");
        }
        auto record = parse_record(access_unit.payload);
        if (!record) return record.error();
        const auto started = std::chrono::steady_clock::now();
        auto selected = select_canvas(access_unit);
        if (!selected) return selected.error();
        const auto& canvas = *canvas_;
        auto& state = *shared_.state;
        const auto precision = precision_of(canvas);
        const auto& parameters = canvas.model->parameters;
        const auto plan = scheduler_->plan(intra);

        if (record.value().q_index == dropped_frame_q_index) {
            // A dropped frame repeats the last presentation without touching
            // reference state (_frame_loop.py frame dropping).
            if (!plan.droppable() || !last_output_) {
                return mlvc_error(ErrorCode::malformed_bitstream,
                                  "MLVC frame cannot be dropped at this position");
            }
            return Frame::copy_from(last_output_->width(), last_output_->height(),
                                    PixelFormat::yuv420p8, last_output_->data(),
                                    packet.timestamp());
        }
        const auto q_index = static_cast<std::uint32_t>(record.value().q_index);
        if (q_index >= parameters.qp_num || access_unit.qp != q_index) {
            return mlvc_error(ErrorCode::malformed_bitstream, "MLVC record has an invalid Q index");
        }
        auto symbols = canvas.entropy->decode(record.value().payload, q_index);
        if (!symbols) return symbols.error();

        if (intra) references_.clear();
        const Tensor* reference = &canvas.zero_feature;
        if (plan.reference && !plan.feature_reset) {
            reference = references_.find(*plan.reference);
            if (reference == nullptr) {
                return mlvc_error(ErrorCode::malformed_bitstream,
                                  "MLVC reference feature is not available");
            }
        }

        auto z = state.allocate(canvas.z_shape, canvas.float_type);
        if (!z) return z.error();
        auto y_0 = state.allocate(canvas.y_shape, canvas.float_type);
        if (!y_0) return y_0.error();
        auto y_1 = state.allocate(canvas.y_shape, canvas.float_type);
        if (!y_1) return y_1.error();
        from_symbols(symbols.value().z, precision, z.value());
        from_symbols(symbols.value().y_0, precision, y_0.value());
        from_symbols(symbols.value().y_1, precision, y_1.value());
        auto q = q_tensor(state, shifted_q_index(parameters, q_index, plan.coded_index));
        if (!q) return q.error();
        auto x_hat = state.allocate(canvas.image_shape, canvas.float_type);
        if (!x_hat) return x_hat.error();
        auto feature = state.allocate(canvas.feature_shape, canvas.float_type);
        if (!feature) return feature.error();
        auto ran = state.run(
            canvas.decoder,
            {{"z_raw", &z.value()}, {"y_raw_0", &y_0.value()}, {"y_raw_1", &y_1.value()},
             {"ref_feature", reference}, {"q_index_shifted", &q.value()}},
            {{"x_hat", &x_hat.value()}, {"feature", &feature.value()}});
        if (!ran) return ran.error();

        auto frame = Frame::create(
            placement_.frame_width, placement_.frame_height, PixelFormat::yuv420p8,
            packet.timestamp());
        if (!frame) return frame.error();
        host_ops::finish_output(x_hat.value().bytes(), placement_, precision, frame.value().data());

        references_.save(plan.coded_index, std::move(feature.value()), plan.mark_as_ltr);
        scheduler_->commit(plan);
        auto retained = Frame::copy_from(
            frame.value().width(), frame.value().height(), PixelFormat::yuv420p8,
            frame.value().data(), frame.value().timestamp());
        if (!retained) return retained.error();
        last_output_ = std::move(retained.value());
        shared_.statistics->record_decode(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started));
        return frame;
    }

    Shared shared_;
    std::shared_ptr<const Canvas> canvas_;
    host_ops::Placement placement_{};
    std::optional<FrameScheduler> scheduler_;
    ReferenceStore<Tensor> references_;
    std::optional<Frame> last_output_;
    std::deque<Frame> outputs_;
    bool flushed_{false};
};

}  // namespace

Result<codec::Sessions> make_sessions(
    const RuntimeConfiguration& configuration,
    std::shared_ptr<session_api::IProviderSession> provider_session) {
    if (!provider_session) {
        return mlvc_error(ErrorCode::dependency_unavailable, "MLVC requires a provider session");
    }
    const auto& root = configuration.artifacts.intra_engine_path;
    if (root.empty()) {
        return mlvc_error(ErrorCode::missing_artifact,
                          "MLVC requires a bundle directory (artifacts.intra_engine_path)");
    }
    try {
        auto bundle = load_bundle(root);
        if (!bundle) return bundle.error();
        Shared shared;
        shared.state = std::make_shared<SharedState>(
            std::move(bundle.value()), std::move(provider_session), configuration.provider.id);
        shared.statistics = std::make_shared<Statistics>();
        shared.logger = make_default_logger_factory(configuration.runtime.log_level)->create("mlvc");
        shared.configuration = configuration;
        shared.logger->log(LogLevel::info, "MLVC bundle " + shared.state->bundle().name + " (" +
                                               shared.state->bundle().target_device + ") loaded");
        codec::Sessions sessions;
        sessions.encoder = std::make_unique<EncoderSession>(shared);
        sessions.decoder = std::make_unique<DecoderSession>(shared);
        sessions.statistics = std::move(shared.statistics);
        return sessions;
    } catch (const std::bad_alloc&) {
        return mlvc_error(ErrorCode::resource_exhausted, "unable to allocate MLVC sessions");
    } catch (const std::exception& exception) {
        return mlvc_error(ErrorCode::internal_error, exception.what());
    }
}

}  // namespace nvcr::mlvc
