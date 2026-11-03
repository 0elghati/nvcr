// DCVC-UF on the generic TensorRT provider session.
//
// Follows pinned upstream's fused inference (dmci_proxy.cpp, dmc_hts_proxy.cpp)
// step for step: the 20 TensorRT stages replace upstream's fused networks, the
// ported upstream glue kernels run between them, and UF rANS runs on the host.
#include "tensorrt_session.hpp"

#include "backend/tensorrt/glue.hpp"
#include "entropy/uf_rans_codec.hpp"
#include "../provider/tensorrt/session.hpp"

#include "../common/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace nvcr::dcvcuf {
namespace {

namespace trt = nvcr::tensorrt;
namespace session_api = provider::experimental;
using Half = std::uint16_t;
using trt::DeviceTensor;

constexpr int ch_y = 256;
constexpr int ch_z = 128;
constexpr int qp_count = 64;
constexpr int min_symbols_per_stream = 32768;  // upstream def_const.h
constexpr int max_entropy_parallel = 8;

Error failure(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "dcvcuf-tensorrt");
}

Result<void> check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return {};
    return failure(ErrorCode::backend_error,
                   std::string(operation) + ": " + cudaGetErrorString(status));
}

int ceil_to(int value, int multiple) { return (value + multiple - 1) / multiple * multiple; }

nvinfer1::Dims dims(int channels, int height, int width) {
    nvinfer1::Dims shape{};
    shape.nbDims = 4;
    shape.d[0] = 1;
    shape.d[1] = channels;
    shape.d[2] = height;
    shape.d[3] = width;
    return shape;
}

int channels(const DeviceTensor& tensor) { return static_cast<int>(tensor.shape.d[1]); }
int height(const DeviceTensor& tensor) { return static_cast<int>(tensor.shape.d[2]); }
int width(const DeviceTensor& tensor) { return static_cast<int>(tensor.shape.d[3]); }
Half* data(const DeviceTensor& tensor) { return static_cast<Half*>(tensor.storage.data); }

glue::HalfTensor view(const DeviceTensor& tensor, int first = 0, int count = 0,
                      glue::Layout layout = glue::Layout::nhwc) {
    const int all = channels(tensor);
    return {data(tensor) + first, count != 0 ? count : all, height(tensor), width(tensor), layout,
            count != 0 && count != all ? all : 0};
}

glue::ConstHalfTensor in(const glue::HalfTensor& tensor) {
    return {tensor.data, tensor.channels, tensor.height, tensor.width, tensor.layout,
            tensor.stride};
}

DeviceTensor named(const DeviceTensor& tensor, std::string name) {
    auto copy = tensor;
    copy.name = std::move(name);
    return copy;
}

// Upstream get_mask_4x: four 64-channel groups, each one 2x2 phase, NHWC bool.
std::array<std::vector<std::uint8_t>, 4> masks(int height, int width) {
    constexpr std::array<std::array<int, 4>, 4> order{{{0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1},
                                                       {1, 0, 3, 2}}};
    std::array<std::vector<std::uint8_t>, 4> output;
    for (int step = 0; step < 4; ++step) {
        auto& mask = output[static_cast<std::size_t>(step)];
        mask.resize(static_cast<std::size_t>(height) * width * ch_y);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int phase = (y % 2) * 2 + x % 2;
                for (int c = 0; c < ch_y; ++c) {
                    const auto group = static_cast<std::size_t>(c / (ch_y / 4));
                    mask[(static_cast<std::size_t>(y) * width + x) * ch_y + c] =
                        order[static_cast<std::size_t>(step)][group] == phase;
                }
            }
        }
    }
    return output;
}

// Per-model tables from tables.bin (see scripts/backends/dcvcuf/add_tensorrt_tables.py).
struct Tables final {
    std::map<std::string, std::vector<Half>> q;  // name -> 64 rows
    std::map<std::string, int> q_width;
    UfRansCdfTable z;
    UfRansCdfTable y;
};

struct TableReader final {
    std::vector<char> bytes;
    std::size_t offset{};

    template <class T>
    std::vector<T> take(std::size_t count) {
        std::vector<T> values(count);
        const auto size = count * sizeof(T);
        if (offset + size > bytes.size()) throw std::runtime_error("tables.bin is truncated");
        std::memcpy(values.data(), bytes.data() + offset, size);
        offset += size;
        return values;
    }

    UfRansCdfTable cdf(std::size_t rows) {
        auto values = take<std::int32_t>(rows * 19);
        auto sizes = take<std::int32_t>(rows);
        UfRansCdfTable table;
        table.sizes = sizes;
        for (std::size_t row = 0; row < rows; ++row) {
            table.values.emplace_back(values.begin() + static_cast<std::ptrdiff_t>(row * 19),
                                      values.begin() + static_cast<std::ptrdiff_t>(row * 19 + 19));
        }
        return table;
    }

    void q(Tables& tables, const std::string& name, int width) {
        tables.q[name] = take<Half>(static_cast<std::size_t>(qp_count) * width);
        tables.q_width[name] = width;
    }
};

std::pair<Tables, Tables> read_tables(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    TableReader reader{{std::istreambuf_iterator<char>(file), {}}};
    Tables image;
    Tables video;
    reader.q(image, "q_scale_enc", 384);
    reader.q(image, "q_scale_dec", 384);
    reader.q(image, "q_scale_y_enc", 256);
    reader.q(image, "q_scale_y_dec", 256);
    image.z = reader.cdf(8192);
    image.y = reader.cdf(128);
    reader.q(video, "q_encoder", 512);
    reader.q(video, "q_decoder", 512);
    reader.q(video, "q_feature", 512);
    video.z = reader.cdf(8192);
    video.y = reader.cdf(128);
    if (reader.offset != reader.bytes.size()) throw std::runtime_error("tables.bin has extra bytes");
    return {std::move(image), std::move(video)};
}

class TensorRTSession final : public ExecutionSession {
public:
    explicit TensorRTSession(std::shared_ptr<trt::TensorRTExecutionSession> session)
        : session_(std::move(session)),
          owner_("dcvcuf-tensorrt-" + std::to_string(++next_owner_)) {}

    Result<void> initialize() {
        try {
            const auto root = session_->bundle_root();
            std::tie(image_, video_) = read_tables(root / "tables.bin");
            auto stages = detail::json::parse_file(root / "stages.json", "dcvcuf-tensorrt");
            if (!stages) return stages.error();
            const auto* list = stages.value().find("stages");
            if (list == nullptr) return failure(ErrorCode::missing_artifact, "stages.json has no stages");
            for (const auto& [name, tensors] : list->object_value) {
                session_api::ExecutableStageDescriptor descriptor;
                descriptor.stage_id = name + ".plan";
                descriptor.artifact.provider_id = "tensorrt";
                descriptor.artifact.codec_id = "dcvc-uf";
                descriptor.artifact.component_id = descriptor.stage_id;
                descriptor.artifact.path = (root / descriptor.stage_id).string();
                for (const auto& tensor : tensors.array_value) {
                    session_api::TensorContract contract;
                    contract.name = tensor.find("name")->string_value;
                    contract.data_type = session_api::TensorDataType::float16;
                    contract.access = tensor.find("access")->string_value == "read" ?
                        session_api::TensorAccess::read : session_api::TensorAccess::write;
                    const auto* profile = tensor.find("profile");
                    for (std::size_t axis = 0; axis < 4; ++axis) {
                        contract.dimensions.push_back(
                            {static_cast<std::int32_t>(
                                 *profile->find("min")->array_value[axis].integer_value),
                             static_cast<std::int32_t>(
                                 *profile->find("max")->array_value[axis].integer_value)});
                    }
                    descriptor.tensors.push_back(std::move(contract));
                }
                auto loaded = session_->load_stage(std::move(descriptor));
                if (!loaded) return loaded.error();
                stages_.emplace(name, loaded.value());
            }
            return session_->finalize_stages();
        } catch (const std::exception& exception) {
            return failure(ErrorCode::missing_artifact, exception.what());
        }
    }

    std::string_view owner_id() const noexcept override { return owner_; }
    Result<session_api::BufferHandle> allocate(std::size_t, session_api::MemoryDomain) override {
        return failure(ErrorCode::not_implemented, "DCVC-UF TensorRT buffers are codec-owned");
    }
    Result<session_api::ExecutableStage> load_stage(session_api::ExecutableStageDescriptor) override {
        return failure(ErrorCode::not_implemented, "DCVC-UF loads its stages from stages.json");
    }
    Result<session_api::Completion> submit(
        const session_api::ExecutableStage&, std::span<const session_api::TensorView>,
        std::span<const session_api::ExecutionDependency>) override {
        return failure(ErrorCode::not_implemented, "DCVC-UF runs whole frames, not single stages");
    }
    Result<void> reset() override { return {}; }
    Result<void> reset_encoder_state() override { return {}; }
    Result<void> reset_decoder_state() override { return {}; }

    Result<Encoded> encode_intra(const Frame& frame, std::uint32_t qp) override {
        return guarded<Encoded>([&]() -> Result<Encoded> { return encode_i(frame, qp); });
    }
    Result<Frame> decode_intra(std::span<const std::byte> bitstream, std::uint8_t parallel,
                               std::uint32_t qp, std::uint32_t width, std::uint32_t height,
                               Timestamp timestamp) override {
        return guarded<Frame>([&]() -> Result<Frame> {
            return decode_i(bitstream, parallel, qp, width, height, timestamp);
        });
    }
    Result<Encoded> encode_predicted(std::span<const Frame> frames, std::uint32_t qp,
                                     bool reset_feature_memory) override {
        if (frames.size() != 8)
            return failure(ErrorCode::invalid_argument, "HT-S encoding requires eight frames");
        return guarded<Encoded>([&]() -> Result<Encoded> {
            return encode_p(frames, qp, reset_feature_memory);
        });
    }
    Result<std::vector<Frame>> decode_predicted(
        std::span<const std::byte> bitstream, std::uint8_t parallel, std::uint32_t qp,
        std::uint32_t width, std::uint32_t height, std::span<const Timestamp> timestamps,
        bool reset_feature_memory) override {
        if (timestamps.empty() || timestamps.size() > 8)
            return failure(ErrorCode::invalid_argument, "HT-S timestamp count must be in [1, 8]");
        return guarded<std::vector<Frame>>([&]() -> Result<std::vector<Frame>> {
            return decode_p(bitstream, parallel, qp, width, height, timestamps,
                            reset_feature_memory);
        });
    }

private:
    // One direction's reference state: the HT-S feature memory and context.
    struct Reference final {
        std::optional<DeviceTensor> feature_i;  // unshuffled I frame or reset feature
        std::optional<DeviceTensor> feature_p;  // last HT-S decoder feature
        std::optional<DeviceTensor> memory;
        std::optional<DeviceTensor> context;
        bool memory_has_value{false};
    };

    template <class T, class Function>
    Result<T> guarded(Function&& function) {
        try {
            const auto scope = session_->allocation_scope();
            auto output = function();
            if (!output) return output;
            auto synchronized = trt::synchronize_stream(stream(), "DCVC-UF frame");
            if (!synchronized) return synchronized.error();
            return output;
        } catch (const std::exception& exception) {
            return failure(ErrorCode::backend_error, exception.what());
        }
    }

    cudaStream_t stream() const { return session_->stream(); }

    // Buffers that depend only on the frame size.
    Result<void> prepare(int frame_width, int frame_height) {
        if (frame_width == width_ && frame_height == height_) return {};
        width_ = frame_width;
        height_ = frame_height;
        h16_ = ceil_to(height_, 16) / 16;
        w16_ = ceil_to(width_, 16) / 16;
        const auto host_masks = masks(h16_, w16_);
        for (std::size_t step = 0; step < 4; ++step) {
            auto mask = trt::allocate_device_tensor("mask", dims(ch_y, h16_, w16_));
            if (!mask) return mask.error();
            auto copied = trt::copy_async(mask.value().storage.data, host_masks[step].data(),
                                          host_masks[step].size(), cudaMemcpyHostToDevice,
                                          stream(), "DCVC-UF mask upload");
            if (!copied) return copied.error();
            masks_[step] = std::move(mask.value());
        }
        auto scratch = trt::allocate_device_tensor(
            "scratch", dims(1, 1, glue::conditional_scratch_ints * 2));
        if (!scratch) return scratch.error();
        scratch_ = std::move(scratch.value());
        auto ones = upload_row(std::vector<Half>(ch_y, 0x3c00), "ones");
        if (!ones) return ones.error();
        ones_ = std::move(ones.value());
        return {};
    }

    const bool* mask(std::size_t step) const {
        return static_cast<const bool*>(masks_[step]->storage.data);
    }
    glue::MaskTensor mask_view(std::size_t step) const {
        return {mask(step), ch_y, h16_, w16_, glue::Layout::nhwc};
    }
    int* scratch() const { return static_cast<int*>(scratch_->storage.data); }

    Result<DeviceTensor> tensor(const char* name, int c, int h, int w) {
        return trt::allocate_device_tensor(name, dims(c, h, w));
    }

    Result<DeviceTensor> upload_row(const std::vector<Half>& row, const char* name) {
        auto output = tensor(name, static_cast<int>(row.size()), 1, 1);
        if (!output) return output.error();
        auto copied = trt::copy_async(output.value().storage.data, row.data(),
                                      row.size() * sizeof(Half), cudaMemcpyHostToDevice,
                                      stream(), "DCVC-UF table upload");
        if (!copied) return copied.error();
        return output;
    }

    Result<DeviceTensor> q_row(const Tables& tables, const char* table, std::uint32_t qp,
                               const char* name) {
        const int row_width = tables.q_width.at(table);
        const auto& values = tables.q.at(table);
        return upload_row(std::vector<Half>(values.begin() + static_cast<std::ptrdiff_t>(qp) * row_width,
                                            values.begin() + static_cast<std::ptrdiff_t>(qp + 1) * row_width),
                          name);
    }

    // Runs one stage and returns its outputs by name.
    Result<std::map<std::string, DeviceTensor>> run(
        const char* stage, std::initializer_list<std::pair<const char*, const DeviceTensor*>> inputs) {
        std::vector<DeviceTensor> bound;
        bound.reserve(inputs.size());
        for (const auto& [name, input] : inputs) bound.push_back(named(*input, name));
        std::vector<const DeviceTensor*> pointers;
        for (const auto& input : bound) pointers.push_back(&input);
        auto outputs = session_->run_stage(stages_.at(stage), pointers);
        if (!outputs) return outputs.error();
        std::map<std::string, DeviceTensor> result;
        for (auto& output : outputs.value()) result.emplace(output.name, std::move(output));
        return result;
    }

    Result<DeviceTensor> upload_frames(std::span<const Frame> frames, glue::Layout layout) {
        const auto frame_bytes = static_cast<std::size_t>(width_) * height_ * 3 / 2;
        std::vector<std::uint8_t> host(frame_bytes * frames.size());
        for (std::size_t index = 0; index < frames.size(); ++index) {
            if (frames[index].width() != static_cast<std::uint32_t>(width_) ||
                frames[index].height() != static_cast<std::uint32_t>(height_))
                return failure(ErrorCode::invalid_argument, "DCVC-UF frames must share one size");
            std::memcpy(host.data() + index * frame_bytes, frames[index].data().data(), frame_bytes);
        }
        auto planes = tensor("planes", 1, 1, static_cast<int>((host.size() + 1) / 2));
        if (!planes) return planes.error();
        auto copied = trt::copy_async(planes.value().storage.data, host.data(), host.size(),
                                      cudaMemcpyHostToDevice, stream(), "DCVC-UF frame upload");
        if (!copied) return copied.error();
        auto output = tensor("frames", 3 * static_cast<int>(frames.size()), h16_ * 16, w16_ * 16);
        if (!output) return output.error();
        auto converted = check(glue::yuv420_to_tensor(
                                   view(output.value(), 0, 0, layout),
                                   static_cast<const std::uint8_t*>(planes.value().storage.data),
                                   width_, height_, stream()),
                               "yuv420_to_tensor");
        if (!converted) return converted.error();
        auto synchronized = trt::synchronize_stream(stream(), "DCVC-UF frame upload");
        if (!synchronized) return synchronized.error();
        return output;
    }

    Result<Frame> download_frame(const DeviceTensor& frame, Timestamp timestamp) {
        const auto frame_bytes = static_cast<std::size_t>(width_) * height_ * 3 / 2;
        auto planes = tensor("planes", 1, 1, static_cast<int>((frame_bytes + 1) / 2));
        if (!planes) return planes.error();
        auto converted = check(
            glue::tensor_to_yuv420(static_cast<std::uint8_t*>(planes.value().storage.data),
                                   in(view(frame, 0, 0, glue::Layout::nchw)), width_, height_,
                                   stream()),
            "tensor_to_yuv420");
        if (!converted) return converted.error();
        std::vector<std::byte> host(frame_bytes);
        auto copied = trt::copy_async(host.data(), planes.value().storage.data, frame_bytes,
                                      cudaMemcpyDeviceToHost, stream(), "DCVC-UF frame download");
        if (!copied) return copied.error();
        auto synchronized = trt::synchronize_stream(stream(), "DCVC-UF frame download");
        if (!synchronized) return synchronized.error();
        return Frame::copy_from(static_cast<std::uint32_t>(width_), static_cast<std::uint32_t>(height_),
                                PixelFormat::yuv420p8, host, timestamp);
    }

    // pre-shuffle 192 channels at 1/8 -> clamped 3-channel frame (upstream pixel_shuffle_8).
    Result<DeviceTensor> shuffle(const DeviceTensor& pre) {
        auto frame = tensor("frame", 3, h16_ * 16, w16_ * 16);
        if (!frame) return frame.error();
        auto shuffled = check(glue::pixel_shuffle_8(view(frame.value(), 0, 0, glue::Layout::nchw),
                                                    in(view(pre)), true, stream()),
                              "pixel_shuffle_8");
        if (!shuffled) return shuffled.error();
        return frame;
    }

    Result<DeviceTensor> pad_y(const DeviceTensor& y) {
        const int ph = ceil_to(h16_, 4);
        const int pw = ceil_to(w16_, 4);
        if (ph == h16_ && pw == w16_) return y;
        auto padded = tensor("y_padded", ch_y, ph, pw);
        if (!padded) return padded.error();
        auto done = check(glue::replicate_pad(view(padded.value()), in(view(y)), ph - h16_,
                                              pw - w16_, stream()),
                          "replicate_pad");
        if (!done) return done.error();
        return padded;
    }

    // Crops a pad4 hyper output to the latent size (crop_hyper_params).
    Result<DeviceTensor> crop(const DeviceTensor& padded) {
        if (height(padded) == h16_ && width(padded) == w16_) return padded;
        auto output = tensor("cropped", channels(padded), h16_, w16_);
        if (!output) return output.error();
        auto done = check(glue::slice(view(output.value()), in(view(padded)), stream()), "slice");
        if (!done) return done.error();
        return output;
    }

    // Compacted symbols or indexes: total count and the first `count` elements.
    template <class T>
    Result<std::vector<T>> fetch(const DeviceTensor& compacted, const DeviceTensor& total) {
        int count = 0;
        auto copied = trt::copy_async(&count, total.storage.data, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream(), "DCVC-UF symbol count");
        if (!copied) return copied.error();
        auto synchronized = trt::synchronize_stream(stream(), "DCVC-UF symbol count");
        if (!synchronized) return synchronized.error();
        std::vector<T> host(static_cast<std::size_t>(count));
        if (count > 0) {
            copied = trt::copy_async(host.data(), compacted.storage.data, host.size() * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream(), "DCVC-UF symbols");
            if (!copied) return copied.error();
            synchronized = trt::synchronize_stream(stream(), "DCVC-UF symbols");
            if (!synchronized) return synchronized.error();
        }
        return host;
    }

    UfRansCodec& coder(bool image, bool encoder, int parallel) {
        auto& slot = coders_[image ? 0 : 1][encoder ? 0 : 1][static_cast<std::size_t>(parallel - 1)];
        if (!slot) {
            slot.emplace();
            const auto& tables = image ? image_ : video_;
            if (!slot->set_entropy_coder_parallel(static_cast<std::size_t>(parallel)) ||
                !slot->add_cdf(tables.z) || !slot->add_cdf(tables.y))
                throw std::runtime_error("cannot configure the UF entropy coder");
        }
        return *slot;
    }

    // Masked quantization step outputs for the encoder: compacted int16 symbols.
    struct Symbols final {
        DeviceTensor values;
        DeviceTensor total;
    };

    Result<Symbols> compact_symbols(const DeviceTensor& symbols, const DeviceTensor& scales,
                                    int count) {
        auto prev = tensor("symbols_prev", 1, 1, count);
        auto condition = tensor("condition", 1, 1, (count + 1) / 2);
        auto values = tensor("symbols", 1, 1, count);
        auto total = tensor("total", 1, 1, 2);
        if (!prev || !condition || !values || !total)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF symbol buffers");
        return compact_symbols(view(symbols), view(scales), count, prev.value(), condition.value(),
                               std::move(values.value()), std::move(total.value()));
    }

    Result<Symbols> compact_symbols(glue::HalfTensor symbols, glue::HalfTensor scales, int count,
                                    const DeviceTensor& prev, const DeviceTensor& condition,
                                    DeviceTensor values, DeviceTensor total) {
        auto* condition_data = static_cast<bool*>(condition.storage.data);
        auto built = check(glue::build_index_enc(in(symbols), in(scales), 0.0F,
                                                 static_cast<std::int16_t*>(prev.storage.data),
                                                 condition_data, stream()),
                           "build_index_enc");
        if (!built) return built.error();
        auto compacted = check(glue::conditional_index_part1(
                                   prev.storage.data, 2, true, condition_data, count,
                                   values.storage.data, static_cast<int*>(total.storage.data),
                                   scratch(), stream()),
                               "conditional_index_part1");
        if (!compacted) return compacted.error();
        return Symbols{std::move(values), std::move(total)};
    }

    Result<std::vector<std::byte>> entropy_encode(bool image, std::span<Symbols> steps,
                                                  const DeviceTensor& z_int8, std::uint32_t qp,
                                                  std::uint8_t& parallel_out) {
        std::vector<std::vector<std::int16_t>> host;
        std::size_t total = 0;
        for (const auto& step : steps) {
            auto symbols = fetch<std::int16_t>(step.values, step.total);
            if (!symbols) return symbols.error();
            total += symbols.value().size();
            host.push_back(std::move(symbols.value()));
        }
        const int parallel = std::clamp(static_cast<int>(total) / min_symbols_per_stream, 1,
                                        max_entropy_parallel);
        auto& codec = coder(image, true, parallel);
        codec.reset_encoder();
        for (auto step = host.rbegin(); step != host.rend(); ++step) {
            auto encoded = codec.encode_y(*step, 1);
            if (!encoded) return encoded.error();
        }
        std::vector<std::int8_t> z(z_int8.storage.bytes);
        auto copied = trt::copy_async(z.data(), z_int8.storage.data, z.size(),
                                      cudaMemcpyDeviceToHost, stream(), "DCVC-UF z symbols");
        if (!copied) return copied.error();
        auto synchronized = trt::synchronize_stream(stream(), "DCVC-UF z symbols");
        if (!synchronized) return synchronized.error();
        z.resize(static_cast<std::size_t>(ch_z) * (ceil_to(h16_, 4) / 4) * (ceil_to(w16_, 4) / 4));
        auto encoded = codec.encode_z(z, 0, static_cast<std::size_t>(qp) * ch_z, ch_z);
        if (!encoded) return encoded.error();
        parallel_out = static_cast<std::uint8_t>(parallel);
        return codec.finish_encode();
    }

    // Decodes z and returns z_hat (FP16, NHWC).
    Result<DeviceTensor> decode_z(UfRansCodec& codec, std::span<const std::byte> bitstream,
                                  std::uint32_t qp) {
        auto stream_set = codec.set_stream(bitstream);
        if (!stream_set) return stream_set.error();
        const int h64 = ceil_to(h16_, 4) / 4;
        const int w64 = ceil_to(w16_, 4) / 4;
        const auto count = static_cast<std::size_t>(ch_z) * h64 * w64;
        auto z = codec.decode_z(count, 0, static_cast<std::size_t>(qp) * ch_z, ch_z);
        if (!z) return z.error();
        auto z_int8 = tensor("z_int8", 1, 1, static_cast<int>((count + 1) / 2));
        auto z_hat = tensor("z_hat", ch_z, h64, w64);
        if (!z_int8 || !z_hat) return failure(ErrorCode::resource_exhausted, "DCVC-UF z buffers");
        auto copied = trt::copy_async(z_int8.value().storage.data, z.value().data(), count,
                                      cudaMemcpyHostToDevice, stream(), "DCVC-UF z upload");
        if (!copied) return copied.error();
        auto converted = check(
            glue::int8_to_dtype(static_cast<const std::int8_t*>(z_int8.value().storage.data),
                                static_cast<int>(count), true, z_hat.value().storage.data, stream()),
            "int8_to_dtype");
        if (!converted) return converted.error();
        return z_hat;
    }

    // Builds decode indexes for `scales` under `condition`, entropy-decodes them
    // and returns the recovered quantized values (FP16, same shape as scales).
    Result<DeviceTensor> decode_y(UfRansCodec& codec, glue::HalfTensor scales, int count) {
        auto prev = tensor("indexes_prev", 1, 1, (count + 1) / 2);
        auto condition = tensor("condition", 1, 1, (count + 1) / 2);
        auto indexes = tensor("indexes", 1, 1, (count + 1) / 2);
        auto total = tensor("total", 1, 1, 2);
        auto decoded = tensor("decoded", 1, 1, (count + 1) / 2);
        auto recovered = tensor("y_q", scales.channels, scales.height, scales.width);
        if (!prev || !condition || !indexes || !total || !decoded || !recovered)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF index buffers");
        auto* condition_data = static_cast<bool*>(condition.value().storage.data);
        auto built = check(glue::build_index_dec(in(scales), 0.0F,
                                                 static_cast<std::uint8_t*>(prev.value().storage.data),
                                                 condition_data, stream()),
                           "build_index_dec");
        if (!built) return built.error();
        auto compacted = check(glue::conditional_index_part1(
                                   prev.value().storage.data, 1, false, condition_data, count,
                                   indexes.value().storage.data,
                                   static_cast<int*>(total.value().storage.data), scratch(),
                                   stream()),
                               "conditional_index_part1");
        if (!compacted) return compacted.error();
        auto host = fetch<std::uint8_t>(indexes.value(), total.value());
        if (!host) return host.error();
        auto values = codec.decode_y(host.value(), 1);
        if (!values) return values.error();
        if (!values.value().empty()) {
            auto copied = trt::copy_async(decoded.value().storage.data, values.value().data(),
                                          values.value().size(), cudaMemcpyHostToDevice, stream(),
                                          "DCVC-UF y upload");
            if (!copied) return copied.error();
        }
        auto restored = check(glue::conditional_recover(
                                  static_cast<const std::int8_t*>(decoded.value().storage.data),
                                  condition_data, scratch(), count, true,
                                  recovered.value().storage.data, stream()),
                              "conditional_recover");
        if (!restored) return restored.error();
        return recovered;
    }

    // HT-S reference from a reconstructed I frame (add_ref_feature_from_frame).
    Result<void> reference_from_frame(Reference& reference, const DeviceTensor& frame,
                                      bool apply_adaptor) {
        auto feature = tensor("feature", 192, h16_ * 2, w16_ * 2);
        if (!feature) return feature.error();
        auto unshuffled = check(glue::pixel_unshuffle(view(feature.value()),
                                                      in(view(frame, 0, 0, glue::Layout::nchw)), 8,
                                                      stream()),
                                "pixel_unshuffle");
        if (!unshuffled) return unshuffled.error();
        reference.feature_i = std::move(feature.value());
        reference.memory_has_value = false;
        if (!apply_adaptor) return {};
        auto adapted = adapt(reference);
        if (!adapted) return adapted;
        return extract(reference);
    }

    // memory from feature_p (adaptor_m) or from feature_i (adaptor_i).
    Result<void> adapt(Reference& reference) {
        if (reference.memory_has_value) {
            auto out = run("p_feature_adaptor_m",
                           {{"memory", &*reference.memory}, {"feature", &*reference.feature_p}});
            if (!out) return out.error();
            reference.memory = out.value().at("memory_out");
        } else {
            auto out = run("p_feature_adaptor_i", {{"feature", &*reference.feature_i}});
            if (!out) return out.error();
            reference.memory = out.value().at("memory");
        }
        return {};
    }

    Result<void> extract(Reference& reference) {
        auto out = run("p_feature_extractor", {{"memory", &*reference.memory}});
        if (!out) return out.error();
        reference.context = out.value().at("context");
        return {};
    }

    // dmci_proxy.cpp compress
    Result<Encoded> encode_i(const Frame& frame, std::uint32_t qp) {
        auto prepared = prepare(static_cast<int>(frame.width()), static_cast<int>(frame.height()));
        if (!prepared) return prepared.error();
        auto x = upload_frames(std::span(&frame, 1), glue::Layout::nchw);
        auto q_enc = q_row(image_, "q_scale_enc", qp, "q_enc");
        if (!x || !q_enc) return failure(ErrorCode::resource_exhausted, "DCVC-UF I inputs");
        auto encoded = run("i_encoder", {{"frame", &x.value()}, {"q_enc", &q_enc.value()}});
        if (!encoded) return encoded.error();
        const auto y = encoded.value().at("y");
        auto y_pad = pad_y(y);
        if (!y_pad) return y_pad.error();
        auto hyper = run("i_hyper_encoder", {{"y_padded", &y_pad.value()}});
        if (!hyper) return hyper.error();
        const auto& z = hyper.value().at("z");
        const int z_count = ch_z * height(z) * width(z);
        auto z_hat = tensor("z_hat", ch_z, height(z), width(z));
        auto z_int8 = tensor("z_int8", 1, 1, (z_count + 1) / 2);
        if (!z_hat || !z_int8) return failure(ErrorCode::resource_exhausted, "DCVC-UF z buffers");
        auto rounded = check(glue::round_z(data(z), z_count, data(z_hat.value()),
                                           static_cast<std::int8_t*>(z_int8.value().storage.data),
                                           stream()),
                             "round_z");
        if (!rounded) return rounded.error();
        auto params = hyper_params_i(z_hat.value());
        if (!params) return params.error();
        auto& [cparams, common] = params.value();

        auto q_y_enc = q_row(image_, "q_scale_y_enc", qp, "q");
        auto q_y_dec = q_row(image_, "q_scale_y_dec", qp, "q");
        if (!q_y_enc || !q_y_dec) return failure(ErrorCode::resource_exhausted, "DCVC-UF q rows");
        auto scaled = check(glue::multiply_with_broadcast(view(y), in(view(y)), data(q_y_enc.value()),
                                                          stream()),
                            "multiply_with_broadcast");
        if (!scaled) return scaled.error();

        auto y_q = tensor("y_q", ch_y, h16_, w16_);
        auto y_hat = tensor("y_hat", ch_y, h16_, w16_);
        auto s_hat = tensor("s_hat", ch_y, h16_, w16_);
        auto so_far = tensor("y_hat", ch_y, h16_, w16_);
        auto y_q_w = tensor("y_q_w", ch_y / 4, h16_, w16_);
        auto s_w = tensor("s_w", ch_y / 4, h16_, w16_);
        if (!y_q || !y_hat || !s_hat || !so_far || !y_q_w || !s_w)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF I buffers");
        const int quarter = ch_y / 4 * h16_ * w16_;
        std::vector<Symbols> steps;
        auto scales = view(cparams, 0, ch_y);
        auto means = view(cparams, ch_y, ch_y);
        std::optional<std::map<std::string, DeviceTensor>> prior;
        for (std::size_t step = 0; step < 4; ++step) {
            if (step > 0) {
                auto out = run(("i_spatial_prior_" + std::to_string(step)).c_str(),
                               {{"y_hat", &so_far.value()}, {"common", &common}});
                if (!out) return out.error();
                prior = std::move(out.value());
                const auto& sm = prior->at("scales_means");
                scales = view(sm, 0, ch_y);
                means = view(sm, ch_y, ch_y);
            }
            auto processed = check(
                glue::process_with_mask(view(y_q.value()), view(y_hat.value()), view(s_hat.value()),
                                        in(view(y)), in(scales), in(means), mask_view(step), 0.0F,
                                        stream()),
                "process_with_mask");
            if (!processed) return processed.error();
            auto written = check(glue::single_part_for_writing_4x(view(y_q_w.value()),
                                                                  in(view(y_q.value())), stream()),
                                 "single_part_for_writing_4x");
            if (!written) return written.error();
            written = check(glue::single_part_for_writing_4x(view(s_w.value()),
                                                             in(view(s_hat.value())), stream()),
                            "single_part_for_writing_4x");
            if (!written) return written.error();
            auto symbols = compact_symbols(y_q_w.value(), s_w.value(), quarter);
            if (!symbols) return symbols.error();
            steps.push_back(std::move(symbols.value()));
            cudaError_t status = cudaSuccess;
            if (step == 0) {
                status = cudaMemcpyAsync(data(so_far.value()), data(y_hat.value()),
                                         y_hat.value().storage.bytes, cudaMemcpyDeviceToDevice, stream());
            } else if (step < 3) {
                status = glue::add_and_multiply_broadcast(view(so_far.value()), in(view(so_far.value())),
                                                          in(view(y_hat.value())), data(*ones_), stream());
            } else {
                status = glue::add_and_multiply_broadcast(view(y_hat.value()), in(view(so_far.value())),
                                                          in(view(y_hat.value())), data(q_y_dec.value()),
                                                          stream());
            }
            auto accumulated = check(status, "y_hat accumulation");
            if (!accumulated) return accumulated.error();
        }
        Encoded output;
        auto bitstream = entropy_encode(true, steps, z_int8.value(), qp, output.entropy_parallel);
        if (!bitstream) return bitstream.error();
        output.bitstream = std::move(bitstream.value());

        auto reconstructed = decode_i_frame(y_hat.value(), qp);
        if (!reconstructed) return reconstructed.error();
        auto referenced = reference_from_frame(encoder_, reconstructed.value(), true);
        if (!referenced) return referenced.error();
        return output;
    }

    // i_hyper_decoder output cropped to the latent size: (params, common).
    Result<std::pair<DeviceTensor, DeviceTensor>> hyper_params_i(const DeviceTensor& z_hat) {
        auto out = run("i_hyper_decoder", {{"z_hat", &z_hat}});
        if (!out) return out.error();
        auto params = crop(out.value().at("params"));
        auto common = crop(out.value().at("common"));
        if (!params) return params.error();
        if (!common) return common.error();
        return std::pair{std::move(params.value()), std::move(common.value())};
    }

    Result<DeviceTensor> decode_i_frame(const DeviceTensor& y_hat, std::uint32_t qp) {
        auto q_dec = q_row(image_, "q_scale_dec", qp, "q_dec");
        if (!q_dec) return q_dec.error();
        auto out = run("i_decoder", {{"y_hat", &y_hat}, {"q_dec", &q_dec.value()}});
        if (!out) return out.error();
        return shuffle(out.value().at("frame_pre_shuffle"));
    }

    // dmci_proxy.cpp decompress
    Result<Frame> decode_i(std::span<const std::byte> bitstream, std::uint8_t parallel,
                           std::uint32_t qp, std::uint32_t frame_width, std::uint32_t frame_height,
                           Timestamp timestamp) {
        if (parallel == 0 || parallel > max_entropy_parallel)
            return failure(ErrorCode::malformed_bitstream, "invalid DCVC-UF entropy parallelism");
        auto prepared = prepare(static_cast<int>(frame_width), static_cast<int>(frame_height));
        if (!prepared) return prepared.error();
        auto& codec = coder(true, false, parallel);
        auto z_hat = decode_z(codec, bitstream, qp);
        if (!z_hat) return z_hat.error();
        auto params = hyper_params_i(z_hat.value());
        if (!params) return params.error();
        auto& [cparams, common] = params.value();
        auto q_y_dec = q_row(image_, "q_scale_y_dec", qp, "q");
        auto so_far = tensor("y_hat", ch_y, h16_, w16_);
        auto y_hat = tensor("y_hat", ch_y, h16_, w16_);
        auto scales_r = tensor("scales_r", ch_y / 4, h16_, w16_);
        if (!q_y_dec || !so_far || !y_hat || !scales_r)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF I buffers");
        const int quarter = ch_y / 4 * h16_ * w16_;
        auto scales = view(cparams, 0, ch_y);
        auto means = view(cparams, ch_y, ch_y);
        std::optional<std::map<std::string, DeviceTensor>> prior;
        for (std::size_t step = 0; step < 4; ++step) {
            if (step > 0) {
                auto out = run(("i_spatial_prior_" + std::to_string(step)).c_str(),
                               {{"y_hat", &so_far.value()}, {"common", &common}});
                if (!out) return out.error();
                prior = std::move(out.value());
                const auto& sm = prior->at("scales_means");
                scales = view(sm, 0, ch_y);
                means = view(sm, ch_y, ch_y);
            }
            auto read = check(glue::single_part_for_reading_4x(view(scales_r.value()), in(scales),
                                                               mask_view(step), stream()),
                              "single_part_for_reading_4x");
            if (!read) return read.error();
            auto y_q = decode_y(codec, view(scales_r.value()), quarter);
            if (!y_q) return y_q.error();
            cudaError_t status = cudaSuccess;
            if (step == 0) {
                status = glue::restore_y_4x(view(so_far.value()), in(view(y_q.value())), in(means),
                                            mask_view(step), stream());
            } else if (step < 3) {
                status = glue::restore_y_4x_and_add_inplace(view(so_far.value()), in(view(y_q.value())),
                                                            in(means), mask_view(step), stream());
            } else {
                status = glue::restore_y_4x_add_and_multiply_broadcast(
                    view(y_hat.value()), in(view(y_q.value())), in(means), mask_view(step),
                    in(view(so_far.value())), data(q_y_dec.value()), stream());
            }
            auto restored = check(status, "restore_y_4x");
            if (!restored) return restored.error();
        }
        auto frame = decode_i_frame(y_hat.value(), qp);
        if (!frame) return frame.error();
        auto referenced = reference_from_frame(decoder_, frame.value(), false);
        if (!referenced) return referenced.error();
        return download_frame(frame.value(), timestamp);
    }

    // p_hyper_decoder + crop + p_prior: (params [q_dec | scales | means], common).
    Result<std::pair<DeviceTensor, DeviceTensor>> hyper_params_p(const DeviceTensor& z_hat,
                                                                 const Reference& reference,
                                                                 const DeviceTensor& q_feature) {
        auto hyper = run("p_hyper_decoder", {{"z_hat", &z_hat}});
        if (!hyper) return hyper.error();
        auto cropped = crop(hyper.value().at("hyper_params"));
        if (!cropped) return cropped.error();
        auto prior = run("p_prior", {{"hyper_params", &cropped.value()},
                                     {"memory", &*reference.memory},
                                     {"q_feature", &q_feature}});
        if (!prior) return prior.error();
        return std::pair{prior.value().at("params"), prior.value().at("common")};
    }

    // dmc_hts_proxy.cpp compress
    Result<Encoded> encode_p(std::span<const Frame> frames, std::uint32_t qp, bool reset_memory) {
        auto& reference = encoder_;
        if (!reference.context)
            return failure(ErrorCode::invalid_state, "HT-S group before an intra frame");
        auto x = upload_frames(frames, glue::Layout::nhwc);
        auto q_encoder = q_row(video_, "q_encoder", qp, "q_encoder");
        auto q_feature = q_row(video_, "q_feature", qp, "q_feature");
        auto q_decoder = q_row(video_, "q_decoder", qp, "q_decoder");
        if (!x || !q_encoder || !q_feature || !q_decoder)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF HT-S inputs");
        auto encoded = run("p_encoder", {{"frames", &x.value()},
                                         {"context", &*reference.context},
                                         {"q_encoder", &q_encoder.value()}});
        if (!encoded) return encoded.error();
        const auto y = encoded.value().at("y");
        auto y_pad = pad_y(y);
        if (!y_pad) return y_pad.error();
        auto hyper = run("p_hyper_encoder", {{"y_padded", &y_pad.value()}});
        if (!hyper) return hyper.error();
        const auto& z = hyper.value().at("z");
        const int z_count = ch_z * height(z) * width(z);
        auto z_hat = tensor("z_hat", ch_z, height(z), width(z));
        auto z_int8 = tensor("z_int8", 1, 1, (z_count + 1) / 2);
        if (!z_hat || !z_int8) return failure(ErrorCode::resource_exhausted, "DCVC-UF z buffers");
        auto rounded = check(glue::round_z(data(z), z_count, data(z_hat.value()),
                                           static_cast<std::int8_t*>(z_int8.value().storage.data),
                                           stream()),
                             "round_z");
        if (!rounded) return rounded.error();
        auto params = hyper_params_p(z_hat.value(), reference, q_feature.value());
        if (!params) return params.error();
        auto& [cparams, common] = params.value();
        const auto q_dec = view(cparams, 0, ch_y);
        const auto scales = view(cparams, ch_y, ch_y);
        const auto means = view(cparams, 2 * ch_y, ch_y);
        auto divided = check(glue::divide_with_clamp_min_inplace(view(y), in(q_dec), stream()),
                             "divide_with_clamp_min_inplace");
        if (!divided) return divided.error();

        auto y_q = tensor("y_q", ch_y, h16_, w16_);
        auto y_hat = tensor("y_hat", ch_y, h16_, w16_);
        if (!y_q || !y_hat) return failure(ErrorCode::resource_exhausted, "DCVC-UF HT-S buffers");
        for (std::size_t step = 0; step < 4; ++step) {
            cudaError_t status = cudaSuccess;
            if (step == 0) {
                status = glue::process_with_mask_no_scale(view(y_q.value()), view(y_hat.value()),
                                                          in(view(y)), in(scales), in(means),
                                                          mask_view(step), 0.0F, stream());
            } else {
                auto prior = run(("p_spatial_prior_" + std::to_string(step)).c_str(),
                                 {{"y_hat", &y_hat.value()}, {"common", &common}});
                if (!prior) return prior.error();
                const auto step_means = in(view(prior.value().at("means")));
                status = step < 3
                    ? glue::process_with_mask_no_scale_add_inplace(
                          view(y_q.value()), view(y_hat.value()), in(view(y)), in(scales),
                          step_means, mask_view(step), 0.0F, stream())
                    : glue::process_with_mask_no_scale_add_and_multiply_inplace(
                          view(y_q.value()), view(y_hat.value()), in(view(y)), in(scales),
                          step_means, mask_view(step), in(q_dec), 0.0F, stream());
            }
            auto processed = check(status, "process_with_mask_no_scale");
            if (!processed) return processed.error();
        }
        const int count = ch_y * h16_ * w16_;
        auto prev = tensor("symbols_prev", 1, 1, count);
        auto condition = tensor("condition", 1, 1, (count + 1) / 2);
        auto values = tensor("symbols", 1, 1, count);
        auto total = tensor("total", 1, 1, 2);
        if (!prev || !condition || !values || !total)
            return failure(ErrorCode::resource_exhausted, "DCVC-UF symbol buffers");
        auto symbols = compact_symbols(view(y_q.value()), scales, count, prev.value(),
                                       condition.value(), std::move(values.value()),
                                       std::move(total.value()));
        if (!symbols) return symbols.error();
        Encoded output;
        auto bitstream = entropy_encode(false, std::span(&symbols.value(), 1), z_int8.value(), qp,
                                        output.entropy_parallel);
        if (!bitstream) return bitstream.error();
        output.bitstream = std::move(bitstream.value());

        auto decoded = run("p_decoder", {{"y_hat", &y_hat.value()},
                                         {"context", &*reference.context},
                                         {"q_decoder", &q_decoder.value()}});
        if (!decoded) return decoded.error();
        reference.feature_p = decoded.value().at("feature");
        if (reset_memory) {
            auto reset = run("p_reset", {{"feature", &*reference.feature_p}});
            if (!reset) return reset.error();
            reference.feature_i = reset.value().at("reset_feature");
            reference.memory_has_value = false;
        } else {
            reference.memory_has_value = true;
        }
        auto adapted = adapt(reference);
        if (!adapted) return adapted.error();
        auto extracted = extract(reference);
        if (!extracted) return extracted.error();
        return output;
    }

    // dmc_hts_proxy.cpp decompress
    Result<std::vector<Frame>> decode_p(std::span<const std::byte> bitstream, std::uint8_t parallel,
                                        std::uint32_t qp, std::uint32_t frame_width,
                                        std::uint32_t frame_height,
                                        std::span<const Timestamp> timestamps, bool reset_memory) {
        auto& reference = decoder_;
        if (!reference.feature_i)
            return failure(ErrorCode::malformed_bitstream, "HT-S group before an intra frame");
        if (parallel == 0 || parallel > max_entropy_parallel)
            return failure(ErrorCode::malformed_bitstream, "invalid DCVC-UF entropy parallelism");
        if (static_cast<int>(frame_width) != width_ || static_cast<int>(frame_height) != height_)
            return failure(ErrorCode::malformed_bitstream, "HT-S group changes the frame size");
        auto adapted = adapt(reference);
        if (!adapted) return adapted.error();
        auto q_feature = q_row(video_, "q_feature", qp, "q_feature");
        auto q_decoder = q_row(video_, "q_decoder", qp, "q_decoder");
        if (!q_feature || !q_decoder) return failure(ErrorCode::resource_exhausted, "DCVC-UF q rows");
        auto& codec = coder(false, false, parallel);
        auto z_hat = decode_z(codec, bitstream, qp);
        if (!z_hat) return z_hat.error();
        auto params = hyper_params_p(z_hat.value(), reference, q_feature.value());
        if (!params) return params.error();
        auto& [cparams, common] = params.value();
        const auto q_dec = view(cparams, 0, ch_y);
        const auto scales = view(cparams, ch_y, ch_y);
        const auto means = view(cparams, 2 * ch_y, ch_y);
        auto y_q = decode_y(codec, scales, ch_y * h16_ * w16_);
        if (!y_q) return y_q.error();
        auto extracted = extract(reference);
        if (!extracted) return extracted.error();

        auto y_hat = tensor("y_hat", ch_y, h16_, w16_);
        if (!y_hat) return y_hat.error();
        for (std::size_t step = 0; step < 4; ++step) {
            cudaError_t status = cudaSuccess;
            if (step == 0) {
                status = glue::restore_y(view(y_hat.value()), in(view(y_q.value())), in(means),
                                         mask_view(step), stream());
            } else {
                auto prior = run(("p_spatial_prior_" + std::to_string(step)).c_str(),
                                 {{"y_hat", &y_hat.value()}, {"common", &common}});
                if (!prior) return prior.error();
                const auto step_means = in(view(prior.value().at("means")));
                status = step < 3
                    ? glue::restore_y_and_add_inplace(view(y_hat.value()), in(view(y_q.value())),
                                                      step_means, mask_view(step), stream())
                    : glue::restore_y_and_add_multiply_inplace(
                          view(y_hat.value()), in(view(y_q.value())), step_means, mask_view(step),
                          in(q_dec), stream());
            }
            auto restored = check(status, "restore_y");
            if (!restored) return restored.error();
        }
        auto decoded = run("p_decoder", {{"y_hat", &y_hat.value()},
                                         {"context", &*reference.context},
                                         {"q_decoder", &q_decoder.value()}});
        if (!decoded) return decoded.error();
        reference.feature_p = decoded.value().at("feature");
        auto heads = run("p_reconstruction", {{"feature", &*reference.feature_p}});
        if (!heads) return heads.error();
        auto reset = run("p_reset", {{"feature", &*reference.feature_p}});
        if (!reset) return reset.error();
        reference.feature_i = reset.value().at("reset_feature");
        reference.memory_has_value = !reset_memory;

        std::vector<Frame> output;
        for (std::size_t index = 0; index < timestamps.size(); ++index) {
            auto frame = shuffle(heads.value().at("frame_" + std::to_string(index) + "_pre_shuffle"));
            if (!frame) return frame.error();
            auto downloaded = download_frame(frame.value(), timestamps[index]);
            if (!downloaded) return downloaded.error();
            output.push_back(std::move(downloaded.value()));
        }
        return output;
    }

    std::shared_ptr<trt::TensorRTExecutionSession> session_;
    std::string owner_;
    std::map<std::string, session_api::ExecutableStage> stages_;
    Tables image_;
    Tables video_;
    std::array<std::array<std::array<std::optional<UfRansCodec>, max_entropy_parallel>, 2>, 2>
        coders_;
    Reference encoder_;
    Reference decoder_;
    std::array<std::optional<DeviceTensor>, 4> masks_;
    std::optional<DeviceTensor> scratch_;
    std::optional<DeviceTensor> ones_;
    int width_{};
    int height_{};
    int h16_{};
    int w16_{};
    static std::atomic<std::uint64_t> next_owner_;
};

std::atomic<std::uint64_t> TensorRTSession::next_owner_{0};

}  // namespace

Result<std::shared_ptr<ExecutionSession>> make_tensorrt_session(
    std::shared_ptr<provider::experimental::IProviderSession> provider) {
    auto session = std::dynamic_pointer_cast<trt::TensorRTExecutionSession>(provider);
    if (!session) return std::shared_ptr<ExecutionSession>{};
    auto output = std::make_shared<TensorRTSession>(std::move(session));
    auto initialized = output->initialize();
    if (!initialized) return initialized.error();
    return std::shared_ptr<ExecutionSession>(std::move(output));
}

}  // namespace nvcr::dcvcuf
