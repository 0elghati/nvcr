#include "native_session.hpp"

#include "diagnostics.hpp"
#include "dmci_proxy.h"
#include "dmc_hts_proxy.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <utility>

namespace nvcr::dcvcuf {
namespace {

using Weights = std::map<std::string, at::Tensor>;
namespace diag = diagnostics;
constexpr std::string_view weight_escape = "__nvcr_dot__";

Error failure(ErrorCode code, std::string message) {
    return Error(code, std::move(message), "dcvcuf-upstream-cuda");
}

std::string decode_name(std::string name) {
    std::size_t position = 0;
    while ((position = name.find(weight_escape, position)) != std::string::npos) {
        name.replace(position, weight_escape.size(), ".");
        ++position;
    }
    return name;
}

Result<std::pair<Weights, Weights>> load_bundle(
    const std::filesystem::path& configured_path, int device) {
    auto path = configured_path;
    if (std::filesystem::is_directory(path))
        path /= "dcvcuf-cvpr2026-hts.pt";
    if (!std::filesystem::is_regular_file(path)) {
        return failure(
            ErrorCode::missing_artifact,
            "DCVC-UF native bundle does not exist: " + path.string());
    }
    try {
        auto module = torch::jit::load(path.string(), at::Device(at::kCPU));
        Weights image;
        Weights video;
        for (const auto& item : module.named_buffers(true)) {
            auto name = decode_name(item.name);
            auto value = item.value;
            if (value.is_floating_point()) {
                value = value.to(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device)));
                value = value.dim() == 4
                    ? value.contiguous(at::MemoryFormat::ChannelsLast)
                    : value.contiguous();
            } else {
                value = value.cpu().contiguous();
            }
            if (name.starts_with("image."))
                image.emplace(name.substr(6), std::move(value));
            else if (name.starts_with("video."))
                video.emplace(name.substr(6), std::move(value));
        }
        if (image.empty() || video.empty())
            return failure(ErrorCode::missing_artifact,
                           "DCVC-UF bundle is missing image or video weights");
        return std::pair{std::move(image), std::move(video)};
    } catch (const std::exception& exception) {
        return failure(
            ErrorCode::missing_artifact,
            std::string("cannot load DCVC-UF native bundle: ") + exception.what());
    }
}

// Builds upstream's model input for N equally sized YUV420P8 frames as
// [1, 3N, H, W]. Only the 4:2:0 planes cross the bus, in one pinned copy; the
// 4:4:4 expansion runs on the device. Nearest-neighbour replication matches
// upstream's scipy.ndimage.zoom(order=0) for a factor of two, and the FP16
// conversion, scaling and centering are upstream get_src_frame's tensor ops.
at::Tensor frames_to_tensor(std::span<const Frame> frames, int device) {
    const auto& first = frames.front();
    const auto width = static_cast<std::int64_t>(first.width());
    const auto height = static_cast<std::int64_t>(first.height());
    const auto count = static_cast<std::int64_t>(frames.size());
    const auto y_size = width * height;
    const auto frame_bytes = y_size * 3 / 2;
    auto staging = at::empty(
        {count, frame_bytes}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    for (std::int64_t index = 0; index < count; ++index) {
        const auto& frame = frames[static_cast<std::size_t>(index)];
        TORCH_CHECK(frame.pixel_format() == PixelFormat::yuv420p8 &&
                        frame.width() == first.width() && frame.height() == first.height() &&
                        static_cast<std::int64_t>(frame.data().size()) >= frame_bytes,
                    "DCVC-UF requires equally sized YUV420P8 frames");
        std::memcpy(staging[index].data_ptr<std::uint8_t>(), frame.data().data(),
                    static_cast<std::size_t>(frame_bytes));
    }
    auto planes = staging.to(at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device)), /*non_blocking=*/true);
    auto y = planes.narrow(1, 0, y_size).reshape({count, 1, height, width});
    auto uv = planes.narrow(1, y_size, y_size / 2)
                  .reshape({count, 2, height / 2, 1, width / 2, 1})
                  .expand({count, 2, height / 2, 2, width / 2, 2})
                  .reshape({count, 2, height, width});
    return at::cat({y, uv}, 1)
        .view({1, 3 * count, height, width})
        .to(at::kHalf)
        .div(255.0)
        .sub(0.5)
        .contiguous(at::MemoryFormat::ChannelsLast);
}

// Converts the first timestamps.size() reconstructions to YUV420P8 frames with
// one device-to-host copy. Per-element arithmetic is unchanged from the
// verified single-frame path: luma is rounded and chroma is 2x2-averaged and
// truncated, as in upstream test_video.py.
std::vector<Frame> tensors_to_frames(
    std::span<const at::Tensor> centered, std::uint32_t width, std::uint32_t height,
    std::span<const Timestamp> timestamps) {
    const auto count = static_cast<std::int64_t>(timestamps.size());
    std::vector<at::Tensor> cropped;
    cropped.reserve(timestamps.size());
    for (std::int64_t index = 0; index < count; ++index) {
        cropped.push_back(centered[static_cast<std::size_t>(index)].index({
            at::indexing::Slice(), at::indexing::Slice(),
            at::indexing::Slice(0, height), at::indexing::Slice(0, width)}));
    }
    auto frames = at::cat(cropped, 0).add(0.5);
    auto y = frames.narrow(1, 0, 1).mul(255).clamp(0, 255).round().to(at::kByte);
    auto uv = frames.narrow(1, 1, 2)
                  .unfold(2, 2, 2).unfold(3, 2, 2)
                  .mean(std::vector<std::int64_t>{4, 5})
                  .mul(255).clamp(0, 255).to(at::kByte);
    auto packed = at::cat({y.reshape({count, -1}), uv.reshape({count, -1})}, 1);
    auto host = at::empty(
        packed.sizes(), at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    host.copy_(packed);
    const auto frame_bytes = static_cast<std::size_t>(packed.size(1));
    std::vector<Frame> output;
    output.reserve(timestamps.size());
    for (std::int64_t index = 0; index < count; ++index) {
        const auto* data = reinterpret_cast<const std::byte*>(
            host[index].const_data_ptr<std::uint8_t>());
        auto frame = Frame::copy_from(
            width, height, PixelFormat::yuv420p8,
            std::span<const std::byte>(data, frame_bytes),
            timestamps[static_cast<std::size_t>(index)]);
        if (!frame) throw std::runtime_error(frame.error().describe());
        output.push_back(std::move(frame.value()));
    }
    return output;
}

class NativeSession final : public ExecutionSession {
public:
    NativeSession(int device, Weights image, Weights video)
        : device_(device),
          owner_("dcvcuf-upstream-cuda-" + std::to_string(++next_owner_)),
          stream_(at::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device))),
          image_weights_(std::move(image)),
          video_weights_(std::move(video)) {}

    Result<void> initialize() {
        try {
            c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_));
            at::cuda::CUDAStreamGuard stream_guard(stream_);
            c10::InferenceMode inference;
            rebuild_encoder();
            rebuild_decoder();
            return {};
        } catch (const std::exception& exception) {
            return failure(ErrorCode::backend_error, exception.what());
        }
    }

    std::string_view owner_id() const noexcept override { return owner_; }

    Result<provider::experimental::BufferHandle> allocate(
        std::size_t, provider::experimental::MemoryDomain) override {
        return failure(ErrorCode::not_implemented,
                       "DCVC-UF native session exposes codec-local fused operations");
    }
    Result<provider::experimental::ExecutableStage> load_stage(
        provider::experimental::ExecutableStageDescriptor) override {
        return failure(ErrorCode::not_implemented,
                       "DCVC-UF native bundle is loaded as one fused codec provider");
    }
    Result<provider::experimental::Completion> submit(
        const provider::experimental::ExecutableStage&,
        std::span<const provider::experimental::TensorView>,
        std::span<const provider::experimental::ExecutionDependency>) override {
        return failure(ErrorCode::not_implemented,
                       "DCVC-UF native bundle uses codec-local fused submission");
    }

    Result<Encoded> encode_intra(
        const Frame& frame, std::uint32_t qp) override {
        return protect<Encoded>([&] {
            const auto padding_r = (16 - static_cast<int>(frame.width()) % 16) % 16;
            const auto padding_b = (16 - static_cast<int>(frame.height()) % 16) % 16;
            observe_step(diag::Direction::encoder, diag::StepKind::intra, false);
            auto result = image_encoder_->compress(
                frames_to_tensor(std::span(&frame, 1), device_), static_cast<int>(qp),
                padding_b, padding_r);
            video_encoder_->add_ref_feature_from_frame(result.reconstruction, true);
            if (auto* sink = diag::observer()) {
                const std::array timestamps{frame.timestamp()};
                sink->frames(diag::Direction::encoder,
                             tensors_to_frames(std::span(&result.reconstruction, 1), frame.width(),
                                               frame.height(), timestamps));
                // Sizes the head used for encoder-side group reconstructions.
                if (!probe_) {
                    probe_ = std::make_unique<DMCHTSProxy>();
                    probe_->set_param(video_weights_, 0.0F);
                }
                probe_->add_ref_feature_from_frame(result.reconstruction, false);
            }
            return Encoded{
                bytes(result.bit_stream), static_cast<std::uint8_t>(result.entropy_coder_parallel)};
        });
    }

    Result<Frame> decode_intra(
        std::span<const std::byte> bitstream, std::uint8_t entropy_parallel,
        std::uint32_t qp, std::uint32_t width, std::uint32_t height,
        Timestamp timestamp) override {
        return protect<Frame>([&] {
            observe_step(diag::Direction::decoder, diag::StepKind::intra, false);
            auto reconstruction = image_decoder_->decompress(
                reinterpret_cast<const std::uint8_t*>(bitstream.data()),
                static_cast<int>(bitstream.size()), static_cast<int>(qp),
                static_cast<int>(height), static_cast<int>(width), entropy_parallel);
            video_decoder_->add_ref_feature_from_frame(reconstruction, false);
            auto output = std::move(tensors_to_frames(
                std::span(&reconstruction, 1), width, height, std::span(&timestamp, 1)).front());
            if (auto* sink = diag::observer())
                sink->frames(diag::Direction::decoder, std::span(&output, 1));
            return output;
        });
    }

    Result<Encoded> encode_predicted(
        std::span<const Frame> frames, std::uint32_t qp, bool reset_feature_memory) override {
        if (frames.size() != 8)
            return failure(ErrorCode::invalid_argument,
                           "DCVC-UF HT-S encoding requires exactly eight frames");
        return protect<Encoded>([&] {
            auto group = frames_to_tensor(frames, device_);
            const auto padding_r = (16 - static_cast<int>(frames.front().width()) % 16) % 16;
            const auto padding_b = (16 - static_cast<int>(frames.front().height()) % 16) % 16;
            auto* sink = diag::observer();
            if (sink != nullptr) {
                sink->step(diag::Direction::encoder, diag::StepKind::predicted, reset_feature_memory);
                observe(diag::Direction::encoder, "memory", video_encoder_->nvcr_memory());
                observe(diag::Direction::encoder, "context", video_encoder_->nvcr_context());
            }
            auto result = video_encoder_->compress(
                group, static_cast<int>(qp), reset_feature_memory, padding_b, padding_r);
            if (sink != nullptr) {
                observe(diag::Direction::encoder, "y_hat", video_encoder_->nvcr_y_hat());
                const auto feature = video_encoder_->nvcr_feature().clone();
                observe(diag::Direction::encoder, "feature", feature);
                // The encoder never runs the head for a group; a separate
                // instance runs it on a copy, leaving encoder state untouched.
                TORCH_CHECK(probe_ != nullptr, "DCVC-UF diagnostics need an intra frame first");
                const auto reconstructed = probe_->nvcr_reconstruct(feature);
                const std::vector<Timestamp> timestamps(frames.size());
                sink->frames(diag::Direction::encoder,
                             tensors_to_frames(reconstructed, frames.front().width(),
                                               frames.front().height(), timestamps));
            }
            return Encoded{
                bytes(result.bit_stream), static_cast<std::uint8_t>(result.entropy_coder_parallel)};
        });
    }

    Result<std::vector<Frame>> decode_predicted(
        std::span<const std::byte> bitstream, std::uint8_t entropy_parallel,
        std::uint32_t qp, std::uint32_t width, std::uint32_t height,
        std::span<const Timestamp> timestamps, bool reset_feature_memory) override {
        if (timestamps.empty() || timestamps.size() > 8)
            return failure(ErrorCode::invalid_argument,
                           "DCVC-UF predicted timestamp count must be in [1, 8]");
        return protect<std::vector<Frame>>([&] {
            // decompress is one upstream call, so a failed group records only
            // its step here; its tensors are reported after it returns.
            observe_step(diag::Direction::decoder, diag::StepKind::predicted, reset_feature_memory);
            auto decoded = video_decoder_->decompress(
                reinterpret_cast<const std::uint8_t*>(bitstream.data()),
                static_cast<int>(bitstream.size()), static_cast<int>(qp),
                static_cast<int>(height), static_cast<int>(width), entropy_parallel,
                reset_feature_memory);
            if (decoded.size() != 8)
                throw std::runtime_error("DCVC-UF HT-S decoder did not return eight frames");
            auto output = tensors_to_frames(decoded, width, height, timestamps);
            if (auto* sink = diag::observer()) {
                observe(diag::Direction::decoder, "memory", video_decoder_->nvcr_memory());
                observe(diag::Direction::decoder, "context", video_decoder_->nvcr_context());
                observe(diag::Direction::decoder, "y_hat", video_decoder_->nvcr_y_hat());
                observe(diag::Direction::decoder, "feature", video_decoder_->nvcr_feature());
                sink->frames(diag::Direction::decoder, output);
            }
            return output;
        });
    }

    // Upstream keeps its fused models for the whole sequence: an intra frame
    // replaces the HT-S reference feature and memory via
    // add_ref_feature_from_frame, and the codec adapter starts every sequence
    // with an intra frame. Rebuilding here would only discard processed
    // weights and captured CUDA graphs.
    Result<void> reset_encoder_state() override { return {}; }
    Result<void> reset_decoder_state() override { return {}; }
    Result<void> reset() override {
        auto encoder = reset_encoder_state();
        if (!encoder) return encoder.error();
        return reset_decoder_state();
    }

private:
    static std::vector<std::byte> bytes(const std::vector<std::uint8_t>& input) {
        std::vector<std::byte> output(input.size());
        std::memcpy(output.data(), input.data(), input.size());
        return output;
    }

    template <class T, class Function>
    Result<T> protect(Function&& function) {
        try {
            c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_));
            at::cuda::CUDAStreamGuard stream_guard(stream_);
            c10::InferenceMode inference;
            if constexpr (std::is_void_v<T>) {
                std::forward<Function>(function)();
                C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
                return {};
            } else {
                auto output = std::forward<Function>(function)();
                C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
                return output;
            }
        } catch (const std::bad_alloc&) {
            return failure(ErrorCode::resource_exhausted,
                           "DCVC-UF native provider allocation failed");
        } catch (const std::exception& exception) {
            return failure(ErrorCode::backend_error, exception.what());
        }
    }

    // C13 G1 diagnostics (diagnostics.hpp); called only with an observer installed.
    static void observe_step(diag::Direction direction, diag::StepKind kind, bool reset) {
        if (auto* sink = diag::observer()) sink->step(direction, kind, reset);
    }
    static void observe(diag::Direction direction, const char* name, const at::Tensor& value) {
        auto* sink = diag::observer();
        if (sink == nullptr) return;
        const auto host = value.cpu().contiguous();
        const auto sizes = host.sizes();
        sink->tensor(direction, name, std::span<const std::int64_t>(sizes.data(), sizes.size()),
                     std::span(reinterpret_cast<const std::byte*>(host.const_data_ptr()),
                               host.nbytes()));
    }

    void rebuild_encoder() {
        image_encoder_ = std::make_unique<DMCIProxy>();
        image_encoder_->set_param(image_weights_, 0.0F);
        video_encoder_ = std::make_unique<DMCHTSProxy>();
        video_encoder_->set_param(video_weights_, 0.0F);
    }
    void rebuild_decoder() {
        image_decoder_ = std::make_unique<DMCIProxy>();
        image_decoder_->set_param(image_weights_, 0.0F);
        video_decoder_ = std::make_unique<DMCHTSProxy>();
        video_decoder_->set_param(video_weights_, 0.0F);
    }

    int device_;
    std::string owner_;
    at::cuda::CUDAStream stream_;
    Weights image_weights_;
    Weights video_weights_;
    std::unique_ptr<DMCIProxy> image_encoder_;
    std::unique_ptr<DMCIProxy> image_decoder_;
    std::unique_ptr<DMCHTSProxy> video_encoder_;
    std::unique_ptr<DMCHTSProxy> video_decoder_;
    std::unique_ptr<DMCHTSProxy> probe_;  // diagnostics only
    static std::atomic<std::uint64_t> next_owner_;
};

std::atomic<std::uint64_t> NativeSession::next_owner_{0};

}  // namespace

Result<std::shared_ptr<provider::experimental::IProviderSession>>
make_native_session(const RuntimeConfiguration& configuration) {
    try {
        const int device = configuration.provider.device_id;
        c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device));
        at::cuda::CUDAStreamGuard stream_guard(
            at::cuda::getStreamFromPool(false, static_cast<c10::DeviceIndex>(device)));
        c10::InferenceMode inference;
        auto loaded = load_bundle(configuration.artifacts.intra_engine_path, device);
        if (!loaded) return loaded.error();
        auto session = std::make_shared<NativeSession>(
            device, std::move(loaded.value().first), std::move(loaded.value().second));
        auto initialized = session->initialize();
        if (!initialized) return initialized.error();
        return std::static_pointer_cast<provider::experimental::IProviderSession>(
            std::move(session));
    } catch (const std::exception& exception) {
        return failure(ErrorCode::backend_error, exception.what());
    }
}

}  // namespace nvcr::dcvcuf
