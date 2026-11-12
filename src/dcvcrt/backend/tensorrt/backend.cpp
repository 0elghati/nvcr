#include "nvcr/dcvcrt/tensorrt_backend.hpp"
#include "nvcr/provider/experimental/session.hpp"
#include "nvcr/runtime/registry.hpp"

#include "../../../provider/tensorrt/session.hpp"
#include "cuda_ops.hpp"
#include "engine_specs.hpp"
#include "../../numerics.hpp"
#include "../../orchestration.hpp"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvcr::dcvcrt {
namespace {

namespace fs = std::filesystem;
using SequenceStateView = codec::SequenceStateView;

constexpr std::string_view subsystem = "dcvcrt.tensorrt";
constexpr std::size_t image_qp_count = 64U;
constexpr std::size_t video_qp_count = 72U;
constexpr std::uint32_t minimum_frame_dimension = 64U;
constexpr std::uint32_t maximum_frame_width = 1920U;
constexpr std::uint32_t maximum_frame_height = 1080U;

namespace session_api = provider::experimental;

// The codec-agnostic TensorRT provider session and its device helpers.
using tensorrt::active_profiler;
using tensorrt::allocate_cuda;
using tensorrt::allocate_cuda_scratch;
using tensorrt::allocate_device_tensor;
using tensorrt::allocate_device_tensor_scratch;
using tensorrt::copy_async;
using tensorrt::CpuProfileScope;
using tensorrt::CudaAllocation;
using tensorrt::CudaEvent;
using tensorrt::DeviceTensor;
using tensorrt::element_count;
using tensorrt::EngineOutputStorage;
using tensorrt::make_dims;
using tensorrt::PinnedHostBuffer;
using tensorrt::read_json_string;
using tensorrt::read_text_file;
using tensorrt::record_cuda_event;
using tensorrt::run_device_stage;
using tensorrt::SessionExecutionScope;
using tensorrt::synchronize_event;
using tensorrt::synchronize_stream;
using tensorrt::tensor_bytes;
using tensorrt::TensorRTExecutionSession;
using tensorrt::TensorRTStageSpec;
using tensorrt::TensorRTTensorSpec;
using tensorrt::valid_sha256;

Error backend_error(std::string message) {
    return Error(ErrorCode::backend_error, std::move(message), std::string(subsystem));
}

Error cuda_error(std::string_view operation, cudaError_t status) {
    return backend_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}

bool supported_frame_dimensions(std::uint32_t width, std::uint32_t height) {
    return width >= minimum_frame_dimension && height >= minimum_frame_dimension &&
        width <= maximum_frame_width && height <= maximum_frame_height &&
        (width & 1U) == 0U && (height & 1U) == 0U;
}




struct ImageDecodeStaging final {
    Result<void> ensure(std::size_t z_count, std::size_t reduced_count) {
        auto ready = z_symbols.ensure(z_count, "image_decode_z_symbols");
        if (!ready) return ready.error();
        for (std::size_t stage = 0; stage < indexes.size(); ++stage) {
            ready = indexes[stage].ensure(reduced_count, "image_decode_indexes");
            if (!ready) return ready.error();
            ready = symbols[stage].ensure(reduced_count, "image_decode_symbols");
            if (!ready) return ready.error();
            if (indexes_ready[stage].data == nullptr) {
                const auto status = cudaEventCreateWithFlags(
                    &indexes_ready[stage].data, cudaEventDisableTiming);
                if (status != cudaSuccess) {
                    return cuda_error("cudaEventCreateWithFlags I decode indexes", status);
                }
            }
        }
        return {};
    }

    PinnedHostBuffer<std::int8_t> z_symbols;
    std::array<PinnedHostBuffer<std::uint8_t>, 4> indexes;
    std::array<PinnedHostBuffer<std::int8_t>, 4> symbols;
    std::array<CudaEvent, 4> indexes_ready;
};

struct VideoDecodeStaging final {
    Result<void> ensure(std::size_t z_count, std::size_t reduced_count) {
        auto ready = z_symbols.ensure(z_count, "video_decode_z_symbols");
        if (!ready) return ready.error();
        for (std::size_t stage = 0; stage < indexes.size(); ++stage) {
            ready = indexes[stage].ensure(reduced_count, "video_decode_indexes");
            if (!ready) return ready.error();
            ready = symbols[stage].ensure(reduced_count, "video_decode_symbols");
            if (!ready) return ready.error();
            if (indexes_ready[stage].data == nullptr) {
                const auto status = cudaEventCreateWithFlags(
                    &indexes_ready[stage].data, cudaEventDisableTiming);
                if (status != cudaSuccess) {
                    return cuda_error("cudaEventCreateWithFlags P decode indexes", status);
                }
            }
        }
        return {};
    }

    PinnedHostBuffer<std::int8_t> z_symbols;
    std::array<PinnedHostBuffer<std::uint8_t>, 2> indexes;
    std::array<PinnedHostBuffer<std::int8_t>, 2> symbols;
    std::array<CudaEvent, 2> indexes_ready;
};


nvinfer1::Dims warmup_shape(
    std::string_view filename, std::string_view name, nvinfer1::Dims shape) {
    const auto select_dynamic_dimension = [&shape](std::int32_t index, std::int64_t value) {
        if (shape.d[index] == -1) shape.d[index] = value;
    };
    if (name == "frame" || name == "reference_frame") {
        select_dynamic_dimension(2, filename.starts_with("p_") ? 192 : 144);
        select_dynamic_dimension(3, filename.starts_with("p_") ? 192 : 176);
    } else if (name == "reference_feature" || name == "temporal_context") {
        select_dynamic_dimension(2, 24);
        select_dynamic_dimension(3, 24);
    } else if (name == "y_padded" || name == "y_hat") {
        select_dynamic_dimension(2, 12);
        select_dynamic_dimension(3, 12);
    } else if (name == "z_hat") {
        select_dynamic_dimension(2, 3);
        select_dynamic_dimension(3, 3);
    } else if (name == "context") {
        select_dynamic_dimension(
            2, filename == "p_spatial_prior.plan" ? 12 :
                (filename.starts_with("p_") ? 24 : 9));
        select_dynamic_dimension(
            3, filename == "p_spatial_prior.plan" ? 12 :
                (filename.starts_with("p_") ? 24 : 11));
    }
    return shape;
}


struct HostTensor final {
    std::string name;
    nvinfer1::Dims shape;
    std::vector<__half> values;
};

HostTensor make_quant_tensor(
    std::span<const float> values, std::uint32_t qp, std::string name);
HostTensor make_video_quant_tensor(
    std::span<const float> values, std::string_view kind, std::uint32_t qp);

struct ImageQuantDeviceCache final {
    std::array<std::optional<DeviceTensor>, image_qp_count> q_enc;
    std::array<std::optional<DeviceTensor>, image_qp_count> q_dec;
};

struct VideoQuantDeviceCache final {
    std::array<std::optional<DeviceTensor>, video_qp_count> q_encoder;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_decoder;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_feature;
    std::array<std::optional<DeviceTensor>, video_qp_count> q_recon;
};


struct DeviceDpb final {
    std::optional<DeviceTensor> frame;
    std::optional<DeviceTensor> feature;
    ReferenceState semantics;

    void clear() noexcept {
        frame.reset();
        feature.reset();
        semantics.reset();
    }
};


Result<DeviceTensor> upload_tensor(
    const HostTensor& input, std::string name, cudaStream_t stream) {
    const auto bytes = input.values.size() * sizeof(__half);
    auto storage = allocate_cuda(bytes);
    if (!storage) return storage.error();
    auto copied = copy_async(
        storage.value().data, input.values.data(), bytes, cudaMemcpyHostToDevice, stream,
        "cudaMemcpyAsync upload");
    if (!copied) return copied.error();
    return DeviceTensor{
        std::move(name), input.shape,
        DeviceTensor::Storage(std::move(storage.value()))};
}

Result<DeviceTensor> upload_int8_tensor_scratch(
    std::span<const std::int8_t> values, nvinfer1::Dims shape, std::string name,
    cudaStream_t stream) {
    auto packed = allocate_cuda_scratch(values.size() * sizeof(std::int8_t));
    if (!packed) return packed.error();
    auto storage = allocate_cuda_scratch(values.size() * sizeof(__half));
    if (!storage) return storage.error();
    auto copied = copy_async(
        packed.value().data, values.data(), values.size() * sizeof(std::int8_t),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync packed symbol upload");
    if (!copied) return copied.error();
    const auto converted = cuda_ops::int8_to_half(
        static_cast<const std::int8_t*>(packed.value().data), storage.value().data,
        values.size(), stream);
    if (converted != cudaSuccess) return cuda_error("cuda_ops::int8_to_half", converted);
    return DeviceTensor{
        std::move(name), shape,
        DeviceTensor::Storage(std::move(storage.value()))};
}

Result<void> populate_image_quant_cache(
    std::span<const float> q_encoder,
    std::span<const float> q_decoder,
    ImageQuantDeviceCache& cache,
    cudaStream_t stream) {
    for (std::uint32_t qp = 0; qp < image_qp_count; ++qp) {
        auto q_enc = upload_tensor(
            make_quant_tensor(q_encoder, qp, "q_enc"), "q_enc", stream);
        if (!q_enc) return q_enc.error();
        cache.q_enc[qp] = std::move(q_enc.value());

        auto q_dec = upload_tensor(
            make_quant_tensor(q_decoder, qp, "q_dec"), "q_dec", stream);
        if (!q_dec) return q_dec.error();
        cache.q_dec[qp] = std::move(q_dec.value());
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize image quant cache");
    if (!synchronized) return synchronized.error();
    return {};
}

Result<void> populate_video_quant_cache(
    std::span<const float> q_encoder_values,
    std::span<const float> q_decoder_values,
    std::span<const float> q_feature_values,
    std::span<const float> q_reconstruction_values,
    VideoQuantDeviceCache& cache,
    cudaStream_t stream) {
    for (std::uint32_t qp = 0; qp < video_qp_count; ++qp) {
        auto q_encoder = upload_tensor(
            make_video_quant_tensor(q_encoder_values, "q_encoder", qp),
            "q_encoder", stream);
        if (!q_encoder) return q_encoder.error();
        cache.q_encoder[qp] = std::move(q_encoder.value());

        auto q_decoder = upload_tensor(
            make_video_quant_tensor(q_decoder_values, "q_decoder", qp),
            "q_decoder", stream);
        if (!q_decoder) return q_decoder.error();
        cache.q_decoder[qp] = std::move(q_decoder.value());

        auto q_feature = upload_tensor(
            make_video_quant_tensor(q_feature_values, "q_feature", qp),
            "q_feature", stream);
        if (!q_feature) return q_feature.error();
        cache.q_feature[qp] = std::move(q_feature.value());

        auto q_recon = upload_tensor(
            make_video_quant_tensor(q_reconstruction_values, "q_recon", qp),
            "q_recon", stream);
        if (!q_recon) return q_recon.error();
        cache.q_recon[qp] = std::move(q_recon.value());
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize quant cache");
    if (!synchronized) return synchronized.error();
    return {};
}


Result<DeviceTensor> take_device_tensor(
    std::vector<DeviceTensor>& tensors, std::string_view name) {
    for (auto& tensor : tensors) {
        if (tensor.name == name) return std::move(tensor);
    }
    return backend_error("TensorRT device output is missing " + std::string(name));
}

Result<std::vector<HostTensor>> download_tensors(
    std::span<const DeviceTensor* const> inputs, cudaStream_t stream) {
    std::vector<HostTensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto* input : inputs) {
        if (input == nullptr) return backend_error("null device tensor download");
        outputs.push_back(HostTensor{input->name, input->shape,
                                     std::vector<__half>(element_count(input->shape))});
        auto copied = copy_async(
            outputs.back().values.data(), input->storage.data,
            outputs.back().values.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream,
            "cudaMemcpyAsync download");
        if (!copied) return copied.error();
    }
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize");
    if (!synchronized) return synchronized.error();
    return outputs;
}

__half to_half(float value) { return __float2half_rn(value); }
float to_float(__half value) { return __half2float(value); }

std::int32_t round_up(std::int32_t value, std::int32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

std::size_t tensor_offset(
    const HostTensor& tensor, std::int32_t channel, std::int32_t y, std::int32_t x) {
    return (static_cast<std::size_t>(channel) * tensor.shape.d[2] + y) *
        tensor.shape.d[3] + x;
}

Result<Frame> ycbcr_to_yuv420p8(
    const HostTensor& tensor, std::uint32_t width, std::uint32_t height, Timestamp timestamp) {
    if ((width & 1U) != 0U || (height & 1U) != 0U) {
        return Error(
            ErrorCode::invalid_argument,
            "YUV420 reconstruction requires even dimensions",
            std::string(subsystem));
    }

    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto uv_width = width / 2U;
    const auto uv_size = static_cast<std::size_t>(uv_width) * (height / 2U);
    std::vector<std::byte> bytes(y_size + 2U * uv_size);
    auto* y_plane = bytes.data();
    auto* u_plane = bytes.data() + y_size;
    auto* v_plane = bytes.data() + y_size + uv_size;

    const auto quantize_y = [](float value) {
        return static_cast<std::byte>(static_cast<unsigned char>(
            std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F)));
    };
    const auto quantize_uv = [](float value) {
        return static_cast<std::byte>(static_cast<unsigned char>(
            std::clamp(value * 255.0F, 0.0F, 255.0F)));
    };

    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            y_plane[static_cast<std::size_t>(y) * width + x] =
                quantize_y(to_float(tensor.values[tensor_offset(tensor, 0, y, x)]));
        }
    }
    for (std::uint32_t y = 0; y < height; y += 2U) {
        for (std::uint32_t x = 0; x < width; x += 2U) {
            float u_sum = 0.0F;
            float v_sum = 0.0F;
            for (std::uint32_t dy = 0; dy < 2U; ++dy) {
                for (std::uint32_t dx = 0; dx < 2U; ++dx) {
                    u_sum += to_float(tensor.values[tensor_offset(tensor, 1, y + dy, x + dx)]);
                    v_sum += to_float(tensor.values[tensor_offset(tensor, 2, y + dy, x + dx)]);
                }
            }
            const auto chroma = static_cast<std::size_t>(y / 2U) * uv_width + x / 2U;
            u_plane[chroma] = quantize_uv(u_sum * 0.25F);
            v_plane[chroma] = quantize_uv(v_sum * 0.25F);
        }
    }
    return Frame::copy_from(width, height, PixelFormat::yuv420p8, bytes, timestamp);
}

Result<Frame> ycbcr_device_to_yuv420p8(
    const DeviceTensor& tensor,
    std::uint32_t width,
    std::uint32_t height,
    Timestamp timestamp,
    PinnedHostBuffer<std::byte>& host_buffer,
    cudaStream_t stream) {
    if ((width & 1U) != 0U || (height & 1U) != 0U) {
        return Error(
            ErrorCode::invalid_argument,
            "YUV420 reconstruction requires even dimensions",
            std::string(subsystem));
    }
    if (tensor.shape.nbDims != 4 || tensor.shape.d[0] != 1 || tensor.shape.d[1] != 3 ||
        tensor.shape.d[2] <= 0 || tensor.shape.d[3] <= 0 ||
        static_cast<std::uint32_t>(tensor.shape.d[2]) < height ||
        static_cast<std::uint32_t>(tensor.shape.d[3]) < width) {
        return backend_error("decoded frame_hat tensor has an invalid visible shape");
    }

    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto byte_count = y_size + y_size / 2U;
    auto output_device = allocate_cuda_scratch(byte_count);
    if (!output_device) return output_device.error();
    const auto converted = cuda_ops::ycbcr_padded_to_yuv420p8(
        tensor.storage.data,
        cuda_ops::Shape4D{1, 3, static_cast<std::int32_t>(tensor.shape.d[2]),
                          static_cast<std::int32_t>(tensor.shape.d[3])},
        static_cast<std::int32_t>(width), static_cast<std::int32_t>(height),
        static_cast<std::uint8_t*>(output_device.value().data), stream);
    if (converted != cudaSuccess) {
        return cuda_error("cuda_ops::ycbcr_padded_to_yuv420p8", converted);
    }
    auto ready = host_buffer.ensure(byte_count, "decoded_yuv420p8_frame");
    if (!ready) return ready.error();
    auto copied = copy_async(
        host_buffer.data, output_device.value().data, byte_count,
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync decoded YUV420P8 frame");
    if (!copied) return copied.error();
    auto synchronized = synchronize_stream(stream, "cudaStreamSynchronize decoded YUV420P8 frame");
    if (!synchronized) return synchronized.error();
    return Frame::copy_from(
        width, height, PixelFormat::yuv420p8,
        std::span<const std::byte>(host_buffer.data, byte_count), timestamp);
}


HostTensor make_quant_tensor(
    std::span<const float> values, std::uint32_t qp, std::string name) {
    HostTensor output{std::move(name), make_dims(368, 1, 1), std::vector<__half>(368)};
    const auto first = static_cast<std::size_t>(qp) * 368;
    for (std::size_t index = 0; index < 368; ++index) {
        output.values[index] = to_half(values[first + index]);
    }
    return output;
}

HostTensor make_video_quant_tensor(
    std::span<const float> values, std::string_view kind, std::uint32_t qp) {
    std::int32_t channels = 256;
    if (kind == "q_recon") channels = 320;
    HostTensor output{std::string(kind), make_dims(channels, 1, 1),
                      std::vector<__half>(static_cast<std::size_t>(channels))};
    const auto first = static_cast<std::size_t>(qp) * channels;
    for (std::int32_t index = 0; index < channels; ++index) {
        output.values[static_cast<std::size_t>(index)] =
            to_half(values[first + static_cast<std::size_t>(index)]);
    }
    return output;
}


Result<HostTensor> take_tensor(std::vector<HostTensor>& tensors, std::string_view name) {
    for (auto& tensor : tensors) {
        if (tensor.name == name) return std::move(tensor);
    }
    return backend_error("TensorRT output is missing " + std::string(name));
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (std::uint32_t shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}


std::vector<std::byte> serialize_dpb(
    const HostTensor& frame, const HostTensor* feature = nullptr) {
    std::vector<std::byte> output;
    const auto frame_bytes = frame.values.size() * sizeof(__half);
    const auto feature_bytes = feature == nullptr ? 0 : feature->values.size() * sizeof(__half);
    output.reserve(24 + frame_bytes + feature_bytes);
    output.insert(output.end(), {std::byte{0x4e}, std::byte{0x56},
                                 std::byte{0x44}, std::byte{0x31}});
    append_u32(output, static_cast<std::uint32_t>(frame.shape.d[2]));
    append_u32(output, static_cast<std::uint32_t>(frame.shape.d[3]));
    append_u32(output, feature == nullptr ? 0U : static_cast<std::uint32_t>(feature->shape.d[2]));
    append_u32(output, feature == nullptr ? 0U : static_cast<std::uint32_t>(feature->shape.d[3]));
    const auto* frame_data = reinterpret_cast<const std::byte*>(frame.values.data());
    output.insert(output.end(), frame_data, frame_data + frame_bytes);
    if (feature != nullptr) {
        const auto* feature_data = reinterpret_cast<const std::byte*>(feature->values.data());
        output.insert(output.end(), feature_data, feature_data + feature_bytes);
    }
    return output;
}

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    TensorRTExecutionSession& engines,
    IntraOrchestration& orchestration,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    cudaStream_t stream);

Result<CodecEncodeResult> encode_intra(
    const Frame& frame,
    std::uint32_t qp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    TensorRTExecutionSession& engines,
    IntraOrchestration& orchestration,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    PinnedHostBuffer<std::int8_t>& image_z_symbols_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes0_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes1_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes2_buffer,
    PinnedHostBuffer<std::int16_t>& image_indexes3_buffer,
    bool verify_reconstruction,
    cudaStream_t stream) {
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

    const auto padded_height = round_up(static_cast<std::int32_t>(frame.height()), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(frame.width()), 16);
    auto input_frame = allocate_device_tensor_scratch("frame", make_dims(3, padded_height, padded_width));
    if (!input_frame) return input_frame.error();
    auto input_bytes_device = allocate_cuda_scratch(frame.data().size());
    if (!input_bytes_device) return input_bytes_device.error();
    auto copied = copy_async(
        input_bytes_device.value().data, frame.data().data(), frame.data().size(),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync I-frame input bytes");
    if (!copied) return copied.error();
    {
        CpuProfileScope stage("i_input_color_convert_gpu");
        cudaError_t converted = cudaSuccess;
        if (frame.pixel_format() == PixelFormat::yuv420p8) {
            converted = cuda_ops::yuv420p8_to_ycbcr_padded(
                input_bytes_device.value().data, static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()), input_frame.value().storage.data,
                padded_height, padded_width, stream);
        } else {
            converted = cuda_ops::rgb24_to_ycbcr_padded(
                input_bytes_device.value().data, static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()), input_frame.value().storage.data,
                padded_height, padded_width, stream);
        }
        if (converted != cudaSuccess) return cuda_error("cuda_ops I-frame input color convert", converted);
    }

    auto& q_enc_opt = image_quant_cache.q_enc[qp];
    if (!q_enc_opt.has_value()) return backend_error("missing cached image q_enc tensor");
    auto* q_enc = &q_enc_opt.value();
    std::array<DeviceTensor*, 2> analysis_inputs{&input_frame.value(), q_enc};
    auto analysis_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::analysis), analysis_inputs,
        EngineOutputStorage::scratch);
    if (!analysis_outputs) return analysis_outputs.error();
    auto y_result = take_device_tensor(analysis_outputs.value(), "y");
    if (!y_result) return y_result.error();
    auto y_device = std::move(y_result.value());
    const auto y_height = static_cast<std::int32_t>(y_device.shape.d[2]);
    const auto y_width = static_cast<std::int32_t>(y_device.shape.d[3]);

    auto y_padded = allocate_device_tensor_scratch(
        "y_padded", make_dims(256, round_up(y_height, 4), round_up(y_width, 4)));
    if (!y_padded) return y_padded.error();
    const cuda_ops::Shape4D y_shape{1, 256, y_height, y_width};
    auto status = cuda_ops::replicate_pad(
        y_device.storage.data, y_padded.value().storage.data, y_shape,
        static_cast<std::int32_t>(y_padded.value().shape.d[2]),
        static_cast<std::int32_t>(y_padded.value().shape.d[3]), stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::replicate_pad", status);

    std::array<DeviceTensor*, 1> hyper_analysis_inputs{&y_padded.value()};
    auto hyper_analysis_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::hyper_analysis),
        hyper_analysis_inputs, EngineOutputStorage::scratch);
    if (!hyper_analysis_outputs) return hyper_analysis_outputs.error();
    auto z_result = take_device_tensor(hyper_analysis_outputs.value(), "z");
    if (!z_result) return z_result.error();
    auto z_device = std::move(z_result.value());
    z_device.name = "z_hat";
    const auto z_count = element_count(z_device.shape);
    auto z_symbol_device = allocate_cuda_scratch(z_count * sizeof(std::int8_t));
    if (!z_symbol_device) return z_symbol_device.error();
    status = cuda_ops::round_to_int8(
        z_device.storage.data, static_cast<std::int8_t*>(z_symbol_device.value().data),
        z_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::round_to_int8", status);

    std::array<DeviceTensor*, 1> hyper_synthesis_inputs{&z_device};
    auto hyper_synthesis_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::hyper_synthesis),
        hyper_synthesis_inputs, EngineOutputStorage::scratch);
    if (!hyper_synthesis_outputs) return hyper_synthesis_outputs.error();
    auto params_padded_result = take_device_tensor(hyper_synthesis_outputs.value(), "params_padded");
    auto common_padded_result = take_device_tensor(hyper_synthesis_outputs.value(), "common_padded");
    if (!params_padded_result) return params_padded_result.error();
    if (!common_padded_result) return common_padded_result.error();
    auto params_padded = std::move(params_padded_result.value());
    auto common_padded = std::move(common_padded_result.value());
    auto params = allocate_device_tensor_scratch("params", make_dims(514, y_height, y_width));
    auto common = allocate_device_tensor_scratch("common", make_dims(256, y_height, y_width));
    if (!params) return params.error();
    if (!common) return common.error();
    status = cuda_ops::crop_spatial(
        params_padded.storage.data, params.value().storage.data,
        cuda_ops::Shape4D{1, 514, static_cast<std::int32_t>(params_padded.shape.d[2]),
                          static_cast<std::int32_t>(params_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial params", status);
    status = cuda_ops::crop_spatial(
        common_padded.storage.data, common.value().storage.data,
        cuda_ops::Shape4D{1, 256, static_cast<std::int32_t>(common_padded.shape.d[2]),
                          static_cast<std::int32_t>(common_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial common", status);

    const auto spatial_count = static_cast<std::size_t>(y_height) * y_width;
    const auto prior_count = static_cast<std::size_t>(256) * spatial_count;
    status = cuda_ops::apply_image_prior_and_quantize(
        params.value().storage.data, y_device.storage.data, spatial_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::apply_image_prior_and_quantize", status);
    auto* params_values = static_cast<__half*>(params.value().storage.data);
    auto* q_dec_values = params_values + spatial_count;
    auto* current_scales = params_values + 2 * spatial_count;
    auto* current_means = params_values + (2 + 256) * spatial_count;

    const auto reduced_shape = make_dims(64, y_height, y_width);
    const auto reduced_count = prior_count / 4;
    std::array<std::optional<DeviceTensor>, 4> encoded_symbols;
    std::array<std::optional<DeviceTensor>, 4> encoded_scales;
    std::optional<DeviceTensor> y_hat_device;
    std::optional<DeviceTensor> spatial_params;
    for (std::int32_t stage_index = 0; stage_index < 4; ++stage_index) {
        const auto stage_slot = static_cast<std::size_t>(stage_index);
        auto mask = allocate_device_tensor_scratch("mask", y_device.shape);
        auto symbols_full = allocate_device_tensor_scratch("symbols_full", y_device.shape);
        auto reconstructed = allocate_device_tensor_scratch("reconstructed", y_device.shape);
        auto scales_full = allocate_device_tensor_scratch("scales_full", y_device.shape);
        if (!mask) return mask.error();
        if (!symbols_full) return symbols_full.error();
        if (!reconstructed) return reconstructed.error();
        if (!scales_full) return scales_full.error();
        status = cuda_ops::make_four_way_mask(mask.value().storage.data, y_shape, stage_index, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::make_four_way_mask", status);
        status = cuda_ops::process_with_mask(
            y_device.storage.data, current_scales, current_means, mask.value().storage.data,
            nullptr, symbols_full.value().storage.data, reconstructed.value().storage.data,
            scales_full.value().storage.data,
            prior_count, 0.0F, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);

        auto symbols = allocate_device_tensor_scratch("symbols", reduced_shape);
        auto scales = allocate_device_tensor_scratch("scales", reduced_shape);
        if (!symbols) return symbols.error();
        if (!scales) return scales.error();
        status = cuda_ops::reduce_channel_quarters(
            symbols_full.value().storage.data, symbols.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_quarters symbols", status);
        status = cuda_ops::reduce_channel_quarters(
            scales_full.value().storage.data, scales.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_quarters scales", status);
        encoded_symbols[stage_slot] = std::move(symbols.value());
        encoded_scales[stage_slot] = std::move(scales.value());

        if (!y_hat_device.has_value()) {
            reconstructed.value().name = "y_hat";
            y_hat_device = std::move(reconstructed.value());
        } else {
            status = cuda_ops::add_in_place(
                y_hat_device->storage.data, reconstructed.value().storage.data,
                prior_count, stream);
            if (status != cudaSuccess) return cuda_error("cuda_ops::add_in_place", status);
        }

        if (stage_index < 3) {
            auto spatial_context = allocate_device_tensor_scratch("context", make_dims(512, y_height, y_width));
            if (!spatial_context) return spatial_context.error();
            const auto y_bytes = prior_count * sizeof(__half);
            copied = copy_async(
                spatial_context.value().storage.data, y_hat_device->storage.data, y_bytes,
                cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync I spatial y");
            if (!copied) return copied.error();
            copied = copy_async(
                static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
                common.value().storage.data, y_bytes, cudaMemcpyDeviceToDevice, stream,
                "cudaMemcpyAsync I spatial common");
            if (!copied) return copied.error();
            std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
            const auto spatial_stage = static_cast<IntraStage>(
                static_cast<std::size_t>(IntraStage::spatial_prior_1) +
                static_cast<std::size_t>(stage_index));
            auto spatial_outputs = run_device_stage(
                engines, orchestration.stage(spatial_stage), spatial_inputs,
                EngineOutputStorage::scratch);
            if (!spatial_outputs) return spatial_outputs.error();
            auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
            if (!spatial_result) return spatial_result.error();
            spatial_params = std::move(spatial_result.value());
            auto* spatial_values = static_cast<__half*>(spatial_params->storage.data);
            current_scales = spatial_values;
            current_means = spatial_values + prior_count;
        }
    }

    status = cuda_ops::multiply_broadcast_in_place(
        y_hat_device->storage.data, q_dec_values, spatial_count, 256, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::multiply_broadcast_in_place", status);
    y_hat_device->name = "y_hat";

    std::array<CudaAllocation, 4> index_devices;
    for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
        auto allocation = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
        if (!allocation) return allocation.error();
        index_devices[stage_index] = std::move(allocation.value());
        status = cuda_ops::build_encode_indexes(
            encoded_symbols[stage_index]->storage.data, encoded_scales[stage_index]->storage.data,
            static_cast<std::int16_t*>(index_devices[stage_index].data), reduced_count,
            0.0F, stream, orchestration.exact_indexes());
        if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    }

    auto z_buffer_ready = image_z_symbols_buffer.ensure(z_count, "image_z_symbols");
    if (!z_buffer_ready) return z_buffer_ready.error();
    std::array<PinnedHostBuffer<std::int16_t>*, 4> index_buffers{
        &image_indexes0_buffer, &image_indexes1_buffer,
        &image_indexes2_buffer, &image_indexes3_buffer};
    for (auto* buffer : index_buffers) {
        auto ready = buffer->ensure(reduced_count, "image_indexes");
        if (!ready) return ready.error();
    }

    std::span<std::int8_t> z_symbols(image_z_symbols_buffer.data, z_count);
    copied = copy_async(
        z_symbols.data(), z_symbol_device.value().data, z_count * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync I z symbols");
    if (!copied) return copied.error();
    for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
        copied = copy_async(
            index_buffers[stage_index]->data, index_devices[stage_index].data,
            reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
            "cudaMemcpyAsync I indexes");
        if (!copied) return copied.error();
    }
    CudaEvent entropy_ready;
    status = cudaEventCreateWithFlags(&entropy_ready.data, cudaEventDisableTiming);
    if (status != cudaSuccess) return cuda_error("cudaEventCreateWithFlags", status);
    status = cudaEventRecord(entropy_ready.data, stream);
    if (status != cudaSuccess) return cuda_error("cudaEventRecord I entropy", status);

    // Reconstruction depends on y_hat and q_dec, but not on the CPU rANS
    // output. Queue it behind the entropy D2H copies so that, once those
    // copies complete, CPU entropy coding can overlap I-frame synthesis just
    // as it already does in the predicted-frame path.
    auto& q_dec_opt = image_quant_cache.q_dec[qp];
    if (!q_dec_opt.has_value()) return backend_error("missing cached image q_dec tensor");
    auto* q_dec = &q_dec_opt.value();
    std::array<DeviceTensor*, 2> synthesis_inputs{&*y_hat_device, q_dec};
    auto synthesis_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::synthesis), synthesis_inputs);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    if (!frame_device_result) return frame_device_result.error();
    auto frame_device = std::move(frame_device_result.value());

    status = cudaEventSynchronize(entropy_ready.data);
    if (status != cudaSuccess) return cuda_error("cudaEventSynchronize I entropy", status);

    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    auto mode = orchestration.begin_encode(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z_device.shape.d[2] * z_device.shape.d[3]);
    auto payload = [&]() -> Result<std::vector<std::byte>> {
        CpuProfileScope entropy_scope("i_entropy_encode");
        auto encoded_z = orchestration.encode_z(z_symbols, qp, per_channel);
        if (!encoded_z) return encoded_z.error();
        for (std::size_t stage_index = 0; stage_index < 4; ++stage_index) {
            std::span<std::int16_t> indexes(index_buffers[stage_index]->data, reduced_count);
            auto encoded_y = orchestration.encode_y(indexes);
            if (!encoded_y) return encoded_y.error();
        }
        return orchestration.finish_encode(
            frame.width(), frame.height(), qp, two_coders);
    }();
    if (!payload) return payload.error();

    std::array<const DeviceTensor*, 1> tensors{&frame_device};
    auto downloaded = download_tensors(tensors, stream);
    if (!downloaded) return downloaded.error();
    auto reconstructed_result = take_tensor(downloaded.value(), "frame_hat");
    if (!reconstructed_result) return reconstructed_result.error();
    auto reconstructed = ycbcr_to_yuv420p8(
        reconstructed_result.value(), frame.width(), frame.height(), frame.timestamp());
    if (!reconstructed) return reconstructed.error();

    frame_device.name = "reference_frame";
    device_dpb.clear();
    device_dpb.frame = std::move(frame_device);
    device_dpb.semantics.commit(state, false);
    std::vector<std::byte> latent_state;

    if (verify_reconstruction) {
        auto conformant_reconstruction = decode_intra(
            payload.value(), frame.timestamp(), state, device_dpb, engines,
            orchestration, image_quant_cache, image_decode_staging, stream);
        if (!conformant_reconstruction) return conformant_reconstruction.error();
        return CodecEncodeResult{
            std::move(payload.value()), std::move(conformant_reconstruction.value().frame),
            std::move(conformant_reconstruction.value().latent_state), qp};
    }

    return CodecEncodeResult{
        std::move(payload.value()), std::move(reconstructed.value()), std::move(latent_state), qp};
}

Result<CodecDecodeResult> decode_intra(
    std::span<const std::byte> payload,
    Timestamp timestamp,
    const SequenceStateView& state,
    DeviceDpb& device_dpb,
    TensorRTExecutionSession& engines,
    IntraOrchestration& orchestration,
    ImageQuantDeviceCache& image_quant_cache,
    ImageDecodeStaging& image_decode_staging,
    cudaStream_t stream) {
    auto parsed = orchestration.parse_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    const auto padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = padded_height / 16;
    const auto y_width = padded_width / 16;
    const auto z_height = round_up(y_height, 4) / 4;
    const auto z_width = round_up(y_width, 4) / 4;

    auto stream_set = orchestration.begin_decode(info);
    if (!stream_set) return stream_set.error();
    const auto per_channel = static_cast<std::size_t>(z_height * z_width);
    auto z_values = [&]() -> Result<std::vector<std::int8_t>> {
        CpuProfileScope entropy_scope("i_entropy_decode_z");
        return orchestration.decode_z(128 * per_channel, info.qp, per_channel);
    }();
    if (!z_values) return z_values.error();
    const auto staging_reduced_count =
        static_cast<std::size_t>(64) * y_height * y_width;
    auto staging_ready = image_decode_staging.ensure(
        z_values.value().size(), staging_reduced_count);
    if (!staging_ready) return staging_ready.error();
    std::copy(
        z_values.value().begin(), z_values.value().end(),
        image_decode_staging.z_symbols.data);
    auto z_device = upload_int8_tensor_scratch(
        std::span<const std::int8_t>(
            image_decode_staging.z_symbols.data, z_values.value().size()),
        make_dims(128, z_height, z_width), "z_hat", stream);
    if (!z_device) return z_device.error();

    std::array<DeviceTensor*, 1> hyper_inputs{&z_device.value()};
    auto hyper_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::hyper_synthesis), hyper_inputs,
        EngineOutputStorage::scratch);
    if (!hyper_outputs) return hyper_outputs.error();
    auto params_padded_result = take_device_tensor(hyper_outputs.value(), "params_padded");
    auto common_padded_result = take_device_tensor(hyper_outputs.value(), "common_padded");
    if (!params_padded_result) return params_padded_result.error();
    if (!common_padded_result) return common_padded_result.error();
    auto params_padded = std::move(params_padded_result.value());
    auto common_padded = std::move(common_padded_result.value());
    auto params = allocate_device_tensor_scratch("params", make_dims(514, y_height, y_width));
    auto common = allocate_device_tensor_scratch("common", make_dims(256, y_height, y_width));
    if (!params) return params.error();
    if (!common) return common.error();
    auto status = cuda_ops::crop_spatial(
        params_padded.storage.data, params.value().storage.data,
        cuda_ops::Shape4D{1, 514, static_cast<std::int32_t>(params_padded.shape.d[2]),
                          static_cast<std::int32_t>(params_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial I params", status);
    status = cuda_ops::crop_spatial(
        common_padded.storage.data, common.value().storage.data,
        cuda_ops::Shape4D{1, 256, static_cast<std::int32_t>(common_padded.shape.d[2]),
                          static_cast<std::int32_t>(common_padded.shape.d[3])},
        y_height, y_width, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::crop_spatial I common", status);

    const auto spatial_count = static_cast<std::size_t>(y_height) * y_width;
    const auto prior_count = static_cast<std::size_t>(256) * spatial_count;
    const auto reduced_shape = make_dims(64, y_height, y_width);
    const auto reduced_count = prior_count / 4;
    const cuda_ops::Shape4D y_shape{1, 256, y_height, y_width};
    status = cuda_ops::apply_image_prior(
        params.value().storage.data, spatial_count, stream);
    if (status != cudaSuccess) {
        return cuda_error("cuda_ops::apply_image_prior I decode", status);
    }
    auto* params_values = static_cast<__half*>(params.value().storage.data);
    auto* q_dec_values = params_values + spatial_count;
    auto* current_scales = params_values + 2 * spatial_count;
    auto* current_means = params_values + (2 + 256) * spatial_count;

    std::optional<DeviceTensor> y_hat_device;
    std::optional<DeviceTensor> spatial_params;
    for (std::int32_t stage = 0; stage < 4; ++stage) {
        auto mask = allocate_device_tensor_scratch("mask", make_dims(256, y_height, y_width));
        auto scales = allocate_device_tensor_scratch("scales", reduced_shape);
        if (!mask) return mask.error();
        if (!scales) return scales.error();
        status = cuda_ops::make_four_way_mask(mask.value().storage.data, y_shape, stage, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::make_four_way_mask I decode", status);
        status = cuda_ops::reduce_masked_quarters(
            current_scales, mask.value().storage.data, scales.value().storage.data,
            y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_quarters I decode scales", status);

        auto indexes_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
        if (!indexes_device) return indexes_device.error();
        status = cuda_ops::build_decode_indexes(
            scales.value().storage.data, static_cast<std::uint8_t*>(indexes_device.value().data),
            reduced_count, 0.0F, stream, orchestration.exact_indexes());
        if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes I", status);
        const auto stage_slot = static_cast<std::size_t>(stage);
        auto& indexes = image_decode_staging.indexes[stage_slot];
        auto copied = copy_async(
            indexes.data, indexes_device.value().data, reduced_count,
            cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync I decode indexes");
        if (!copied) return copied.error();
        auto recorded = record_cuda_event(
            image_decode_staging.indexes_ready[stage_slot].data, stream,
            "cudaEventRecord I decode indexes");
        if (!recorded) return recorded.error();
        auto synchronized = synchronize_event(
            image_decode_staging.indexes_ready[stage_slot].data,
            "cudaEventSynchronize I decode indexes");
        if (!synchronized) return synchronized.error();

        auto decoded = [&]() -> Result<std::vector<std::int8_t>> {
            CpuProfileScope entropy_scope("i_entropy_decode_y");
            return orchestration.decode_y(
                std::span<const std::uint8_t>(indexes.data, reduced_count));
        }();
        if (!decoded) return decoded.error();
        auto& symbols = image_decode_staging.symbols[stage_slot];
        std::copy(decoded.value().begin(), decoded.value().end(), symbols.data);
        auto symbols_device = upload_int8_tensor_scratch(
            std::span<const std::int8_t>(symbols.data, decoded.value().size()),
            reduced_shape, "symbols", stream);
        if (!symbols_device) return symbols_device.error();
        auto reconstructed = allocate_device_tensor_scratch("reconstructed", make_dims(256, y_height, y_width));
        if (!reconstructed) return reconstructed.error();
        status = cuda_ops::restore_four_way(
            symbols_device.value().storage.data, current_means, mask.value().storage.data,
            reconstructed.value().storage.data, y_shape, stream);
        if (status != cudaSuccess) return cuda_error("cuda_ops::restore_four_way I", status);
        if (!y_hat_device.has_value()) {
            reconstructed.value().name = "y_hat";
            y_hat_device = std::move(reconstructed.value());
        } else {
            status = cuda_ops::add_in_place(
                y_hat_device->storage.data, reconstructed.value().storage.data,
                prior_count, stream);
            if (status != cudaSuccess) return cuda_error("cuda_ops::add_in_place I decode", status);
        }

        if (stage < 3) {
            auto spatial_context = allocate_device_tensor_scratch("context", make_dims(512, y_height, y_width));
            if (!spatial_context) return spatial_context.error();
            const auto y_bytes = prior_count * sizeof(__half);
            copied = copy_async(
                spatial_context.value().storage.data, y_hat_device->storage.data, y_bytes,
                cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync I decode spatial y");
            if (!copied) return copied.error();
            copied = copy_async(
                static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
                common.value().storage.data, y_bytes, cudaMemcpyDeviceToDevice, stream,
                "cudaMemcpyAsync I decode spatial common");
            if (!copied) return copied.error();
            std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
            const auto spatial_stage = static_cast<IntraStage>(
                static_cast<std::size_t>(IntraStage::spatial_prior_1) +
                static_cast<std::size_t>(stage));
            auto spatial_outputs = run_device_stage(
                engines, orchestration.stage(spatial_stage), spatial_inputs,
                EngineOutputStorage::scratch);
            if (!spatial_outputs) return spatial_outputs.error();
            auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
            if (!spatial_result) return spatial_result.error();
            spatial_params = std::move(spatial_result.value());
            auto* spatial_values = static_cast<__half*>(spatial_params->storage.data);
            current_scales = spatial_values;
            current_means = spatial_values + prior_count;
        }
    }

    auto entropy_complete = orchestration.end_decode();
    if (!entropy_complete) return entropy_complete.error();

    status = cuda_ops::multiply_broadcast_in_place(
        y_hat_device->storage.data, q_dec_values, spatial_count, 256, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::multiply_broadcast_in_place I decode", status);
    y_hat_device->name = "y_hat";
    auto& q_dec_opt = image_quant_cache.q_dec[info.qp];
    if (!q_dec_opt.has_value()) return backend_error("missing cached image q_dec tensor");
    auto* q_dec = &q_dec_opt.value();
    std::array<DeviceTensor*, 2> synthesis_inputs{&*y_hat_device, q_dec};
    auto synthesis_outputs = run_device_stage(
        engines, orchestration.stage(IntraStage::synthesis), synthesis_inputs);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    if (!frame_device_result) return frame_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    std::array<const DeviceTensor*, 1> reconstruction_inputs{&frame_device};
    auto reconstruction = download_tensors(reconstruction_inputs, stream);
    if (!reconstruction) return reconstruction.error();
    auto frame_result = take_tensor(reconstruction.value(), "frame_hat");
    if (!frame_result) return frame_result.error();
    auto frame = ycbcr_to_yuv420p8(frame_result.value(), info.width, info.height, timestamp);
    if (!frame) return frame.error();
    frame_device.name = "reference_frame";
    device_dpb.clear();
    device_dpb.frame = std::move(frame_device);
    device_dpb.semantics.commit(state, false);
    auto latent_state = serialize_dpb(frame_result.value());
    return CodecDecodeResult{std::move(frame.value()), std::move(latent_state)};
}

Result<CodecEncodeResult> encode_predicted(
    const Frame& frame, std::uint32_t base_qp, const SequenceStateView& state,
    DeviceDpb& device_dpb,
    TensorRTExecutionSession& engines, PredictedOrchestration& orchestration,
    VideoQuantDeviceCache& quant_cache,
    PinnedHostBuffer<std::int8_t>& z_symbols_buffer,
    PinnedHostBuffer<std::int16_t>& indexes0_buffer,
    PinnedHostBuffer<std::int16_t>& indexes1_buffer,
    bool verify_reconstruction, cudaStream_t stream) {
    if (frame.pixel_format() != PixelFormat::rgb24 &&
        frame.pixel_format() != PixelFormat::yuv420p8) {
        return Error(ErrorCode::invalid_argument, "native P-frame encoding requires RGB24 or YUV420P8",
                     std::string(subsystem));
    }
    if (!supported_frame_dimensions(frame.width(), frame.height())) {
        return Error(ErrorCode::invalid_argument,
                     "native P-frame dimensions must be even and within 64x64 and 1920x1080",
                     std::string(subsystem));
    }
    if (!device_dpb.frame) {
        return Error(ErrorCode::invalid_state, "encoder device DPB is unavailable",
                     std::string(subsystem));
    }
    auto decision = orchestration.select_frame(base_qp, state, device_dpb.semantics);
    if (!decision) return decision.error();
    const auto qp = decision.value().qp;
    const auto use_frame_reference = decision.value().use_frame_reference;
    if (!use_frame_reference && !device_dpb.feature) {
        return Error(ErrorCode::invalid_state,
                     "P-frame feature storage is unavailable",
                     std::string(subsystem));
    }

    const std::string reference_name = use_frame_reference ? "reference_frame" : "reference_feature";
    auto* reference_device = use_frame_reference ? &*device_dpb.frame : &*device_dpb.feature;
    reference_device->name = reference_name;
    auto& q_feature_opt = quant_cache.q_feature[qp];
    if (!q_feature_opt.has_value()) return backend_error("missing cached q_feature tensor");
    auto* q_feature_device = &q_feature_opt.value();
    std::array<DeviceTensor*, 2> reference_inputs{
        reference_device, q_feature_device};
    const auto reference_stage = use_frame_reference ?
        PredictedStage::reference_frame : PredictedStage::reference_feature;
    auto reference_outputs = run_device_stage(
        engines, orchestration.stage(reference_stage), reference_inputs,
        EngineOutputStorage::scratch);
    if (!reference_outputs) return reference_outputs.error();
    auto context_result = take_device_tensor(reference_outputs.value(), "context");
    auto temporal_result = take_device_tensor(reference_outputs.value(), "temporal_context");
    if (!context_result) return context_result.error();
    if (!temporal_result) return temporal_result.error();
    auto context = std::move(context_result.value());
    auto temporal_context = std::move(temporal_result.value());

    const auto padded_height = round_up(static_cast<std::int32_t>(frame.height()), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(frame.width()), 16);
    auto input_frame = allocate_device_tensor_scratch(
        "frame", make_dims(3, padded_height, padded_width));
    if (!input_frame) return input_frame.error();
    auto input_bytes_device = allocate_cuda_scratch(frame.data().size());
    if (!input_bytes_device) return input_bytes_device.error();
    auto input_bytes_copied = copy_async(
        input_bytes_device.value().data, frame.data().data(), frame.data().size(),
        cudaMemcpyHostToDevice, stream, "cudaMemcpyAsync frame input bytes");
    if (!input_bytes_copied) return input_bytes_copied.error();
    {
        CpuProfileScope stage("p_input_color_convert_gpu");
        cudaError_t converted = cudaSuccess;
        if (frame.pixel_format() == PixelFormat::yuv420p8) {
            converted = cuda_ops::yuv420p8_to_ycbcr_padded(
                input_bytes_device.value().data,
                static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()),
                input_frame.value().storage.data,
                padded_height,
                padded_width,
                stream);
        } else {
            converted = cuda_ops::rgb24_to_ycbcr_padded(
                input_bytes_device.value().data,
                static_cast<std::int32_t>(frame.width()),
                static_cast<std::int32_t>(frame.height()),
                input_frame.value().storage.data,
                padded_height,
                padded_width,
                stream);
        }
        if (converted != cudaSuccess) {
            return cuda_error("cuda_ops input color convert", converted);
        }
    }
    auto& q_encoder_opt = quant_cache.q_encoder[qp];
    if (!q_encoder_opt.has_value()) return backend_error("missing cached q_encoder tensor");
    auto* q_encoder = &q_encoder_opt.value();
    std::array<DeviceTensor*, 3> analysis_inputs{
        &input_frame.value(), &context, q_encoder};
    auto analysis_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::analysis), analysis_inputs,
        EngineOutputStorage::scratch);
    if (!analysis_outputs) return analysis_outputs.error();
    auto y_result = take_device_tensor(analysis_outputs.value(), "y");
    if (!y_result) return y_result.error();
    auto y_device = std::move(y_result.value());
    const auto y_height = static_cast<std::int32_t>(y_device.shape.d[2]);
    const auto y_width = static_cast<std::int32_t>(y_device.shape.d[3]);

    auto y_padded = allocate_device_tensor_scratch(
        "y_padded", make_dims(128, round_up(y_height, 4), round_up(y_width, 4)));
    if (!y_padded) return y_padded.error();
    const cuda_ops::Shape4D y_shape{1, 128, y_height, y_width};
    const auto padded = cuda_ops::replicate_pad(
        y_device.storage.data, y_padded.value().storage.data, y_shape,
        static_cast<std::int32_t>(y_padded.value().shape.d[2]),
        static_cast<std::int32_t>(y_padded.value().shape.d[3]), stream);
    if (padded != cudaSuccess) return cuda_error("cuda_ops::replicate_pad", padded);
    std::array<DeviceTensor*, 1> hyper_inputs{&y_padded.value()};
    auto hyper_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::hyper_analysis), hyper_inputs,
        EngineOutputStorage::scratch);
    if (!hyper_outputs) return hyper_outputs.error();
    auto z_result = take_device_tensor(hyper_outputs.value(), "z");
    if (!z_result) return z_result.error();
    auto z_device = std::move(z_result.value());
    z_device.name = "z_hat";
    const auto z_count = element_count(z_device.shape);
    auto z_symbol_device = allocate_cuda_scratch(z_count * sizeof(std::int8_t));
    if (!z_symbol_device) return z_symbol_device.error();
    const auto rounded = cuda_ops::round_to_int8(
        z_device.storage.data, static_cast<std::int8_t*>(z_symbol_device.value().data),
        z_count, stream);
    if (rounded != cudaSuccess) return cuda_error("cuda_ops::round_to_int8", rounded);

    std::array<DeviceTensor*, 2> prior_inputs{&z_device, &temporal_context};
    auto prior_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::prior), prior_inputs,
        EngineOutputStorage::scratch);
    if (!prior_outputs) return prior_outputs.error();
    auto params_result = take_device_tensor(prior_outputs.value(), "params");
    if (!params_result) return params_result.error();
    auto params_device = std::move(params_result.value());

    const cuda_ops::Shape4D prior_shape{1, 128, y_height, y_width};
    const auto prior_count = static_cast<std::size_t>(128) * y_height * y_width;
    auto* params_values = static_cast<__half*>(params_device.storage.data);
    auto* q_dec_values = params_values;
    auto* scales_values = params_values + prior_count;
    auto* means_values = params_values + 2 * prior_count;
    const auto quantized = cuda_ops::clamp_reciprocal_with_quant(
        q_dec_values, y_device.storage.data, prior_count, stream);
    if (quantized != cudaSuccess) {
        return cuda_error("cuda_ops::clamp_reciprocal_with_quant", quantized);
    }

    auto mask0 = allocate_device_tensor_scratch("mask0", y_device.shape);
    auto symbols_full0 = allocate_device_tensor_scratch("symbols_full0", y_device.shape);
    auto y_hat_device = allocate_device_tensor_scratch("y_hat", y_device.shape);
    auto scales_full0 = allocate_device_tensor_scratch("scales_full0", y_device.shape);
    if (!mask0) return mask0.error();
    if (!symbols_full0) return symbols_full0.error();
    if (!y_hat_device) return y_hat_device.error();
    if (!scales_full0) return scales_full0.error();
    auto status = cuda_ops::make_two_way_mask(mask0.value().storage.data, prior_shape, 0, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::process_with_mask(
        y_device.storage.data, scales_values, means_values, mask0.value().storage.data,
        nullptr, symbols_full0.value().storage.data, y_hat_device.value().storage.data,
        scales_full0.value().storage.data, prior_count,
        0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);

    const auto reduced_shape = make_dims(64, y_height, y_width);
    auto symbols0 = allocate_device_tensor_scratch("symbols0", reduced_shape);
    auto scales0 = allocate_device_tensor_scratch("scales0", reduced_shape);
    if (!symbols0) return symbols0.error();
    if (!scales0) return scales0.error();
    status = cuda_ops::reduce_channel_halves(
        symbols_full0.value().storage.data, symbols0.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::reduce_channel_halves(
        scales_full0.value().storage.data, scales0.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);

    auto spatial_context = allocate_device_tensor_scratch(
        "context", make_dims(512, y_height, y_width));
    if (!spatial_context) return spatial_context.error();
    const auto y_bytes = prior_count * sizeof(__half);
    const auto params_bytes = 3 * y_bytes;
    auto copied = copy_async(
        spatial_context.value().storage.data, y_hat_device.value().storage.data, y_bytes,
        cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync spatial y");
    if (!copied) return copied.error();
    copied = copy_async(
        static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
        params_device.storage.data, params_bytes, cudaMemcpyDeviceToDevice, stream,
        "cudaMemcpyAsync spatial params");
    if (!copied) return copied.error();
    std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
    auto spatial_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::spatial_prior), spatial_inputs,
        EngineOutputStorage::scratch);
    if (!spatial_outputs) return spatial_outputs.error();
    auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
    if (!spatial_result) return spatial_result.error();
    auto spatial_params = std::move(spatial_result.value());
    auto* spatial_values = static_cast<__half*>(spatial_params.storage.data);

    auto mask1 = allocate_device_tensor_scratch("mask1", y_device.shape);
    auto symbols_full1 = allocate_device_tensor_scratch("symbols_full1", y_device.shape);
    auto reconstructed1 = allocate_device_tensor_scratch("reconstructed1", y_device.shape);
    auto scales_full1 = allocate_device_tensor_scratch("scales_full1", y_device.shape);
    if (!mask1) return mask1.error();
    if (!symbols_full1) return symbols_full1.error();
    if (!reconstructed1) return reconstructed1.error();
    if (!scales_full1) return scales_full1.error();
    status = cuda_ops::make_two_way_mask(mask1.value().storage.data, prior_shape, 1, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::process_with_mask(
        y_device.storage.data, spatial_values, spatial_values + prior_count,
        mask1.value().storage.data, nullptr, symbols_full1.value().storage.data,
        reconstructed1.value().storage.data,
        scales_full1.value().storage.data, prior_count, 0.0F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::process_with_mask", status);
    auto symbols1 = allocate_device_tensor_scratch("symbols1", reduced_shape);
    auto scales1 = allocate_device_tensor_scratch("scales1", reduced_shape);
    if (!symbols1) return symbols1.error();
    if (!scales1) return scales1.error();
    status = cuda_ops::reduce_channel_halves(
        symbols_full1.value().storage.data, symbols1.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::reduce_channel_halves(
        scales_full1.value().storage.data, scales1.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_channel_halves", status);
    status = cuda_ops::add_and_multiply(
        y_hat_device.value().storage.data, reconstructed1.value().storage.data, q_dec_values,
        prior_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::add_and_multiply", status);

    const auto reduced_count = prior_count / 2;
    auto indexes0_device = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
    auto indexes1_device = allocate_cuda_scratch(reduced_count * sizeof(std::int16_t));
    if (!indexes0_device) return indexes0_device.error();
    if (!indexes1_device) return indexes1_device.error();
    status = cuda_ops::build_encode_indexes(
        symbols0.value().storage.data, scales0.value().storage.data,
        static_cast<std::int16_t*>(indexes0_device.value().data), reduced_count, 0.0F, stream,
        orchestration.exact_indexes());
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    status = cuda_ops::build_encode_indexes(
        symbols1.value().storage.data, scales1.value().storage.data,
        static_cast<std::int16_t*>(indexes1_device.value().data), reduced_count, 0.0F, stream,
        orchestration.exact_indexes());
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_encode_indexes", status);
    auto z_buffer_ready = z_symbols_buffer.ensure(z_count, "z_symbols");
    if (!z_buffer_ready) return z_buffer_ready.error();
    auto indexes0_ready = indexes0_buffer.ensure(reduced_count, "indexes0");
    if (!indexes0_ready) return indexes0_ready.error();
    auto indexes1_ready = indexes1_buffer.ensure(reduced_count, "indexes1");
    if (!indexes1_ready) return indexes1_ready.error();

    std::span<std::int8_t> z_symbols(z_symbols_buffer.data, z_count);
    std::span<std::int16_t> indexes0(indexes0_buffer.data, reduced_count);
    std::span<std::int16_t> indexes1(indexes1_buffer.data, reduced_count);
    copied = copy_async(
        z_symbols.data(), z_symbol_device.value().data, z_count * sizeof(std::int8_t),
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync z symbols");
    if (!copied) return copied.error();
    copied = copy_async(
        indexes0.data(), indexes0_device.value().data,
        reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
        "cudaMemcpyAsync indexes0");
    if (!copied) return copied.error();
    copied = copy_async(
        indexes1.data(), indexes1_device.value().data,
        reduced_count * sizeof(std::int16_t), cudaMemcpyDeviceToHost, stream,
        "cudaMemcpyAsync indexes1");
    if (!copied) return copied.error();

    CudaEvent entropy_ready;
    status = cudaEventCreateWithFlags(&entropy_ready.data, cudaEventDisableTiming);
    if (status != cudaSuccess) return cuda_error("cudaEventCreateWithFlags", status);
    status = cudaEventRecord(entropy_ready.data, stream);
    if (status != cudaSuccess) return cuda_error("cudaEventRecord entropy", status);

    auto& q_decoder_opt = quant_cache.q_decoder[qp];
    if (!q_decoder_opt.has_value()) return backend_error("missing cached q_decoder tensor");
    auto& q_recon_opt = quant_cache.q_recon[qp];
    if (!q_recon_opt.has_value()) return backend_error("missing cached q_recon tensor");
    auto* q_decoder = &q_decoder_opt.value();
    auto* q_recon = &q_recon_opt.value();
    std::array<DeviceTensor*, 4> synthesis_inputs{
        &y_hat_device.value(), &context, q_decoder, q_recon};
    auto synthesis_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::synthesis), synthesis_inputs);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    auto feature_device_result = take_device_tensor(synthesis_outputs.value(), "feature");
    if (!frame_device_result) return frame_device_result.error();
    if (!feature_device_result) return feature_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    auto feature_device = std::move(feature_device_result.value());
    status = cudaEventSynchronize(entropy_ready.data);
    if (status != cudaSuccess) return cuda_error("cudaEventSynchronize entropy", status);

    const auto entropy_start = std::chrono::steady_clock::now();
    const bool two_coders =
        static_cast<std::uint64_t>(frame.width()) * frame.height() >= 1280ULL * 720ULL;
    auto mode = orchestration.begin_encode(two_coders);
    if (!mode) return mode.error();
    const auto per_channel = static_cast<std::size_t>(z_device.shape.d[2] * z_device.shape.d[3]);
    auto encoded_z = orchestration.encode_z(z_symbols, qp, per_channel);
    if (!encoded_z) return encoded_z.error();
    auto encoded_y0 = orchestration.encode_y(indexes0);
    if (!encoded_y0) return encoded_y0.error();
    auto encoded_y1 = orchestration.encode_y(indexes1);
    if (!encoded_y1) return encoded_y1.error();
    auto payload = orchestration.finish_encode(
        frame.width(), frame.height(), qp, two_coders, use_frame_reference);
    if (!payload) return payload.error();
    if (active_profiler != nullptr && active_profiler->enabled()) {
        active_profiler->record_cpu_stage(
            "p_entropy_encode",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - entropy_start).count());
    }
    Frame reconstructed_frame;
    if (verify_reconstruction) {
        std::array<const DeviceTensor*, 1> tensors{&frame_device};
        auto downloaded = download_tensors(tensors, stream);
        if (!downloaded) return downloaded.error();
        auto reconstructed = ycbcr_to_yuv420p8(
            downloaded.value().front(), frame.width(), frame.height(), frame.timestamp());
        if (!reconstructed) return reconstructed.error();
        reconstructed_frame = std::move(reconstructed.value());
    }
    device_dpb.frame = std::move(frame_device);
    device_dpb.feature = std::move(feature_device);
    device_dpb.semantics.commit(state, true);
    std::vector<std::byte> latent_state;
    return CodecEncodeResult{
        std::move(payload.value()), std::move(reconstructed_frame),
        std::move(latent_state), qp};
}

Result<CodecDecodeResult> decode_predicted(
    std::span<const std::byte> payload, Timestamp timestamp, const SequenceStateView& state,
    DeviceDpb& device_dpb,
    TensorRTExecutionSession& engines, PredictedOrchestration& orchestration,
    VideoQuantDeviceCache& quant_cache,
    VideoDecodeStaging& decode_staging,
    PinnedHostBuffer<std::byte>& output_frame_buffer,
    cudaStream_t stream) {
    auto parsed = orchestration.parse_payload(payload);
    if (!parsed) return parsed.error();
    const auto info = parsed.value();
    if (!device_dpb.frame) {
        return Error(ErrorCode::invalid_state, "decoder device DPB is unavailable",
                     std::string(subsystem));
    }
    auto reference_valid = orchestration.validate_decode_reference(
        info.use_frame_reference, state, device_dpb.semantics);
    if (!reference_valid) return reference_valid.error();
    if (!info.use_frame_reference && !device_dpb.feature) {
        return Error(ErrorCode::invalid_state,
                     "P-frame feature storage is unavailable",
                     std::string(subsystem));
    }
    auto* reference_device = info.use_frame_reference ? &*device_dpb.frame :
                                                        &*device_dpb.feature;
    const std::string reference_name = info.use_frame_reference ?
        "reference_frame" : "reference_feature";
    reference_device->name = reference_name;
    auto& q_feature_opt = quant_cache.q_feature[info.qp];
    if (!q_feature_opt.has_value()) return backend_error("missing cached q_feature tensor");
    auto* q_feature_device = &q_feature_opt.value();
    std::array<DeviceTensor*, 2> reference_inputs{
        reference_device, q_feature_device};
    const auto reference_stage = info.use_frame_reference ?
        PredictedStage::reference_frame : PredictedStage::reference_feature;
    auto reference_outputs = run_device_stage(
        engines, orchestration.stage(reference_stage), reference_inputs,
        EngineOutputStorage::scratch);
    if (!reference_outputs) return reference_outputs.error();
    auto context_result = take_device_tensor(reference_outputs.value(), "context");
    auto temporal_result = take_device_tensor(reference_outputs.value(), "temporal_context");
    if (!context_result) return context_result.error();
    if (!temporal_result) return temporal_result.error();
    auto context = std::move(context_result.value());
    auto temporal_context = std::move(temporal_result.value());

    const auto padded_height = round_up(static_cast<std::int32_t>(info.height), 16);
    const auto padded_width = round_up(static_cast<std::int32_t>(info.width), 16);
    const auto y_height = padded_height / 16;
    const auto y_width = padded_width / 16;
    const auto z_height = round_up(y_height, 4) / 4;
    const auto z_width = round_up(y_width, 4) / 4;
    auto stream_set = orchestration.begin_decode(info);
    if (!stream_set) return stream_set.error();
    const auto per_channel = static_cast<std::size_t>(z_height * z_width);
    auto z_values = [&]() -> Result<std::vector<std::int8_t>> {
        CpuProfileScope entropy_scope("p_entropy_decode_z");
        return orchestration.decode_z(128 * per_channel, info.qp, per_channel);
    }();
    if (!z_values) return z_values.error();
    const auto reduced_count = static_cast<std::size_t>(64) * y_height * y_width;
    auto staging_ready = decode_staging.ensure(z_values.value().size(), reduced_count);
    if (!staging_ready) return staging_ready.error();
    std::copy(
        z_values.value().begin(), z_values.value().end(), decode_staging.z_symbols.data);
    auto z_device = upload_int8_tensor_scratch(
        std::span<const std::int8_t>(
            decode_staging.z_symbols.data, z_values.value().size()),
        make_dims(128, z_height, z_width), "z_hat", stream);
    if (!z_device) return z_device.error();
    std::array<DeviceTensor*, 2> prior_inputs{&z_device.value(), &temporal_context};
    auto prior_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::prior), prior_inputs,
        EngineOutputStorage::scratch);
    if (!prior_outputs) return prior_outputs.error();
    auto params_device_result = take_device_tensor(prior_outputs.value(), "params");
    if (!params_device_result) return params_device_result.error();
    auto params_device = std::move(params_device_result.value());
    const cuda_ops::Shape4D prior_shape{1, 128, y_height, y_width};
    const auto prior_count = static_cast<std::size_t>(128) * y_height * y_width;
    const auto reduced_shape = make_dims(64, y_height, y_width);
    auto* params_values = static_cast<__half*>(params_device.storage.data);
    auto* q_dec_values = params_values;
    auto* scales_values = params_values + prior_count;
    auto* means_values = params_values + 2 * prior_count;
    auto status = cuda_ops::clamp_min(q_dec_values, prior_count, 0.5F, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::clamp_min", status);

    auto mask0 = allocate_device_tensor_scratch("mask0", make_dims(128, y_height, y_width));
    auto scales0 = allocate_device_tensor_scratch("scales0", reduced_shape);
    if (!mask0) return mask0.error();
    if (!scales0) return scales0.error();
    status = cuda_ops::make_two_way_mask(mask0.value().storage.data, prior_shape, 0, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::reduce_masked_halves(
        scales_values, mask0.value().storage.data, scales0.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_halves", status);
    auto indexes0_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
    if (!indexes0_device) return indexes0_device.error();
    status = cuda_ops::build_decode_indexes(
        scales0.value().storage.data, static_cast<std::uint8_t*>(indexes0_device.value().data),
        reduced_count, 0.0F, stream, orchestration.exact_indexes());
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes", status);
    auto& indexes0 = decode_staging.indexes[0];
    auto copied = copy_async(
        indexes0.data, indexes0_device.value().data, reduced_count,
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync indexes0");
    if (!copied) return copied.error();
    auto recorded = record_cuda_event(
        decode_staging.indexes_ready[0].data, stream, "cudaEventRecord indexes0");
    if (!recorded) return recorded.error();
    auto synchronized = synchronize_event(
        decode_staging.indexes_ready[0].data, "cudaEventSynchronize indexes0");
    if (!synchronized) return synchronized.error();
    auto decoded0 = [&]() -> Result<std::vector<std::int8_t>> {
        CpuProfileScope entropy_scope("p_entropy_decode_y0");
        return orchestration.decode_y(
            std::span<const std::uint8_t>(indexes0.data, reduced_count));
    }();
    if (!decoded0) return decoded0.error();
    auto& symbols0_host = decode_staging.symbols[0];
    std::copy(decoded0.value().begin(), decoded0.value().end(), symbols0_host.data);
    auto symbols0 = upload_int8_tensor_scratch(
        std::span<const std::int8_t>(symbols0_host.data, decoded0.value().size()),
        reduced_shape, "symbols0", stream);
    if (!symbols0) return symbols0.error();
    auto y_hat_device = allocate_device_tensor_scratch("y_hat", make_dims(128, y_height, y_width));
    if (!y_hat_device) return y_hat_device.error();
    status = cuda_ops::restore_two_way(
        symbols0.value().storage.data, means_values, mask0.value().storage.data,
        y_hat_device.value().storage.data, prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::restore_two_way", status);

    auto spatial_context = allocate_device_tensor_scratch(
        "context", make_dims(512, y_height, y_width));
    if (!spatial_context) return spatial_context.error();
    const auto y_bytes = prior_count * sizeof(__half);
    const auto params_bytes = 3 * y_bytes;
    copied = copy_async(
        spatial_context.value().storage.data, y_hat_device.value().storage.data, y_bytes,
        cudaMemcpyDeviceToDevice, stream, "cudaMemcpyAsync spatial y");
    if (!copied) return copied.error();
    copied = copy_async(
        static_cast<std::byte*>(spatial_context.value().storage.data) + y_bytes,
        params_device.storage.data, params_bytes, cudaMemcpyDeviceToDevice, stream,
        "cudaMemcpyAsync spatial params");
    if (!copied) return copied.error();
    std::array<DeviceTensor*, 1> spatial_inputs{&spatial_context.value()};
    auto spatial_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::spatial_prior), spatial_inputs,
        EngineOutputStorage::scratch);
    if (!spatial_outputs) return spatial_outputs.error();
    auto spatial_result = take_device_tensor(spatial_outputs.value(), "scales_means");
    if (!spatial_result) return spatial_result.error();
    auto spatial_params = std::move(spatial_result.value());
    auto* spatial_values = static_cast<__half*>(spatial_params.storage.data);

    auto mask1 = allocate_device_tensor_scratch("mask1", make_dims(128, y_height, y_width));
    auto scales1 = allocate_device_tensor_scratch("scales1", reduced_shape);
    if (!mask1) return mask1.error();
    if (!scales1) return scales1.error();
    status = cuda_ops::make_two_way_mask(mask1.value().storage.data, prior_shape, 1, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::make_two_way_mask", status);
    status = cuda_ops::reduce_masked_halves(
        spatial_values, mask1.value().storage.data, scales1.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::reduce_masked_halves", status);
    auto indexes1_device = allocate_cuda_scratch(reduced_count * sizeof(std::uint8_t));
    if (!indexes1_device) return indexes1_device.error();
    status = cuda_ops::build_decode_indexes(
        scales1.value().storage.data, static_cast<std::uint8_t*>(indexes1_device.value().data),
        reduced_count, 0.0F, stream, orchestration.exact_indexes());
    if (status != cudaSuccess) return cuda_error("cuda_ops::build_decode_indexes", status);
    auto& indexes1 = decode_staging.indexes[1];
    copied = copy_async(
        indexes1.data, indexes1_device.value().data, reduced_count,
        cudaMemcpyDeviceToHost, stream, "cudaMemcpyAsync indexes1");
    if (!copied) return copied.error();
    recorded = record_cuda_event(
        decode_staging.indexes_ready[1].data, stream, "cudaEventRecord indexes1");
    if (!recorded) return recorded.error();
    synchronized = synchronize_event(
        decode_staging.indexes_ready[1].data, "cudaEventSynchronize indexes1");
    if (!synchronized) return synchronized.error();
    auto decoded1 = [&]() -> Result<std::vector<std::int8_t>> {
        CpuProfileScope entropy_scope("p_entropy_decode_y1");
        return orchestration.decode_y(
            std::span<const std::uint8_t>(indexes1.data, reduced_count));
    }();
    if (!decoded1) return decoded1.error();
    auto entropy_complete = orchestration.end_decode();
    if (!entropy_complete) return entropy_complete.error();
    auto& symbols1_host = decode_staging.symbols[1];
    std::copy(decoded1.value().begin(), decoded1.value().end(), symbols1_host.data);
    auto symbols1 = upload_int8_tensor_scratch(
        std::span<const std::int8_t>(symbols1_host.data, decoded1.value().size()),
        reduced_shape, "symbols1", stream);
    if (!symbols1) return symbols1.error();
    auto reconstructed1 = allocate_device_tensor_scratch(
        "reconstructed1", make_dims(128, y_height, y_width));
    if (!reconstructed1) return reconstructed1.error();
    status = cuda_ops::restore_two_way(
        symbols1.value().storage.data, spatial_values + prior_count,
        mask1.value().storage.data, reconstructed1.value().storage.data,
        prior_shape, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::restore_two_way", status);
    status = cuda_ops::add_and_multiply(
        y_hat_device.value().storage.data, reconstructed1.value().storage.data,
        q_dec_values, prior_count, stream);
    if (status != cudaSuccess) return cuda_error("cuda_ops::add_and_multiply", status);

    auto& q_decoder_opt = quant_cache.q_decoder[info.qp];
    if (!q_decoder_opt.has_value()) return backend_error("missing cached q_decoder tensor");
    auto& q_recon_opt = quant_cache.q_recon[info.qp];
    if (!q_recon_opt.has_value()) return backend_error("missing cached q_recon tensor");
    auto* q_decoder = &q_decoder_opt.value();
    auto* q_recon = &q_recon_opt.value();
    std::array<DeviceTensor*, 4> synthesis_inputs{
        &y_hat_device.value(), &context, q_decoder, q_recon};
    auto synthesis_outputs = run_device_stage(
        engines, orchestration.stage(PredictedStage::synthesis), synthesis_inputs);
    if (!synthesis_outputs) return synthesis_outputs.error();
    auto frame_device_result = take_device_tensor(synthesis_outputs.value(), "frame_hat");
    auto feature_device_result = take_device_tensor(synthesis_outputs.value(), "feature");
    if (!frame_device_result) return frame_device_result.error();
    if (!feature_device_result) return feature_device_result.error();
    auto frame_device = std::move(frame_device_result.value());
    auto feature_device = std::move(feature_device_result.value());
    auto frame = ycbcr_device_to_yuv420p8(
        frame_device, info.width, info.height, timestamp, output_frame_buffer, stream);
    if (!frame) return frame.error();
    device_dpb.frame = std::move(frame_device);
    device_dpb.feature = std::move(feature_device);
    device_dpb.semantics.commit(state, true);
    std::vector<std::byte> latent_state;
    return CodecDecodeResult{std::move(frame.value()), std::move(latent_state)};
}

// DCVC-RT's runtime bundle: exactly these files, digest-verified by the session.
constexpr std::array<std::string_view, 20> dcvcrt_bundle_files{
    "i_analysis.plan", "i_hyper_analysis.plan", "i_hyper_synthesis.plan",
    "i_spatial_prior_1.plan", "i_spatial_prior_2.plan", "i_spatial_prior_3.plan",
    "i_synthesis.plan", "p_reference_frame.plan", "p_reference_feature.plan",
    "p_analysis.plan", "p_hyper_analysis.plan", "p_prior.plan",
    "p_spatial_prior.plan", "p_synthesis.plan", "i_entropy.bin", "i_quant.bin",
    "i_frame_manifest.json", "p_entropy.bin", "p_quant.bin", "p_frame_manifest.json",
};

Result<void> validate_dcvcrt_bundle(const TensorRTExecutionSession& session) {
    const auto& files = session.bundle_files();
    if (files.size() != dcvcrt_bundle_files.size()) {
        return backend_error("engine.sha256 does not describe the exact runtime bundle");
    }
    for (const auto required : dcvcrt_bundle_files) {
        if (!files.contains(std::string(required))) {
            return backend_error("engine.sha256 is missing " + std::string(required));
        }
    }
    auto text = read_text_file(session.bundle_root() / "engine_manifest.json");
    if (!text) return text.error();
    auto i_manifest = read_json_string(text.value(), "i_model_manifest_sha256");
    if (!i_manifest) return i_manifest.error();
    auto p_manifest = read_json_string(text.value(), "p_model_manifest_sha256");
    if (!p_manifest) return p_manifest.error();
    if (!valid_sha256(i_manifest.value()) || !valid_sha256(p_manifest.value()) ||
        files.at("i_frame_manifest.json") != i_manifest.value() ||
        files.at("p_frame_manifest.json") != p_manifest.value()) {
        return backend_error("engine bundle does not match its model manifest identities");
    }
    return {};
}

// Loads DCVC-RT's stages in bundle order with its warm-up shapes, then
// finalizes the session.
Result<void> load_dcvcrt_stages(TensorRTExecutionSession& session) {
    auto valid = validate_dcvcrt_bundle(session);
    if (!valid) return valid.error();
    for (const auto& engine : engine_specs) {
        TensorRTStageSpec specification;
        specification.filename = std::string(engine.filename);
        specification.tensors.reserve(engine.tensors.size());
        for (const auto& tensor : engine.tensors) {
            nvinfer1::Dims shape{};
            shape.nbDims = 4;
            for (std::int32_t axis = 0; axis < 4; ++axis) {
                shape.d[axis] = tensor.dimensions[static_cast<std::size_t>(axis)];
            }
            const auto warmup = warmup_shape(engine.filename, tensor.name, shape);
            TensorRTTensorSpec spec{std::string(tensor.name), tensor.mode, tensor.dimensions, {}};
            for (std::size_t axis = 0; axis < 4U; ++axis) {
                spec.warmup_dimensions[axis] = warmup.d[static_cast<std::int32_t>(axis)];
            }
            specification.tensors.push_back(std::move(spec));
        }
        auto loaded = session.load_stage(std::move(specification), "dcvcrt");
        if (!loaded) return loaded.error();
    }
    return session.finalize_stages();
}

// Codec-owned device state used across DCVC-RT frames.
struct DcvcrtDeviceState final {
    ImageQuantDeviceCache image_quant_cache;
    VideoQuantDeviceCache video_quant_cache;
    ImageDecodeStaging image_decode_staging;
    VideoDecodeStaging video_decode_staging;
    PinnedHostBuffer<std::int8_t> z_symbols;
    PinnedHostBuffer<std::int8_t> image_z_symbols;
    PinnedHostBuffer<std::int16_t> image_indexes0;
    PinnedHostBuffer<std::int16_t> image_indexes1;
    PinnedHostBuffer<std::int16_t> image_indexes2;
    PinnedHostBuffer<std::int16_t> image_indexes3;
    PinnedHostBuffer<std::int16_t> indexes0;
    PinnedHostBuffer<std::int16_t> indexes1;
    PinnedHostBuffer<std::byte> decoded_frame;
};

Result<void> initialize_device_state(
    TensorRTExecutionSession& session,
    DcvcrtDeviceState& state,
    const IntraOrchestration& intra,
    const PredictedOrchestration& predicted) {
    const auto scope = session.allocation_scope();
    auto image_ready = populate_image_quant_cache(
        intra.encoder_quantization(), intra.decoder_quantization(),
        state.image_quant_cache, session.stream());
    if (!image_ready) return image_ready.error();
    return populate_video_quant_cache(
        predicted.encoder_quantization(), predicted.decoder_quantization(),
        predicted.feature_quantization(), predicted.reconstruction_quantization(),
        state.video_quant_cache, session.stream());
}

class TensorRTBackend final : public CodecBackend {
public:
    explicit TensorRTBackend(std::shared_ptr<TensorRTExecutionSession> session = {})
        : session_(std::move(session)) {}

    ~TensorRTBackend() override {
        if (session_) {
            static_cast<void>(session_->reset());
        }
        encoder_dpb_.clear();
        decoder_dpb_.clear();
    }

    Result<void> initialize(const RuntimeConfiguration& configuration) override {
#ifndef NDEBUG
        std::clog << "[nvcr.dcvcrt] warning: Debug build; codec timings are not "
                     "representative. Configure with -DCMAKE_BUILD_TYPE=Release.\n";
#endif
        if (initialized_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is already initialized",
                std::string(subsystem));
        }
        try {
            if (!session_) {
                auto session = std::make_shared<TensorRTExecutionSession>();
                auto session_ready = session->initialize(configuration);
                if (!session_ready) return session_ready.error();
                session_ = std::move(session);
            }
            if (session_->loaded_stages().empty()) {
                auto stages_loaded = load_dcvcrt_stages(*session_);
                if (!stages_loaded) return stages_loaded.error();
                if (configuration.runtime.log_level <= LogLevel::info) {
                    std::clog << "[nvcr.dcvcrt] [info] TensorRT mode: "
                              << session_->execution_mode() << '\n';
                }
            }
            // Normative stages bind ahead of the plans with the same stage id.
            auto normative = normative_stages(configuration);
            if (!normative) return normative.error();
            std::vector<session_api::ExecutableStage> loaded_stages;
            if (normative.value()) {
                for (auto descriptor : normative.value()->descriptors) {
                    auto loaded = session_->load_stage(std::move(descriptor));
                    if (!loaded) return loaded.error();
                    loaded_stages.push_back(std::move(loaded.value()));
                }
            }
            for (auto& stage : session_->loaded_stages()) loaded_stages.push_back(std::move(stage));

            IntraOrchestration intra_orchestration;
            auto intra_ready = intra_orchestration.initialize(
                configuration.artifacts.intra_engine_path);
            if (!intra_ready) return intra_ready.error();
            intra_orchestration.set_exact_indexes(normative.value().has_value());
            auto stages_ready = intra_orchestration.bind_stages(loaded_stages);
            if (!stages_ready) return stages_ready.error();

            PredictedOrchestration predicted_orchestration;
            auto predicted_ready = predicted_orchestration.initialize(
                configuration.artifacts.intra_engine_path);
            if (!predicted_ready) return predicted_ready.error();
            predicted_orchestration.set_exact_indexes(normative.value().has_value());
            stages_ready = predicted_orchestration.bind_stages(loaded_stages);
            if (!stages_ready) return stages_ready.error();

            auto buffers_ready = initialize_device_state(
                *session_, state_, intra_orchestration, predicted_orchestration);
            if (!buffers_ready) return buffers_ready.error();

            intra_orchestration_ = std::move(intra_orchestration);
            predicted_orchestration_ = std::move(predicted_orchestration);
            intra_qp_ = configuration.codec.intra_qp;
            verify_encoder_reconstruction_ =
                configuration.codec.verify_encoder_reconstruction;
            initialized_ = true;
            return {};
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("TensorRT initialization failed: ") + exception.what());
        } catch (...) {
            return backend_error("TensorRT initialization failed");
        }
    }

    Result<CodecEncodeResult> encode(
        const Frame& frame,
        FrameType frame_type,
        const SequenceStateView& state) override {
        if (!initialized_ || !session_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        try {
            constexpr std::uint64_t graph_area_limit = 640ULL * 360ULL;
            const bool allow_cuda_graphs =
                static_cast<std::uint64_t>(frame.width()) * frame.height() <=
                graph_area_limit;
            SessionExecutionScope execution(
                *session_, "encode", frame_type, state.frame_index, allow_cuda_graphs);
            const auto stream = session_->stream();
            Result<CodecEncodeResult> result = frame_type == FrameType::intra ?
                encode_intra(
                    frame, intra_qp_, state, encoder_dpb_, *session_,
                    intra_orchestration_,
                    state_.image_quant_cache, state_.image_decode_staging,
                    state_.image_z_symbols, state_.image_indexes0,
                    state_.image_indexes1, state_.image_indexes2,
                    state_.image_indexes3, verify_encoder_reconstruction_,
                    stream) :
                encode_predicted(
                    frame, intra_qp_, state, encoder_dpb_, *session_,
                    predicted_orchestration_,
                    state_.video_quant_cache, state_.z_symbols,
                    state_.indexes0, state_.indexes1,
                    verify_encoder_reconstruction_, stream);
            execution.finish();
            return result;
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("native frame encoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native frame encoding failed");
        }
    }

    Result<CodecDecodeResult> decode(
        std::span<const std::byte> payload,
        FrameType frame_type,
        Timestamp timestamp,
        const SequenceStateView& state) override {
        if (!initialized_ || !session_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        try {
            bool allow_cuda_graphs = false;
            if (frame_type == FrameType::predicted) {
                auto predicted_info = predicted_orchestration_.parse_payload(payload);
                if (!predicted_info) return predicted_info.error();
                constexpr std::uint64_t graph_area_limit = 640ULL * 360ULL;
                allow_cuda_graphs =
                    static_cast<std::uint64_t>(predicted_info.value().width) *
                        predicted_info.value().height <= graph_area_limit;
            }
            SessionExecutionScope execution(
                *session_, "decode", frame_type, state.frame_index, allow_cuda_graphs);
            const auto stream = session_->stream();
            Result<CodecDecodeResult> result = frame_type == FrameType::intra ?
                decode_intra(
                    payload, timestamp, state, decoder_dpb_, *session_,
                    intra_orchestration_,
                    state_.image_quant_cache, state_.image_decode_staging,
                    stream) :
                decode_predicted(
                    payload, timestamp, state, decoder_dpb_, *session_,
                    predicted_orchestration_,
                    state_.video_quant_cache, state_.video_decode_staging,
                    state_.decoded_frame, stream);
            execution.finish();
            return result;
        } catch (const std::exception& exception) {
            return backend_error(
                std::string("native frame decoding failed: ") + exception.what());
        } catch (...) {
            return backend_error("native frame decoding failed");
        }
    }

    Result<void> flush() override {
        if (!initialized_ || !session_) {
            return Error(
                ErrorCode::invalid_state,
                "TensorRT backend is not initialized",
                std::string(subsystem));
        }
        return session_->reset();
    }

    void reset() noexcept override {
        if (session_) static_cast<void>(session_->reset());
        encoder_dpb_.clear();
        decoder_dpb_.clear();
    }

    void reset_encoder() noexcept override {
        if (session_) static_cast<void>(session_->reset());
        encoder_dpb_.clear();
    }

    void reset_decoder() noexcept override {
        if (session_) static_cast<void>(session_->reset());
        decoder_dpb_.clear();
    }

private:
    std::shared_ptr<TensorRTExecutionSession> session_;
    DcvcrtDeviceState state_;
    IntraOrchestration intra_orchestration_;
    PredictedOrchestration predicted_orchestration_;
    DeviceDpb encoder_dpb_;
    DeviceDpb decoder_dpb_;
    std::uint32_t intra_qp_{};
    bool verify_encoder_reconstruction_{};
    bool initialized_{};
};

}  // namespace

Result<std::unique_ptr<CodecBackend>> make_tensorrt_backend() {
    try {
        return std::unique_ptr<CodecBackend>(std::make_unique<TensorRTBackend>());
    } catch (const std::exception& exception) {
        return backend_error(
            std::string("failed to allocate TensorRT backend: ") + exception.what());
    } catch (...) {
        return backend_error("failed to allocate TensorRT backend");
    }
}

Result<std::unique_ptr<CodecBackend>> make_tensorrt_backend(
    std::shared_ptr<provider::experimental::IProviderSession> provider_session) {
    try {
        auto tensorrt_session =
            std::dynamic_pointer_cast<TensorRTExecutionSession>(provider_session);
        if (!tensorrt_session) {
            return Error(
                ErrorCode::incompatible_target,
                "DCVC-RT requires a TensorRT provider session",
                std::string(subsystem));
        }
        return std::unique_ptr<CodecBackend>(
            std::make_unique<TensorRTBackend>(std::move(tensorrt_session)));
    } catch (const std::bad_alloc&) {
        return Error(
            ErrorCode::resource_exhausted,
            "unable to allocate TensorRT backend",
            std::string(subsystem));
    } catch (const std::exception& exception) {
        return backend_error(
            std::string("failed to allocate TensorRT backend: ") + exception.what());
    } catch (...) {
        return backend_error("failed to allocate TensorRT backend");
    }
}

provider::ProviderDescriptor tensorrt_provider_descriptor() {
    return tensorrt::provider_descriptor();
}

void register_tensorrt_provider() {
    runtime::Registry::instance().register_provider({
        tensorrt_provider_descriptor(),
        provider::ProviderCapabilities{
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_dynamic_shapes = true,
            .supports_cuda_graphs = true,
            .target_device = "cuda",
        },
        {},
        [](const RuntimeConfiguration&) -> Result<codec::Components> {
            auto backend = make_tensorrt_backend();
            if (!backend) return backend.error();
            return codec::Components{std::move(backend.value())};
        },
        tensorrt::make_provider_session,
    });
}

}  // namespace nvcr::dcvcrt
