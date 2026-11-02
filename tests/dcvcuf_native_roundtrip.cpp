#include "dmci_proxy.h"
#include "dmc_hts_proxy.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::string_view escape = "__nvcr_dot__";

std::string decode_name(std::string name) {
    std::size_t position = 0;
    while ((position = name.find(escape, position)) != std::string::npos) {
        name.replace(position, escape.size(), ".");
        ++position;
    }
    return name;
}

using Weights = std::map<std::string, at::Tensor>;

std::pair<Weights, Weights> load_bundle(const std::string& path, int device) {
    auto module = torch::jit::load(path, at::Device(at::kCPU));
    Weights image;
    Weights video;
    for (const auto& item : module.named_buffers(true)) {
        auto name = decode_name(item.name);
        auto value = item.value;
        if (value.is_floating_point()) {
            value = value.to(at::Device(at::kCUDA, device));
            value = value.dim() == 4
                ? value.contiguous(at::MemoryFormat::ChannelsLast)
                : value.contiguous();
        } else
            value = value.cpu().contiguous();
        if (name.starts_with("image."))
            image.emplace(name.substr(6), std::move(value));
        else if (name.starts_with("video."))
            video.emplace(name.substr(6), std::move(value));
    }
    if (image.empty() || video.empty())
        throw std::runtime_error("native bundle is missing image or video weights");
    return {std::move(image), std::move(video)};
}

std::vector<std::byte> read_bytes(const std::string& path, std::size_t size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    std::vector<std::byte> bytes(size);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (static_cast<std::size_t>(input.gcount()) != size)
        throw std::runtime_error("input does not contain nine complete frames");
    return bytes;
}

at::Tensor yuv_frame(const std::byte* data, int width, int height, int device) {
    const auto y_size = static_cast<std::size_t>(width) * height;
    const auto uv_width = width / 2;
    const auto uv_height = height / 2;
    const auto uv_size = static_cast<std::size_t>(uv_width) * uv_height;
    const auto* y = reinterpret_cast<const std::uint8_t*>(data);
    const auto* u = y + y_size;
    const auto* v = u + uv_size;
    std::vector<float> values(3 * y_size);
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const auto index = static_cast<std::size_t>(row) * width + column;
            const auto uv_index = static_cast<std::size_t>(row / 2) * uv_width + column / 2;
            values[index] = static_cast<float>(y[index]) / 255.0F - 0.5F;
            values[y_size + index] = static_cast<float>(u[uv_index]) / 255.0F - 0.5F;
            values[2 * y_size + index] = static_cast<float>(v[uv_index]) / 255.0F - 0.5F;
        }
    }
    return at::from_blob(values.data(), {1, 3, height, width},
                         at::TensorOptions().dtype(at::kFloat))
        .clone()
        .to(at::Device(at::kCUDA, device), at::kHalf)
        .contiguous(at::MemoryFormat::ChannelsLast);
}

void write_frame(std::ofstream& output, const at::Tensor& centered, int width, int height) {
    auto frame = centered.index({
        at::indexing::Slice(), at::indexing::Slice(),
        at::indexing::Slice(0, height), at::indexing::Slice(0, width)}).add(0.5);
    auto y = frame.narrow(1, 0, 1).mul(255).clamp(0, 255).round()
                 .to(at::kByte).contiguous().cpu();
    auto uv = frame.narrow(1, 1, 2)
                  .unfold(2, 2, 2).unfold(3, 2, 2)
                  .mean(std::vector<std::int64_t>{4, 5})
                  .mul(255).clamp(0, 255).to(at::kByte).contiguous().cpu();
    output.write(reinterpret_cast<const char*>(y.const_data_ptr<std::uint8_t>()),
                 static_cast<std::streamsize>(y.numel()));
    output.write(reinterpret_cast<const char*>(uv.const_data_ptr<std::uint8_t>()),
                 static_cast<std::streamsize>(uv.numel()));
    if (!output) throw std::runtime_error("failed writing reconstruction");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: dcvcuf-native-roundtrip BUNDLE INPUT.yuv OUTPUT.yuv WIDTH HEIGHT\n";
        return 2;
    }
    try {
        const int width = std::stoi(argv[4]);
        const int height = std::stoi(argv[5]);
        if (width <= 0 || height <= 0 || width % 2 || height % 2)
            throw std::invalid_argument("dimensions must be positive and even");
        constexpr int device = 0;
        constexpr int qp = 32;
        c10::cuda::CUDAGuard guard(device);
        at::cuda::CUDAStreamGuard stream_guard(at::cuda::getStreamFromPool(false, device));
        torch::InferenceMode inference;
        auto [image_weights, video_weights] = load_bundle(argv[1], device);

        const auto frame_size = static_cast<std::size_t>(width) * height * 3 / 2;
        const auto input = read_bytes(argv[2], frame_size * 9);
        std::vector<at::Tensor> frames;
        frames.reserve(9);
        for (std::size_t index = 0; index < 9; ++index)
            frames.push_back(yuv_frame(input.data() + index * frame_size, width, height, device));

        DMCIProxy image_encoder;
        DMCIProxy image_decoder;
        image_encoder.set_param(image_weights, 0.0F);
        image_decoder.set_param(image_weights, 0.0F);
        auto intra = image_encoder.compress(frames.front(), qp, 0, 0);
        if (intra.bit_stream.empty() || !intra.reconstruction.defined())
            throw std::runtime_error("native image encoder returned no result");
        auto decoded_intra = image_decoder.decompress(
            intra.bit_stream.data(), static_cast<int>(intra.bit_stream.size()),
            qp, height, width, intra.entropy_coder_parallel);
        if (!decoded_intra.defined() || !at::equal(intra.reconstruction, decoded_intra))
            throw std::runtime_error("native image encode/decode reconstruction differs");

        DMCHTSProxy video_encoder;
        DMCHTSProxy video_decoder;
        video_encoder.set_param(video_weights, 0.0F);
        video_decoder.set_param(video_weights, 0.0F);
        video_encoder.add_ref_feature_from_frame(intra.reconstruction, true);
        video_decoder.add_ref_feature_from_frame(decoded_intra, false);

        std::vector<at::Tensor> predicted_inputs(frames.begin() + 1, frames.end());
        auto group = at::cat(predicted_inputs, 1)
                         .contiguous(at::MemoryFormat::ChannelsLast);
        auto predicted = video_encoder.compress(group, qp, false, 0, 0);
        if (predicted.bit_stream.empty())
            throw std::runtime_error("native HT-S encoder returned no result");
        auto decoded_predicted = video_decoder.decompress(
            predicted.bit_stream.data(), static_cast<int>(predicted.bit_stream.size()),
            qp, height, width, predicted.entropy_coder_parallel, false);
        if (decoded_predicted.size() != 8 ||
            std::any_of(decoded_predicted.begin(), decoded_predicted.end(),
                        [](const at::Tensor& value) { return !value.defined(); }))
            throw std::runtime_error("native HT-S decoder did not return eight frames");

        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create output");
        write_frame(output, decoded_intra, width, height);
        for (const auto& frame : decoded_predicted)
            write_frame(output, frame, width, height);

        std::cout << "{\"image_payload_bytes\":" << intra.bit_stream.size()
                  << ",\"image_entropy_parallel\":" << intra.entropy_coder_parallel
                  << ",\"predicted_payload_bytes\":" << predicted.bit_stream.size()
                  << ",\"predicted_entropy_parallel\":" << predicted.entropy_coder_parallel
                  << ",\"decoded_frames\":9,\"image_roundtrip_exact\":true}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "dcvcuf-native-roundtrip: " << exception.what() << '\n';
        return 1;
    }
}
