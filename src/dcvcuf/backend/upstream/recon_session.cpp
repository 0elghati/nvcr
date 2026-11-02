#include "recon_session.hpp"
#include "dmc_hts_proxy.h"
#include "../../../common/sha256.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>
#include <atomic>
#include <limits>
#include <optional>

namespace nvcr::dcvcuf {
namespace {
using namespace provider::experimental;
Error invalid(std::string message) {
    return {ErrorCode::invalid_argument, std::move(message), "dcvcuf-upstream"};
}

constexpr std::string_view weight_name_escape = "__nvcr_dot__";

std::string digest_hex(std::span<const std::byte> digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const auto value = std::to_integer<unsigned char>(digest[i]);
        result[2 * i] = digits[value >> 4U];
        result[2 * i + 1] = digits[value & 0x0fU];
    }
    return result;
}

std::string decode_weight_name(std::string name) {
    std::size_t position = 0;
    while ((position = name.find(weight_name_escape, position)) != std::string::npos) {
        name.replace(position, weight_name_escape.size(), ".");
        ++position;
    }
    return name;
}

Result<Weights> load_weights(const provider::ArtifactDescriptor& artifact, int device) {
    if (artifact.path.empty() || artifact.expected_digest.size() != 32)
        return invalid("UF stage requires a path and SHA-256 digest");
    auto actual = detail::sha256_file(artifact.path, "dcvcuf-upstream");
    if (!actual) return invalid(std::string(actual.error().message()));
    if (actual.value() != digest_hex(artifact.expected_digest))
        return invalid("UF weight artifact SHA-256 mismatch");
    try {
        auto module = torch::jit::load(artifact.path, at::Device(at::kCUDA, device));
        Weights result;
        for (const auto& item : module.named_buffers(true))
            result.emplace(decode_weight_name(item.name), item.value);
        if (result.empty()) return invalid("UF weight artifact contains no tensors");
        return result;
    } catch (const std::exception& exception) {
        return invalid(std::string("cannot load UF weight artifact: ") + exception.what());
    }
}

Weights submodule_weights(const Weights& weights, std::string_view prefix) {
    Weights result;
    for (const auto& [name, tensor] : weights) {
        if (name.starts_with(prefix))
            result.emplace(name.substr(prefix.size()), tensor);
    }
    TORCH_CHECK(!result.empty(), "UF artifact is missing ", prefix, " weights");
    for (const auto& [name, tensor] : result)
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kHalf,
                    prefix, name, " must be CUDA FP16");
    return result;
}

class Buffer final : public IProviderBuffer {
public:
    Buffer(std::string owner, std::size_t size, int device)
        : owner_(std::move(owner)), bytes(at::empty(
              {static_cast<std::int64_t>(size)},
              at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, device)))) {}
    std::size_t size_bytes() const noexcept override { return bytes.numel(); }
    MemoryDomain domain() const noexcept override { return MemoryDomain::provider_device; }
    std::string_view owner_id() const noexcept override { return owner_; }
private:
    std::string owner_;
public:
    at::Tensor bytes;
};

class Done final : public ICompletion {
public:
    explicit Done(std::string owner) : owner_(std::move(owner)) {}
    std::string_view owner_id() const noexcept override { return owner_; }
    bool ready() const noexcept override { return true; }
    Result<void> wait() override { return {}; }
private:
    std::string owner_;
};

class Stage final : public IExecutableStage {
public:
    Stage(std::string owner, ExecutableStageDescriptor descriptor,
          const Weights& weights, int device)
        : owner_(std::move(owner)), descriptor_(std::move(descriptor)) {
        const auto decoder_stage = descriptor_.stage_id == "dcvcuf.hts.decoder";
        const auto stage_weights = submodule_weights(
            weights, decoder_stage ? "decoder." : "recon_head.");
        const auto& dims = descriptor_.tensors.back().dimensions;
        auto options = at::TensorOptions().dtype(at::kHalf)
                           .device(at::Device(at::kCUDA, device))
                           .memory_format(at::MemoryFormat::ChannelsLast);
        const auto h = dims[1].minimum;
        const auto w = dims[2].minimum;
        if (descriptor_.stage_id == "dcvcuf.hts.decoder") {
            decoder = std::make_unique<DMCHTSDecoderProxy>();
            decoder->set_param(stage_weights);
            auto latent = at::empty({1, 256, h / 2, w / 2}, options);
            decoder_cat = at::empty({1, 1024, h, w}, options);
            auto up = decoder_cat.narrow(1, 0, 512);
            context = decoder_cat.narrow(1, 512, 512);
            storage = at::empty({1, 512, h, w}, options);
            decoder->pre_allocate_tensors(latent, up, decoder_cat, storage);
        } else {
            recon = std::make_unique<DMCHTSReconHeadProxy>();
            recon->set_param(stage_weights);
            auto input = at::empty({1, 512, h, w}, options);
            storage = at::empty({1, 192, h, w}, options);
            recon->pre_allocate_tensors(input, storage);
        }
    }
    const ExecutableStageDescriptor& descriptor() const noexcept override { return descriptor_; }
    std::string_view owner_id() const noexcept override { return owner_; }
    void forward(const std::vector<at::Tensor>& bound) const {
        if (decoder) {
            context.copy_(bound[1].permute({0, 3, 1, 2}));
            auto feature = decoder->forward(bound[0].permute({0, 3, 1, 2}), decoder_cat,
                                            bound[2].permute({0, 3, 1, 2}));
            bound[3].copy_(feature.permute({0, 2, 3, 1}));
        } else {
            auto [frames, reset_feature] = recon->forward(bound[0].permute({0, 3, 1, 2}));
            for (std::size_t i = 0; i < frames.size(); ++i)
                bound[1].narrow(0, i, 1).copy_(frames[i].permute({0, 2, 3, 1}));
            bound[2].copy_(reset_feature.permute({0, 2, 3, 1}));
        }
    }
private:
    std::string owner_;
    ExecutableStageDescriptor descriptor_;
    std::unique_ptr<DMCHTSReconHeadProxy> recon;
    std::unique_ptr<DMCHTSDecoderProxy> decoder;
    at::Tensor storage, decoder_cat, context;
};

class Session final : public IProviderSession {
public:
    explicit Session(int device) : device_(device) {
        static std::atomic<std::uint64_t> next{0};
        owner_ = "dcvcuf-upstream-" + std::to_string(++next);
    }
    std::string_view owner_id() const noexcept override { return owner_; }
    Result<BufferHandle> allocate(std::size_t bytes, MemoryDomain domain) override {
        if (domain != MemoryDomain::provider_device || !bytes ||
            bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            return invalid("expected nonempty device allocation");
        try {
            c10::cuda::CUDAGuard guard(device_);
            return BufferHandle(std::make_shared<Buffer>(owner_, bytes, device_));
        } catch (const std::exception& e) { return invalid(e.what()); }
    }
    Result<ExecutableStage> load_stage(ExecutableStageDescriptor desc) override {
        const bool decoder = desc.stage_id == "dcvcuf.hts.decoder";
        if ((!decoder && desc.stage_id != "dcvcuf.hts.reconstruction") ||
            desc.tensors.size() != (decoder ? 4u : 3u) ||
            desc.tensors[0].dimensions.size() != 4)
            return invalid("invalid synthesis stage contract");
        const auto h = desc.tensors[0].dimensions[1].minimum;
        const auto w = desc.tensors[0].dimensions[2].minimum;
        if (h <= 0 || w <= 0 || h > (decoder ? 128 : 256) ||
            w > (decoder ? 128 : 256))
            return invalid("prototype supports feature dimensions 1 through 256");
        const auto& a = desc.artifact;
        if (a.path.empty() || a.expected_digest.size() != 32)
            return invalid("UF stage requires a verified artifact");
        const auto expected = decoder
            ? decoder_descriptor(h * 2, w * 2, a.path, a.expected_digest)
            : recon_descriptor(h, w, a.path, a.expected_digest);
        const auto& b = expected.artifact;
        if (desc.stage_id != expected.stage_id || a.codec_id != b.codec_id ||
            a.model_set_id != b.model_set_id || a.component_id != b.component_id ||
            a.provider_id != b.provider_id || a.precision != b.precision ||
            a.engine_profile_id != b.engine_profile_id ||
            a.codec_api_version.major != 1 || a.provider_api_version.major != 1 ||
            a.model_set_version.major != 1 || a.model_set_version.minor != 0 ||
            a.manifest_schema_version.major != 2 ||
            a.manifest_schema_version.minor != 0)
            return invalid("unsupported reconstruction artifact identity");
        for (std::size_t i = 0; i < expected.tensors.size(); ++i) {
            const auto& t = desc.tensors[i];
            const auto& e = expected.tensors[i];
            if (t.name != e.name || t.data_type != e.data_type ||
                t.access != e.access || t.dimensions.size() != e.dimensions.size())
                return invalid("invalid tensor contract");
            for (std::size_t j = 0; j < t.dimensions.size(); ++j)
                if (t.dimensions[j].minimum != e.dimensions[j].minimum ||
                    t.dimensions[j].maximum != e.dimensions[j].maximum)
                    return invalid("invalid fixed dimensions");
        }
        if (weights_) {
            if (artifact_path_ != a.path || artifact_digest_ != a.expected_digest)
                return invalid("all UF stages in a session must use one verified artifact");
        } else {
            auto loaded = load_weights(a, device_);
            if (!loaded) return loaded.error();
            weights_.emplace(std::move(loaded.value()));
            artifact_path_ = a.path;
            artifact_digest_ = a.expected_digest;
        }
        try {
            c10::cuda::CUDAGuard guard(device_);
            return ExecutableStage(std::make_shared<Stage>(
                owner_, std::move(desc), *weights_, device_));
        } catch (const std::exception& e) {
            return invalid(e.what());
        }
    }
    Result<Completion> submit(const ExecutableStage& handle,
                              std::span<const TensorView> tensors,
                              std::span<const ExecutionDependency> dependencies) override {
        auto stage = std::dynamic_pointer_cast<const Stage>(handle);
        if (!stage || stage->owner_id() != owner_ || tensors.size() != stage->descriptor().tensors.size())
            return invalid("foreign stage or wrong binding count");
        for (const auto& dep : dependencies)
            if (!dep.completion || dep.completion->owner_id() != owner_)
                return invalid("foreign completion");
        std::vector<at::Tensor> bound;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            const auto& e = stage->descriptor().tensors[i];
            auto buffer = std::dynamic_pointer_cast<Buffer>(t.storage.buffer);
            if (!buffer || buffer->owner_id() != owner_ || t.name != e.name ||
                t.data_type != e.data_type || t.access != e.access || t.shape.size() != 4)
                return invalid("invalid tensor binding or foreign buffer");
            std::size_t bytes = 2;
            for (std::size_t j = 0; j < 4; ++j) {
                if (t.shape[j] != e.dimensions[j].minimum) return invalid("invalid tensor shape");
                bytes *= static_cast<std::size_t>(t.shape[j]);
            }
            const auto& s = t.storage;
            if (s.offset_bytes % 2 || s.size_bytes != bytes ||
                s.offset_bytes > buffer->size_bytes() || bytes > buffer->size_bytes() - s.offset_bytes)
                return invalid("invalid tensor byte bounds");
            for (std::size_t j = 0; j < i; ++j) {
                const auto& other = tensors[j].storage;
                if (other.buffer == s.buffer && s.offset_bytes < other.offset_bytes + other.size_bytes &&
                    other.offset_bytes < s.offset_bytes + s.size_bytes)
                    return invalid("overlapping bindings");
            }
            bound.push_back(buffer->bytes.narrow(0, s.offset_bytes, bytes).view(at::kHalf).view(t.shape));
        }
        try {
            c10::cuda::CUDAGuard guard(device_);
            for (const auto& dep : dependencies) {
                auto waited = dep.completion->wait();
                if (!waited) return waited.error();
            }
            stage->forward(bound);
            // Bounded correctness bridge: synchronous completion. Async
            // execution can be introduced only with its own lifetime tests.
            C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(device_)));
            return Completion(std::make_shared<Done>(owner_));
        } catch (const std::exception& e) {
            return Error(ErrorCode::backend_error, e.what(), "dcvcuf-upstream");
        }
    }
    Result<void> reset() override { return {}; } // Every submit completes before returning.
private:
    int device_;
    std::string owner_;
    std::optional<Weights> weights_;
    std::string artifact_path_;
    std::vector<std::byte> artifact_digest_;
};
}  // namespace

std::shared_ptr<provider::experimental::IProviderSession> make_recon_session(
    int device) {
    return std::make_shared<Session>(device);
}

provider::experimental::ExecutableStageDescriptor recon_descriptor(
    int h, int w, std::string artifact_path,
    std::vector<std::byte> expected_digest) {
    using namespace provider::experimental;
    TORCH_CHECK(h > 0 && w > 0 && h <= 256 && w <= 256,
                "prototype feature dimensions must be in [1, 256]");
    ExecutableStageDescriptor d;
    d.stage_id = "dcvcuf.hts.reconstruction";
    d.artifact.codec_id = "dcvc-uf";
    d.artifact.model_set_id = "dcvcuf-cvpr2026-hts";
    d.artifact.component_id = "hts-reconstruction";
    d.artifact.provider_id = "dcvcuf-upstream-cuda";
    d.artifact.precision = "fp16";
    d.artifact.path = std::move(artifact_path);
    d.artifact.expected_digest = std::move(expected_digest);
    d.tensors = {
        {"feature", TensorDataType::float16, TensorAccess::read, {{1,1},{h,h},{w,w},{512,512}}},
        {"frames", TensorDataType::float16, TensorAccess::write, {{8,8},{h*8,h*8},{w*8,w*8},{3,3}}},
        {"reset_feature", TensorDataType::float16, TensorAccess::write, {{1,1},{h,h},{w,w},{192,192}}},
    };
    return d;
}

provider::experimental::ExecutableStageDescriptor decoder_descriptor(
    int h, int w, std::string artifact_path,
    std::vector<std::byte> expected_digest) {
    using namespace provider::experimental;
    auto d = recon_descriptor(h, w, std::move(artifact_path),
                              std::move(expected_digest));
    TORCH_CHECK(h % 2 == 0 && w % 2 == 0, "decoder feature dimensions must be even");
    d.stage_id = "dcvcuf.hts.decoder";
    d.artifact.component_id = "hts-decoder";
    d.tensors = {
        {"latent", TensorDataType::float16, TensorAccess::read, {{1,1},{h/2,h/2},{w/2,w/2},{256,256}}},
        {"context", TensorDataType::float16, TensorAccess::read, {{1,1},{h,h},{w,w},{512,512}}},
        {"quant", TensorDataType::float16, TensorAccess::read, {{1,1},{1,1},{1,1},{512,512}}},
        {"feature", TensorDataType::float16, TensorAccess::write, {{1,1},{h,h},{w,w},{512,512}}},
    };
    return d;
}

void upload(const provider::experimental::BufferHandle& handle, const at::Tensor& value) {
    auto buffer = std::dynamic_pointer_cast<Buffer>(handle);
    TORCH_CHECK(buffer && value.scalar_type() == at::kHalf && value.is_contiguous() &&
                value.nbytes() == buffer->size_bytes(), "invalid upload");
    c10::cuda::CUDAGuard guard(buffer->bytes.device());
    buffer->bytes.copy_(value.view(at::kByte).reshape({-1}));
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}
at::Tensor download(const provider::experimental::BufferHandle& handle,
                    const std::vector<std::int64_t>& shape) {
    auto buffer = std::dynamic_pointer_cast<Buffer>(handle);
    TORCH_CHECK(buffer, "invalid download");
    return buffer->bytes.view(at::kHalf).view(shape).cpu();
}
}  // namespace nvcr::dcvcuf
