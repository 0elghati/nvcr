#include <torch/extension.h>
#include "recon_session.hpp"
#include "uf_rans_codec.hpp"
#include "capture_hts.h"

namespace {
using namespace nvcr;
using namespace provider::experimental;
template<class T> T checked(Result<T> result) {
    TORCH_CHECK(result, result.error().describe());
    return std::move(result.value());
}
void checked(Result<void> result) { TORCH_CHECK(result, result.error().describe()); }

std::vector<std::byte> digest_bytes(std::string_view hex) {
    TORCH_CHECK(hex.size() == 64, "SHA-256 must contain 64 hex characters");
    auto digit = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        TORCH_CHECK(false, "invalid SHA-256 character");
        return 0;
    };
    std::vector<std::byte> result(32);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<std::byte>((digit(hex[2 * i]) << 4U) | digit(hex[2 * i + 1]));
    return result;
}

dcvcuf::UfRansCdfTable cdf_table(const at::Tensor& values, const at::Tensor& sizes) {
    TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kInt &&
                values.dim() == 2 && values.is_contiguous(), "CDF values must be contiguous CPU int32");
    TORCH_CHECK(sizes.device().is_cpu() && sizes.scalar_type() == at::kInt &&
                sizes.dim() == 1 && sizes.is_contiguous() && sizes.size(0) == values.size(0),
                "CDF sizes must be contiguous CPU int32 rows");
    dcvcuf::UfRansCdfTable table;
    const auto* value = values.const_data_ptr<std::int32_t>();
    const auto* size = sizes.const_data_ptr<std::int32_t>();
    table.values.resize(values.size(0));
    table.sizes.assign(size, size + sizes.numel());
    for (std::int64_t row = 0; row < values.size(0); ++row) {
        table.values[row].assign(value + row * values.size(1), value + (row + 1) * values.size(1));
    }
    return table;
}

std::tuple<at::Tensor, at::Tensor> decode_entropy(
    const at::Tensor& z_cdf, const at::Tensor& z_sizes,
    const at::Tensor& y_cdf, const at::Tensor& y_sizes,
    const at::Tensor& stream, const at::Tensor& y_indexes,
    std::size_t parallel, std::size_t qp, std::size_t z_channels,
    std::size_t z_height, std::size_t z_width) {
    TORCH_CHECK(stream.device().is_cpu() && stream.scalar_type() == at::kByte &&
                stream.dim() == 1 && stream.is_contiguous(), "stream must be contiguous CPU uint8");
    TORCH_CHECK(y_indexes.device().is_cpu() && y_indexes.scalar_type() == at::kByte &&
                y_indexes.dim() == 1 && y_indexes.is_contiguous(),
                "y indexes must be contiguous CPU uint8");
    dcvcuf::UfRansCodec codec;
    checked(codec.set_entropy_coder_parallel(parallel));
    const auto z_group = checked(codec.add_cdf(cdf_table(z_cdf, z_sizes)));
    const auto y_group = checked(codec.add_cdf(cdf_table(y_cdf, y_sizes)));
    checked(codec.set_stream(std::span(
        reinterpret_cast<const std::byte*>(stream.const_data_ptr<std::uint8_t>()), stream.numel())));
    const auto plane = z_height * z_width;
    auto z = checked(codec.decode_z(z_channels * plane, z_group, qp * z_channels, z_channels));
    auto y = checked(codec.decode_y(std::span(y_indexes.const_data_ptr<std::uint8_t>(),
                                             y_indexes.numel()), y_group));
    auto options = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
    return {at::from_blob(z.data(), {static_cast<std::int64_t>(z.size())}, options).clone(),
            at::from_blob(y.data(), {static_cast<std::int64_t>(y.size())}, options).clone()};
}

class Probe {
public:
    Probe(const std::string& artifact_path, const std::string& artifact_sha256,
          int h, int w, int device)
        : session(dcvcuf::make_recon_session(device)),
          stage(checked(session->load_stage(dcvcuf::recon_descriptor(
              h, w, artifact_path, digest_bytes(artifact_sha256))))),
          decoder(checked(session->load_stage(dcvcuf::decoder_descriptor(
              h, w, artifact_path, digest_bytes(artifact_sha256))))) {
        bindings = allocate(stage);
        decoder_bindings = allocate(decoder);
        // The synthesis output is the reconstruction input: one opaque device
        // allocation shared by two stage submissions, with no host transfer.
        decoder_bindings.back().storage = bindings[0].storage;
    }
    std::vector<TensorView> allocate(const ExecutableStage& executable) {
        std::vector<TensorView> result;
        for (const auto& contract : executable->descriptor().tensors) {
            std::vector<std::int64_t> shape;
            std::size_t size = 2;
            for (const auto& dim : contract.dimensions) {
                shape.push_back(dim.minimum);
                size *= dim.minimum;
            }
            auto buffer = checked(session->allocate(size, MemoryDomain::provider_device));
            result.push_back({contract.name, contract.data_type, std::move(shape),
                                contract.access, {std::move(buffer), 0, size}});
        }
        return result;
    }
    std::vector<at::Tensor> run(const at::Tensor& feature) {
        dcvcuf::upload(bindings[0].storage.buffer, feature);
        auto done = checked(session->submit(stage, bindings, {}));
        TORCH_CHECK(done->ready(), "synchronous completion must be ready");
        checked(done->wait());
        std::vector<at::Tensor> outputs;
        for (const auto& t : bindings) outputs.push_back(dcvcuf::download(t.storage.buffer, t.shape));
        return outputs;
    }
    std::vector<at::Tensor> run_pipeline(const at::Tensor& latent, const at::Tensor& context,
                                        const at::Tensor& quant) {
        dcvcuf::upload(decoder_bindings[0].storage.buffer, latent);
        dcvcuf::upload(decoder_bindings[1].storage.buffer, context);
        dcvcuf::upload(decoder_bindings[2].storage.buffer, quant);
        auto decoded = checked(session->submit(decoder, decoder_bindings, {}));
        std::vector<ExecutionDependency> dependencies{{decoded}};
        checked(checked(session->submit(stage, bindings, dependencies))->wait());
        std::vector<at::Tensor> outputs;
        for (std::size_t i = 0; i < 3; ++i) {
            const auto& t = decoder_bindings[i];
            outputs.push_back(dcvcuf::download(t.storage.buffer, t.shape));
        }
        for (const auto& t : bindings) outputs.push_back(dcvcuf::download(t.storage.buffer, t.shape));
        return outputs;
    }
    void reset() { checked(session->reset()); }
    std::vector<std::string> rejection_checks(Probe& other) {
        std::vector<std::string> passed;
        auto check_stage = [&](const ExecutableStage& stage, const std::vector<TensorView>& bindings,
                               const ExecutableStage& foreign_stage,
                               const std::vector<TensorView>& foreign_bindings) {
            auto reject = [&](std::string name, const auto& result) {
                TORCH_CHECK(!result, name, " was accepted");
                TORCH_CHECK(result.error().code() == ErrorCode::invalid_argument,
                            name, " returned unexpected error");
                passed.push_back(stage->descriptor().stage_id + ":" + name);
            };
            reject("foreign_stage", session->submit(foreign_stage, bindings, {}));
            reject("missing_binding", session->submit(stage, std::span(bindings).first(2), {}));
            for (int kind = 0; kind < 8; ++kind) {
                auto bad = bindings;
                const std::vector<std::string> names{"foreign_buffer", "shape", "name", "dtype",
                                                      "access", "size", "offset", "null_buffer"};
                if (kind == 0) bad[0].storage.buffer = foreign_bindings[0].storage.buffer;
                if (kind == 1) ++bad[0].shape[1];
                if (kind == 2) bad[0].name = "wrong";
                if (kind == 3) bad[0].data_type = TensorDataType::float32;
                if (kind == 4) bad[0].access = TensorAccess::write;
                if (kind == 5) --bad[0].storage.size_bytes;
                if (kind == 6) bad[0].storage.offset_bytes = 2;
                if (kind == 7) bad[0].storage.buffer.reset();
                reject(names[kind], session->submit(stage, bad, {}));
            }
            auto alias = bindings;
            alias[2].storage.buffer = alias[0].storage.buffer;
            reject("overlap", session->submit(stage, alias, {}));
            auto foreign_done = checked(other.session->submit(foreign_stage, foreign_bindings, {}));
            std::vector<ExecutionDependency> deps{{foreign_done}};
            reject("foreign_dependency", session->submit(stage, bindings, deps));
            auto done = checked(session->submit(stage, bindings, {}));
            deps[0].completion = done;
            checked(checked(session->submit(stage, bindings, deps))->wait());
            passed.push_back(stage->descriptor().stage_id + ":same_session_dependency");
            auto desc = stage->descriptor();
            desc.artifact.model_set_id = "wrong";
            reject("model_identity", session->load_stage(desc));
            desc = stage->descriptor();
            desc.artifact.path = "unverified-artifact";
            reject("unverified_artifact", session->load_stage(desc));
            reject("unsupported_memory", session->allocate(16, MemoryDomain::host));
            desc = stage->descriptor();
            desc.tensors.back().dimensions[3].minimum += 1;
            reject("output_contract", session->load_stage(desc));
            desc = stage->descriptor();
            desc.stage_id = "unknown";
            reject("unknown_stage", session->load_stage(desc));
        };
        check_stage(stage, bindings, other.stage, other.bindings);
        check_stage(decoder, decoder_bindings, other.decoder, other.decoder_bindings);
        return passed;
    }
private:
    std::shared_ptr<IProviderSession> session;
    ExecutableStage stage, decoder;
    std::vector<TensorView> bindings, decoder_bindings;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NVCRCaptureHTSProxy>(m, "CaptureHTS")
        .def(py::init<>())
        .def("set_param", &NVCRCaptureHTSProxy::set_param)
        .def("add_ref_feature_from_frame", &NVCRCaptureHTSProxy::add_ref_feature_from_frame)
        .def("decompress", &NVCRCaptureHTSProxy::decompress)
        .def("snapshot", &NVCRCaptureHTSProxy::snapshot)
        .def("entropy_snapshot", &NVCRCaptureHTSProxy::entropy_snapshot);
    m.def("decode_entropy", &decode_entropy);
    py::class_<Probe>(m, "SessionProbe")
        .def(py::init<const std::string&, const std::string&, int, int, int>())
        .def("run", &Probe::run)
        .def("run_pipeline", &Probe::run_pipeline)
        .def("reset", &Probe::reset)
        .def("rejection_checks", &Probe::rejection_checks);
}
