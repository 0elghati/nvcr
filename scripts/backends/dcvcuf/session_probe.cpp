#include <torch/extension.h>
#include "recon_session.hpp"
#include "capture_hts.h"

namespace {
using namespace nvcr;
using namespace provider::experimental;
template<class T> T checked(Result<T> result) {
    TORCH_CHECK(result, result.error().describe());
    return std::move(result.value());
}
void checked(Result<void> result) { TORCH_CHECK(result, result.error().describe()); }

class Probe {
public:
    Probe(const dcvcuf::Weights& weights, const dcvcuf::Weights& decoder_weights,
          int h, int w, int device)
        : session(dcvcuf::make_recon_session(weights, decoder_weights, device)),
          stage(checked(session->load_stage(dcvcuf::recon_descriptor(h, w)))),
          decoder(checked(session->load_stage(dcvcuf::decoder_descriptor(h, w)))) {
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
        .def("snapshot", &NVCRCaptureHTSProxy::snapshot);
    py::class_<Probe>(m, "SessionProbe")
        .def(py::init<const dcvcuf::Weights&, const dcvcuf::Weights&, int, int, int>())
        .def("run", &Probe::run)
        .def("run_pipeline", &Probe::run_pipeline)
        .def("reset", &Probe::reset)
        .def("rejection_checks", &Probe::rejection_checks);
}
