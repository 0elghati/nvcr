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
    Probe(const dcvcuf::Weights& weights, int h, int w, int device)
        : session(dcvcuf::make_recon_session(weights, device)),
          stage(checked(session->load_stage(dcvcuf::recon_descriptor(h, w)))) {
        for (const auto& contract : stage->descriptor().tensors) {
            std::vector<std::int64_t> shape;
            std::size_t size = 2;
            for (const auto& dim : contract.dimensions) {
                shape.push_back(dim.minimum);
                size *= dim.minimum;
            }
            auto buffer = checked(session->allocate(size, MemoryDomain::provider_device));
            bindings.push_back({contract.name, contract.data_type, std::move(shape),
                                contract.access, {std::move(buffer), 0, size}});
        }
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
    void reset() { checked(session->reset()); }
    std::vector<std::string> rejection_checks(Probe& other) {
        std::vector<std::string> passed;
        auto reject = [&](std::string name, const auto& result) {
            TORCH_CHECK(!result, name, " was accepted");
            TORCH_CHECK(result.error().code() == ErrorCode::invalid_argument,
                        name, " returned unexpected error");
            passed.push_back(std::move(name));
        };
        reject("foreign_stage", session->submit(other.stage, bindings, {}));
        reject("missing_binding", session->submit(stage, std::span(bindings).first(2), {}));
        for (int kind = 0; kind < 8; ++kind) {
            auto bad = bindings;
            const std::vector<std::string> names{"foreign_buffer", "shape", "name", "dtype",
                                                  "access", "size", "offset", "null_buffer"};
            if (kind == 0) bad[0].storage.buffer = other.bindings[0].storage.buffer;
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
        auto foreign_done = checked(other.session->submit(other.stage, other.bindings, {}));
        std::vector<ExecutionDependency> deps{{foreign_done}};
        reject("foreign_dependency", session->submit(stage, bindings, deps));
        auto done = checked(session->submit(stage, bindings, {}));
        deps[0].completion = done;
        checked(checked(session->submit(stage, bindings, deps))->wait());
        passed.push_back("same_session_dependency");
        auto desc = stage->descriptor();
        desc.artifact.model_set_id = "wrong";
        reject("model_identity", session->load_stage(desc));
        desc = stage->descriptor();
        desc.artifact.path = "unverified-artifact";
        reject("unverified_artifact", session->load_stage(desc));
        reject("unsupported_memory", session->allocate(16, MemoryDomain::host));
        return passed;
    }
private:
    std::shared_ptr<IProviderSession> session;
    ExecutableStage stage;
    std::vector<TensorView> bindings;
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
        .def(py::init<const dcvcuf::Weights&, int, int, int>())
        .def("run", &Probe::run)
        .def("reset", &Probe::reset)
        .def("rejection_checks", &Probe::rejection_checks);
}
