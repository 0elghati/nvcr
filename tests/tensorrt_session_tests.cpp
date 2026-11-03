// Codec-agnostic TensorRT session contract: a stage described only by a
// generic descriptor is validated against its engine, and lifecycle misuse is
// rejected. Uses one DCVC-RT plan from an installed bundle as the fixture.

#include "provider/tensorrt/session.hpp"

#include <nvcr/configuration/configuration.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace session_api = nvcr::provider::experimental;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

session_api::DimensionBounds dynamic_axis() {
    return {1, std::numeric_limits<std::int32_t>::max()};
}

session_api::TensorContract tensor(
    std::string name, session_api::TensorAccess access,
    std::vector<session_api::DimensionBounds> dimensions) {
    return {std::move(name), session_api::TensorDataType::float16, access,
            std::move(dimensions)};
}

// DCVC-RT's I-frame synthesis plan: y_hat [1,256,H,W], q_dec [1,368,1,1] -> frame_hat [1,3,16H,16W].
session_api::ExecutableStageDescriptor synthesis_descriptor(const std::filesystem::path& root) {
    session_api::ExecutableStageDescriptor descriptor;
    descriptor.stage_id = "i_synthesis.plan";
    descriptor.artifact.provider_id = "tensorrt";
    descriptor.artifact.codec_id = "contract-test";
    descriptor.artifact.component_id = "i_synthesis.plan";
    descriptor.artifact.path = (root / "i_synthesis.plan").string();
    descriptor.tensors = {
        tensor("y_hat", session_api::TensorAccess::read,
               {{1, 1}, {256, 256}, dynamic_axis(), dynamic_axis()}),
        tensor("q_dec", session_api::TensorAccess::read, {{1, 1}, {368, 368}, {1, 1}, {1, 1}}),
        tensor("frame_hat", session_api::TensorAccess::write,
               {{1, 1}, {3, 3}, dynamic_axis(), dynamic_axis()}),
    };
    return descriptor;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: nvcr_tensorrt_session_tests ENGINE_DIRECTORY\n";
        return 2;
    }
    const auto root = std::filesystem::absolute(argv[1]);
    nvcr::RuntimeConfiguration configuration;
    configuration.artifacts.intra_engine_path = root;
    configuration.artifacts.model_id = "dcvcrt-cvpr2025";
    configuration.provider.id = "tensorrt";

    auto created = nvcr::tensorrt::make_provider_session(configuration);
    if (!created) {
        std::cerr << created.error().describe() << '\n';
        return 1;
    }
    auto session =
        std::dynamic_pointer_cast<nvcr::tensorrt::TensorRTExecutionSession>(created.value());
    expect(session != nullptr, "provider session is a TensorRT session");
    if (!session) return 1;
    expect(session->loaded_stages().empty(), "a new provider session holds no codec stages");
    expect(session->bundle_files().contains("i_synthesis.plan"),
           "the session lists the verified bundle files");

    auto wrong_name = synthesis_descriptor(root);
    wrong_name.tensors[0].name = "not_y_hat";
    expect(!session->load_stage(wrong_name), "a contract that disagrees with the engine is rejected");

    auto wrong_channels = synthesis_descriptor(root);
    wrong_channels.tensors[0].dimensions[1] = {128, 128};
    expect(!session->load_stage(wrong_channels), "a wrong fixed axis is rejected");

    auto outside = synthesis_descriptor(root);
    outside.stage_id = outside.artifact.component_id = "missing.plan";
    outside.artifact.path = (root / "missing.plan").string();
    expect(!session->load_stage(outside), "a file outside the verified bundle is rejected");

    auto foreign = synthesis_descriptor(root);
    foreign.artifact.provider_id = "other";
    expect(!session->load_stage(foreign), "a stage for another provider is rejected");

    auto stage = session->load_stage(synthesis_descriptor(root));
    expect(static_cast<bool>(stage), "a matching generic descriptor loads");
    expect(!session->load_stage(synthesis_descriptor(root)), "a stage loads at most once");
    expect(!session->submit(stage ? stage.value() : nullptr, {}, {}),
           "execution before finalize is rejected");

    auto finalized = session->finalize_stages();
    expect(static_cast<bool>(finalized), "the session finalizes its loaded stages");
    if (!finalized) std::cerr << finalized.error().describe() << '\n';
    expect(!session->finalize_stages(), "a session finalizes once");
    auto late = synthesis_descriptor(root);
    late.stage_id = late.artifact.component_id = "i_analysis.plan";
    late.artifact.path = (root / "i_analysis.plan").string();
    expect(!session->load_stage(late), "stages cannot be loaded after finalize");

    if (stage && finalized) {
        const auto scope = session->allocation_scope();
        auto y_hat = nvcr::tensorrt::allocate_device_tensor(
            "y_hat", nvcr::tensorrt::make_dims(256, 4, 4));
        auto q_dec = nvcr::tensorrt::allocate_device_tensor(
            "q_dec", nvcr::tensorrt::make_dims(368, 1, 1));
        expect(y_hat && q_dec, "device inputs allocate");
        if (y_hat && q_dec) {
            const nvcr::tensorrt::DeviceTensor* inputs[] = {&y_hat.value(), &q_dec.value()};
            auto outputs = session->run_stage(stage.value(), inputs);
            expect(outputs && outputs.value().size() == 1U &&
                       outputs.value().front().shape.d[2] == 64 &&
                       outputs.value().front().shape.d[3] == 64,
                   "the generic stage executes with engine-resolved output shapes");
            expect(static_cast<bool>(
                       nvcr::tensorrt::synchronize_stream(session->stream(), "test")),
                   "the stage completes");
        }
    }

    if (failures != 0) return 1;
    std::cout << "TensorRT session contracts validated\n";
    return 0;
}
