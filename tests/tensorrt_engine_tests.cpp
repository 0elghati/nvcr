#include <nvcr/configuration/configuration.hpp>
#include <nvcr/dcvcrt/tensorrt_backend.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace fs = std::filesystem;

constexpr std::array<std::string_view, 20> runtime_files{
    "i_analysis.plan", "i_hyper_analysis.plan", "i_hyper_synthesis.plan",
    "i_spatial_prior_1.plan", "i_spatial_prior_2.plan", "i_spatial_prior_3.plan",
    "i_synthesis.plan", "p_reference_frame.plan", "p_reference_feature.plan",
    "p_analysis.plan", "p_hyper_analysis.plan", "p_prior.plan",
    "p_spatial_prior.plan", "p_synthesis.plan", "i_entropy.bin", "i_quant.bin",
    "i_frame_manifest.json", "p_entropy.bin", "p_quant.bin", "p_frame_manifest.json",
};

struct TemporaryDirectory final {
    fs::path path;
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

bool initialize_is_rejected(
    const nvcr::RuntimeConfiguration& configuration,
    std::string_view case_name) {
    auto backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) {
        std::cerr << backend.error().describe() << '\n';
        return false;
    }
    auto initialized = backend.value()->initialize(configuration);
    if (initialized) {
        std::cerr << case_name << " was accepted\n";
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: nvcr_tensorrt_engine_tests ENGINE_DIRECTORY\n";
        return 2;
    }

    const fs::path engine_root = fs::absolute(argv[1]);
    nvcr::RuntimeConfiguration configuration;
    configuration.intra_engine_path = engine_root;
    configuration.device_id = 0;

    auto wrong_model = configuration;
    wrong_model.model_id = "different-dcvcrt-model";
    if (!initialize_is_rejected(wrong_model, "wrong-model engine bundle")) return 1;

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    TemporaryDirectory temporary{
        fs::temp_directory_path() / ("nvcr-engine-contract-" + std::to_string(nonce))};
    std::error_code filesystem_error;
    fs::create_directories(temporary.path, filesystem_error);
    if (filesystem_error) {
        std::cerr << "cannot create engine test directory: " << filesystem_error.message() << '\n';
        return 1;
    }

    {
        std::ofstream manifest(temporary.path / "engine_manifest.json", std::ios::binary);
        manifest << "{\"format\": 2, \"schema\": \"corrupt\"}\n";
    }
    auto corrupt_manifest = configuration;
    corrupt_manifest.intra_engine_path = temporary.path;
    if (!initialize_is_rejected(corrupt_manifest, "corrupt engine manifest")) return 1;

    for (const auto name : runtime_files) {
        fs::create_symlink(engine_root / name, temporary.path / name, filesystem_error);
        if (filesystem_error) {
            std::cerr << "cannot create engine test symlink for " << name << ": "
                      << filesystem_error.message() << '\n';
            return 1;
        }
    }
    fs::copy_file(
        engine_root / "engine_manifest.json",
        temporary.path / "engine_manifest.json",
        fs::copy_options::overwrite_existing,
        filesystem_error);
    if (filesystem_error) {
        std::cerr << "cannot copy engine manifest: " << filesystem_error.message() << '\n';
        return 1;
    }
    {
        std::ofstream checksums(temporary.path / "engine.sha256", std::ios::binary);
        checksums << "corrupt\n";
    }
    if (!initialize_is_rejected(corrupt_manifest, "corrupt engine checksum manifest")) return 1;

    auto backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) {
        std::cerr << backend.error().describe() << '\n';
        return 1;
    }
    auto initialized = backend.value()->initialize(configuration);
    if (!initialized) {
        std::cerr << initialized.error().describe() << '\n';
        return 1;
    }

    auto flushed = backend.value()->flush();
    if (!flushed) {
        std::cerr << flushed.error().describe() << '\n';
        return 1;
    }

    auto initialized_twice = backend.value()->initialize(configuration);
    if (initialized_twice ||
        initialized_twice.error().code() != nvcr::ErrorCode::invalid_state) {
        std::cerr << "second TensorRT initialization was not rejected\n";
        return 1;
    }

    std::cout << "TensorRT bundle identity, integrity, and I/P contracts validated\n";
    return 0;
}
