#include <nvcr/configuration/configuration.hpp>
#include <nvcr/dcvcrt/tensorrt_backend.hpp>

#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: nvcr_tensorrt_engine_tests ENGINE_DIRECTORY\n";
        return 2;
    }

    auto backend = nvcr::dcvcrt::make_tensorrt_backend();
    if (!backend) {
        std::cerr << backend.error().describe() << '\n';
        return 1;
    }

    nvcr::RuntimeConfiguration configuration;
    configuration.intra_engine_path = std::filesystem::path(argv[1]);
    configuration.device_id = 0;
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

    std::cout << "TensorRT I/P-frame engine contracts validated\n";
    return 0;
}
