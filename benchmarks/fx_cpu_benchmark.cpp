// nvcr.fx.v1 CPU executor: time per stage on seeded inputs (tests/fx_inputs.hpp).
//
// usage: nvcr_fx_cpu_benchmark PROGRAM_DIR [--repeat N] [--reference] WIDTHxHEIGHT...
//   PROGRAM_DIR: a bundle's normative/ directory (any codec's fx programs).
//   --reference: time the reference kernels instead of the tiled ones.
// Prints one line per stage and size: the median and minimum wall time of N
// runs after one warm-up, and whether the stage failed on these inputs.

#include "deterministic/fx_program.hpp"
#include "fx_inputs.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace det = nvcr::deterministic::fx;
using Clock = std::chrono::steady_clock;

std::vector<std::string> stages(const std::filesystem::path& directory) {
    std::vector<std::string> names;
    const std::string suffix = ".fxprog.json";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.size() > suffix.size() && name.ends_with(suffix)) names.push_back(name.substr(0, name.size() - suffix.size()));
    }
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: nvcr_fx_cpu_benchmark PROGRAM_DIR [--repeat N] [--reference] WIDTHxHEIGHT...\n";
        return 2;
    }
    const std::filesystem::path directory = argv[1];
    int repeat = 3;
    auto kernels = det::Kernels::tiled;
    std::vector<std::pair<int, int>> sizes;
    for (int a = 2; a < argc; ++a) {
        const std::string arg = argv[a];
        int width = 0, height = 0;
        if (arg == "--repeat" && a + 1 < argc) {
            repeat = std::max(1, std::stoi(argv[++a]));
        } else if (arg == "--reference") {
            kernels = det::Kernels::reference;
        } else if (std::sscanf(arg.c_str(), "%dx%d", &width, &height) == 2 && width > 0 && height > 0) {
            sizes.emplace_back(width, height);
        } else {
            std::cerr << "bad argument: " << arg << '\n';
            return 2;
        }
    }
    std::cout << (kernels == det::Kernels::tiled ? "tiled" : "reference") << " kernels, hardware threads "
              << std::thread::hardware_concurrency() << ", repeat " << repeat << '\n';
    std::cout << "stage,size,median_ms,min_ms,status\n";
    for (const auto& stage : stages(directory)) {
        auto program = det::load_program(directory, stage);
        if (!program) {
            std::cerr << stage << ": " << program.error().describe() << '\n';
            return 1;
        }
        for (const auto& [width, height] : sizes) {
            std::mt19937 rng(7);
            const auto inputs = nvcr::test::fx::stage_inputs(program.value(), width, height, rng);
            bool ok = det::run(program.value(), inputs, kernels).has_value();  // warm-up
            std::vector<double> ms;
            for (int r = 0; r < repeat; ++r) {
                const auto start = Clock::now();
                ok = det::run(program.value(), inputs, kernels).has_value() && ok;
                ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            }
            std::sort(ms.begin(), ms.end());
            std::printf("%s,%dx%d,%.1f,%.1f,%s\n", stage.c_str(), width, height, ms[ms.size() / 2], ms.front(),
                        ok ? "ok" : "stage_failure");
            std::fflush(stdout);
        }
    }
    return 0;
}
