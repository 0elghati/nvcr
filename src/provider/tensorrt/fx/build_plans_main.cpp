// nvcr-fx-plans: compiles nvcr.fx.v1 programs into TensorRT plans for one bundle.
//
// usage: nvcr-fx-plans PROGRAM_DIR SHAPES.json OUT_DIR
//   SHAPES.json: {"<stage>": {"<input>": [channels, height, width], ...}, ...}
// Writes OUT_DIR/<stage>.fx.plan and OUT_DIR/<stage>.fx.plan.json, which names the
// SHA-256 of the program the plan was compiled from and every tensor's fixed shape.

#include "../../../common/json.hpp"
#include "plugins.hpp"

#include <fstream>
#include <iostream>

namespace fx = nvcr::tensorrt::fx;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: nvcr-fx-plans PROGRAM_DIR SHAPES.json OUT_DIR\n";
        return 2;
    }
    const std::filesystem::path programs = argv[1];
    const std::filesystem::path out = argv[3];
    auto shapes = nvcr::detail::json::parse_file(argv[2], "nvcr-fx-plans");
    if (!shapes) {
        std::cerr << shapes.error().describe() << '\n';
        return 1;
    }
    for (const auto& [stage, inputs] : shapes.value().object_value) {
        auto program = nvcr::deterministic::fx::load_program(programs, stage);
        auto digest = nvcr::deterministic::fx::program_digest(programs, stage);
        if (!program || !digest) {
            std::cerr << (program ? digest.error() : program.error()).describe() << '\n';
            return 1;
        }
        std::vector<fx::InputShape> list;
        for (const auto& [name, dims] : inputs.object_value) {
            const auto& d = dims.array_value;
            if (d.size() != 3 || !d[0].integer_value || !d[1].integer_value || !d[2].integer_value) {
                std::cerr << stage << ": shape of " << name << " must be [channels, height, width]\n";
                return 1;
            }
            list.push_back({name, static_cast<int>(*d[0].integer_value), static_cast<int>(*d[1].integer_value),
                            static_cast<int>(*d[2].integer_value)});
        }
        auto plan = fx::build_plan(program.value(), list);
        if (!plan) {
            std::cerr << plan.error().describe() << '\n';
            return 1;
        }
        const auto& bytes = plan.value().bytes;
        std::ofstream(out / (stage + ".fx.plan"), std::ios::binary)
            .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        std::ofstream sidecar(out / (stage + ".fx.plan.json"));
        sidecar << "{\"program_sha256\": \"" << digest.value() << "\", \"tensors\": {";
        const char* separator = "";
        for (const auto* shapes : {&list, &plan.value().outputs}) {
            for (const auto& s : *shapes) {
                sidecar << separator << '"' << s.name << "\": [1, " << s.channels << ", " << s.height << ", " << s.width
                        << ']';
                separator = ", ";
            }
        }
        sidecar << "}}\n";
        std::cout << stage << ": " << bytes.size() << " bytes\n";
    }
    return 0;
}
