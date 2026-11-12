// nvcr.fx-program.v1: loading, the spec's load-time refusals, the data digest
// and running a small program through the CPU reference.
#include "common/sha256.hpp"
#include "deterministic/fx_program.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fx = nvcr::deterministic::fx;
namespace fs = std::filesystem;

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

struct Data final {
    std::string bytes;

    template <typename T>
    std::string add(const std::vector<T>& values) {
        const auto offset = bytes.size();
        bytes.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
        return "[" + std::to_string(offset) + ", " + std::to_string(values.size()) + "]";
    }
};

// x (1 channel) -> im2col 3x3 -> cols (9) -> gemm (centre weight w, 2 rows summed) -> y (1)
//   -> pixel_shuffle check: z = pixel_unshuffle(pixel_shuffle(q)) for q (4 channels).
void write_program(const fs::path& dir, std::int16_t weight, std::int32_t shift) {
    Data data;
    std::vector<std::int16_t> w(18, 0);
    w[4] = weight;
    w[13] = 1;
    const auto weights = data.add(w);
    const auto bias = data.add(std::vector<std::int64_t>{0, 0});
    const auto shifts = data.add(std::vector<std::int32_t>{shift, shift});
    const auto dc = data.add(std::vector<std::int16_t>{7, 7, 7});
    std::ofstream(dir / "t.fxw", std::ios::binary) << data.bytes;
    auto digest = nvcr::detail::sha256_file(dir / "t.fxw", "test");
    std::ofstream(dir / "t.fxprog.json") << R"({"format": "nvcr.fx-program.v1", "profile": "nvcr.fx.v1", "fa": 14,
 "data": {"file": "t.fxw", "sha256": ")" << digest.value() << R"("},
 "tensors": [{"name": "x", "channels": 1}, {"name": "cols", "channels": 9}, {"name": "y", "channels": 1},
             {"name": "q", "channels": 4}, {"name": "up", "channels": 1}, {"name": "z", "channels": 4}],
 "inputs": [0, 3], "outputs": [2, 5],
 "delta_tables": [{"h": 2, "shift": 1, "dc": )" << dc << R"(}],
 "nodes": [
  {"op": "im2col", "inputs": [0], "output": 1, "kernel": 3, "stride": 1},
  {"op": "gemm", "inputs": [1], "output": 2, "in_channels": 9, "out_channels": 2, "group": 2, "input_shift": 0,
   "delta_table": 0, "weights": )" << weights << R"(, "bias": )" << bias << R"(, "shift": )" << shifts << R"(},
  {"op": "pixel_shuffle", "inputs": [3], "output": 4, "factor": 2},
  {"op": "pixel_unshuffle", "inputs": [4], "output": 5, "factor": 2}
 ]})";
}

void runs() {
    const auto dir = fs::temp_directory_path() / "nvcr_fx_program_tests";
    fs::create_directories(dir);
    write_program(dir, 2, 1);
    auto program = fx::load_program(dir, "t");
    expect(program.has_value(), "a valid program loads");
    if (!program) {
        std::cerr << program.error().describe() << '\n';
        return;
    }
    // x = 5 on a 1x1 image: row 0 = rsr(2 * 5, 1) = 5 and row 1 = rsr(5, 1) = 3, both above
    // the delta table's h = 2, so the activation keeps them; the group sums them to 8.
    const fx::Tensor x{1, 1, 1, {5}};
    const fx::Tensor q{4, 1, 1, {1, 2, 3, 4}};
    const fx::Tensor inputs[] = {x, q};
    auto outputs = fx::run(program.value(), inputs);
    expect(outputs && outputs.value()[0].values == std::vector<std::int32_t>{8}, "im2col, gemm and grouping");
    expect(outputs && outputs.value()[1].values == q.values, "pixel_unshuffle undoes pixel_shuffle");

    const fx::Tensor big{1, 1, 1, {(1 << 30) + 1}};
    const fx::Tensor bad_inputs[] = {big, q};
    expect(!fx::run(program.value(), bad_inputs), "an input above 2^30 fails the stage");

    auto digest = fx::program_digest(dir, "t");
    write_program(dir, 3, 1);
    auto changed = fx::program_digest(dir, "t");
    expect(digest && changed && digest.value() != changed.value(), "the digest covers the weights");

    write_program(dir, 32640, 1);
    expect(!fx::load_program(dir, "t"), "a gemm weight above two digits is refused");
    write_program(dir, 2, 32);
    expect(!fx::load_program(dir, "t"), "a shift of 32 is refused");

    write_program(dir, 2, 1);
    std::ofstream(dir / "t.fxw", std::ios::binary | std::ios::app) << 'x';
    auto tampered = fx::load_program(dir, "t");
    expect(!tampered && tampered.error().code() == nvcr::ErrorCode::digest_mismatch,
           "a changed data file is refused");
    fs::remove_all(dir);
}

}  // namespace

int main() {
    runs();
    if (failures == 0) std::cout << "nvcr.fx-program.v1 tests passed\n";
    return failures == 0 ? 0 : 1;
}
