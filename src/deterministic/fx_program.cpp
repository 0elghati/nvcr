#include "fx_program.hpp"

#include "../common/json.hpp"
#include "../common/sha256.hpp"

#include <cstring>
#include <fstream>
#include <iterator>
#include <map>

namespace nvcr::deterministic::fx {
namespace {

constexpr std::string_view subsystem = "deterministic.fx";
constexpr std::string_view format = "nvcr.fx-program.v1";
using Json = detail::json::Value;

Error invalid(std::string message) {
    return Error(ErrorCode::invalid_argument, "fx program: " + std::move(message), std::string(subsystem));
}

Error stage_failure(std::string message) {
    return Error(ErrorCode::malformed_bitstream, "nvcr.fx.v1: " + std::move(message), std::string(subsystem));
}

Result<std::int64_t> integer(const Json& object, std::string_view key) {
    const auto* value = object.find(key);
    if (value == nullptr || !value->integer_value) return invalid("field '" + std::string(key) + "' must be an integer");
    return *value->integer_value;
}

Result<int> small(const Json& object, std::string_view key, std::int64_t lo, std::int64_t hi) {
    auto value = integer(object, key);
    if (!value) return value.error();
    if (value.value() < lo || value.value() > hi) return invalid("field '" + std::string(key) + "' out of range");
    return static_cast<int>(value.value());
}

// A [byte offset, element count] reference into the data file.
template <typename T>
Result<std::vector<T>> array(const Json& object, std::string_view key, const std::string& data) {
    const auto* value = object.find(key);
    if (value == nullptr || value->array_value.size() != 2 || !value->array_value[0].integer_value ||
        !value->array_value[1].integer_value) {
        return invalid("field '" + std::string(key) + "' must be [offset, count]");
    }
    const auto offset = *value->array_value[0].integer_value;
    const auto count = *value->array_value[1].integer_value;
    if (offset < 0 || count < 0 || static_cast<std::uint64_t>(offset) > data.size() ||
        static_cast<std::uint64_t>(count) > (data.size() - static_cast<std::size_t>(offset)) / sizeof(T)) {
        return invalid("field '" + std::string(key) + "' lies outside the data file");
    }
    std::vector<T> out(static_cast<std::size_t>(count));
    std::memcpy(out.data(), data.data() + offset, out.size() * sizeof(T));
    return out;
}

template <typename T>
bool within(const std::vector<T>& values, std::int64_t lo, std::int64_t hi) {
    for (const auto v : values) {
        if (v < lo || v > hi) return false;
    }
    return true;
}

const std::map<std::string, Op, std::less<>> op_names{
    {"gemm", Op::gemm}, {"depthwise3x3", Op::depthwise3x3}, {"mul", Op::mul},
    {"affine", Op::affine}, {"add", Op::add}, {"sat_table", Op::sat_table},
    {"delta", Op::delta}, {"clamp", Op::clamp}, {"concat", Op::concat}, {"slice", Op::slice},
    {"pixel_shuffle", Op::pixel_shuffle}, {"pixel_unshuffle", Op::pixel_unshuffle}, {"im2col", Op::im2col},
    {"param_row", Op::param_row}, {"crop_like", Op::crop_like},
};

std::size_t input_count(Op op) {
    switch (op) {
    case Op::mul:
    case Op::add:
    case Op::concat:
    case Op::crop_like:
        return 2;
    default:
        return 1;
    }
}

// Reads one node's parameters and checks the spec's load-time bounds.
Result<Node> parse_node(const Json& json, const Program& program, const std::string& data) {
    Node node;
    const auto* op = json.find("op");
    const auto found = op ? op_names.find(op->string_value) : op_names.end();
    if (found == op_names.end()) return invalid("unknown op");
    node.op = found->second;
    const auto* inputs = json.find("inputs");
    if (inputs == nullptr) return invalid("node without inputs");
    for (const auto& input : inputs->array_value) {
        if (!input.integer_value || *input.integer_value < 0 ||
            static_cast<std::size_t>(*input.integer_value) >= program.tensor_names.size()) {
            return invalid("node input out of range");
        }
        node.inputs.push_back(static_cast<std::size_t>(*input.integer_value));
    }
    const std::size_t extra = node.op == Op::gemm ? 2 : 0;  // gemm: up to two residuals
    if (node.inputs.size() < input_count(node.op) || node.inputs.size() > input_count(node.op) + extra) {
        return invalid("wrong number of node inputs");
    }
    auto output = small(json, "output", 0, static_cast<std::int64_t>(program.tensor_names.size()) - 1);
    if (!output) return output.error();
    node.output = static_cast<std::size_t>(output.value());
    const auto delta_ok = [&](int index) { return index >= -1 && index < static_cast<int>(program.delta_tables.size()); };

    switch (node.op) {
    case Op::gemm: {
        auto& g = node.gemm;
        auto in = small(json, "in_channels", 1, 1 << 20);
        auto out = small(json, "out_channels", 1, 1 << 20);
        auto group = small(json, "group", 1, 4);
        auto input_shift = small(json, "input_shift", 0, 30);
        auto table = small(json, "delta_table", -1, 1 << 20);
        auto weights = array<std::int16_t>(json, "weights", data);
        auto bias = array<std::int64_t>(json, "bias", data);
        auto shift = array<std::int32_t>(json, "shift", data);
        for (const auto* check : {&in, &out, &group, &input_shift, &table}) {
            if (!*check) return check->error();
        }
        if (!weights) return weights.error();
        if (!bias) return bias.error();
        if (!shift) return shift.error();
        g.in_channels = in.value();
        g.out_channels = out.value();
        g.group = group.value();
        g.input_shift = input_shift.value();
        node.delta_table = table.value();
        g.weights.assign(weights.value().begin(), weights.value().end());
        g.bias = std::move(bias.value());
        g.shift = std::move(shift.value());
        const auto rows = static_cast<std::size_t>(g.out_channels);
        if (g.group == 3 || g.out_channels % g.group != 0 || !delta_ok(node.delta_table) ||
            g.weights.size() != rows * static_cast<std::size_t>(g.in_channels) || g.bias.size() != rows ||
            g.shift.size() != rows) {
            return invalid("gemm parameters do not match its shape");
        }
        if (!within(g.weights, -digit2_max, digit2_max) || !within(g.bias, -(std::int64_t{1} << 52), std::int64_t{1} << 52) ||
            !within(g.shift, 1, 31)) {
            return invalid("gemm weights, bias or shifts exceed the profile's bounds");
        }
        break;
    }
    case Op::depthwise3x3: {
        auto weights = array<std::int32_t>(json, "weights", data);
        auto bias = array<std::int64_t>(json, "bias", data);
        auto shift = array<std::int32_t>(json, "shift", data);
        if (!weights) return weights.error();
        if (!bias) return bias.error();
        if (!shift) return shift.error();
        node.depthwise = {std::move(weights.value()), std::move(bias.value()), std::move(shift.value())};
        const auto channels = static_cast<std::size_t>(program.tensor_channels[node.inputs[0]]);
        const auto& d = node.depthwise;
        if (d.weights.size() != channels * 9 || d.bias.size() != channels || d.shift.size() != channels ||
            !within(d.weights, -(1 << 24), 1 << 24) || !within(d.bias, -(std::int64_t{1} << 52), std::int64_t{1} << 52) ||
            !within(d.shift, 0, 31)) {
            return invalid("depthwise parameters exceed the profile's bounds");
        }
        break;
    }
    case Op::mul: {
        auto shift = small(json, "shift", 0, 31);
        if (!shift) return shift.error();
        node.shift = shift.value();
        const auto b = program.tensor_channels[node.inputs[1]];
        if (b != 1 && b != program.tensor_channels[node.inputs[0]]) return invalid("mul factor channels");
        break;
    }
    case Op::param_row: {
        auto rows = array<std::int64_t>(json, "rows", data);
        if (!rows) return rows.error();
        node.rows = std::move(rows.value());
        const auto channels = static_cast<std::size_t>(program.tensor_channels[node.output]);
        if (program.tensor_channels[node.inputs[0]] != 1 || node.rows.empty() || node.rows.size() % channels != 0 ||
            !within(node.rows, -(1 << 24), 1 << 24)) {
            return invalid("param_row table does not match its output or exceeds 2^24");
        }
        break;
    }
    case Op::affine: {
        auto alpha = integer(json, "alpha");
        auto beta = integer(json, "beta");
        auto shift = small(json, "shift", 0, 31);
        if (!alpha) return alpha.error();
        if (!beta) return beta.error();
        if (!shift) return shift.error();
        node.alpha = alpha.value();
        node.beta = beta.value();
        node.shift = shift.value();
        if (node.alpha < -(1 << 24) || node.alpha > (1 << 24) || node.beta < -(std::int64_t{1} << 52) ||
            node.beta > (std::int64_t{1} << 52)) {
            return invalid("affine parameters exceed the profile's bounds");
        }
        break;
    }
    case Op::sat_table: {
        auto table = array<std::int32_t>(json, "table", data);
        auto top = integer(json, "top");
        if (!table) return table.error();
        if (!top) return top.error();
        node.table = std::move(table.value());
        node.hi = top.value();
        if (node.table.empty() || node.table.size() % 2 != 0 || node.hi < INT32_MIN || node.hi > INT32_MAX) {
            return invalid("sat_table needs an even, non-empty table");
        }
        break;
    }
    case Op::delta: {
        auto table = small(json, "delta_table", 0, 1 << 20);
        if (!table) return table.error();
        node.delta_table = table.value();
        if (!delta_ok(node.delta_table)) return invalid("delta table out of range");
        break;
    }
    case Op::clamp: {
        auto lo = integer(json, "lo");
        auto hi = integer(json, "hi");
        if (!lo) return lo.error();
        if (!hi) return hi.error();
        node.lo = lo.value();
        node.hi = hi.value();
        if (node.lo > node.hi || node.lo < INT32_MIN || node.hi > INT32_MAX) return invalid("clamp bounds invalid");
        break;
    }
    case Op::slice: {
        auto begin = small(json, "begin", 0, 1 << 20);
        auto end = small(json, "end", 1, 1 << 20);
        if (!begin) return begin.error();
        if (!end) return end.error();
        node.begin = begin.value();
        node.end = end.value();
        if (node.begin >= node.end || node.end > program.tensor_channels[node.inputs[0]]) {
            return invalid("slice range invalid");
        }
        break;
    }
    case Op::pixel_shuffle:
    case Op::pixel_unshuffle: {
        auto factor = small(json, "factor", 2, 16);
        if (!factor) return factor.error();
        node.factor = factor.value();
        break;
    }
    case Op::im2col: {
        auto kernel = small(json, "kernel", 2, 3);
        auto stride = small(json, "stride", 1, 2);
        if (!kernel) return kernel.error();
        if (!stride) return stride.error();
        node.kernel = kernel.value();
        node.stride = stride.value();
        if (node.kernel == 2 && node.stride != 2) return invalid("im2col 2x2 needs stride 2");
        break;
    }
    case Op::add:
    case Op::concat:
    case Op::crop_like:
        break;
    }
    return node;
}

Tensor concat(const Tensor& a, const Tensor& b) {
    Tensor y{a.channels + b.channels, a.height, a.width, a.values};
    y.values.insert(y.values.end(), b.values.begin(), b.values.end());
    return y;
}

Tensor slice(const Tensor& x, int begin, int end) {
    const auto plane = x.plane();
    Tensor y{end - begin, x.height, x.width, {}};
    y.values.assign(x.values.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(begin) * plane),
                    x.values.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(end) * plane));
    return y;
}

// PyTorch pixel_shuffle / pixel_unshuffle: in[c r^2 + i r + j][y][x] <-> out[c][y r + i][x r + j].
Tensor shuffle(const Tensor& x, int r, bool up) {
    Tensor y = up ? Tensor{x.channels / (r * r), x.height * r, x.width * r, {}}
                  : Tensor{x.channels * r * r, x.height / r, x.width / r, {}};
    y.values.resize(x.values.size());
    const Tensor& deep = up ? x : y;  // the tensor with r^2 channels per output channel
    for (std::int32_t c = 0; c < deep.channels; ++c) {
        const int base = c / (r * r);
        const int i = (c / r) % r;
        const int j = c % r;
        for (std::int32_t yy = 0; yy < deep.height; ++yy) {
            for (std::int32_t xx = 0; xx < deep.width; ++xx) {
                const auto d = (static_cast<std::size_t>(c) * static_cast<std::size_t>(deep.height) + static_cast<std::size_t>(yy)) *
                        static_cast<std::size_t>(deep.width) + static_cast<std::size_t>(xx);
                const auto wide_w = static_cast<std::size_t>(deep.width) * static_cast<std::size_t>(r);
                const auto wide_h = static_cast<std::size_t>(deep.height) * static_cast<std::size_t>(r);
                const auto s = (static_cast<std::size_t>(base) * wide_h + static_cast<std::size_t>(yy * r + i)) * wide_w +
                    static_cast<std::size_t>(xx * r + j);
                if (up) {
                    y.values[s] = x.values[d];
                } else {
                    y.values[d] = x.values[s];
                }
            }
        }
    }
    return y;
}

// Channels (c, i, j) of the k x k taps; zero outside. 3x3: padding 1; 2x2: padding 0.
Tensor im2col(const Tensor& x, int k, int stride) {
    const int pad = k == 3 ? 1 : 0;
    const int h = (x.height + 2 * pad - k) / stride + 1;
    const int w = (x.width + 2 * pad - k) / stride + 1;
    Tensor y{x.channels * k * k, h, w, {}};
    y.values.assign(static_cast<std::size_t>(y.channels) * y.plane(), 0);
    for (std::int32_t c = 0; c < x.channels; ++c) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                const auto out_c = static_cast<std::size_t>((c * k + i) * k + j);
                for (int oy = 0; oy < h; ++oy) {
                    for (int ox = 0; ox < w; ++ox) {
                        const int iy = oy * stride - pad + i;
                        const int ix = ox * stride - pad + j;
                        if (iy < 0 || iy >= x.height || ix < 0 || ix >= x.width) continue;
                        y.values[out_c * y.plane() + static_cast<std::size_t>(oy * w + ox)] =
                            x.values[static_cast<std::size_t>(c) * x.plane() + static_cast<std::size_t>(iy * x.width + ix)];
                    }
                }
            }
        }
    }
    return y;
}

}  // namespace

Result<std::string> program_digest(const std::filesystem::path& directory, std::string_view stage) {
    return detail::sha256_file(directory / (std::string(stage) + ".fxprog.json"), subsystem);
}

Result<Program> load_program(const std::filesystem::path& directory, std::string_view stage) {
    auto json = detail::json::parse_file(directory / (std::string(stage) + ".fxprog.json"), subsystem);
    if (!json) return json.error();
    const auto& root = json.value();
    const auto* format_value = root.find("format");
    const auto* profile = root.find("profile");
    const auto* data_info = root.find("data");
    if (format_value == nullptr || format_value->string_value != format || profile == nullptr ||
        profile->string_value != "nvcr.fx.v1" || data_info == nullptr) {
        return invalid("not an nvcr.fx-program.v1 program for nvcr.fx.v1");
    }
    const auto* file = data_info->find("file");
    const auto* sha = data_info->find("sha256");
    if (file == nullptr || sha == nullptr || file->string_value.find('/') != std::string::npos) {
        return invalid("data file reference invalid");
    }
    const auto data_path = directory / file->string_value;
    auto data_digest = detail::sha256_file(data_path, subsystem);
    if (!data_digest) return data_digest.error();
    if (data_digest.value() != sha->string_value) {
        return Error(ErrorCode::digest_mismatch, "fx program data digest mismatch: " + data_path.string(),
                     std::string(subsystem));
    }
    std::ifstream stream(data_path, std::ios::binary);
    const std::string data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    Program program;
    program.name = std::string(stage);
    auto fa = small(root, "fa", 0, 24);
    if (!fa) return fa.error();
    program.fa = fa.value();
    const auto* tensors = root.find("tensors");
    if (tensors == nullptr || tensors->array_value.empty()) return invalid("program without tensors");
    for (const auto& tensor : tensors->array_value) {
        const auto* name = tensor.find("name");
        auto channels = small(tensor, "channels", 1, 1 << 20);
        if (name == nullptr || !channels) return invalid("tensor needs a name and channels");
        program.tensor_names.push_back(name->string_value);
        program.tensor_channels.push_back(channels.value());
    }
    for (const auto* key : {"inputs", "outputs"}) {
        const auto* list = root.find(key);
        if (list == nullptr || list->array_value.empty()) return invalid(std::string(key) + " missing");
        for (const auto& id : list->array_value) {
            if (!id.integer_value || *id.integer_value < 0 ||
                static_cast<std::size_t>(*id.integer_value) >= program.tensor_names.size()) {
                return invalid(std::string(key) + " out of range");
            }
            (std::string_view(key) == "inputs" ? program.inputs : program.outputs)
                .push_back(static_cast<std::size_t>(*id.integer_value));
        }
    }
    if (const auto* tables = root.find("delta_tables")) {
        for (const auto& table : tables->array_value) {
            auto h = integer(table, "h");
            auto shift = small(table, "shift", 0, 30);
            auto dc = array<std::int16_t>(table, "dc", data);
            if (!h || !shift || !dc) return invalid("delta table needs h, shift and dc");
            if (h.value() < 0 || h.value() > stored_max ||
                dc.value().size() != static_cast<std::size_t>(((2 * h.value()) >> shift.value()) + 1)) {
                return invalid("delta table size does not match h and shift");
            }
            program.delta_tables.push_back({h.value(), shift.value(), std::move(dc.value())});
        }
    }
    const auto* nodes = root.find("nodes");
    if (nodes == nullptr || nodes->array_value.empty()) return invalid("program without nodes");
    for (const auto& json_node : nodes->array_value) {
        auto node = parse_node(json_node, program, data);
        if (!node) return node.error();
        program.nodes.push_back(std::move(node.value()));
    }
    return program;
}

Result<std::vector<Tensor>> run(const Program& program, std::span<const Tensor> inputs) {
    if (inputs.size() != program.inputs.size()) return invalid(program.name + ": wrong number of inputs");
    std::vector<std::optional<Tensor>> tensors(program.tensor_names.size());
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        const auto id = program.inputs[slot];
        const auto& x = inputs[slot];
        if (x.channels != program.tensor_channels[id] || x.values.size() != static_cast<std::size_t>(x.channels) * x.plane()) {
            return invalid(program.name + ": input " + program.tensor_names[id] + " has the wrong shape");
        }
        for (const auto v : x.values) {
            if (v > stored_max || v < -stored_max) return stage_failure(program.name + ": input value out of range");
        }
        tensors[id] = x;
    }
    for (const auto& node : program.nodes) {
        std::vector<const Tensor*> in;
        for (const auto id : node.inputs) {
            if (!tensors[id]) return invalid(program.name + ": tensor read before it is written");
            in.push_back(&*tensors[id]);
        }
        const Tensor& x = *in[0];
        const bool sized = node.op == Op::gemm || node.op == Op::add || node.op == Op::concat;
        for (const auto* t : in) {
            if (sized && (t->height != x.height || t->width != x.width)) {
                return invalid(program.name + ": node inputs differ in size");
            }
        }
        Result<Tensor> y = Tensor{};
        switch (node.op) {
        case Op::gemm: {
            Gemm op = node.gemm;
            op.activation = node.delta_table >= 0 ? &program.delta_tables[static_cast<std::size_t>(node.delta_table)] : nullptr;
            if (x.channels != op.in_channels) return invalid(program.name + ": gemm input channels");
            for (std::size_t r = 1; r < in.size(); ++r) {
                if (in[r]->channels != op.out_channels / op.group) return invalid(program.name + ": residual channels");
            }
            y = gemm(op, x, std::span<const Tensor* const>(in).subspan(1));
            break;
        }
        case Op::depthwise3x3: y = depthwise3x3(node.depthwise, x); break;
        case Op::mul: {
            const Tensor& b = *in[1];
            if (b.plane() != 1 && (b.height != x.height || b.width != x.width)) {
                return invalid(program.name + ": mul factor size");
            }
            y = mul(x, b, node.shift);
            break;
        }
        case Op::param_row: {
            const auto channels = static_cast<std::size_t>(program.tensor_channels[node.output]);
            if (x.values.size() != 1 || x.values[0] < 0 ||
                static_cast<std::size_t>(x.values[0]) >= node.rows.size() / channels) {
                return stage_failure(program.name + ": qp outside the parameter table");
            }
            const auto row = static_cast<std::size_t>(x.values[0]) * channels;
            Tensor out{static_cast<std::int32_t>(channels), 1, 1, {}};
            for (std::size_t c = 0; c < channels; ++c) out.values.push_back(static_cast<std::int32_t>(node.rows[row + c]));
            y = std::move(out);
            break;
        }
        case Op::crop_like: {
            const Tensor& ref = *in[1];
            if (ref.height > x.height || ref.width > x.width) return invalid(program.name + ": crop larger than input");
            Tensor out{x.channels, ref.height, ref.width, {}};
            for (std::int32_t c = 0; c < x.channels; ++c) {
                for (std::int32_t row = 0; row < ref.height; ++row) {
                    const auto from = x.values.begin() +
                        static_cast<std::ptrdiff_t>((static_cast<std::size_t>(c) * static_cast<std::size_t>(x.height) +
                                                     static_cast<std::size_t>(row)) * static_cast<std::size_t>(x.width));
                    out.values.insert(out.values.end(), from, from + ref.width);
                }
            }
            y = std::move(out);
            break;
        }
        case Op::affine: y = affine(x, node.alpha, node.beta, node.shift); break;
        case Op::add:
            if (in[1]->channels != x.channels) return invalid(program.name + ": add channels");
            y = add(x, *in[1]);
            break;
        case Op::sat_table: y = sat_table(x, node.table, static_cast<std::int32_t>(node.hi)); break;
        case Op::delta: y = delta(x, program.delta_tables[static_cast<std::size_t>(node.delta_table)]); break;
        case Op::clamp: y = clamp(x, static_cast<std::int32_t>(node.lo), static_cast<std::int32_t>(node.hi)); break;
        case Op::concat: y = concat(x, *in[1]); break;
        case Op::slice: y = slice(x, node.begin, node.end); break;
        case Op::pixel_shuffle:
        case Op::pixel_unshuffle: {
            const int r = node.factor;
            const bool up = node.op == Op::pixel_shuffle;
            if (up ? x.channels % (r * r) != 0 : (x.height % r != 0 || x.width % r != 0)) {
                return invalid(program.name + ": pixel shuffle size");
            }
            y = shuffle(x, r, up);
            break;
        }
        case Op::im2col: y = im2col(x, node.kernel, node.stride); break;
        }
        if (!y) return y.error();
        if (y.value().channels != program.tensor_channels[node.output]) {
            return invalid(program.name + ": node output has the wrong channel count");
        }
        tensors[node.output] = std::move(y.value());
    }
    std::vector<Tensor> outputs;
    for (const auto id : program.outputs) {
        if (!tensors[id]) return invalid(program.name + ": output not produced");
        outputs.push_back(*tensors[id]);
    }
    return outputs;
}

}  // namespace nvcr::deterministic::fx
