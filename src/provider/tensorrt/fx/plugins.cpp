// TensorRT plugins implementing nvcr.fx.v1 (docs/spec/numerics-fx-v1.md) for the
// TensorRT provider: NvcrFxGemm and NvcrFxDepthwise run Triton kernels compiled
// ahead of time; NvcrFxOp runs the CUDA elementwise and data-movement kernels.
// Each plugin serializes its integer parameters into the plan. A kernel that
// meets a failure condition sets the device failure word (ops.hpp).

#include "plugins.hpp"

#include "../../../deterministic/fx_program.hpp"
#include "kernel_table.hpp"
#include "ops.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nvcr::tensorrt::fx {
namespace {

using namespace nvinfer1;
using deterministic::fx::Op;

constexpr char plugin_version[] = "1";
constexpr char plugin_namespace[] = "";
constexpr std::int32_t gemm_limit = (1 << 29) - 1;

std::size_t field_bytes(PluginFieldType type) {
    switch (type) {
    case PluginFieldType::kINT8: return 1;
    case PluginFieldType::kINT16: return 2;
    case PluginFieldType::kINT32: return 4;
    case PluginFieldType::kINT64: return 8;
    default: return 0;
    }
}

struct Field final {
    PluginFieldType type{};
    std::int32_t length{};
    std::vector<unsigned char> data;
};

// Fields shared by a plugin and its clones, with per-device copies of the array fields.
struct Fields final {
    std::map<std::string, Field> values;
    std::mutex mutex;
    std::map<std::pair<int, std::string>, void*> device;

    ~Fields() {
        for (auto& entry : device) cudaFree(entry.second);
    }

    std::int64_t integer(const std::string& name, std::int64_t fallback = 0) const {
        const auto found = values.find(name);
        if (found == values.end() || found->second.data.empty()) return fallback;
        const auto& f = found->second;
        std::int64_t v = 0;
        switch (f.type) {
        case PluginFieldType::kINT8: v = *reinterpret_cast<const std::int8_t*>(f.data.data()); break;
        case PluginFieldType::kINT16: v = *reinterpret_cast<const std::int16_t*>(f.data.data()); break;
        case PluginFieldType::kINT32: v = *reinterpret_cast<const std::int32_t*>(f.data.data()); break;
        case PluginFieldType::kINT64: v = *reinterpret_cast<const std::int64_t*>(f.data.data()); break;
        default: return fallback;
        }
        return v;
    }

    std::int32_t length(const std::string& name) const {
        const auto found = values.find(name);
        return found == values.end() ? 0 : found->second.length;
    }

    void* on_device(const std::string& name) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        const auto key = std::make_pair(dev, name);
        if (auto found = device.find(key); found != device.end()) return found->second;
        const auto field = values.find(name);
        if (field == values.end() || field->second.data.empty()) return nullptr;
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, field->second.data.size()) != cudaSuccess) return nullptr;
        // TensorRT may run layers on other streams: finish the copy before any kernel can read it.
        if (cudaMemcpy(pointer, field->second.data.data(), field->second.data.size(), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(pointer);
            return nullptr;
        }
        device[key] = pointer;
        return pointer;
    }

    bool upload_all() {
        for (const auto& entry : values) {
            if (entry.second.length > 1 && on_device(entry.first) == nullptr) return false;
        }
        return true;
    }
};

std::shared_ptr<Fields> read_fields(const PluginFieldCollection* fc) {
    auto fields = std::make_shared<Fields>();
    for (std::int32_t index = 0; fc != nullptr && index < fc->nbFields; ++index) {
        const auto& f = fc->fields[index];
        Field copy{f.type, f.length, {}};
        const auto bytes = field_bytes(f.type) * static_cast<std::size_t>(f.length);
        if (bytes > 0 && f.data != nullptr) {
            const auto* begin = static_cast<const unsigned char*>(f.data);
            copy.data.assign(begin, begin + bytes);
        }
        fields->values[f.name] = std::move(copy);
    }
    return fields;
}

std::int64_t volume(Dims const& d, int from = 0) {
    std::int64_t v = 1;
    for (int index = from; index < d.nbDims; ++index) v *= d.d[index];
    return v;
}

Shape shape_of(Dims const& d) {
    return {static_cast<int>(d.d[1]), static_cast<int>(d.d[2]), static_cast<int>(d.d[3])};
}

class Plugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    Plugin(std::string name, std::shared_ptr<Fields> fields) : name_(std::move(name)), fields_(std::move(fields)) {
        for (const auto& [key, value] : fields_->values) {
            serialize_.emplace_back(key.c_str(), value.data.empty() ? nullptr : value.data.data(), value.type,
                                    value.length);
        }
        collection_.nbFields = static_cast<std::int32_t>(serialize_.size());
        collection_.fields = serialize_.data();
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    AsciiChar const* getPluginVersion() const noexcept override { return plugin_version; }
    AsciiChar const* getPluginNamespace() const noexcept override { return plugin_namespace; }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) noexcept override {
        return 0;
    }
    int32_t getOutputDataTypes(DataType* types, int32_t nbOutputs, DataType const*, int32_t) const noexcept override {
        for (int32_t index = 0; index < nbOutputs; ++index) types[index] = DataType::kINT32;
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t,
                                   int32_t) noexcept override {
        return inOut[pos].desc.type == DataType::kINT32 && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override {
        return 64;  // int32 maxima the GEMM and depthwise kernels report
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
        if (!fields_->upload_all()) {
            report("cannot upload the layer's parameters to the device");
            return nullptr;
        }
        return clone();
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &collection_; }

    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        std::uint32_t* word = nullptr;
        if (failure_word(&word) != cudaSuccess) return report("failure word unavailable");
        return run(in, out, inputs, outputs, static_cast<std::int32_t*>(workspace), word, stream);
    }

protected:
    virtual int run(PluginTensorDesc const* in, PluginTensorDesc const* out, void const* const* inputs,
                    void* const* outputs, std::int32_t* amax, std::uint32_t* word, cudaStream_t stream) = 0;

    int report(const std::string& what) const {
        std::fprintf(stderr, "nvcr.fx.v1 %s: %s\n", name_.c_str(), what.c_str());
        return 1;
    }

    int cuda(cudaError_t status) const { return status == cudaSuccess ? 0 : report(cudaGetErrorString(status)); }

    // Loads and launches an ahead-of-time kernel by key.
    int launch_kernel(const std::string& key, unsigned gx, unsigned gy, unsigned gz, cudaStream_t stream,
                      void** args, int nargs) const {
        Kernel kernel{};
        if (const auto status = load_kernel(key, kernel); status != CUDA_SUCCESS) {
            return report("kernel " + key + ": " + kernel_error(status));
        }
        if (const auto status = launch(kernel, gx, gy, gz, stream, args, nargs); status != CUDA_SUCCESS) {
            return report("launch " + key + ": " + kernel_error(status));
        }
        return 0;
    }

    std::string name_;
    std::shared_ptr<Fields> fields_;

private:
    std::vector<PluginField> serialize_;
    PluginFieldCollection collection_{};
};

// ---- gemm: spec 4.1 ------------------------------------------------------------------------------------------------
class GemmPlugin final : public Plugin {
public:
    using Plugin::Plugin;
    AsciiChar const* getPluginName() const noexcept override { return "NvcrFxGemm"; }
    IPluginV3* clone() noexcept override { return new GemmPlugin(name_, fields_); }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t,
                            IExprBuilder& b) noexcept override {
        outputs[0] = inputs[0];
        outputs[0].d[1] = b.constant(fields_->integer("O") / fields_->integer("g", 1));
        return 0;
    }

protected:
    int run(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
            std::int32_t* amax, std::uint32_t* word, cudaStream_t stream) override {
        const auto& f = *fields_;
        std::int32_t O = static_cast<std::int32_t>(f.integer("O"));
        std::int32_t K = static_cast<std::int32_t>(in[0].dims.d[1]);
        std::int32_t N = static_cast<std::int32_t>(volume(in[0].dims, 2));
        std::int32_t stride = O * K;
        std::int32_t nlut = static_cast<std::int32_t>(f.integer("h"));
        const bool ws = f.integer("wsilu") != 0;
        if (K != f.integer("K")) return report("input channels do not match the weights");
        char key[200];
        std::snprintf(key, sizeof(key), "gemm_g%d_r%d_ws%d_wsh%d_xs%d_bm%d_bn%d_bk%d_w%d_s%d_ho%d_hk%d_hs%d_hl%d_hn%d",
                      static_cast<int>(f.integer("g", 1)), static_cast<int>(f.integer("nres")), ws ? 1 : 0,
                      ws ? static_cast<int>(f.integer("wsh")) : 0, static_cast<int>(f.integer("xs")),
                      static_cast<int>(f.integer("bm")), static_cast<int>(f.integer("bn")),
                      static_cast<int>(f.integer("bk")), static_cast<int>(f.integer("warps")),
                      static_cast<int>(f.integer("stages")), O % 16 == 0, K % 16 == 0, stride % 16 == 0,
                      ws && nlut % 16 == 0 && nlut != 0, N % 16 == 0);
        void* wd = fields_->on_device("wd");
        void* clo = fields_->on_device("clo");
        void* chi = fields_->on_device("chi");
        void* s = fields_->on_device("s");
        void* lut = ws ? fields_->on_device("dc") : clo;
        if (!wd || !clo || !chi || !s || !lut) return report("parameters unavailable on the device");
        void* x = const_cast<void*>(inputs[0]);
        void* out = outputs[0];
        const auto nres = f.integer("nres");
        void* r1 = nres > 0 ? const_cast<void*>(inputs[1]) : out;
        void* r2 = nres > 1 ? const_cast<void*>(inputs[2]) : out;
        if (cudaMemsetAsync(amax, 0, 16, stream) != cudaSuccess) return report("cannot clear the maxima");
        void* args[] = {&wd, &x, &clo, &chi, &s, &lut, &r1, &r2, &out, &amax, &O, &K, &N, &stride, &nlut};
        const auto bm = static_cast<unsigned>(f.integer("bm"));
        const auto bn = static_cast<unsigned>(f.integer("bn"));
        if (const int status = launch_kernel(key, (static_cast<unsigned>(O) + bm - 1) / bm,
                                             (static_cast<unsigned>(N) + bn - 1) / bn, 1, stream, args, 15)) {
            return status;
        }
        if (const int status = cuda(check_max(amax, 1, 1 << 30, word, stream))) return status;
        return cuda(check_max(amax, 2, gemm_limit - 1, word, stream));
    }
};

// ---- depthwise3x3: spec 4.2 ----------------------------------------------------------------------------------------
class DepthwisePlugin final : public Plugin {
public:
    using Plugin::Plugin;
    AsciiChar const* getPluginName() const noexcept override { return "NvcrFxDepthwise"; }
    IPluginV3* clone() noexcept override { return new DepthwisePlugin(name_, fields_); }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t,
                            IExprBuilder&) noexcept override {
        outputs[0] = inputs[0];
        return 0;
    }

protected:
    int run(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
            std::int32_t* amax, std::uint32_t* word, cudaStream_t stream) override {
        const auto C = static_cast<unsigned>(in[0].dims.d[1]);
        std::int32_t H = static_cast<std::int32_t>(in[0].dims.d[2]);
        std::int32_t W = static_cast<std::int32_t>(in[0].dims.d[3]);
        char key[96];
        std::snprintf(key, sizeof(key), "dw_th4_bw128_w4_hh%d_hw%d", H % 16 == 0, W % 16 == 0);
        void* w = fields_->on_device("w");
        void* bias = fields_->on_device("b");
        void* s = fields_->on_device("s");
        if (!w || !bias || !s) return report("parameters unavailable on the device");
        void* x = const_cast<void*>(inputs[0]);
        void* out = outputs[0];
        if (cudaMemsetAsync(amax, 0, 8, stream) != cudaSuccess) return report("cannot clear the maxima");
        void* args[] = {&x, &w, &bias, &s, &out, &amax, &H, &W};
        if (const int status = launch_kernel(key, C, (static_cast<unsigned>(H) + 3) / 4,
                                             (static_cast<unsigned>(W) + 127) / 128, stream, args, 8)) {
            return status;
        }
        return cuda(check_max(amax, 1, 1 << 30, word, stream));
    }
};

// ---- the other ops: spec 4.3-4.7 -----------------------------------------------------------------------------------
class OpPlugin final : public Plugin {
public:
    using Plugin::Plugin;
    AsciiChar const* getPluginName() const noexcept override { return "NvcrFxOp"; }
    IPluginV3* clone() noexcept override { return new OpPlugin(name_, fields_); }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t,
                            IExprBuilder& b) noexcept override {
        outputs[0] = inputs[0];
        auto& o = outputs[0];
        const auto r = static_cast<std::int32_t>(fields_->integer("factor", 1));
        const auto k = static_cast<std::int32_t>(fields_->integer("kernel", 3));
        const auto s = static_cast<std::int32_t>(fields_->integer("stride", 1));
        const auto pad = k == 3 ? 1 : 0;
        const auto scale = [&](IDimensionExpr const* d, DimensionOperation op, std::int32_t v) {
            return b.operation(op, *d, *b.constant(v));
        };
        switch (static_cast<Op>(fields_->integer("op"))) {
        case Op::param_row:
            o.d[1] = b.constant(fields_->integer("channels"));
            o.d[2] = b.constant(1);
            o.d[3] = b.constant(1);
            break;
        case Op::crop_like:
            o.d[2] = inputs[1].d[2];
            o.d[3] = inputs[1].d[3];
            break;
        case Op::pixel_shuffle:
            o.d[1] = scale(inputs[0].d[1], DimensionOperation::kFLOOR_DIV, r * r);
            o.d[2] = scale(inputs[0].d[2], DimensionOperation::kPROD, r);
            o.d[3] = scale(inputs[0].d[3], DimensionOperation::kPROD, r);
            break;
        case Op::pixel_unshuffle:
            o.d[1] = scale(inputs[0].d[1], DimensionOperation::kPROD, r * r);
            o.d[2] = scale(inputs[0].d[2], DimensionOperation::kFLOOR_DIV, r);
            o.d[3] = scale(inputs[0].d[3], DimensionOperation::kFLOOR_DIV, r);
            break;
        case Op::im2col:
            o.d[1] = scale(inputs[0].d[1], DimensionOperation::kPROD, k * k);
            for (int axis = 2; axis < 4; ++axis) {
                const auto* padded = scale(inputs[0].d[axis], DimensionOperation::kSUM, 2 * pad - k);
                o.d[axis] = scale(scale(padded, DimensionOperation::kFLOOR_DIV, s), DimensionOperation::kSUM, 1);
            }
            break;
        default:
            break;
        }
        return 0;
    }

protected:
    int run(PluginTensorDesc const* in, PluginTensorDesc const* out_desc, void const* const* inputs,
            void* const* outputs, std::int32_t*, std::uint32_t* word, cudaStream_t stream) override {
        const auto& f = *fields_;
        const auto* x = static_cast<const std::int32_t*>(inputs[0]);
        auto* out = static_cast<std::int32_t*>(outputs[0]);
        const auto count = volume(in[0].dims);
        const auto shift = static_cast<int>(f.integer("shift"));
        switch (static_cast<Op>(f.integer("op"))) {
        case Op::mul:
            return cuda(mul(x, static_cast<const std::int32_t*>(inputs[1]), out, shape_of(in[0].dims),
                            shape_of(in[1].dims), shift, word, stream));
        case Op::affine:
            return cuda(affine(x, out, count, f.integer("alpha"), f.integer("beta"), shift, word, stream));
        case Op::add:
            return cuda(add(x, static_cast<const std::int32_t*>(inputs[1]), out, count, word, stream));
        case Op::sat_table: {
            const auto* table = static_cast<const std::int32_t*>(fields_->on_device("table"));
            if (table == nullptr) return report("table unavailable on the device");
            return cuda(sat_table(x, out, count, table, f.length("table") / 2,
                                  static_cast<std::int32_t>(f.integer("top")), stream));
        }
        case Op::delta: {
            const auto* dc = static_cast<const std::int16_t*>(fields_->on_device("dc"));
            if (dc == nullptr) return report("table unavailable on the device");
            return cuda(delta(x, out, count, dc, f.integer("h"), static_cast<int>(f.integer("wsh")), stream));
        }
        case Op::clamp:
            return cuda(clamp(x, out, count, static_cast<std::int32_t>(f.integer("lo")),
                              static_cast<std::int32_t>(f.integer("hi")), stream));
        case Op::param_row: {
            const auto* rows = static_cast<const std::int64_t*>(fields_->on_device("rows"));
            const auto channels = static_cast<int>(f.integer("channels"));
            if (rows == nullptr) return report("rows unavailable on the device");
            return cuda(param_row(x, rows, out, channels, f.length("rows") / channels, word, stream));
        }
        case Op::crop_like:
            return cuda(crop(x, out, shape_of(in[0].dims), static_cast<int>(out_desc[0].dims.d[2]),
                             static_cast<int>(out_desc[0].dims.d[3]), stream));
        case Op::pixel_shuffle:
        case Op::pixel_unshuffle:
            return cuda(shuffle(x, out, shape_of(in[0].dims), static_cast<int>(f.integer("factor")),
                                static_cast<Op>(f.integer("op")) == Op::pixel_shuffle, stream));
        case Op::im2col:
            return cuda(im2col(x, out, shape_of(in[0].dims), static_cast<int>(f.integer("kernel")),
                               static_cast<int>(f.integer("stride")), stream));
        default:
            return report("op is not a plugin op");
        }
    }
};

template <typename T>
class Creator final : public IPluginCreatorV3One {
public:
    explicit Creator(const char* name) : name_(name) {}
    AsciiChar const* getPluginName() const noexcept override { return name_; }
    AsciiChar const* getPluginVersion() const noexcept override { return plugin_version; }
    AsciiChar const* getPluginNamespace() const noexcept override { return plugin_namespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &empty_; }
    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        return new T(name, read_fields(fc));
    }

private:
    const char* name_;
    PluginFieldCollection empty_{};
};

Creator<GemmPlugin> gemm_creator("NvcrFxGemm");
Creator<DepthwisePlugin> depthwise_creator("NvcrFxDepthwise");
Creator<OpPlugin> op_creator("NvcrFxOp");

}  // namespace

bool register_plugins() {
    static const bool registered = [] {
        auto* registry = getPluginRegistry();
        bool ok = registry != nullptr;
        for (IPluginCreatorInterface* creator :
             {static_cast<IPluginCreatorInterface*>(&gemm_creator), static_cast<IPluginCreatorInterface*>(&depthwise_creator),
              static_cast<IPluginCreatorInterface*>(&op_creator)}) {
            ok = ok && registry->registerCreator(*creator, plugin_namespace);
        }
        return ok;
    }();
    return registered;
}

bool kernels_available(int major, int minor) noexcept {
    return kernel_entry_count > 0 && std::string(kernel_target) == "cuda:" + std::to_string(major * 10 + minor);
}

}  // namespace nvcr::tensorrt::fx
