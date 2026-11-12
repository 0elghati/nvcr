#include "kernel_table.hpp"

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace nvcr::tensorrt::fx {
namespace {

std::mutex cache_mutex;
std::map<std::tuple<CUcontext, std::string>, Kernel> cache;
std::map<std::tuple<CUcontext, std::string>, CUmodule> modules;

}  // namespace

CUresult load_kernel(std::string_view key, Kernel& out) {
    CUcontext context{};
    CUresult status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS) return status;
    if (context == nullptr) {
        // The runtime API (TensorRT, cudaMalloc) uses the primary context; make it current for this thread.
        CUdevice device{};
        status = cuCtxGetDevice(&device);
        if (status != CUDA_SUCCESS) {
            status = cuDeviceGet(&device, 0);
            if (status != CUDA_SUCCESS) return status;
        }
        status = cuDevicePrimaryCtxRetain(&context, device);
        if (status != CUDA_SUCCESS) return status;
        status = cuCtxSetCurrent(context);
        if (status != CUDA_SUCCESS) return status;
    }
    const auto id = std::make_tuple(context, std::string(key));
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (auto found = cache.find(id); found != cache.end()) {
        out = found->second;
        return CUDA_SUCCESS;
    }
    const KernelEntry* entry = nullptr;
    for (std::size_t index = 0; index < kernel_entry_count; ++index) {
        if (key == kernel_entries[index].key) {
            entry = &kernel_entries[index];
            break;
        }
    }
    if (entry == nullptr) return CUDA_ERROR_NOT_FOUND;
    CUmodule module{};
    status = cuModuleLoadData(&module, entry->cubin);
    if (status != CUDA_SUCCESS) return status;
    Kernel kernel{};
    status = cuModuleGetFunction(&kernel.function, module, entry->entry);
    if (status != CUDA_SUCCESS) {
        cuModuleUnload(module);
        return status;
    }
    kernel.shared_bytes = entry->shared_bytes;
    kernel.warps = entry->warps;
    if (kernel.shared_bytes > 49152) {
        CUdevice device{};
        int optin = 0;
        status = cuCtxGetDevice(&device);
        if (status == CUDA_SUCCESS) {
            status = cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);
        }
        if (status == CUDA_SUCCESS && optin < kernel.shared_bytes) status = CUDA_ERROR_INVALID_VALUE;
        if (status == CUDA_SUCCESS) {
            status = cuFuncSetAttribute(kernel.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        kernel.shared_bytes);
        }
        if (status != CUDA_SUCCESS) {
            cuModuleUnload(module);
            return status;
        }
    }
    modules[id] = module;
    cache[id] = kernel;
    out = kernel;
    return CUDA_SUCCESS;
}

CUresult launch(const Kernel& kernel, unsigned grid_x, unsigned grid_y, unsigned grid_z, CUstream stream,
                void** args, int nargs) {
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) return CUDA_SUCCESS;
    CUdeviceptr global_scratch = 0;
    CUdeviceptr profile_scratch = 0;
    std::vector<void*> all(args, args + nargs);
    all.push_back(&global_scratch);
    all.push_back(&profile_scratch);
    return cuLaunchKernel(kernel.function, grid_x, grid_y, grid_z, static_cast<unsigned>(kernel.warps) * 32U, 1, 1,
                          static_cast<unsigned>(kernel.shared_bytes), stream, all.data(), nullptr);
}

std::string kernel_error(CUresult status) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &text);
    return std::string(name ? name : "CUDA_ERROR") + ": " + (text ? text : "unknown");
}

}  // namespace nvcr::tensorrt::fx
